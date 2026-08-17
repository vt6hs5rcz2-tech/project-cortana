"""Continuous conversation loop for Project Cortana."""

import logging
from collections.abc import Callable

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient, generate_response
from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.commands import CommandOutcome, handle_slash_command, parse_slash_input
from src.config import MAX_CONVERSATION_MESSAGE_CHARS
from src.conversation import (
    STARTUP_GREETING,
    SHUTDOWN_MESSAGE,
    MESSAGE_TOO_LONG,
    ApiInputMessage,
    ConversationHistory,
    is_exit_command,
)
from src.conversation_intelligence import (
    ConversationIntelligence,
    safe_interpret,
)
from src.conversation_state import ConversationState, InteractionMode
from src.speech_delivery import SpeechDeliveryState
from src.document_chunker import DocumentChunker
from src.document_extractor import TextExtractor
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import DocumentVault
from src.evidence_store import EvidenceStore
from src.incident_analysis_audit import InMemoryIncidentAnalysisAuditLog
from src.incident_analysis_repository import InMemoryIncidentAnalysisRepository
from src.incident_repository import IncidentRepository
from src.memory_store import MemoryStore
from src.calendar_service import CalendarService
from src.reminder_service import ReminderService
from src.retrieval_session import RetrievalSession
from src.settings import Settings
from src.study_service import StudyPartnerService
from src.vision_input import VisualInputLoader
from src.vision_service import VisualAnalysisService
from src.voice_input import MicrophoneCaptureAdapter, poll_windows_console_enter_stop
from src.voice_service import VoiceService
from src.tool_executor import DefensiveToolExecutor
from src.tool_registry import ToolRegistry
from src.tool_repository import ToolControlRepository
from src.workflow_executor import WorkflowExecutor
from src.workflow_registry import WorkflowRegistry
from src.workflow_repository import WorkflowRunRepository

BLANK_INPUT_MESSAGE = "Cortana: Please enter a message."
THINKING_MESSAGE = "Cortana: Thinking..."
AI_AUTH_FAILURE = (
    "Cortana: The API key was rejected. Check OPENAI_API_KEY in your .env file."
)
AI_NETWORK_FAILURE = (
    "Cortana: I couldn't reach the AI service. Check your connection and try again."
)
AI_TEMPORARY_FAILURE = (
    "Cortana: The AI service is temporarily unavailable. Try again shortly."
)
AI_GENERIC_FAILURE = "Cortana: I couldn't complete that request."


def default_voice_stop_signal() -> bool:
    """Poll for Enter on the Windows console during a voice-turn listen phase.

    Distinct from the conversation loop ``read_input`` seam so scripted outer
    loop inputs are never consumed as the voice stop signal. Uses non-blocking
    console polling so no stdin waiter can outlive ``capture()``.
    """
    return poll_windows_console_enter_stop()


def end_conversation(*, logger: logging.Logger) -> None:
    """Print the shutdown message and record session termination."""
    print(SHUTDOWN_MESSAGE)
    logger.info("Conversation session ended by user.")


def read_session_input(input_reader: Callable[[], str]) -> str | None:
    """Read one line of user input, or None when the session should end."""
    try:
        return input_reader().strip()
    except (EOFError, KeyboardInterrupt):
        return None


def classify_ai_failure(error: BaseException) -> str:
    """Map a model failure to a bounded user-facing message.

    Classification uses exception type and status_code only. Provider
    message text is never shown.
    """
    status = getattr(error, "status_code", None)
    name = type(error).__name__.lower()

    if status in {401, 403} or "auth" in name or "permission" in name:
        return AI_AUTH_FAILURE
    if status == 429 or "ratelimit" in name:
        return AI_TEMPORARY_FAILURE
    if isinstance(status, int) and status >= 500:
        return AI_TEMPORARY_FAILURE
    if "internalserver" in name:
        return AI_TEMPORARY_FAILURE
    if isinstance(error, (ConnectionError, TimeoutError)):
        return AI_NETWORK_FAILURE
    if "apiconnection" in name or "timeout" in name or "connection" in name:
        return AI_NETWORK_FAILURE
    return AI_GENERIC_FAILURE


def process_conversation_turn(
    *,
    client: OpenAIClient,
    settings: Settings,
    user_message: str,
    logger: logging.Logger,
    conversation_history: ConversationHistory | None = None,
    active_memory_context: ActiveMemoryContext | None = None,
    conversation_state: ConversationState | None = None,
    conversation_intelligence: ConversationIntelligence | None = None,
    interaction_mode: InteractionMode = "text",
    classified_failure: list[str] | None = None,
) -> str | None:
    """Generate one ordinary conversational answer without printing it.

    On success, writes the user message and assistant answer to history when a
    history object is provided. On failure, returns None and leaves history
    unchanged. Oversized user input returns MESSAGE_TOO_LONG without calling
    the model or mutating history or conversational state.

    Milestone 27 conversational intelligence is advisory only. Internal
    failures degrade to the ordinary conversation path.
    """
    if len(user_message.strip()) > MAX_CONVERSATION_MESSAGE_CHARS:
        return MESSAGE_TOO_LONG

    active_memories = (
        active_memory_context.list_active()
        if active_memory_context is not None
        else None
    )
    memory_boundary_token = (
        active_memory_context.boundary_token
        if active_memory_context is not None
        else None
    )

    guidance = None
    context_messages: list[ApiInputMessage] | None = None
    intelligence = conversation_intelligence
    state = conversation_state
    prior_state: ConversationState | None = None
    if state is not None:
        prior_state = state.clone()
        if intelligence is None:
            intelligence = ConversationIntelligence()
        state.set_interaction_mode(interaction_mode)
        guidance = safe_interpret(intelligence, user_message, state)
        if guidance is not None:
            context_messages = [
                {"role": message["role"], "content": message["content"]}
                for message in intelligence.build_context_messages(guidance, state)
            ]

    try:
        answer = generate_response(
            client=client,
            settings=settings,
            user_message=user_message,
            conversation_history=conversation_history,
            active_memories=active_memories,
            memory_boundary_token=memory_boundary_token,
            conversational_context_messages=context_messages,
        )
    except Exception as error:
        if prior_state is not None and state is not None:
            state.restore(prior_state)
        logger.error(
            "The OpenAI request failed with error type: %s",
            type(error).__name__,
        )
        if classified_failure is not None:
            classified_failure.append(classify_ai_failure(error))
        return None

    if conversation_history is not None:
        conversation_history.add_user_message(user_message.strip())
        conversation_history.add_assistant_message(answer)

    if state is not None and intelligence is not None and guidance is not None:
        try:
            intelligence.observe_assistant_reply(answer, state, guidance)
        except Exception:
            logger.error(
                "Conversational intelligence observe failed error_type=Exception"
            )

    logger.info("Response completed.")
    return answer


def handle_message(
    *,
    client: OpenAIClient,
    settings: Settings,
    user_message: str,
    logger: logging.Logger,
    conversation_history: ConversationHistory | None = None,
    active_memory_context: ActiveMemoryContext | None = None,
    conversation_state: ConversationState | None = None,
    conversation_intelligence: ConversationIntelligence | None = None,
) -> None:
    """Generate and display one Cortana response."""
    if len(user_message.strip()) > MAX_CONVERSATION_MESSAGE_CHARS:
        print(MESSAGE_TOO_LONG)
        return
    print(THINKING_MESSAGE)
    failures: list[str] = []
    answer = process_conversation_turn(
        client=client,
        settings=settings,
        user_message=user_message,
        logger=logger,
        conversation_history=conversation_history,
        active_memory_context=active_memory_context,
        conversation_state=conversation_state,
        conversation_intelligence=conversation_intelligence,
        classified_failure=failures,
    )
    if answer is None:
        print(failures[0] if failures else AI_GENERIC_FAILURE)
        return
    if answer == MESSAGE_TOO_LONG:
        print(MESSAGE_TOO_LONG)
        return

    print(f"Cortana: {answer}")


def run_conversation_loop(
    *,
    client: OpenAIClient,
    settings: Settings,
    logger: logging.Logger,
    memory_store: MemoryStore,
    active_memory_context: ActiveMemoryContext,
    document_vault: DocumentVault,
    document_extractor: TextExtractor,
    document_chunker: DocumentChunker | None = None,
    document_retriever: LexicalDocumentRetriever | None = None,
    retrieval_session: RetrievalSession | None = None,
    incident_repository: IncidentRepository | None = None,
    evidence_store: EvidenceStore | None = None,
    tool_registry: ToolRegistry | None = None,
    tool_repository: ToolControlRepository | None = None,
    tool_executor: DefensiveToolExecutor | None = None,
    workflow_registry: WorkflowRegistry | None = None,
    workflow_run_repository: WorkflowRunRepository | None = None,
    workflow_executor: WorkflowExecutor | None = None,
    analysis_repository: InMemoryIncidentAnalysisRepository | None = None,
    analysis_audit_log: InMemoryIncidentAnalysisAuditLog | None = None,
    reminder_service: ReminderService | None = None,
    calendar_service: CalendarService | None = None,
    study_service: StudyPartnerService | None = None,
    vision_loader: VisualInputLoader | None = None,
    vision_service: VisualAnalysisService | None = None,
    voice_capture: MicrophoneCaptureAdapter | None = None,
    voice_service: VoiceService | None = None,
    read_input: Callable[[], str] | None = None,
    stop_signal: Callable[[], bool] | None = None,
    conversation_history: ConversationHistory | None = None,
    conversation_state: ConversationState | None = None,
    conversation_intelligence: ConversationIntelligence | None = None,
    speech_delivery_state: SpeechDeliveryState | None = None,
) -> None:
    """Run the interactive conversation until the user exits."""
    input_reader = read_input or (lambda: input("You: "))
    voice_stop_signal = stop_signal or default_voice_stop_signal
    history = conversation_history or ConversationHistory()
    state = conversation_state if conversation_state is not None else ConversationState()
    delivery_state = (
        speech_delivery_state
        if speech_delivery_state is not None
        else SpeechDeliveryState()
    )
    intelligence = (
        conversation_intelligence
        if conversation_intelligence is not None
        else ConversationIntelligence()
    )
    chunker = document_chunker or DocumentChunker()
    retriever = document_retriever or LexicalDocumentRetriever(chunker=chunker)
    session = retrieval_session or RetrievalSession()
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=memory_store,
        document_vault=document_vault,
        document_retriever=retriever,
        incident_repository=incident_repository,
    )

    print(STARTUP_GREETING)

    while True:
        user_message = read_session_input(input_reader)

        if user_message is None:
            end_conversation(logger=logger)
            return

        if not user_message:
            print(BLANK_INPUT_MESSAGE)
            continue

        if is_exit_command(user_message):
            end_conversation(logger=logger)
            return

        if parse_slash_input(user_message) is not None:
            command_result = handle_slash_command(
                user_message,
                settings=settings,
                conversation_history=history,
                memory_store=memory_store,
                active_memory_context=active_memory_context,
                document_vault=document_vault,
                document_extractor=document_extractor,
                document_retriever=retriever,
                retrieval_session=session,
                incident_repository=incident_repository,
                evidence_store=evidence_store,
                tool_registry=tool_registry,
                tool_repository=tool_repository,
                tool_executor=tool_executor,
                workflow_registry=workflow_registry,
                workflow_run_repository=workflow_run_repository,
                workflow_executor=workflow_executor,
                analysis_repository=analysis_repository,
                analysis_audit_log=analysis_audit_log,
                reminder_service=reminder_service,
                calendar_service=calendar_service,
                study_service=study_service,
                vision_loader=vision_loader,
                vision_service=vision_service,
                voice_capture=voice_capture,
                voice_service=voice_service,
                stop_signal=voice_stop_signal,
                client=client,
                conversation_state=state,
                speech_delivery_state=delivery_state,
            )
            if command_result.message:
                print(command_result.message)
            if command_result.outcome == CommandOutcome.EXIT:
                end_conversation(logger=logger)
                return
            continue

        orchestration_result = orchestrator.try_handle(user_message)
        if orchestration_result is not None:
            print(orchestration_result.safe_user_message)
            continue

        handle_message(
            client=client,
            settings=settings,
            user_message=user_message,
            logger=logger,
            conversation_history=history,
            active_memory_context=active_memory_context,
            conversation_state=state,
            conversation_intelligence=intelligence,
        )
