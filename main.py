"""Project Cortana application entry point."""

import logging
from typing import cast

from src.active_memory import ActiveMemoryContext
from src.ai_service import OpenAIClient
from src.config import (
    APP_NAME,
    STUDY_PARTNER_ENABLED,
    VISION_ANALYSIS_ENABLED,
    VERSION,
    get_default_document_vault_file_path,
    get_default_evidence_store_dir_path,
    get_default_incident_repository_file_path,
    get_default_memory_file_path,
    get_default_reminder_repository_file_path,
    get_default_calendar_repository_file_path,
    get_default_study_repository_file_path,
    get_default_tool_control_repository_file_path,
    get_default_workflow_repository_file_path,
)
from src.conversation_loop import run_conversation_loop
from src.document_chunker import DocumentChunker
from src.document_extractor import DefaultTextExtractor
from src.document_knowledge_service import DocumentKnowledgeService
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.evidence_store import LocalEvidenceStore
from src.incident_analysis_audit import InMemoryIncidentAnalysisAuditLog
from src.incident_analysis_repository import InMemoryIncidentAnalysisRepository
from src.incident_repository import JsonIncidentRepository
from src.logger import setup_logging
from src.memory_store import JsonMemoryStore
from src.openai_client import create_openai_client
from src.retrieval_session import RetrievalSession
from src.settings import Settings, load_settings
from src.calendar_commands import create_default_calendar_service
from src.reminder_commands import create_default_reminder_service
from src.study_commands import create_default_study_service
from src.study_service import StudyPartnerService
from src.vision_commands import create_default_vision_services
from src.vision_input import VisualInputLoader
from src.vision_service import VisualAnalysisService
from src.tool_commands import create_default_tool_services
from src.tool_repository import JsonToolControlRepository
from src.workflow_commands import create_default_workflow_services


def initialize_ai(
    logger: logging.Logger,
) -> tuple[Settings, OpenAIClient] | None:
    """Load settings and create the OpenAI client."""
    try:
        settings = load_settings()
    except ValueError as error:
        logger.error("%s", error)
        print(
            "Cortana is not connected to OpenAI yet. "
            "Add your API key to the private .env file."
        )
        return None

    client = cast(
        OpenAIClient,
        create_openai_client(settings),
    )

    return settings, client


def main() -> None:
    """Initialize Project Cortana and run the conversation loop."""
    logger = setup_logging()
    logger.info("Starting %s v%s", APP_NAME, VERSION)

    initialized = initialize_ai(logger)

    if initialized is None:
        return

    settings, client = initialized
    print("Cortana's AI connection is configured.")

    memory_store = JsonMemoryStore(get_default_memory_file_path())
    active_memory_context = ActiveMemoryContext()
    document_vault = JsonDocumentVault(get_default_document_vault_file_path())
    document_extractor = DefaultTextExtractor()
    document_chunker = DocumentChunker()
    document_retriever = LexicalDocumentRetriever(chunker=document_chunker)
    retrieval_session = RetrievalSession()
    incident_repository = JsonIncidentRepository(
        get_default_incident_repository_file_path()
    )
    evidence_store = LocalEvidenceStore(get_default_evidence_store_dir_path())
    tool_repository = JsonToolControlRepository(
        get_default_tool_control_repository_file_path()
    )
    tool_registry, tool_executor = create_default_tool_services(
        tool_repository=tool_repository,
        incident_repository=incident_repository,
    )
    workflow_registry, workflow_run_repository, workflow_executor = (
        create_default_workflow_services(
            tool_registry=tool_registry,
            tool_repository=tool_repository,
            tool_executor=tool_executor,
            incident_repository=incident_repository,
            workflow_repository_file_path=get_default_workflow_repository_file_path(),
        )
    )
    analysis_repository = InMemoryIncidentAnalysisRepository()
    analysis_audit_log = InMemoryIncidentAnalysisAuditLog()
    reminder_service = create_default_reminder_service(
        repository_file_path=get_default_reminder_repository_file_path(),
    )
    calendar_service = create_default_calendar_service(
        repository_file_path=get_default_calendar_repository_file_path(),
        oauth_client_file=settings.google_oauth_client_file,
    )
    study_service: StudyPartnerService | None = None
    if STUDY_PARTNER_ENABLED:
        knowledge_service = DocumentKnowledgeService(
            vault=document_vault,
            retriever=document_retriever,
            retrieval_session=retrieval_session,
            settings=settings,
            client=client,
            chunker=document_chunker,
        )
        study_service = create_default_study_service(
            vault=document_vault,
            knowledge_service=knowledge_service,
            settings=settings,
            client=client,
            repository_file_path=get_default_study_repository_file_path(),
            chunker=document_chunker,
            document_retriever=document_retriever,
            retrieval_session=retrieval_session,
        )

    vision_loader: VisualInputLoader | None = None
    vision_service: VisualAnalysisService | None = None
    if VISION_ANALYSIS_ENABLED:
        vision_loader, vision_service = create_default_vision_services(
            settings=settings,
            client=client,
        )

    run_conversation_loop(
        client=client,
        settings=settings,
        logger=logger,
        memory_store=memory_store,
        active_memory_context=active_memory_context,
        document_vault=document_vault,
        document_extractor=document_extractor,
        document_chunker=document_chunker,
        document_retriever=document_retriever,
        retrieval_session=retrieval_session,
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
    )


if __name__ == "__main__":
    main()
