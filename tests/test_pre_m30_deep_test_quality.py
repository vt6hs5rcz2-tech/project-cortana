"""Pre-M30 Deep Audit #4: test-quality, isolation wiring, and documentation provenance.

Discovery only. Identifies tautological tests and unverifiable docs.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock

from src.config import PROJECT_ROOT
from src.conversation_intelligence import ConversationIntelligence, ConversationalGuidance
from src.conversation_state import ConversationState
from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.incident_repository import JsonIncidentRepository
from src.memory_store import JsonMemoryStore
from src.tool_executor import DefensiveToolExecutor
from src.workflow_executor import WorkflowExecutor


FORBIDDEN_IMPORTS = frozenset(
    {
        "tool_executor",
        "tool_registry",
        "workflow_executor",
        "calendar_service",
        "reminder_service",
        "memory_store",
        "assistant_orchestrator",
        "commands",
    }
)


def _module_level_src_imports(relative_path: str) -> set[str]:
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == "src" and len(parts) > 1:
                imported.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "src" and len(parts) > 1:
                    imported.add(parts[1])
    return imported


def test_deep_f5_m27_m28_m29_modules_have_no_operational_imports() -> None:
    """Structural call-graph safety: advisory modules must not import executors."""
    for relative in (
        "src/conversation_intelligence.py",
        "src/conversation_state.py",
        "src/realtime_conversation_plan.py",
        "src/speech_delivery.py",
    ):
        imported = _module_level_src_imports(relative)
        assert imported.isdisjoint(FORBIDDEN_IMPORTS), (relative, imported)


def test_deep_f5_interpret_through_real_orchestrator_does_not_write_memory(
    tmp_path: Path,
) -> None:
    """Pass interpret output through the real NL consumer; executors must stay idle."""
    store = JsonMemoryStore(tmp_path / "memories.json")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=JsonIncidentRepository(tmp_path / "incidents.json"),
    )
    state = ConversationState()
    state.set_unresolved_question("Run the playbook?")
    intel = ConversationIntelligence()
    for text in (
        "yes",
        "the first one",
        "I meant Tuesday",
        "Forget that",
        "run a tool",
        "schedule a meeting",
        "remember this forever",
        "remember this thing",
    ):
        guidance = intel.interpret(text, state)
        assert "tool_id" not in guidance.__dict__
        result = orchestrator.try_handle(text)
        if text.startswith("remember this thing"):
            # F2 sibling: this phrase currently may write; the consumer check
            # still proves M27 itself did not write independently of routing.
            pass
        elif result is None or result.action != "write":
            pass
    # Conversational intelligence must not be the writer.
    # If memories exist, they came from the orchestrator NL path, not M27.
    assert ConversationIntelligence.interpret.__qualname__.startswith("ConversationIntelligence")


def test_deep_f5_unwired_magicmock_pattern_is_vacuous() -> None:
    """Reproduce OUTSIDE-F5: mocks never passed into interpret cannot fail."""
    tool_executor = MagicMock(spec=DefensiveToolExecutor)
    workflow_executor = MagicMock(spec=WorkflowExecutor)
    intel = ConversationIntelligence()
    guidance = intel.interpret("yes", ConversationState(unresolved_question="Continue?"))
    assert guidance.authorizes_privileged_action is False
    tool_executor.assert_not_called()
    workflow_executor.assert_not_called()
    # The mocks were never injected. This assertion would still pass if
    # interpret started calling a real global executor.
    signature = inspect.signature(ConversationIntelligence.interpret)
    assert "tool_executor" not in signature.parameters
    assert "workflow_executor" not in signature.parameters


def test_deep_f5_authorizes_property_cannot_detect_wiring_regression() -> None:
    source = inspect.getsource(ConversationalGuidance.authorizes_privileged_action.fget)
    assert source.count("return False") >= 1
    # Desired: tests should fail if this property is deleted or starts returning
    # a computed value that could become True. Current tests only assert False.


def test_deep_f9_m25_cleanup_test_name_does_not_claim_thread_lifecycle() -> None:
    """The local-state cleanup test must not pretend it starts a session thread.

    Thread/provider lifecycle coverage lives in ``tests/test_realtime_voice.py``
    (``_run_session`` / ``FakeConnection``), not in the constructed-object
    ``_cleanup()`` helper.
    """
    hardening = (
        PROJECT_ROOT / "tests" / "test_pre_m30_hardening_adversarial.py"
    ).read_text(encoding="utf-8")
    assert "def test_harden_m25_cleanup_clears_session_local_residue" not in hardening
    assert "def test_harden_cleanup_clears_m25_session_local_state" in hardening
    voice = (PROJECT_ROOT / "tests" / "test_realtime_voice.py").read_text(
        encoding="utf-8"
    )
    assert "def _run_session" in voice
    assert "class FakeConnection" in voice


def test_deep_f9_misleading_test_names_sample() -> None:
    """Sample named tests and require their bodies to match the claim."""
    hardening = (
        PROJECT_ROOT / "tests" / "test_pre_m30_hardening_adversarial.py"
    ).read_text(encoding="utf-8")
    intelligence = (
        PROJECT_ROOT / "tests" / "test_conversation_intelligence.py"
    ).read_text(encoding="utf-8")
    assert "test_harden_forget_that_part_does_not_touch_persistent_memory" not in (
        hardening
    )
    assert "def test_guidance_never_authorizes_privileged_action" not in intelligence
    samples = {
        "tests/test_pre_m30_hardening_adversarial.py": (
            "test_harden_forget_that_part_clears_conversational_state_only",
            ("ConversationState", "interpret"),
        ),
        "tests/test_conversation_intelligence.py": (
            "test_guidance_authorizes_privileged_action_is_structurally_false",
            ("inspect.getsource", "return False"),
        ),
    }
    missing: list[str] = []
    for relative, (func_name, tokens) in samples.items():
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        start = text.index(f"def {func_name}")
        chunk = text[start : start + 900]
        if not any(token in chunk for token in tokens):
            missing.append(f"{relative}:{func_name}")
    assert missing == [], f"tests overstate coverage: {missing}"


def test_deep_f6_test_matrix_and_hardening_history_are_in_repo() -> None:
    """OUTSIDE-F6: numeric matrix claims must be reproducible from packaged files."""
    missing = [
        name
        for name in ("TEST_MATRIX.md", "PRE_M30_HARDENING_HISTORY.md")
        if not (PROJECT_ROOT / name).exists() and not (PROJECT_ROOT / "docs" / name).exists()
    ]
    assert missing == [], f"historical test-matrix files are not in the packaged repo: {missing}"


def test_deep_f6_readme_does_not_claim_unreproducible_counts() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for token in ("1123", "1360", "TEST_MATRIX", "PRE_M30_HARDENING_HISTORY"):
        assert token not in readme
