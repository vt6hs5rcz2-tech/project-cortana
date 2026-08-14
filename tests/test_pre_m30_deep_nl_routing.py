"""Pre-M30 Deep Audit #4: NL operational routing sibling search.

Discovery only. Failures document underspecified conversational text that
creates persistent side effects.
"""

from __future__ import annotations

from pathlib import Path

from src.assistant_orchestrator import UnifiedAssistantOrchestrator
from src.document_retrieval import LexicalDocumentRetriever
from src.document_vault import JsonDocumentVault
from src.incident_repository import JsonIncidentRepository
from src.memory_store import JsonMemoryStore


def _orchestrator(tmp_path: Path) -> tuple[UnifiedAssistantOrchestrator, JsonMemoryStore]:
    store = JsonMemoryStore(tmp_path / "memories.json")
    orchestrator = UnifiedAssistantOrchestrator(
        memory_store=store,
        document_vault=JsonDocumentVault(tmp_path / "documents.json"),
        document_retriever=LexicalDocumentRetriever(),
        incident_repository=JsonIncidentRepository(tmp_path / "incidents.json"),
    )
    return orchestrator, store


def test_deep_f2_known_deictic_phrases_are_blocked(tmp_path: Path) -> None:
    """Control: the current deictic allowlist still rejects exact this/that/it."""
    orchestrator, store = _orchestrator(tmp_path)
    for phrase in (
        "remember this",
        "remember that",
        "remember it",
        "remember this forever",
        "remember that forever",
        "remember it forever",
        "remember this.",
        "remember that!",
    ):
        assert orchestrator.try_handle(phrase) is None, phrase
    assert store.list_memories() == []


def test_deep_f2_remember_this_thing_must_not_persist(tmp_path: Path) -> None:
    """OUTSIDE-F2: 'remember this thing' is too vague to persist."""
    orchestrator, store = _orchestrator(tmp_path)
    result = orchestrator.try_handle("remember this thing")
    assert result is None
    assert store.list_memories() == []


def test_deep_f2_remember_that_fact_must_not_persist(tmp_path: Path) -> None:
    """OUTSIDE-F2: 'remember that fact' is too vague to persist."""
    orchestrator, store = _orchestrator(tmp_path)
    result = orchestrator.try_handle("remember that fact")
    assert result is None
    assert store.list_memories() == []


def test_deep_f2_remember_it_please_must_not_persist(tmp_path: Path) -> None:
    """OUTSIDE-F2: trailing 'please' must not defeat the deictic filter."""
    orchestrator, store = _orchestrator(tmp_path)
    result = orchestrator.try_handle("remember it please")
    assert result is None
    assert store.list_memories() == []


def test_deep_f2_polite_and_duration_suffixes_must_not_persist(tmp_path: Path) -> None:
    """Sibling of OUTSIDE-F2: polite/duration suffixes on deictic payloads."""
    orchestrator, store = _orchestrator(tmp_path)
    phrases = (
        "remember this forever please",
        "remember that forever please",
        "remember this for me",
        "remember that for me",
        "remember it for me",
        "remember this right now",
        "remember that please",
        "remember this please",
        "remember the above",
        "remember the previous thing",
        "remember what I just said",
    )
    leaked: list[str] = []
    for phrase in phrases:
        result = orchestrator.try_handle(phrase)
        if result is not None:
            leaked.append(phrase)
    assert leaked == [], f"deictic/low-content remember phrases persisted: {leaked}"
    assert store.list_memories() == []


def test_deep_explicit_remember_payload_still_persists(tmp_path: Path) -> None:
    """Control: a concrete remember payload remains operational."""
    orchestrator, store = _orchestrator(tmp_path)
    result = orchestrator.try_handle("remember the VPN is 10.0.0.1")
    assert result is not None
    assert result.domain == "memory"
    assert result.action == "write"
    assert [memory.text for memory in store.list_memories()] == ["the VPN is 10.0.0.1"]


def test_deep_nl_pronoun_operations_do_not_execute_privileged_domains(
    tmp_path: Path,
) -> None:
    """Pronoun-heavy operational lookalikes must not become side-effecting."""
    orchestrator, store = _orchestrator(tmp_path)
    phrases = (
        "run that",
        "execute that",
        "approve that",
        "delete that",
        "schedule that",
        "remind me about that",
        "search that",
        "open that",
        "run that please",
        "execute that forever",
        "approve that for me",
        "delete that right now",
    )
    for phrase in phrases:
        result = orchestrator.try_handle(phrase)
        assert result is None or result.action not in {"write", "run", "execute", "delete"}
        if result is not None:
            assert result.domain in {
                "memory",
                "documents",
                "incident",
                "evidence",
                "tool",
                "workflow",
                "analyst",
                "reminder",
                "calendar",
                "study",
            }
            # Guidance-only domains must not persist memories.
            if result.domain != "memory":
                assert store.list_memories() == []
    assert store.list_memories() == []


def test_deep_guidance_phrases_are_exact_and_suffix_sensitive(tmp_path: Path) -> None:
    """Guidance handlers are whole-message exact; suffixes fall through to chat."""
    orchestrator, _store = _orchestrator(tmp_path)
    exact = (
        "run a tool",
        "run the playbook",
        "set a reminder",
        "list reminders",
        "show my calendar",
        "schedule a meeting",
        "help me study",
        "quiz me",
        "search evidence",
    )
    for phrase in exact:
        result = orchestrator.try_handle(phrase)
        assert result is not None, phrase
        assert result.action != "write"
    for phrase in (
        "please run a tool",
        "run a tool please",
        "run the playbook please",
        "set a reminder for me",
        "help me study please",
    ):
        assert orchestrator.try_handle(phrase) is None, phrase


def test_deep_search_documents_for_pronoun_is_operational(tmp_path: Path) -> None:
    """Document search with a pronoun is operational (read-only), not a write."""
    orchestrator, store = _orchestrator(tmp_path)
    result = orchestrator.try_handle("search documents for it")
    assert result is not None
    assert result.domain == "documents"
    assert store.list_memories() == []


def test_deep_case_and_punctuation_do_not_bypass_memory_write_anchor(
    tmp_path: Path,
) -> None:
    """Anchored remember pattern is whole-message; prefix chatter must fall through."""
    orchestrator, store = _orchestrator(tmp_path)
    for phrase in (
        "please remember the VPN is 10.0.0.1",
        "I should remember the VPN is 10.0.0.1",
        "can you remember the VPN is 10.0.0.1",
        "Do you remember this?",
    ):
        assert orchestrator.try_handle(phrase) is None, phrase
    assert store.list_memories() == []
    result = orchestrator.try_handle("REMEMBER the VPN is 10.0.0.1")
    assert result is not None
    assert [memory.text for memory in store.list_memories()] == ["the VPN is 10.0.0.1"]
