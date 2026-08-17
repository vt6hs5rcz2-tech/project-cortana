"""Milestone 30 Batch 2: data-dir isolation, demo reset, and metadata gate."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.commands import CORE_HELP_TEXT, HELP_TEXT, format_diagnostics
from src.config import (
    CORTANA_OWNED_STORE_DIRNAMES,
    CORTANA_OWNED_STORE_FILENAMES,
    DATA_DIR,
    LOG_DIR,
    PROJECT_ROOT,
    cortana_owned_store_paths,
    data_profile_label,
    get_app_data_dir,
    get_builtin_app_data_dir,
    get_default_app_data_dir,
    get_default_calendar_repository_file_path,
    get_default_document_vault_file_path,
    get_default_evidence_store_dir_path,
    get_default_incident_repository_file_path,
    get_default_memory_file_path,
    get_default_reminder_repository_file_path,
    get_default_study_repository_file_path,
    get_default_tool_control_repository_file_path,
    get_default_tool_process_scratch_dir_path,
    get_default_workflow_repository_file_path,
    is_custom_data_profile,
)
from src.memory_store import JsonMemoryStore
from src.pilot_demo import (
    REFUSE_CONFIRMATION,
    REFUSE_DEFAULT_APP_DATA,
    REFUSE_FILESYSTEM_ROOT,
    REFUSE_HOME_ROOT,
    REFUSE_MISSING_OVERRIDE,
    REFUSE_REPOSITORY_ROOT,
    prepare_pilot_demo,
)
from src.readiness import ReadinessOutcome, evaluate_readiness, format_startup_banner
from src.realtime_metadata_gate import (
    CORTANA_GENERATION_META,
    CORTANA_USER_ITEM_META,
    build_gate_metadata,
    evaluate_metadata_echo,
    fail_metadata_gate,
    format_gate_report,
    format_latency_line,
    run_live_metadata_validation,
    session_update_payload,
)
from src.settings import INVALID_DATA_DIR_MESSAGE, Settings

STORE_GETTERS = (
    get_default_memory_file_path,
    get_default_document_vault_file_path,
    get_default_incident_repository_file_path,
    get_default_evidence_store_dir_path,
    get_default_tool_control_repository_file_path,
    get_default_workflow_repository_file_path,
    get_default_reminder_repository_file_path,
    get_default_calendar_repository_file_path,
    get_default_study_repository_file_path,
    get_default_tool_process_scratch_dir_path,
)


def _settings() -> Settings:
    return Settings(openai_api_key="test-api-key", openai_model="test-model")


def test_diagnostics_reports_default_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORTANA_DATA_DIR", raising=False)
    text = format_diagnostics(_settings())
    assert "Data profile: default" in text
    assert "Realtime metadata gate: PASS" in text
    assert "not measured" not in text
    assert "cortana_user_item_id" not in text
    assert "cortana_generation" not in text
    assert "create-send" not in text
    assert "response-created" not in text
    assert "test-api-key" not in text


def test_diagnostics_metadata_gate_uses_release_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.realtime_metadata_gate import (
        REALTIME_METADATA_GATE_RELEASE_OUTCOME,
        realtime_metadata_gate_diagnostics_line,
    )

    monkeypatch.delenv("CORTANA_DATA_DIR", raising=False)
    assert REALTIME_METADATA_GATE_RELEASE_OUTCOME == "PASS"
    assert realtime_metadata_gate_diagnostics_line() == (
        "Realtime metadata gate: PASS"
    )
    text = format_diagnostics(_settings())
    assert realtime_metadata_gate_diagnostics_line() in text
    assert text.count("Realtime metadata gate:") == 1


def test_unset_data_dir_keeps_builtin_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORTANA_DATA_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Example\AppData\Local")
    assert is_custom_data_profile() is False
    assert data_profile_label() == "default"
    assert get_app_data_dir() == get_builtin_app_data_dir()
    assert get_default_app_data_dir() == get_builtin_app_data_dir()
    path = get_default_memory_file_path()
    assert "ProjectCortana" in path.parts
    assert PROJECT_ROOT not in path.parents


def test_custom_data_dir_moves_all_owned_stores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "pilot-profile"
    monkeypatch.setenv("CORTANA_DATA_DIR", str(profile))
    resolved = profile.resolve()
    assert is_custom_data_profile() is True
    assert data_profile_label() == "custom"
    assert get_app_data_dir() == resolved
    for getter in STORE_GETTERS:
        assert getter().parent == resolved
    owned = cortana_owned_store_paths()
    assert {path.name for path in owned} == set(
        CORTANA_OWNED_STORE_FILENAMES + CORTANA_OWNED_STORE_DIRNAMES
    )
    assert all(path.parent == resolved for path in owned)


def test_custom_profiles_do_not_see_each_other(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    monkeypatch.setenv("CORTANA_DATA_DIR", str(profile_a))
    store_a = JsonMemoryStore(get_default_memory_file_path())
    store_a.add_memory("alpha-only memory")
    assert get_default_memory_file_path().is_relative_to(profile_a.resolve())

    monkeypatch.setenv("CORTANA_DATA_DIR", str(profile_b))
    store_b = JsonMemoryStore(get_default_memory_file_path())
    assert store_b.list_memories() == []
    assert not get_default_memory_file_path().exists()
    assert not (profile_b / "memories.json").exists()
    assert (profile_a / "memories.json").exists()


def test_relative_data_dir_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CORTANA_DATA_DIR", "relative-profile")
    assert get_app_data_dir() == (tmp_path / "relative-profile").resolve()
    assert get_default_memory_file_path() == (
        tmp_path / "relative-profile" / "memories.json"
    ).resolve()


def test_unwritable_custom_data_dir_blocks_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file", encoding="utf-8")
    monkeypatch.setenv("CORTANA_DATA_DIR", str(blocked))
    report = evaluate_readiness(settings=_settings())
    assert report.outcome is ReadinessOutcome.BLOCKED_BY_REQUIRED_CONFIGURATION
    assert "invalid_data_directory" in report.required_issues
    assert report.data_profile == "custom"
    banner = format_startup_banner(report)
    assert INVALID_DATA_DIR_MESSAGE in banner
    assert str(blocked) not in banner


def test_diagnostics_reports_custom_without_full_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "secret-profile-name-xyz"
    profile.mkdir()
    monkeypatch.setenv("CORTANA_DATA_DIR", str(profile))
    text = format_diagnostics(_settings())
    assert "Data profile: custom" in text
    assert "Realtime metadata gate: PASS" in text
    assert "not measured" not in text
    assert str(profile) not in text
    assert "secret-profile-name-xyz" not in text
    assert "test-api-key" not in text
    assert "cortana_user_item_id" not in text
    assert "cortana_generation" not in text


def test_readiness_does_not_write_probe_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "empty-profile"
    profile.mkdir()
    before = {path.name for path in profile.iterdir()}
    monkeypatch.setenv("CORTANA_DATA_DIR", str(profile))
    evaluate_readiness(settings=_settings())
    after = {path.name for path in profile.iterdir()}
    assert before == after


def test_repository_paths_are_not_redirected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CORTANA_DATA_DIR", str(tmp_path / "profile"))
    assert LOG_DIR == PROJECT_ROOT / "logs"
    assert DATA_DIR == PROJECT_ROOT / "data"
    memory_path = get_default_memory_file_path()
    assert memory_path.parent == (tmp_path / "profile").resolve()
    assert memory_path.parent != PROJECT_ROOT
    assert memory_path.parent != PROJECT_ROOT / "data"


def test_demo_reset_refuses_without_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORTANA_DATA_DIR", raising=False)
    result = prepare_pilot_demo(confirm=True)
    assert result.allowed is False
    assert result.performed is False
    assert result.reason == REFUSE_MISSING_OVERRIDE


def test_demo_reset_refuses_default_app_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORTANA_DATA_DIR", str(get_builtin_app_data_dir()))
    result = prepare_pilot_demo(confirm=True)
    assert result.allowed is False
    assert result.performed is False
    assert result.reason == REFUSE_DEFAULT_APP_DATA


def test_demo_reset_refuses_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORTANA_DATA_DIR", str(PROJECT_ROOT))
    result = prepare_pilot_demo(confirm=True)
    assert result.allowed is False
    assert result.performed is False
    assert result.reason == REFUSE_REPOSITORY_ROOT
    assert (PROJECT_ROOT / "main.py").exists()


def test_demo_reset_refuses_filesystem_root(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(PROJECT_ROOT.resolve().anchor)
    monkeypatch.setenv("CORTANA_DATA_DIR", str(root))
    result = prepare_pilot_demo(confirm=True)
    assert result.allowed is False
    assert result.performed is False
    assert result.reason == REFUSE_FILESYSTEM_ROOT


def test_demo_reset_refuses_home_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORTANA_DATA_DIR", str(Path.home()))
    result = prepare_pilot_demo(confirm=True)
    assert result.allowed is False
    assert result.performed is False
    assert result.reason == REFUSE_HOME_ROOT


def test_demo_reset_refuses_without_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "demo"
    profile.mkdir()
    target = profile / "memories.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CORTANA_DATA_DIR", str(profile))
    result = prepare_pilot_demo(confirm=False)
    assert result.allowed is True
    assert result.performed is False
    assert result.reason == REFUSE_CONFIRMATION
    assert target.exists()
    assert "memories.json" in result.store_names


def test_demo_reset_lists_intended_stores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CORTANA_DATA_DIR", str(tmp_path / "demo"))
    result = prepare_pilot_demo(confirm=False)
    for name in CORTANA_OWNED_STORE_FILENAMES + CORTANA_OWNED_STORE_DIRNAMES:
        assert name in result.store_names


def test_demo_reset_deletes_only_known_stores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "demo"
    profile.mkdir()
    (profile / "memories.json").write_text("{}", encoding="utf-8")
    (profile / "reminders.json").write_text("{}", encoding="utf-8")
    evidence = profile / "evidence"
    evidence.mkdir()
    (evidence / "copy.bin").write_text("x", encoding="utf-8")
    sentinel = profile / "unrelated-sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    outside = tmp_path / "source-document.md"
    outside.write_text("do not delete", encoding="utf-8")
    monkeypatch.setenv("CORTANA_DATA_DIR", str(profile))
    result = prepare_pilot_demo(confirm=True)
    assert result.performed is True
    assert not (profile / "memories.json").exists()
    assert not (profile / "reminders.json").exists()
    assert not evidence.exists()
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert outside.exists()
    assert outside.read_text(encoding="utf-8") == "do not delete"


def test_demo_reset_is_idempotent_when_store_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "demo"
    profile.mkdir()
    monkeypatch.setenv("CORTANA_DATA_DIR", str(profile))
    first = prepare_pilot_demo(confirm=True)
    second = prepare_pilot_demo(confirm=True)
    assert first.performed is True
    assert second.performed is True
    assert second.removed_names == ()


def test_help_does_not_advertise_demo_or_live_scripts() -> None:
    combined = CORE_HELP_TEXT + "\n" + HELP_TEXT
    assert "prepare_pilot_demo" not in combined
    assert "validate_realtime_metadata" not in combined
    assert "--confirm-demo-reset" not in combined


def test_metadata_gate_pass_requires_exact_echo() -> None:
    sent = build_gate_metadata()
    result = evaluate_metadata_echo(sent=sent, received=dict(sent))
    assert result.outcome == "PASS"


def test_metadata_gate_mismatch_fails() -> None:
    sent = build_gate_metadata(user_item_id="item-a", generation="1")
    received = build_gate_metadata(user_item_id="item-b", generation="1")
    result = evaluate_metadata_echo(sent=sent, received=received)
    assert result.outcome == "FAIL"
    assert result.reason == "metadata mismatch"


def test_metadata_gate_missing_metadata_fails() -> None:
    sent = build_gate_metadata()
    assert evaluate_metadata_echo(sent=sent, received=None).outcome == "FAIL"
    assert evaluate_metadata_echo(sent=sent, received={"other": "x"}).outcome == "FAIL"


def test_metadata_gate_exceptions_fail() -> None:
    result = fail_metadata_gate("connection error (TimeoutError)")
    assert result.outcome == "FAIL"
    report = format_gate_report(result)
    assert "FAIL" in report
    assert "test-api-key" not in report
    assert "OPENAI_API_KEY" not in report


def test_metadata_gate_timestamps_use_monotonic_values() -> None:
    assert format_latency_line("commit -> create-send", 10.0, 10.25) == (
        "commit -> create-send: 250.0 ms"
    )
    assert format_latency_line("commit -> create-send", None, 10.0) == (
        "commit -> create-send: NOT MEASURED"
    )
    source = Path("src/realtime_metadata_gate.py").read_text(encoding="utf-8")
    assert "time.monotonic" in source


def test_metadata_gate_source_does_not_print_secrets() -> None:
    module_source = Path("src/realtime_metadata_gate.py").read_text(encoding="utf-8")
    script_source = Path("scripts/validate_realtime_metadata.py").read_text(
        encoding="utf-8"
    )
    assert "print(settings.openai_api_key)" not in module_source
    assert "print(os.environ" not in module_source
    assert "print(os.environ" not in script_source
    assert 'if __name__ == "__main__"' in script_source


def test_metadata_gate_import_does_not_run_live_or_write_stores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    profile = tmp_path / "gate-profile"
    profile.mkdir()
    monkeypatch.setenv("CORTANA_DATA_DIR", str(profile))
    importlib.reload(importlib.import_module("src.realtime_metadata_gate"))
    spec = importlib.util.spec_from_file_location(
        "validate_realtime_metadata_import_check",
        Path("scripts/validate_realtime_metadata.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert {path.name for path in profile.iterdir()} == set()


def test_metadata_gate_fake_connection_pass_and_fail() -> None:
    settings = _settings()
    payload = session_update_payload(settings)
    audio = payload["audio"]
    assert isinstance(audio, dict)
    audio_input = audio["input"]
    assert isinstance(audio_input, dict)
    turn_detection = audio_input["turn_detection"]
    assert isinstance(turn_detection, dict)
    assert turn_detection["type"] == "server_vad"
    assert turn_detection["create_response"] is False
    assert turn_detection["interrupt_response"] is True

    sent = build_gate_metadata()
    passing = _fake_client(sent)
    assert run_live_metadata_validation(settings, passing).outcome == "PASS"

    mismatch = _fake_client(
        {CORTANA_USER_ITEM_META: "other", CORTANA_GENERATION_META: "9"}
    )
    assert run_live_metadata_validation(settings, mismatch).outcome == "FAIL"

    class BrokenClient:
        class realtime:
            @staticmethod
            def connect(**_kwargs: object) -> object:
                raise RuntimeError("no network")

    failed = run_live_metadata_validation(settings, BrokenClient())
    assert failed.outcome == "FAIL"
    assert "RuntimeError" in failed.reason


def _fake_client(metadata: dict[str, str]) -> object:
    event = SimpleNamespace(
        type="response.created",
        response=SimpleNamespace(metadata=metadata),
    )

    class Connection:
        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def recv(self, timeout: float | None = None) -> object:
            return event

        class session:
            @staticmethod
            def update(**_kwargs: object) -> None:
                return None

        class response:
            @staticmethod
            def create(**_kwargs: object) -> None:
                return None

    class Client:
        class realtime:
            @staticmethod
            def connect(**_kwargs: object) -> Connection:
                return Connection()

    return Client()
