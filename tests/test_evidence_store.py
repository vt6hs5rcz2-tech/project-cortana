"""Tests for Milestone 8 local evidence file storage."""

import logging
import os
from hashlib import sha256
from pathlib import Path

import pytest

from src.config import MAX_EVIDENCE_SOURCE_BYTES
from src.evidence_store import (
    EvidenceStoreError,
    LocalEvidenceStore,
    evidence_storage_filename,
)
from src.security_evidence import create_evidence_record
from src.security_custody import create_custody_entry
from tests.security_helpers import evidence_store, incident_repository


def test_streaming_copy_preserves_bytes_and_source(
    tmp_path: Path,
) -> None:
    """Copied evidence bytes should match the source hash and leave source unchanged."""
    source = tmp_path / "source payload.bin"
    payload = b"evidence-bytes-12345"
    source.write_bytes(payload)
    store = evidence_store(tmp_path)
    evidence_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    digest, size, filename = store.copy_from_path(
        str(source),
        evidence_id=evidence_id,
    )

    assert digest == sha256(payload).hexdigest()
    assert size == len(payload)
    assert filename == "source payload.bin"
    assert source.read_bytes() == payload
    stored = store.directory_path / evidence_storage_filename(evidence_id)
    assert stored.read_bytes() == payload
    assert store.verify_stored_hash(evidence_id, digest) == "match"


def test_source_filename_does_not_control_destination(tmp_path: Path) -> None:
    """Internal storage names must come from evidence IDs only."""
    source = tmp_path / "evil..name.txt"
    source.write_text("hello", encoding="utf-8")
    store = evidence_store(tmp_path)
    evidence_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    store.copy_from_path(str(source), evidence_id=evidence_id)

    assert (store.directory_path / "evil..name.txt").exists() is False
    assert (store.directory_path / evidence_storage_filename(evidence_id)).exists()


def test_oversized_evidence_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized evidence sources should be rejected before storage succeeds."""
    monkeypatch.setattr("src.evidence_store.MAX_EVIDENCE_SOURCE_BYTES", 16)
    monkeypatch.setattr("src.config.MAX_EVIDENCE_SOURCE_BYTES", 16)
    source = tmp_path / "big.bin"
    source.write_bytes(b"x" * 64)
    store = evidence_store(tmp_path)

    with pytest.raises(EvidenceStoreError):
        store.copy_from_path(
            str(source),
            evidence_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
        )

    assert list(store.directory_path.glob("*.bin")) == []
    assert list(store.directory_path.glob(".*.partial")) == []


def test_symlink_rejected_when_os_allows(
    tmp_path: Path,
) -> None:
    """Symlinked sources should be rejected, skipping only when OS denies creation."""
    target = tmp_path / "target.txt"
    target.write_text("symlink target content", encoding="utf-8")
    link_path = tmp_path / "link.txt"
    try:
        os.symlink(target, link_path)
    except OSError as error:
        pytest.skip(
            "Operating system refused symlink creation "
            f"({type(error).__name__}). "
            "Enable symlink privileges or Developer Mode to run this coverage."
        )

    store = evidence_store(tmp_path)
    with pytest.raises(EvidenceStoreError) as error_info:
        store.copy_from_path(
            str(link_path),
            evidence_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
        )

    assert "symlink target content" not in error_info.value.user_message
    assert str(link_path) not in error_info.value.user_message


def test_directory_rejected_as_non_regular(tmp_path: Path) -> None:
    """Directories must not be registered as evidence."""
    directory = tmp_path / "folder"
    directory.mkdir()
    store = evidence_store(tmp_path)

    with pytest.raises(EvidenceStoreError) as error_info:
        store.copy_from_path(
            str(directory),
            evidence_id="eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
        )

    assert str(directory) not in error_info.value.user_message


def test_existing_destination_collision(tmp_path: Path) -> None:
    """Existing evidence objects must not be overwritten silently."""
    source = tmp_path / "one.txt"
    source.write_text("one", encoding="utf-8")
    store = evidence_store(tmp_path)
    evidence_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
    store.copy_from_path(str(source), evidence_id=evidence_id)

    second = tmp_path / "two.txt"
    second.write_text("two", encoding="utf-8")
    with pytest.raises(EvidenceStoreError):
        store.copy_from_path(str(second), evidence_id=evidence_id)

    stored = store.directory_path / evidence_storage_filename(evidence_id)
    assert stored.read_text(encoding="utf-8") == "one"


def test_verification_mismatch_and_missing(
    tmp_path: Path,
) -> None:
    """Verification should report mismatch and missing without repairing files."""
    source = tmp_path / "payload.txt"
    source.write_text("payload", encoding="utf-8")
    store = evidence_store(tmp_path)
    evidence_id = "12121212-1212-1212-1212-121212121212"
    digest, _size, _name = store.copy_from_path(str(source), evidence_id=evidence_id)

    stored = store.directory_path / evidence_storage_filename(evidence_id)
    stored.write_text("tampered", encoding="utf-8")
    assert store.verify_stored_hash(evidence_id, digest) == "mismatch"

    stored.unlink()
    assert store.verify_stored_hash(evidence_id, digest) == "missing"


def test_no_path_leakage_in_errors_or_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Evidence errors and logs must not expose absolute paths."""
    missing = tmp_path / "missing-evidence-file.bin"
    store = evidence_store(tmp_path)

    with caplog.at_level(logging.ERROR, logger="ProjectCortana"):
        with pytest.raises(EvidenceStoreError) as error_info:
            store.copy_from_path(
                str(missing),
                evidence_id="34343434-3434-3434-3434-343434343434",
            )

    assert str(missing) not in error_info.value.user_message
    assert str(missing) not in caplog.text
    assert str(store.directory_path) not in error_info.value.user_message


def test_register_flow_records_copied_status(tmp_path: Path) -> None:
    """Repository evidence metadata should claim copied only after a stored copy."""
    source = tmp_path / "note.txt"
    source.write_text("note-body", encoding="utf-8")
    store = evidence_store(tmp_path)
    repo = incident_repository(tmp_path)
    evidence_id = "56565656-5656-5656-5656-565656565656"

    digest, size, filename = store.copy_from_path(str(source), evidence_id=evidence_id)
    evidence = create_evidence_record(
        evidence_id=evidence_id,
        evidence_type="file",
        title="Note",
        description="Body",
        sha256_hash=digest,
        source_size_bytes=size,
        collector="analyst",
        storage_status="copied",
        original_filename=filename,
    )
    registered = create_custody_entry(
        evidence_id=evidence_id,
        action="registered",
        actor="analyst",
        reason="test",
        resulting_hash=digest,
    )
    saved = repo.add_evidence(evidence, [registered])
    assert saved.storage_status == "copied"
    assert MAX_EVIDENCE_SOURCE_BYTES > 0
