"""Tests for Milestone 20 secret store."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from keyring.errors import KeyringError, NoKeyringError, PasswordDeleteError

from src.config import PROJECT_ROOT
from src.secret_store import (
    InMemorySecretStore,
    KeyringSecretStore,
    SecretStoreError,
    google_refresh_token_secret_key,
)


def test_inmemory_set_get_delete() -> None:
    store = InMemorySecretStore()
    store.set_secret("k", "v")
    assert store.get_secret("k") == "v"
    store.delete_secret("k")
    assert store.get_secret("k") is None


def test_google_refresh_token_secret_key() -> None:
    assert (
        google_refresh_token_secret_key("abc")
        == "cortana.calendar.google.abc.refresh_token"
    )


def test_keyring_set_get_delete_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    backend: dict[tuple[str, str], str] = {}

    def set_password(service: str, key: str, value: str) -> None:
        backend[(service, key)] = value

    def get_password(service: str, key: str) -> str | None:
        return backend.get((service, key))

    def delete_password(service: str, key: str) -> None:
        if (service, key) not in backend:
            raise PasswordDeleteError("missing")
        del backend[(service, key)]

    monkeypatch.setattr("src.secret_store.keyring.set_password", set_password)
    monkeypatch.setattr("src.secret_store.keyring.get_password", get_password)
    monkeypatch.setattr("src.secret_store.keyring.delete_password", delete_password)

    store = KeyringSecretStore(service_name="TestCortana")
    store.set_secret("token-key", "refresh-value")
    assert store.get_secret("token-key") == "refresh-value"
    store.delete_secret("token-key")
    assert store.get_secret("token-key") is None


def test_delete_nonexistent_secret_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delete_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "src.secret_store.keyring.get_password",
        lambda *_args, **_kwargs: None,
    )

    def delete_password(service: str, key: str) -> None:
        delete_calls.append((service, key))

    monkeypatch.setattr("src.secret_store.keyring.delete_password", delete_password)

    store = KeyringSecretStore(service_name="TestCortana")
    store.delete_secret("missing-key")
    assert delete_calls == []


def test_delete_existing_secret_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend: dict[tuple[str, str], str] = {
        ("TestCortana", "token-key"): "refresh-token-value",
    }

    def get_password(service: str, key: str) -> str | None:
        return backend.get((service, key))

    def delete_password(service: str, key: str) -> None:
        del backend[(service, key)]

    monkeypatch.setattr("src.secret_store.keyring.get_password", get_password)
    monkeypatch.setattr("src.secret_store.keyring.delete_password", delete_password)

    store = KeyringSecretStore(service_name="TestCortana")
    store.delete_secret("token-key")
    assert backend.get(("TestCortana", "token-key")) is None


def test_delete_existing_secret_password_delete_error_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.secret_store.keyring.get_password",
        lambda *_args, **_kwargs: "refresh-token-value",
    )

    def delete_password(*_args: object, **_kwargs: object) -> None:
        raise PasswordDeleteError("backend delete failed")

    monkeypatch.setattr("src.secret_store.keyring.delete_password", delete_password)

    store = KeyringSecretStore(service_name="TestCortana")
    with pytest.raises(SecretStoreError, match="PasswordDeleteError"):
        store.delete_secret("token-key")


def test_delete_existing_secret_other_keyring_error_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.secret_store.keyring.get_password",
        lambda *_args, **_kwargs: "refresh-token-value",
    )

    def delete_password(*_args: object, **_kwargs: object) -> None:
        raise KeyringError("generic backend failure")

    monkeypatch.setattr("src.secret_store.keyring.delete_password", delete_password)

    store = KeyringSecretStore(service_name="TestCortana")
    with pytest.raises(SecretStoreError, match="KeyringError"):
        store.delete_secret("token-key")


def test_delete_precheck_get_password_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get_password(*_args: object, **_kwargs: object) -> str | None:
        raise KeyringError("precheck failed")

    monkeypatch.setattr("src.secret_store.keyring.get_password", get_password)

    store = KeyringSecretStore(service_name="TestCortana")
    with pytest.raises(SecretStoreError, match="KeyringError"):
        store.delete_secret("token-key")


def test_delete_postcheck_still_present_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.secret_store.keyring.get_password",
        lambda *_args, **_kwargs: "refresh-token-value",
    )
    monkeypatch.setattr(
        "src.secret_store.keyring.delete_password",
        lambda *_args, **_kwargs: None,
    )

    store = KeyringSecretStore(service_name="TestCortana")
    with pytest.raises(SecretStoreError, match="still present"):
        store.delete_secret("token-key")


def test_keyring_error_maps_to_secret_store_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise KeyringError("nope")

    monkeypatch.setattr("src.secret_store.keyring.set_password", boom)
    store = KeyringSecretStore()
    with pytest.raises(SecretStoreError):
        store.set_secret("k", "v")


def test_no_keyring_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise NoKeyringError("missing backend")

    monkeypatch.setattr("src.secret_store.keyring.get_password", boom)
    store = KeyringSecretStore()
    with pytest.raises(SecretStoreError):
        store.get_secret("k")


def test_malformed_stored_secret_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.secret_store.keyring.get_password",
        lambda *_args, **_kwargs: "   ",
    )
    store = KeyringSecretStore()
    with pytest.raises(SecretStoreError, match="malformed"):
        store.get_secret("k")


def test_keyring_secret_store_does_not_write_files() -> None:
    source = (PROJECT_ROOT / "src" / "secret_store.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"open", "write_text", "write_bytes", "Path"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                pytest.fail("KeyringSecretStore must not call open()")
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "write_text",
                "write_bytes",
            }:
                pytest.fail("KeyringSecretStore must not write files")
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KeyringSecretStore"
    )
    class_source = ast.get_source_segment(source, class_node) or ""
    assert "open(" not in class_source
    assert "write_text" not in class_source
    assert "write_bytes" not in class_source
    assert "Never falls back to plaintext file storage." in source
    assert "keyring.set_password" in class_source
    _ = forbidden
    _ = Path
