"""OS credential-store abstraction for Milestone 20 secrets.

Stores only opaque secret strings (for calendar: Google refresh tokens).
Never falls back to plaintext file storage.
"""

from __future__ import annotations

import logging
from typing import Protocol

import keyring
from keyring.errors import KeyringError

from src.config import SECRET_STORE_SERVICE_NAME

logger = logging.getLogger("ProjectCortana")


class SecretStoreError(RuntimeError):
    """Raised when the credential store cannot complete an operation safely."""

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or (
            "Cortana: Secure credential storage is unavailable."
        )


class SecretStore(Protocol):
    """Persistence boundary for opaque secret strings."""

    def set_secret(self, key: str, value: str) -> None:
        """Store or replace one secret value for the given key."""

    def get_secret(self, key: str) -> str | None:
        """Return the secret for the key, or None when absent."""

    def delete_secret(self, key: str) -> None:
        """Delete the secret for the key. Absent keys succeed silently."""


class InMemorySecretStore:
    """Test-only secret store kept entirely in process memory."""

    def __init__(self) -> None:
        self._secrets: dict[str, str] = {}

    def set_secret(self, key: str, value: str) -> None:
        cleaned_key = _require_key(key)
        cleaned_value = _require_value(value)
        self._secrets[cleaned_key] = cleaned_value

    def get_secret(self, key: str) -> str | None:
        cleaned_key = _require_key(key)
        return self._secrets.get(cleaned_key)

    def delete_secret(self, key: str) -> None:
        cleaned_key = _require_key(key)
        self._secrets.pop(cleaned_key, None)


class KeyringSecretStore:
    """Production secret store backed by the OS keyring / Credential Manager."""

    def __init__(self, *, service_name: str = SECRET_STORE_SERVICE_NAME) -> None:
        cleaned = service_name.strip()
        if not cleaned:
            raise SecretStoreError("Secret store service name cannot be blank.")
        self._service_name = cleaned

    def set_secret(self, key: str, value: str) -> None:
        cleaned_key = _require_key(key)
        cleaned_value = _require_value(value)
        try:
            keyring.set_password(self._service_name, cleaned_key, cleaned_value)
        except KeyringError as error:
            logger.error(
                "SecretStore set failed error_type=%s",
                type(error).__name__,
            )
            raise SecretStoreError(
                f"SecretStore set failed: {type(error).__name__}",
            ) from error

    def get_secret(self, key: str) -> str | None:
        cleaned_key = _require_key(key)
        try:
            value = keyring.get_password(self._service_name, cleaned_key)
        except KeyringError as error:
            logger.error(
                "SecretStore get failed error_type=%s",
                type(error).__name__,
            )
            raise SecretStoreError(
                f"SecretStore get failed: {type(error).__name__}",
            ) from error
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise SecretStoreError(
                "Stored secret is malformed.",
                user_message=(
                    "Cortana: Stored calendar credentials are invalid. "
                    "Reconnect with /calendar-connect."
                ),
            )
        return value

    def delete_secret(self, key: str) -> None:
        cleaned_key = _require_key(key)
        try:
            existing = keyring.get_password(self._service_name, cleaned_key)
        except KeyringError as error:
            logger.error(
                "SecretStore delete precheck failed error_type=%s",
                type(error).__name__,
            )
            raise SecretStoreError(
                f"SecretStore delete failed: {type(error).__name__}",
            ) from error

        if existing is None:
            return
        if not isinstance(existing, str) or not existing.strip():
            raise SecretStoreError(
                "Stored secret is malformed.",
                user_message=(
                    "Cortana: Stored calendar credentials are invalid. "
                    "Reconnect with /calendar-connect."
                ),
            )

        try:
            keyring.delete_password(self._service_name, cleaned_key)
        except KeyringError as error:
            logger.error(
                "SecretStore delete failed error_type=%s",
                type(error).__name__,
            )
            raise SecretStoreError(
                f"SecretStore delete failed: {type(error).__name__}",
            ) from error

        try:
            remaining = keyring.get_password(self._service_name, cleaned_key)
        except KeyringError as error:
            logger.error(
                "SecretStore delete verification failed error_type=%s",
                type(error).__name__,
            )
            raise SecretStoreError(
                f"SecretStore delete failed: {type(error).__name__}",
            ) from error
        if remaining is not None:
            raise SecretStoreError(
                "SecretStore delete verification failed: secret still present.",
            )


def google_refresh_token_secret_key(account_id: str) -> str:
    """Return the namespaced SecretStore key for one Google refresh token."""
    cleaned = account_id.strip()
    if not cleaned:
        raise SecretStoreError("Account ID for secret key cannot be blank.")
    return f"cortana.calendar.google.{cleaned}.refresh_token"


def _require_key(key: str) -> str:
    if not isinstance(key, str):
        raise SecretStoreError("Secret key must be a string.")
    cleaned = key.strip()
    if not cleaned:
        raise SecretStoreError("Secret key cannot be blank.")
    if len(cleaned) > 256:
        raise SecretStoreError("Secret key is too long.")
    return cleaned


def _require_value(value: str) -> str:
    if not isinstance(value, str):
        raise SecretStoreError("Secret value must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise SecretStoreError("Secret value cannot be blank.")
    return cleaned
