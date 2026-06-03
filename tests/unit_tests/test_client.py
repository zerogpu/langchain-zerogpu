"""Unit tests for credential resolution and SecretStr handling (no network)."""

import pytest
from pydantic import SecretStr, ValidationError
from zerogpu.core.api_error import ApiError

from langchain_zerogpu import (
    ZeroGPUAuthError,
    ZeroGPUChatTool,
    ZeroGPUClient,
    ZeroGPUError,
)
from langchain_zerogpu._client import resolve_api_key, resolve_project_id

VALID_KEY = "zgpu-api-secret-value"
VALID_PROJECT = "00000000-0000-4000-8000-000000000000"


def test_resolve_api_key_from_argument() -> None:
    resolved = resolve_api_key(VALID_KEY)
    assert isinstance(resolved, SecretStr)
    assert resolved.get_secret_value() == VALID_KEY


def test_resolve_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEROGPU_API_KEY", VALID_KEY)
    assert resolve_api_key(None).get_secret_value() == VALID_KEY


def test_resolve_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZEROGPU_API_KEY", raising=False)
    with pytest.raises(ZeroGPUAuthError):
        resolve_api_key(None)


def test_resolve_api_key_bad_prefix() -> None:
    with pytest.raises(ZeroGPUAuthError):
        resolve_api_key("not-a-zgpu-key")


def test_resolve_project_id_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZEROGPU_PROJECT_ID", raising=False)
    with pytest.raises(ZeroGPUAuthError):
        resolve_project_id(None)


def test_client_does_not_expose_key_in_repr() -> None:
    client = ZeroGPUClient(api_key=VALID_KEY, project_id=VALID_PROJECT)
    assert "secret-value" not in repr(client.__dict__["_api_key"])


def test_tool_constructed_from_args_resolves_client() -> None:
    tool = ZeroGPUChatTool(api_key=VALID_KEY, project_id=VALID_PROJECT)
    assert isinstance(tool.client, ZeroGPUClient)


def test_tool_missing_credentials_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZEROGPU_API_KEY", raising=False)
    monkeypatch.delenv("ZEROGPU_PROJECT_ID", raising=False)
    # The ZeroGPUAuthError raised inside the before-validator is surfaced by
    # pydantic as a ValidationError that preserves the actionable message.
    with pytest.raises(ValidationError) as exc_info:
        ZeroGPUChatTool()
    assert "ZEROGPU_API_KEY" in str(exc_info.value)


def test_map_error_402_payment_required() -> None:
    err = ZeroGPUClient._map_error(ApiError(status_code=402, body=None))
    assert isinstance(err, ZeroGPUError)
    assert not isinstance(err, ZeroGPUAuthError)
    assert "402" in str(err)


def test_map_error_402_includes_body_detail() -> None:
    body = {"error": {"message": "insufficient_quota"}}
    err = ZeroGPUClient._map_error(ApiError(status_code=402, body=body))
    assert "insufficient_quota" in str(err)
