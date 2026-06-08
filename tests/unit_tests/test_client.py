"""Unit tests for credential resolution and SecretStr handling (no network)."""

import pytest
from pydantic import SecretStr, ValidationError
from zerogpu.core.parse_error import ParsingError

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


# -- request construction ----------------------------------------------------


def test_build_kwargs_sends_plain_string_input() -> None:
    kwargs = ZeroGPUClient._build_kwargs(
        model="m", text="hello", system=None, metadata=None, additional_body=None
    )
    assert kwargs == {"model": "m", "input": "hello"}


def test_build_kwargs_carries_system_as_instructions() -> None:
    kwargs = ZeroGPUClient._build_kwargs(
        model="m",
        text="hello",
        system="Reply briefly.",
        metadata=None,
        additional_body=None,
    )
    # The passage stays a plain string; the system prompt rides as instructions.
    assert kwargs["input"] == "hello"
    body = kwargs["request_options"]["additional_body_parameters"]
    assert body == {"instructions": "Reply briefly."}


def test_build_kwargs_merges_instructions_with_additional_body() -> None:
    additional_body = {"categories": ["a", "b"]}
    kwargs = ZeroGPUClient._build_kwargs(
        model="m",
        text="hello",
        system="sys",
        metadata=None,
        additional_body=additional_body,
    )
    body = kwargs["request_options"]["additional_body_parameters"]
    assert body == {"categories": ["a", "b"], "instructions": "sys"}
    # The caller's dict must not be mutated.
    assert additional_body == {"categories": ["a", "b"]}


# -- response recovery -------------------------------------------------------


def test_recover_text_reads_output_from_parsing_error_body() -> None:
    body = {"output": [{"content": [{"type": "output_text", "text": "recovered"}]}]}
    err = ParsingError(status_code=200, body=body)
    assert ZeroGPUClient._recover_text(err) == "recovered"


def test_recover_text_raises_on_empty_output() -> None:
    err = ParsingError(status_code=200, body={"output": []})
    with pytest.raises(ZeroGPUError):
        ZeroGPUClient._recover_text(err)
