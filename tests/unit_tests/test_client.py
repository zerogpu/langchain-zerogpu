"""Unit tests for credential resolution and SecretStr handling (no network)."""

import pytest
from pydantic import SecretStr, ValidationError
from zerogpu.core.api_error import ApiError
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


def test_resolve_project_id_from_argument() -> None:
    assert resolve_project_id(VALID_PROJECT) == VALID_PROJECT


def test_resolve_project_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZEROGPU_PROJECT_ID", VALID_PROJECT)
    assert resolve_project_id(None) == VALID_PROJECT


def test_resolve_project_id_missing_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # x-project-id is optional in the API spec: an unset project id leaves the
    # request unscoped rather than failing.
    monkeypatch.delenv("ZEROGPU_PROJECT_ID", raising=False)
    assert resolve_project_id(None) == ""


def test_tool_constructs_without_a_project_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZEROGPU_PROJECT_ID", raising=False)
    tool = ZeroGPUChatTool(api_key=VALID_KEY)
    assert isinstance(tool.client, ZeroGPUClient)


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


# -- error mapping -----------------------------------------------------------


def test_map_error_402_payment_required() -> None:
    err = ZeroGPUClient._map_error(ApiError(status_code=402, body=None))
    assert isinstance(err, ZeroGPUError)
    assert not isinstance(err, ZeroGPUAuthError)
    assert "402" in str(err)


def test_map_error_402_includes_body_detail() -> None:
    body = {"error": {"message": "insufficient_quota"}}
    err = ZeroGPUClient._map_error(ApiError(status_code=402, body=body))
    assert "insufficient_quota" in str(err)


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


def test_summarize_routes_through_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # llama-3.1-8b-instruct-fast rejects the Responses API's plain-string
    # input, so the summarizer must use chat completions -- with the
    # instruction that keeps it summarizing rather than chatting back.
    from langchain_zerogpu import ZeroGPUSummarizeTool
    from langchain_zerogpu.tools import MODEL_SUMMARIZE, SUMMARIZE_INSTRUCTION

    calls: dict[str, object] = {}
    tool = ZeroGPUSummarizeTool(api_key=VALID_KEY, project_id=VALID_PROJECT)

    def fake_chat(*, model: str, text: str, system: str | None = None) -> str:
        calls.update(model=model, text=text, system=system)
        return "a summary"

    def fail_responses(**kwargs: object) -> str:
        raise AssertionError("summarize must not use the Responses API")

    monkeypatch.setattr(tool.client, "chat", fake_chat)
    monkeypatch.setattr(tool.client, "responses", fail_responses)

    assert tool.invoke({"text": "a long passage"}) == "a summary"
    assert calls["model"] == MODEL_SUMMARIZE
    assert calls["system"] == SUMMARIZE_INSTRUCTION


# -- response recovery -------------------------------------------------------


def test_recover_text_reads_output_from_parsing_error_body() -> None:
    body = {"output": [{"content": [{"type": "output_text", "text": "recovered"}]}]}
    err = ParsingError(status_code=200, body=body)
    assert ZeroGPUClient._recover_text(err) == "recovered"


def test_recover_text_raises_on_empty_output() -> None:
    err = ParsingError(status_code=200, body={"output": []})
    with pytest.raises(ZeroGPUError):
        ZeroGPUClient._recover_text(err)


def test_select_text_skips_the_reasoning_trace() -> None:
    # gpt-oss-120b emits its scratchpad as the first output item; the answer is
    # the output_text block that follows.
    output = [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "User asks for a color."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Azure"}],
        },
    ]
    assert ZeroGPUClient._select_text(output) == "Azure"


def test_select_text_falls_back_to_untyped_blocks() -> None:
    output = [{"content": [{"text": "plain"}]}]
    assert ZeroGPUClient._select_text(output) == "plain"


def test_select_text_ignores_a_lone_reasoning_item() -> None:
    output = [
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "hmm"}]}
    ]
    with pytest.raises(ZeroGPUError):
        ZeroGPUClient._select_text(output)


# -- chat completions (qwen3-30b-a3b-fp8) ------------------------------------


def test_chat_messages_sends_a_lone_user_turn() -> None:
    messages = ZeroGPUClient._chat_messages("hello", None)
    assert [(m.role, m.content) for m in messages] == [("user", "hello")]


def test_chat_messages_prepends_the_system_turn() -> None:
    messages = ZeroGPUClient._chat_messages("hello", "Be brief.")
    assert [(m.role, m.content) for m in messages] == [
        ("system", "Be brief."),
        ("user", "hello"),
    ]


def test_choice_text_reads_the_first_message_content() -> None:
    choices = [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]
    assert ZeroGPUClient._choice_text(choices) == "hi"


def test_choice_text_raises_on_empty_choices() -> None:
    with pytest.raises(ZeroGPUError):
        ZeroGPUClient._choice_text([])


def test_recover_choice_text_reads_from_parsing_error_body() -> None:
    body = {"choices": [{"message": {"role": "assistant", "content": "recovered"}}]}
    err = ParsingError(status_code=200, body=body)
    assert ZeroGPUClient._recover_choice_text(err) == "recovered"
