"""Thin wrapper around the official ZeroGPU Python SDK (``zerogpu-api``).

This module centralises three concerns that every tool in the package shares:

* **Credential resolution** -- read the API key / project id from an explicit
  constructor argument first, then fall back to environment variables, and raise
  a clear error if neither is present.
* **SDK access** -- lazily construct (and reuse) the synchronous
  :class:`zerogpu.ZerogpuApi` and asynchronous :class:`zerogpu.AsyncZerogpuApi`
  clients so that a single :class:`ZeroGPUClient` can serve both ``_run`` and
  ``_arun`` code paths.
* **Error mapping** -- translate the SDK's HTTP errors (401/403/429/5xx) and
  network failures into short, actionable messages instead of leaking raw
  stack traces to the calling agent.

The API key is always stored as a :class:`pydantic.SecretStr` so it is never
rendered in logs or ``repr`` output.
"""

from __future__ import annotations

import json
import os
import typing as t

from pydantic import SecretStr
from zerogpu import AsyncZerogpuApi, ChatMessage, ZerogpuApi
from zerogpu.core.api_error import ApiError

API_KEY_ENV = "ZEROGPU_API_KEY"
"""Environment variable read when no ``api_key`` is passed explicitly."""

PROJECT_ID_ENV = "ZEROGPU_PROJECT_ID"
"""Environment variable read when no ``project_id`` is passed explicitly."""

API_KEY_PREFIX = "zgpu-api-"
"""Every valid ZeroGPU API key starts with this prefix."""


class ZeroGPUAuthError(ValueError):
    """Raised when credentials are missing, malformed, or rejected (401/403)."""


class ZeroGPUError(RuntimeError):
    """Raised for non-auth ZeroGPU failures (rate limits, 5xx, network)."""


def resolve_api_key(api_key: str | SecretStr | None) -> SecretStr:
    """Resolve and validate the ZeroGPU API key.

    Args:
        api_key: An explicit key (``str`` or :class:`~pydantic.SecretStr`), or
            ``None`` to fall back to the ``ZEROGPU_API_KEY`` environment
            variable.

    Returns:
        The key wrapped in a :class:`~pydantic.SecretStr`.

    Raises:
        ZeroGPUAuthError: If no key can be found, or the key does not start
            with the ``zgpu-api-`` prefix.
    """
    if api_key is None:
        raw = os.environ.get(API_KEY_ENV)
    elif isinstance(api_key, SecretStr):
        raw = api_key.get_secret_value()
    else:
        raw = api_key

    if not raw:
        raise ZeroGPUAuthError(
            "No ZeroGPU API key provided. Pass api_key=... or set the "
            f"{API_KEY_ENV} environment variable."
        )
    if not raw.startswith(API_KEY_PREFIX):
        raise ZeroGPUAuthError(
            f"Invalid ZeroGPU API key: it must start with {API_KEY_PREFIX!r}."
        )
    return SecretStr(raw)


def resolve_project_id(project_id: str | None) -> str:
    """Resolve the ZeroGPU project id.

    Args:
        project_id: An explicit project id, or ``None`` to fall back to the
            ``ZEROGPU_PROJECT_ID`` environment variable.

    Returns:
        The resolved project id string.

    Raises:
        ZeroGPUAuthError: If no project id can be found.
    """
    raw = project_id if project_id else os.environ.get(PROJECT_ID_ENV)
    if not raw:
        raise ZeroGPUAuthError(
            "No ZeroGPU project id provided. Pass project_id=... or set the "
            f"{PROJECT_ID_ENV} environment variable."
        )
    return raw


def _error_detail(body: t.Any) -> str:
    """Best-effort extraction of a human message from an SDK error body."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return str(err["message"])
        if isinstance(body.get("message"), str):
            return str(body["message"])
    if isinstance(body, str) and body:
        return body
    return ""


def maybe_json(text: str) -> t.Any:
    """Parse ``text`` as JSON, returning the original string on failure.

    ZeroGPU classification / extraction models return their structured payload
    as a JSON-encoded string. Parsing it yields a native ``dict``/``list`` that
    is far more useful to an agent, while plain-text outputs pass through
    unchanged.

    Args:
        text: The raw output text from a ZeroGPU response.

    Returns:
        The parsed object, or the original string if it is not valid JSON.
    """
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


class ZeroGPUClient:
    """Reusable, credential-bound wrapper over the ZeroGPU SDK.

    A single instance can be shared across many tools. The underlying
    synchronous and asynchronous SDK clients are created lazily on first use.

    Args:
        api_key: ZeroGPU API key, or ``None`` to read ``ZEROGPU_API_KEY``.
        project_id: ZeroGPU project id, or ``None`` to read
            ``ZEROGPU_PROJECT_ID``.
        base_url: Optional override for the API base URL (defaults to the SDK's
            production environment).
        timeout: Optional per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        project_id: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key: SecretStr = resolve_api_key(api_key)
        self._project_id: str = resolve_project_id(project_id)
        self._base_url = base_url
        self._timeout = timeout
        self._sync: ZerogpuApi | None = None
        self._async: AsyncZerogpuApi | None = None

    # -- SDK client accessors ------------------------------------------------

    @property
    def sync_client(self) -> ZerogpuApi:
        """The lazily-constructed synchronous :class:`zerogpu.ZerogpuApi`."""
        if self._sync is None:
            self._sync = ZerogpuApi(
                api_key=self._api_key.get_secret_value(),
                project_id=self._project_id,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._sync

    @property
    def async_client(self) -> AsyncZerogpuApi:
        """The lazily-constructed async :class:`zerogpu.AsyncZerogpuApi`."""
        if self._async is None:
            self._async = AsyncZerogpuApi(
                api_key=self._api_key.get_secret_value(),
                project_id=self._project_id,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._async

    # -- error handling ------------------------------------------------------

    @staticmethod
    def _map_error(err: Exception) -> Exception:
        """Translate an SDK/transport exception into a clear package error."""
        if isinstance(err, ApiError):
            status = err.status_code
            detail = _error_detail(err.body)
            suffix = f" {detail}" if detail else ""
            if status == 401:
                return ZeroGPUAuthError(
                    "ZeroGPU authentication failed (401): the API key was "
                    f"rejected. Check {API_KEY_ENV}.{suffix}"
                )
            if status == 403:
                return ZeroGPUAuthError(
                    "ZeroGPU access denied (403): this project or API key does "
                    "not have access to the requested model. Check "
                    f"{PROJECT_ID_ENV} and your model entitlements.{suffix}"
                )
            if status == 429:
                return ZeroGPUError(
                    "ZeroGPU rate limit exceeded (429): slow down or check your "
                    f"plan quota.{suffix}"
                )
            if status is not None and status >= 500:
                return ZeroGPUError(
                    f"ZeroGPU server error ({status}): the backend failed to "
                    f"process the request.{suffix}"
                )
            return ZeroGPUError(f"ZeroGPU request failed ({status}).{suffix}")
        return ZeroGPUError(f"ZeroGPU request could not be completed: {err}")

    # -- response parsing ----------------------------------------------------

    @staticmethod
    def _response_text(response: t.Any) -> str:
        """Extract the first output text block from a Responses API result."""
        for message in getattr(response, "output", None) or []:
            for block in getattr(message, "content", None) or []:
                text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    return text
        raise ZeroGPUError("ZeroGPU returned an empty response.")

    @staticmethod
    def _chat_text(response: t.Any) -> str:
        """Extract the assistant message content from a Chat API result."""
        choices = getattr(response, "choices", None) or []
        if choices:
            message = (
                choices[0].get("message") if isinstance(choices[0], dict) else None
            )
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    return content
        raise ZeroGPUError("ZeroGPU returned an empty chat response.")

    # -- request helpers -----------------------------------------------------

    @staticmethod
    def _request_options(
        additional_body: dict[str, t.Any] | None,
    ) -> dict[str, t.Any] | None:
        if additional_body:
            return {"additional_body_parameters": additional_body}
        return None

    def responses(
        self,
        *,
        model: str,
        text: str,
        metadata: dict[str, t.Any] | None = None,
        additional_body: dict[str, t.Any] | None = None,
    ) -> str:
        """Call ``POST /v1/responses`` synchronously and return output text.

        Args:
            model: ZeroGPU model identifier.
            text: The input passage (sent as the request ``input`` string).
            metadata: Optional model-specific metadata (e.g. GLiNER ``usecase``,
                ``schema``, ``labels``, ``threshold``, ``mask``).
            additional_body: Optional top-level body fields injected via the
                SDK's ``additional_body_parameters`` (e.g. ``categories`` for
                zero-shot classification).

        Returns:
            The first output text block.
        """
        kwargs: dict[str, t.Any] = {"model": model, "input": text}
        if metadata:
            kwargs["metadata"] = metadata
        options = self._request_options(additional_body)
        if options:
            kwargs["request_options"] = options
        try:
            response = self.sync_client.responses.create_response(**kwargs)
        except Exception as err:  # noqa: BLE001 - re-raised as mapped error
            raise self._map_error(err) from err
        return self._response_text(response)

    async def aresponses(
        self,
        *,
        model: str,
        text: str,
        metadata: dict[str, t.Any] | None = None,
        additional_body: dict[str, t.Any] | None = None,
    ) -> str:
        """Async counterpart of :meth:`responses`."""
        kwargs: dict[str, t.Any] = {"model": model, "input": text}
        if metadata:
            kwargs["metadata"] = metadata
        options = self._request_options(additional_body)
        if options:
            kwargs["request_options"] = options
        try:
            response = await self.async_client.responses.create_response(**kwargs)
        except Exception as err:  # noqa: BLE001 - re-raised as mapped error
            raise self._map_error(err) from err
        return self._response_text(response)

    def _chat_messages(self, text: str, system: str | None) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=text))
        return messages

    def chat(
        self,
        *,
        model: str,
        text: str,
        system: str | None = None,
    ) -> str:
        """Call ``POST /v1/chat/completions`` synchronously and return content.

        Args:
            model: ZeroGPU chat model identifier.
            text: The user message.
            system: Optional system prompt prepended to the conversation.

        Returns:
            The assistant message content.
        """
        try:
            response = self.sync_client.chat.create_chat_completion(
                model=model, messages=self._chat_messages(text, system)
            )
        except Exception as err:  # noqa: BLE001 - re-raised as mapped error
            raise self._map_error(err) from err
        return self._chat_text(response)

    async def achat(
        self,
        *,
        model: str,
        text: str,
        system: str | None = None,
    ) -> str:
        """Async counterpart of :meth:`chat`."""
        try:
            response = await self.async_client.chat.create_chat_completion(
                model=model, messages=self._chat_messages(text, system)
            )
        except Exception as err:  # noqa: BLE001 - re-raised as mapped error
            raise self._map_error(err) from err
        return self._chat_text(response)
