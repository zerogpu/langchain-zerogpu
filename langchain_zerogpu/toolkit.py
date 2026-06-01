"""Toolkit bundling all eleven ZeroGPU tools behind a single shared client."""

from __future__ import annotations

import typing as t

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_zerogpu._client import ZeroGPUClient
from langchain_zerogpu.tools import ALL_TOOL_CLASSES


class ZeroGPUToolkit(BaseToolkit):
    """Bundle of all ZeroGPU tools wired to one shared SDK client.

    Construct the toolkit once with your credentials and call
    :meth:`get_tools` to obtain every :class:`~langchain_core.tools.BaseTool`,
    each sharing a single :class:`~langchain_zerogpu._client.ZeroGPUClient`
    (and therefore a single set of pooled SDK connections).

    Example:
        >>> from langchain_zerogpu import ZeroGPUToolkit
        >>> toolkit = ZeroGPUToolkit(
        ...     api_key="zgpu-api-...", project_id="your-project-id"
        ... )
        >>> tools = toolkit.get_tools()

    Args:
        api_key: Explicit API key; falls back to ``ZEROGPU_API_KEY``.
        project_id: Explicit project id; falls back to ``ZEROGPU_PROJECT_ID``.
        base_url: Optional base URL override for the ZeroGPU API.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: ZeroGPUClient = Field(..., exclude=True)
    """Shared, credential-bound SDK wrapper passed to every tool."""

    api_key: SecretStr | None = Field(default=None, exclude=True)
    """Explicit API key; falls back to ``ZEROGPU_API_KEY`` when omitted."""

    project_id: str | None = Field(default=None, exclude=True)
    """Explicit project id; falls back to ``ZEROGPU_PROJECT_ID`` when omitted."""

    base_url: str | None = Field(default=None, exclude=True)
    """Optional base URL override for the ZeroGPU API."""

    @model_validator(mode="before")
    @classmethod
    def _ensure_client(cls, data: t.Any) -> t.Any:
        """Build a single :class:`ZeroGPUClient` shared by all tools."""
        if isinstance(data, dict) and not data.get("client"):
            data = dict(data)
            data["client"] = ZeroGPUClient(
                api_key=data.get("api_key"),
                project_id=data.get("project_id"),
                base_url=data.get("base_url"),
            )
        return data

    def get_tools(self) -> list[BaseTool]:
        """Return all eleven ZeroGPU tools sharing this toolkit's client.

        Returns:
            A list of every ZeroGPU :class:`~langchain_core.tools.BaseTool`,
            each constructed with the toolkit's shared client.
        """
        # Each concrete subclass supplies default ``name``/``description``; the
        # base type mypy sees here does not, hence the call-arg ignore.
        return [
            tool_cls(client=self.client)  # type: ignore[call-arg]
            for tool_cls in ALL_TOOL_CLASSES
        ]
