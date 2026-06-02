"""LangChain tools for ZeroGPU.

ZeroGPU is a compute-efficient inference provider for apps and agents. It runs
purpose-built small and nano language models across an edge network for the
high-volume tasks you run constantly -- classification, extraction, moderation,
routing, summarization -- at ~10x lower latency and 50%+ lower cost than
frontier-model workflows.

This package exposes the eleven ZeroGPU task models -- chat, summarization,
classification, entity / JSON extraction, and PII extraction / redaction -- as
first-class LangChain :class:`~langchain_core.tools.BaseTool` subclasses, plus a
:class:`ZeroGPUToolkit` that bundles them behind a single shared client.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from langchain_zerogpu._client import ZeroGPUAuthError, ZeroGPUClient, ZeroGPUError
from langchain_zerogpu.toolkit import ZeroGPUToolkit
from langchain_zerogpu.tools import (
    ZeroGPUChatThinkingTool,
    ZeroGPUChatTool,
    ZeroGPUClassifyIABEnrichedTool,
    ZeroGPUClassifyIABTool,
    ZeroGPUClassifyStructuredTool,
    ZeroGPUClassifyZeroShotTool,
    ZeroGPUExtractEntitiesTool,
    ZeroGPUExtractJSONTool,
    ZeroGPUExtractPIITool,
    ZeroGPURedactPIITool,
    ZeroGPUSummarizeTool,
)

try:
    # Single source of truth: the version declared in pyproject.toml and
    # baked into the installed distribution's metadata. Avoids drift between
    # this module and pyproject.
    __version__ = version("langchain-zerogpu")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "ZeroGPUChatTool",
    "ZeroGPUChatThinkingTool",
    "ZeroGPUSummarizeTool",
    "ZeroGPUClassifyIABTool",
    "ZeroGPUClassifyIABEnrichedTool",
    "ZeroGPUClassifyZeroShotTool",
    "ZeroGPUClassifyStructuredTool",
    "ZeroGPUExtractEntitiesTool",
    "ZeroGPUExtractPIITool",
    "ZeroGPURedactPIITool",
    "ZeroGPUExtractJSONTool",
    "ZeroGPUToolkit",
    "ZeroGPUClient",
    "ZeroGPUAuthError",
    "ZeroGPUError",
]
