"""LangChain tools for ZeroGPU.

ZeroGPU is a compute-efficient inference provider for apps and agents. It runs
purpose-built small and nano language models across an edge network for the
high-volume tasks you run constantly -- classification, extraction, moderation,
routing, summarization -- at ~10x lower latency and 50%+ lower cost than
frontier-model workflows.

This package exposes the ZeroGPU task models -- chat, reasoning, moderation,
summarization, classification, routing decisions, signal / entity / JSON
extraction, and PII extraction / redaction -- as first-class LangChain
:class:`~langchain_core.tools.BaseTool` subclasses, plus a
:class:`ZeroGPUToolkit` that bundles all 26 behind a single shared
client.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from langchain_zerogpu._client import ZeroGPUAuthError, ZeroGPUClient, ZeroGPUError
from langchain_zerogpu.toolkit import ZeroGPUToolkit
from langchain_zerogpu.tools import (
    ZeroGPUChatThinkingTool,
    ZeroGPUChatTool,
    ZeroGPUClassifyDomainTool,
    ZeroGPUClassifyIABEnrichedTool,
    ZeroGPUClassifyIABTool,
    ZeroGPUClassifyStructuredTool,
    ZeroGPUClassifyZeroShotTool,
    ZeroGPUDecideTool,
    ZeroGPUDecideToolsTool,
    ZeroGPUExtractEntitiesTool,
    ZeroGPUExtractJSONTool,
    ZeroGPUExtractPIITool,
    ZeroGPUExtractSignalsTool,
    ZeroGPUModerateTool,
    ZeroGPUReasonDeepseekTool,
    ZeroGPUReasonGLMTool,
    ZeroGPUReasonGPT6LunaTool,
    ZeroGPUReasonGPTLunaTool,
    ZeroGPUReasonGPTMiniTool,
    ZeroGPUReasonGPTNanoTool,
    ZeroGPUReasonMultilingualTool,
    ZeroGPUReasonQwenTool,
    ZeroGPUReasonTool,
    ZeroGPURedactPIITool,
    ZeroGPUSummarize3BTool,
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
    "ZeroGPUReasonTool",
    "ZeroGPUReasonMultilingualTool",
    "ZeroGPUReasonQwenTool",
    "ZeroGPUReasonGLMTool",
    "ZeroGPUReasonDeepseekTool",
    "ZeroGPUReasonGPTLunaTool",
    "ZeroGPUReasonGPT6LunaTool",
    "ZeroGPUReasonGPTMiniTool",
    "ZeroGPUReasonGPTNanoTool",
    "ZeroGPUModerateTool",
    "ZeroGPUSummarizeTool",
    "ZeroGPUSummarize3BTool",
    "ZeroGPUClassifyIABTool",
    "ZeroGPUClassifyIABEnrichedTool",
    "ZeroGPUClassifyDomainTool",
    "ZeroGPUExtractSignalsTool",
    "ZeroGPUDecideToolsTool",
    "ZeroGPUClassifyZeroShotTool",
    "ZeroGPUClassifyStructuredTool",
    "ZeroGPUDecideTool",
    "ZeroGPUExtractEntitiesTool",
    "ZeroGPUExtractPIITool",
    "ZeroGPURedactPIITool",
    "ZeroGPUExtractJSONTool",
    "ZeroGPUToolkit",
    "ZeroGPUClient",
    "ZeroGPUAuthError",
    "ZeroGPUError",
]
