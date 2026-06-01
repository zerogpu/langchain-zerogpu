"""LangChain tools for ZeroGPU nano-model tasks.

Exposes the eleven ZeroGPU task models -- chat, summarization, classification,
entity / JSON extraction, and PII extraction / redaction -- as first-class
LangChain :class:`~langchain_core.tools.BaseTool` subclasses, plus a
:class:`ZeroGPUToolkit` that bundles them behind a single shared client.
"""

from __future__ import annotations

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

__version__ = "0.1.0"

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
