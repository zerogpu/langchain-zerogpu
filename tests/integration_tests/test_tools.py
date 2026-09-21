"""Standard LangChain ``ToolsIntegrationTests`` for every ZeroGPU tool.

These make real calls against a live ZeroGPU project and are skipped unless both
``ZEROGPU_API_KEY`` and ``ZEROGPU_PROJECT_ID`` are set in the environment.
"""

import os
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_zerogpu import (
    ZeroGPUChatThinkingTool,
    ZeroGPUChatTool,
    ZeroGPUClassifyDomainTool,
    ZeroGPUClassifyIABEnrichedTool,
    ZeroGPUClassifyIABTool,
    ZeroGPUClassifyStructuredTool,
    ZeroGPUClassifyZeroShotTool,
    ZeroGPUExtractEntitiesTool,
    ZeroGPUExtractJSONTool,
    ZeroGPUExtractPIITool,
    ZeroGPUExtractSignalsTool,
    ZeroGPUModerateTool,
    ZeroGPUReasonDeepseekTool,
    ZeroGPUReasonGLMTool,
    ZeroGPUReasonGPTLunaTool,
    ZeroGPUReasonGPTMiniTool,
    ZeroGPUReasonGPTNanoTool,
    ZeroGPUReasonLongContextTool,
    ZeroGPUReasonMultilingualTool,
    ZeroGPUReasonTool,
    ZeroGPURedactPIITool,
    ZeroGPUSummarizeTool,
)

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ZEROGPU_API_KEY") and os.environ.get("ZEROGPU_PROJECT_ID")),
    reason="ZEROGPU_API_KEY and ZEROGPU_PROJECT_ID required for integration tests",
)


class TestZeroGPUChatToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUChatTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Say hello in one word."}


class TestZeroGPUChatThinkingToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUChatThinkingTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Is 17 prime? Think it through."}


class TestZeroGPUReasonToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUReasonTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Three ways to handle third-party rate limits, one line each."}


class TestZeroGPUReasonMultilingualToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUReasonMultilingualTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "¿Cuál es la capital de Japón? Explica brevemente."}


class TestZeroGPUReasonLongContextToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUReasonLongContextTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Summarize the architecture described above in one line."}


class TestZeroGPUReasonGLMToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUReasonGLMTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Rewrite this callback-style function to use async/await."}


class TestZeroGPUReasonDeepseekToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUReasonDeepseekTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Plan the steps to migrate this service to async I/O."}


class TestZeroGPUReasonGPTLunaToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUReasonGPTLunaTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Draft a one-paragraph release note for a caching change."}


class TestZeroGPUReasonGPTMiniToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUReasonGPTMiniTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "List the fields this JSON payload is missing, one per line."}


class TestZeroGPUReasonGPTNanoToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUReasonGPTNanoTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Route this ticket to billing, support, or sales. One word."}


class TestZeroGPUModerateToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUModerateTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "How do I reset my password? I forgot it again."}


class TestZeroGPUSummarizeToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUSummarizeTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "text": (
                "ZeroGPU runs small task models at the edge so that agents can "
                "offload cheap NLP work instead of spending frontier tokens. "
                "It supports classification, extraction, summarization and more."
            )
        }


class TestZeroGPUClassifyIABToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUClassifyIABTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "The Lakers won in overtime last night."}


class TestZeroGPUClassifyIABEnrichedToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUClassifyIABEnrichedTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "A review of the latest electric SUV models for 2026."}


class TestZeroGPUClassifyDomainToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUClassifyDomainTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"domain": "nytimes.com"}


class TestZeroGPUExtractSignalsToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUExtractSignalsTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Shopping for a family EV with a 300-mile range."}


class TestZeroGPUClassifyZeroShotToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUClassifyZeroShotTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "text": "The new GPU smashes every benchmark.",
            "labels": ["tech", "politics", "sports"],
        }


class TestZeroGPUClassifyStructuredToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUClassifyStructuredTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "text": "The support team resolved my billing issue quickly.",
            "schema": {
                "sentiment": ["positive", "negative"],
                "topic": ["billing", "support"],
            },
        }


class TestZeroGPUExtractEntitiesToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUExtractEntitiesTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "text": "Ada Lovelace worked at Analytical Engines in 1843.",
            "labels": ["person", "company", "date"],
        }


class TestZeroGPUExtractPIIToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUExtractPIITool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Email me at jane@example.com or call 555-0100."}


class TestZeroGPURedactPIIToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPURedactPIITool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Email me at jane@example.com or call 555-0100."}


class TestZeroGPUExtractJSONToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUExtractJSONTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "text": "Jane Doe, CTO at Acme, jane@example.com",
            "schema": {
                "contact": [
                    "name::str::Full name",
                    "title::str::Job title",
                    "email::str::Email address",
                ]
            },
        }
