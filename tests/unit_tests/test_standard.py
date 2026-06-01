"""Standard LangChain ``ToolsUnitTests`` for every ZeroGPU tool.

These run with sockets disabled (see ``pytest --disable-socket``) and exercise
construction, naming, and input-schema validation without any network access.
"""

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_zerogpu import (
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

_CONSTRUCTOR_PARAMS = {
    "api_key": "zgpu-api-unit-test",
    "project_id": "00000000-0000-4000-8000-000000000000",
}


class _ZeroGPUToolUnitTests(ToolsUnitTests):
    """Shared constructor params for all ZeroGPU unit-test subclasses."""

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return dict(_CONSTRUCTOR_PARAMS)


class TestZeroGPUChatToolUnit(_ZeroGPUToolUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUChatTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Say hello in one word."}


class TestZeroGPUChatThinkingToolUnit(_ZeroGPUToolUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUChatThinkingTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Is 17 prime? Think it through."}


class TestZeroGPUSummarizeToolUnit(_ZeroGPUToolUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUSummarizeTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "A long passage that needs condensing into a TL;DR."}


class TestZeroGPUClassifyIABToolUnit(_ZeroGPUToolUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUClassifyIABTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "The Lakers won in overtime last night."}


class TestZeroGPUClassifyIABEnrichedToolUnit(_ZeroGPUToolUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUClassifyIABEnrichedTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "A review of the latest electric SUV models for 2026."}


class TestZeroGPUClassifyZeroShotToolUnit(_ZeroGPUToolUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUClassifyZeroShotTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "text": "The new GPU smashes every benchmark.",
            "labels": ["tech", "politics", "sports"],
        }


class TestZeroGPUClassifyStructuredToolUnit(_ZeroGPUToolUnitTests):
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


class TestZeroGPUExtractEntitiesToolUnit(_ZeroGPUToolUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUExtractEntitiesTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "text": "Ada Lovelace worked at Analytical Engines in 1843.",
            "labels": ["person", "company", "date"],
        }


class TestZeroGPUExtractPIIToolUnit(_ZeroGPUToolUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPUExtractPIITool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Email me at jane@example.com or call 555-0100."}


class TestZeroGPURedactPIIToolUnit(_ZeroGPUToolUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return ZeroGPURedactPIITool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"text": "Email me at jane@example.com or call 555-0100."}


class TestZeroGPUExtractJSONToolUnit(_ZeroGPUToolUnitTests):
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
