"""Compile-only smoke test: the package imports and the toolkit builds tools."""

import pytest

from langchain_zerogpu import ZeroGPUToolkit
from langchain_zerogpu.tools import ALL_TOOL_CLASSES

_PARAMS = {
    "api_key": "zgpu-api-compile-test",
    "project_id": "00000000-0000-4000-8000-000000000000",
}


@pytest.mark.compile
def test_toolkit_compiles_all_tools() -> None:
    toolkit = ZeroGPUToolkit(**_PARAMS)
    tools = toolkit.get_tools()
    assert len(tools) == len(ALL_TOOL_CLASSES) == 11
    for tool in tools:
        assert tool.name
        assert tool.description
        assert tool.args_schema is not None
