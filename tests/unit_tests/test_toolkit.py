"""Unit tests for ``ZeroGPUToolkit`` (no network)."""

from langchain_zerogpu import ZeroGPUToolkit
from langchain_zerogpu.tools import ALL_TOOL_CLASSES

_PARAMS = {
    "api_key": "zgpu-api-unit-test",
    "project_id": "00000000-0000-4000-8000-000000000000",
}


def test_get_tools_returns_eleven_tools() -> None:
    toolkit = ZeroGPUToolkit(**_PARAMS)
    tools = toolkit.get_tools()
    assert len(tools) == 11
    assert {type(tool) for tool in tools} == set(ALL_TOOL_CLASSES)


def test_tools_share_a_single_client() -> None:
    toolkit = ZeroGPUToolkit(**_PARAMS)
    tools = toolkit.get_tools()
    shared = toolkit.client
    for tool in tools:
        assert tool.client is shared, f"{type(tool).__name__} has its own client"


def test_each_tool_has_unique_stable_name() -> None:
    toolkit = ZeroGPUToolkit(**_PARAMS)
    names = [tool.name for tool in toolkit.get_tools()]
    assert len(names) == len(set(names))
    assert all(name.startswith("zerogpu_") for name in names)
