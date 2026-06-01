"""Confirm every public class is importable from the package root."""

from langchain_zerogpu import __all__

EXPECTED = [
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
]


def test_all_public_names_exported() -> None:
    for name in EXPECTED:
        assert name in __all__, f"{name} missing from __all__"


def test_eleven_tool_classes_importable() -> None:
    import langchain_zerogpu as pkg

    tool_names = [name for name in EXPECTED if name != "ZeroGPUToolkit"]
    assert len(tool_names) == 11
    for name in tool_names:
        assert hasattr(pkg, name), f"{name} not importable from package root"


def test_toolkit_importable() -> None:
    from langchain_zerogpu import ZeroGPUToolkit

    assert ZeroGPUToolkit is not None
