"""LangChain ``BaseTool`` implementations for ZeroGPU tasks.

Each tool routes a single text-in / structured-out task to a purpose-built
small or nano ZeroGPU language model through the official ``zerogpu-api`` SDK --
the repeatable, high-volume work frontier models shouldn't run, at ~10x lower
latency and 50%+ lower cost. All eleven tools share the same
credential-resolution and error-handling behaviour via
:class:`~langchain_zerogpu._client.ZeroGPUClient`.

The classes here mirror the capabilities exposed by the ZeroGPU Claude Code
plugin (``zerogpu-router``): same tasks, same models, re-expressed as
first-class LangChain tools.
"""

from __future__ import annotations

import typing as t

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_zerogpu._client import ZeroGPUClient, maybe_json
from langchain_zerogpu._schemas import (
    ChatInput,
    ExtractEntitiesInput,
    ExtractJSONInput,
    ExtractPIIInput,
    RedactPIIInput,
    StructuredClassifyInput,
    TextInput,
    ZeroShotInput,
)

# -- ZeroGPU model identifiers ------------------------------------------------

MODEL_CHAT = "LFM2.5-1.2B-Instruct"
MODEL_CHAT_THINKING = "LFM2.5-1.2B-Thinking"
MODEL_SUMMARIZE = "llama-3.1-8b-instruct-fast"
MODEL_IAB = "zlm-v1-iab-classify-edge"
MODEL_IAB_ENRICHED = "zlm-v1-iab-classify-edge-enriched"
MODEL_ZERO_SHOT = "deberta-v3-small"
MODEL_GLINER = "gliner2-base-v1"
MODEL_PII = "gliner-multi-pii-v1"


class _BaseZeroGPUTool(BaseTool):
    """Common credential wiring shared by every ZeroGPU tool.

    A tool may be constructed either with an explicit
    :class:`~langchain_zerogpu._client.ZeroGPUClient` (``client=...``) -- which
    is how :class:`~langchain_zerogpu.toolkit.ZeroGPUToolkit` shares one client
    across all tools -- or with ``api_key`` / ``project_id`` arguments (or
    neither, to resolve from ``ZEROGPU_API_KEY`` / ``ZEROGPU_PROJECT_ID``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: ZeroGPUClient = Field(..., exclude=True)
    """Shared, credential-bound SDK wrapper."""

    api_key: SecretStr | None = Field(default=None, exclude=True)
    """Explicit API key; falls back to ``ZEROGPU_API_KEY`` when omitted."""

    project_id: str | None = Field(default=None, exclude=True)
    """Explicit project id; falls back to ``ZEROGPU_PROJECT_ID`` when omitted."""

    base_url: str | None = Field(default=None, exclude=True)
    """Optional base URL override for the ZeroGPU API."""

    @model_validator(mode="before")
    @classmethod
    def _ensure_client(cls, data: t.Any) -> t.Any:
        """Build a :class:`ZeroGPUClient` from credentials when none is given."""
        if isinstance(data, dict) and not data.get("client"):
            data = dict(data)
            data["client"] = ZeroGPUClient(
                api_key=data.get("api_key"),
                project_id=data.get("project_id"),
                base_url=data.get("base_url"),
            )
        return data


class ZeroGPUChatTool(_BaseZeroGPUTool):
    """Generate a short, single-turn chat reply with a fast nano model.

    Routes to ``LFM2.5-1.2B-Instruct``. Use for quick replies that do not need
    Claude-level reasoning, multi-step planning, or prior conversation context.
    """

    name: str = "zerogpu_chat"
    description: str = (
        "Generate a short single-turn chat reply to a piece of text using a "
        "fast, cheap nano model. Best for simple, self-contained prompts that "
        "do not require multi-step reasoning or conversation history."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.responses(model=MODEL_CHAT, text=text, system=system)

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.aresponses(model=MODEL_CHAT, text=text, system=system)


class ZeroGPUChatThinkingTool(_BaseZeroGPUTool):
    """Chat reply that includes a visible step-by-step reasoning trace.

    Routes to ``LFM2.5-1.2B-Thinking``. The returned text contains the model's
    reasoning followed by its answer.
    """

    name: str = "zerogpu_chat_thinking"
    description: str = (
        "Generate a chat reply that includes a visible reasoning trace, using "
        "a nano 'thinking' model. Use when you want the small model's "
        "intermediate reasoning, not just the final answer."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.responses(
            model=MODEL_CHAT_THINKING, text=text, system=system
        )

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.aresponses(
            model=MODEL_CHAT_THINKING, text=text, system=system
        )


class ZeroGPUSummarizeTool(_BaseZeroGPUTool):
    """Condense a passage into a short summary.

    Routes to ``llama-3.1-8b-instruct-fast``. Best for passages up to a few
    paragraphs.
    """

    name: str = "zerogpu_summarize"
    description: str = (
        "Summarize / condense a passage of text into a short TL;DR using a "
        "fast model. Best for passages up to a few paragraphs."
    )
    args_schema: type[TextInput] = TextInput

    def _run(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.responses(model=MODEL_SUMMARIZE, text=text)

    async def _arun(
        self,
        text: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.aresponses(model=MODEL_SUMMARIZE, text=text)


class ZeroGPUClassifyIABTool(_BaseZeroGPUTool):
    """Classify text into the IAB content taxonomy.

    Routes to ``zlm-v1-iab-classify-edge``. Returns the parsed IAB
    classification result.
    """

    name: str = "zerogpu_classify_iab"
    description: str = (
        "Classify a passage into the IAB content taxonomy (the standard ad / "
        "content category taxonomy). Returns structured IAB categories."
    )
    args_schema: type[TextInput] = TextInput

    def _run(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(self.client.responses(model=MODEL_IAB, text=text))

    async def _arun(
        self,
        text: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(await self.client.aresponses(model=MODEL_IAB, text=text))


class ZeroGPUClassifyIABEnrichedTool(_BaseZeroGPUTool):
    """Classify text into the enriched IAB taxonomy with topics and intent.

    Routes to ``zlm-v1-iab-classify-edge-enriched``. Returns the richer
    classification payload (IAB categories plus topics / keywords / intent).
    """

    name: str = "zerogpu_classify_iab_enriched"
    description: str = (
        "Classify a passage into the enriched IAB taxonomy, returning IAB "
        "categories together with topics, keywords, and intent signals."
    )
    args_schema: type[TextInput] = TextInput

    def _run(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(self.client.responses(model=MODEL_IAB_ENRICHED, text=text))

    async def _arun(
        self,
        text: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(
            await self.client.aresponses(model=MODEL_IAB_ENRICHED, text=text)
        )


class ZeroGPUClassifyZeroShotTool(_BaseZeroGPUTool):
    """Zero-shot classification against a caller-supplied flat list of labels.

    Routes to ``deberta-v3-small``. The candidate ``labels`` are sent as the
    request's ``categories`` and the result includes per-label scores.
    """

    name: str = "zerogpu_classify_zero_shot"
    description: str = (
        "Zero-shot classify text against a custom flat list of labels (e.g. "
        '["tech", "politics", "sports"]). Returns each label with a score so '
        "you can pick the best match."
    )
    args_schema: type[ZeroShotInput] = ZeroShotInput

    def _run(
        self,
        text: str,
        labels: list[str],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(
            self.client.responses(
                model=MODEL_ZERO_SHOT,
                text=text,
                additional_body={"categories": labels},
            )
        )

    async def _arun(
        self,
        text: str,
        labels: list[str],
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(
            await self.client.aresponses(
                model=MODEL_ZERO_SHOT,
                text=text,
                additional_body={"categories": labels},
            )
        )


class ZeroGPUClassifyStructuredTool(_BaseZeroGPUTool):
    """Multi-axis classification driven by a labelled schema.

    Routes to ``gliner2-base-v1`` with ``usecase="classification"``. The
    ``schema`` maps each axis to its candidate labels.
    """

    name: str = "zerogpu_classify_structured"
    description: str = (
        "Classify text along multiple labelled axes at once. Pass a schema "
        'mapping each axis to its candidate labels, e.g. {"sentiment": '
        '["positive", "negative"], "topic": ["billing", "support"]}.'
    )
    args_schema: type[StructuredClassifyInput] = StructuredClassifyInput

    def _run(
        self,
        text: str,
        schema: dict[str, t.Any],
        threshold: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> t.Any:
        metadata = _gliner_metadata(
            usecase="classification", schema=schema, threshold=threshold
        )
        return maybe_json(
            self.client.responses(model=MODEL_GLINER, text=text, metadata=metadata)
        )

    async def _arun(
        self,
        text: str,
        schema: dict[str, t.Any],
        threshold: float | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> t.Any:
        metadata = _gliner_metadata(
            usecase="classification", schema=schema, threshold=threshold
        )
        return maybe_json(
            await self.client.aresponses(
                model=MODEL_GLINER, text=text, metadata=metadata
            )
        )


class ZeroGPUExtractEntitiesTool(_BaseZeroGPUTool):
    """Custom-label named-entity recognition.

    Routes to ``gliner2-base-v1`` with ``usecase="ner"``. Extracts spans for
    each entity type listed in ``labels``.
    """

    name: str = "zerogpu_extract_entities"
    description: str = (
        "Extract named entities of custom types from text. Pass the entity "
        'types as labels, e.g. ["person", "company", "date"]. Returns the '
        "matched spans grouped by label."
    )
    args_schema: type[ExtractEntitiesInput] = ExtractEntitiesInput

    def _run(
        self,
        text: str,
        labels: list[str],
        threshold: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> t.Any:
        metadata = _gliner_metadata(usecase="ner", labels=labels, threshold=threshold)
        return maybe_json(
            self.client.responses(model=MODEL_GLINER, text=text, metadata=metadata)
        )

    async def _arun(
        self,
        text: str,
        labels: list[str],
        threshold: float | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> t.Any:
        metadata = _gliner_metadata(usecase="ner", labels=labels, threshold=threshold)
        return maybe_json(
            await self.client.aresponses(
                model=MODEL_GLINER, text=text, metadata=metadata
            )
        )


class ZeroGPUExtractPIITool(_BaseZeroGPUTool):
    """Extract PII entities from text, grouped by category.

    Routes to ``gliner-multi-pii-v1`` with ``usecase="extract-pii"``.
    """

    name: str = "zerogpu_extract_pii"
    description: str = (
        "Detect and extract personally identifiable information (PII) from "
        "text, grouped by category. Optionally restrict to specific categories "
        '(e.g. ["identity", "contact"]). Returns the detected PII as JSON.'
    )
    args_schema: type[ExtractPIIInput] = ExtractPIIInput

    def _run(
        self,
        text: str,
        categories: list[str] | None = None,
        threshold: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> t.Any:
        metadata = _gliner_metadata(
            usecase="extract-pii", labels=categories, threshold=threshold
        )
        return maybe_json(
            self.client.responses(model=MODEL_PII, text=text, metadata=metadata)
        )

    async def _arun(
        self,
        text: str,
        categories: list[str] | None = None,
        threshold: float | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> t.Any:
        metadata = _gliner_metadata(
            usecase="extract-pii", labels=categories, threshold=threshold
        )
        return maybe_json(
            await self.client.aresponses(model=MODEL_PII, text=text, metadata=metadata)
        )


class ZeroGPURedactPIITool(_BaseZeroGPUTool):
    """Mask PII inline, replacing each match with a ``[LABEL]`` placeholder.

    Routes to ``gliner-multi-pii-v1`` with ``usecase="redact"`` and
    ``mask="label"``. Returns the redacted text.
    """

    name: str = "zerogpu_redact_pii"
    description: str = (
        "Redact personally identifiable information (PII) from text inline, "
        "replacing each match with a labelled placeholder such as [PHONE] or "
        "[EMAIL]. Returns the redacted text."
    )
    args_schema: type[RedactPIIInput] = RedactPIIInput

    def _run(
        self,
        text: str,
        categories: list[str] | None = None,
        threshold: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        metadata = _gliner_metadata(
            usecase="redact", labels=categories, threshold=threshold, mask="label"
        )
        return self.client.responses(model=MODEL_PII, text=text, metadata=metadata)

    async def _arun(
        self,
        text: str,
        categories: list[str] | None = None,
        threshold: float | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        metadata = _gliner_metadata(
            usecase="redact", labels=categories, threshold=threshold, mask="label"
        )
        return await self.client.aresponses(
            model=MODEL_PII, text=text, metadata=metadata
        )


class ZeroGPUExtractJSONTool(_BaseZeroGPUTool):
    """Schema-driven JSON extraction.

    Routes to ``gliner2-base-v1`` with ``usecase="json"``. The grouped
    ``schema`` declares the fields to pull out.
    """

    name: str = "zerogpu_extract_json"
    description: str = (
        "Extract structured fields from text into JSON. Pass a grouped schema "
        'mapping a group to "field::type::description" specs, e.g. '
        '{"contact": ["name::str::Full name", "email::str::Email address"]}.'
    )
    args_schema: type[ExtractJSONInput] = ExtractJSONInput

    def _run(
        self,
        text: str,
        schema: dict[str, t.Any],
        threshold: float | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> t.Any:
        metadata = _gliner_metadata(usecase="json", schema=schema, threshold=threshold)
        return maybe_json(
            self.client.responses(model=MODEL_GLINER, text=text, metadata=metadata)
        )

    async def _arun(
        self,
        text: str,
        schema: dict[str, t.Any],
        threshold: float | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> t.Any:
        metadata = _gliner_metadata(usecase="json", schema=schema, threshold=threshold)
        return maybe_json(
            await self.client.aresponses(
                model=MODEL_GLINER, text=text, metadata=metadata
            )
        )


def _gliner_metadata(
    *,
    usecase: str,
    schema: dict[str, t.Any] | None = None,
    labels: list[str] | None = None,
    threshold: float | None = None,
    mask: str | None = None,
) -> dict[str, t.Any]:
    """Assemble the GLiNER ``metadata`` block, omitting empty fields."""
    metadata: dict[str, t.Any] = {"usecase": usecase}
    if schema is not None:
        metadata["schema"] = schema
    if labels:
        metadata["labels"] = labels
    if threshold is not None:
        metadata["threshold"] = threshold
    if mask is not None:
        metadata["mask"] = mask
    return metadata


#: Every tool class exported by the package, in a stable order.
ALL_TOOL_CLASSES: list[type[_BaseZeroGPUTool]] = [
    ZeroGPUChatTool,
    ZeroGPUChatThinkingTool,
    ZeroGPUSummarizeTool,
    ZeroGPUClassifyIABTool,
    ZeroGPUClassifyIABEnrichedTool,
    ZeroGPUClassifyZeroShotTool,
    ZeroGPUClassifyStructuredTool,
    ZeroGPUExtractEntitiesTool,
    ZeroGPUExtractPIITool,
    ZeroGPURedactPIITool,
    ZeroGPUExtractJSONTool,
]
