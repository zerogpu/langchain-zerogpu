"""LangChain ``BaseTool`` implementations for ZeroGPU tasks.

Each tool routes a single text-in / structured-out task to a purpose-built
small or nano ZeroGPU language model through the official ``zerogpu-api`` SDK --
the repeatable, high-volume work frontier models shouldn't run, at ~10x lower
latency and 50%+ lower cost. All fifteen tools share the same
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
    DomainInput,
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
MODEL_REASON = "gpt-oss-120b"
MODEL_REASON_MULTILINGUAL = "qwen3-30b-a3b-fp8"
MODEL_SUMMARIZE = "llama-3.1-8b-instruct-fast"
MODEL_FOLLOWUP = "zlm-v1-followup-questions-edge"
MODEL_IAB = "zlm-v1-iab-classify-edge"
MODEL_IAB_ENRICHED = "zlm-v2-iab-classify-edge-enriched"
MODEL_IAB_DOMAIN = "zlm-v1-iab-domain-classifier"
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
    """Optional project id; falls back to ``ZEROGPU_PROJECT_ID`` when omitted."""

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


class ZeroGPUReasonTool(_BaseZeroGPUTool):
    """Answer a hard prompt with a large open-weight reasoning model.

    Routes to ``gpt-oss-120b`` (117B parameters, 131K context) over the
    Responses API. Use when a task needs more reasoning headroom than the nano
    edge models provide; the returned text is the final answer.
    """

    name: str = "zerogpu_reason"
    description: str = (
        "Answer a prompt that needs real reasoning headroom -- multi-step "
        "analysis, tricky logic, longer context -- using a large open-weight "
        "model. Slower and pricier than the nano chat tools, so prefer those "
        "for simple prompts."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.responses(model=MODEL_REASON, text=text, system=system)

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.aresponses(
            model=MODEL_REASON, text=text, system=system
        )


class ZeroGPUReasonMultilingualTool(_BaseZeroGPUTool):
    """Reasoning answer in a lighter, multilingual model.

    Routes to ``qwen3-30b-a3b-fp8`` (30.5B parameters, 100+ languages). This
    model is served on the Chat Completions endpoint only, so it is the one
    tool that does not use the Responses API.
    """

    name: str = "zerogpu_reason_multilingual"
    description: str = (
        "Answer a prompt that needs reasoning, in any of 100+ languages, using "
        "a mid-size multilingual model. Prefer this over zerogpu_reason when "
        "the prompt or the expected answer is not in English."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(
            model=MODEL_REASON_MULTILINGUAL, text=text, system=system
        )

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(
            model=MODEL_REASON_MULTILINGUAL, text=text, system=system
        )


#: Steers the summarizer, which otherwise replies conversationally to a bare
#: passage and runs into the completion-token cap mid-sentence.
SUMMARIZE_INSTRUCTION = (
    "Summarize the user's text faithfully and concisely. Reply with the "
    "summary only -- no preamble, commentary, or follow-up questions."
)


class ZeroGPUSummarizeTool(_BaseZeroGPUTool):
    """Condense a passage into a short summary.

    Routes to ``llama-3.1-8b-instruct-fast`` over Chat Completions. That model
    currently rejects the Responses API's plain-string ``input`` with
    ``invalid_prompt`` ("required properties at '/' are 'prompt' ...
    'messages'"), so the passage is sent as a user message instead.
    """

    name: str = "zerogpu_summarize"
    description: str = (
        "Summarize / condense a passage of text into a short TL;DR using a "
        "fast model. Handles anything from a few paragraphs to a full "
        "document."
    )
    args_schema: type[TextInput] = TextInput

    def _run(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(
            model=MODEL_SUMMARIZE, text=text, system=SUMMARIZE_INSTRUCTION
        )

    async def _arun(
        self,
        text: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(
            model=MODEL_SUMMARIZE, text=text, system=SUMMARIZE_INSTRUCTION
        )


class ZeroGPUFollowUpQuestionsTool(_BaseZeroGPUTool):
    """Generate the questions a reader would naturally ask next.

    Routes to ``zlm-v1-followup-questions-edge``, which emits one ready-to-render
    question per line. The lines are returned as a list of strings.
    """

    name: str = "zerogpu_followup_questions"
    description: str = (
        "Generate a short set of natural follow-up questions a reader would ask "
        "next about a piece of content (an article, an answer, a chat turn). "
        "Returns a list of ready-to-render questions."
    )
    args_schema: type[TextInput] = TextInput

    def _run(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[str]:
        return _questions(self.client.responses(model=MODEL_FOLLOWUP, text=text))

    async def _arun(
        self,
        text: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[str]:
        return _questions(await self.client.aresponses(model=MODEL_FOLLOWUP, text=text))


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


class ZeroGPUClassifyDomainTool(_BaseZeroGPUTool):
    """Classify a bare domain name into the IAB taxonomy.

    Routes to ``zlm-v1-iab-domain-classifier``. Where the page-level IAB tools
    take the body of an article, this one takes only the domain (for example
    ``nytimes.com``) and characterises the site as a whole.
    """

    name: str = "zerogpu_classify_domain"
    description: str = (
        'Classify a domain name (e.g. "nytimes.com") into the IAB content '
        "taxonomy without fetching the page. Returns scored categories, topics, "
        "keywords, and inferred user intent for the site as a whole. Use the "
        "page-level IAB tools instead when you have the actual content."
    )
    args_schema: type[DomainInput] = DomainInput

    def _run(
        self,
        domain: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(self.client.responses(model=MODEL_IAB_DOMAIN, text=domain))

    async def _arun(
        self,
        domain: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(
            await self.client.aresponses(model=MODEL_IAB_DOMAIN, text=domain)
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


def _questions(text: str) -> list[str]:
    """Normalise the follow-up model's output into a list of questions.

    The model returns a JSON array of question strings; the newline-delimited
    form shown in the docs is accepted as a fallback.
    """
    parsed = maybe_json(text)
    if isinstance(parsed, list):
        return [str(question).strip() for question in parsed if str(question).strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


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
    ZeroGPUReasonTool,
    ZeroGPUReasonMultilingualTool,
    ZeroGPUSummarizeTool,
    ZeroGPUFollowUpQuestionsTool,
    ZeroGPUClassifyIABTool,
    ZeroGPUClassifyIABEnrichedTool,
    ZeroGPUClassifyDomainTool,
    ZeroGPUClassifyZeroShotTool,
    ZeroGPUClassifyStructuredTool,
    ZeroGPUExtractEntitiesTool,
    ZeroGPUExtractPIITool,
    ZeroGPURedactPIITool,
    ZeroGPUExtractJSONTool,
]
