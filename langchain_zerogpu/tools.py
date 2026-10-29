"""LangChain ``BaseTool`` implementations for ZeroGPU tasks.

Each tool routes a single text-in / structured-out task to a purpose-built
small or nano ZeroGPU language model through the official ``zerogpu-api`` SDK --
the repeatable, high-volume work frontier models shouldn't run, at ~10x lower
latency and 50%+ lower cost. All 26 tools share the same
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
MODEL_REASON_QWEN = "qwen3.6-35b-a3b"
MODEL_REASON_GLM = "glm-5.3-flash"
MODEL_REASON_DEEPSEEK = "deepseek-v4.1-flash"
MODEL_REASON_GPT_LUNA = "gpt-5.6-luna"
MODEL_REASON_GPT6_LUNA = "gpt-6-luna"
MODEL_REASON_GPT_MINI = "gpt-4.1-mini"
MODEL_REASON_GPT_NANO = "gpt-5.4-nano"
MODEL_MODERATE = "llama-guard-4-12b"
MODEL_SUMMARIZE = "llama-3.1-8b-instruct-fast"
MODEL_SUMMARIZE_3B = "llama-3.2-3b-instruct-fast"
MODEL_IAB = "zlm-v1-iab-classify-edge"
MODEL_IAB_ENRICHED = "zlm-v2-iab-classify-edge-enriched"
MODEL_IAB_DOMAIN = "zlm-v1-iab-domain-classifier"
MODEL_EXTRACT_SIGNALS = "zlm-v1-signal-extract"
MODEL_DECIDE_TOOLS = "zlm-v1-tool-decisions"
MODEL_ZERO_SHOT = "deberta-v3-small"
MODEL_GLINER = "gliner2.5-base-v1"
MODEL_DECIDE = "gliner2.5-decide-v1"
MODEL_PII = "gliner2.5-multi-v1"


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

    Routes to ``gpt-oss-120b`` (120B parameters, 131K context) over the
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

    Routes to ``qwen3-30b-a3b-fp8`` (30B parameters, 100+ languages). Served
    on the Chat Completions endpoint only, so it does not use the Responses API.
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


class ZeroGPUReasonQwenTool(_BaseZeroGPUTool):
    """Reasoning answer from a fast, open-weight multimodal model.

    Routes to ``qwen3.6-35b-a3b`` (35B-parameter mixture-of-experts, 16K-token
    context). Served on the Chat Completions endpoint only, so it does not use
    the Responses API. Built for document, screenshot, and content
    understanding; this tool sends the text side of that work.
    """

    name: str = "zerogpu_reason_qwen"
    description: str = (
        "Answer a chat, document-understanding, content-extraction, or "
        "tool-calling prompt with a fast 35B mixture-of-experts model that "
        "supports structured outputs. Its 16K-token context is the smallest "
        "of the reasoning tools, so prefer zerogpu_reason_glm for large "
        "inputs."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(model=MODEL_REASON_QWEN, text=text, system=system)

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(
            model=MODEL_REASON_QWEN, text=text, system=system
        )


class ZeroGPUReasonGLMTool(_BaseZeroGPUTool):
    """Reasoning answer for coding and long-horizon agent work.

    Routes to ``glm-5.3-flash`` (hybrid sparse and linear attention,
    1,048,576-token context). Served on the Chat Completions endpoint only, so
    it does not use the Responses API.
    """

    name: str = "zerogpu_reason_glm"
    description: str = (
        "Answer a coding or long-horizon agent prompt -- reading a codebase, "
        "writing or porting code, planning and running multi-step work -- with "
        "Z.ai's efficient open-weight model, whose hybrid attention keeps "
        "long-context behaviour accurate across a 1M-token window, with "
        "function calling and adjustable reasoning effort. It is the cheapest "
        "tool with a 1M context, so prefer it for very large inputs."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(model=MODEL_REASON_GLM, text=text, system=system)

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(model=MODEL_REASON_GLM, text=text, system=system)


class ZeroGPUReasonDeepseekTool(_BaseZeroGPUTool):
    """Reasoning answer from the V4.1 generation of DeepSeek Flash.

    Routes to ``deepseek-v4.1-flash`` (sparse mixture-of-experts, 8B parameters
    active on input and 16B on output, 1,048,576-token context). Served on the
    Chat Completions endpoint only, so it does not use the Responses API.
    """

    name: str = "zerogpu_reason_deepseek"
    description: str = (
        "Answer a chat, reasoning, or agentic prompt using DeepSeek's V4.1 "
        "Flash model -- a sparse mixture-of-experts model with a 1M-token "
        "context window, function calling, and both fast and higher-effort "
        "reasoning modes. It costs about three times as much per input token "
        "and three and a half times as much per output token as "
        "zerogpu_reason_glm, which has the same context window, so prefer "
        "that tool unless you specifically want the V4.1 generation."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(model=MODEL_REASON_DEEPSEEK, text=text, system=system)

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(
            model=MODEL_REASON_DEEPSEEK, text=text, system=system
        )


class ZeroGPUReasonGPTLunaTool(_BaseZeroGPUTool):
    """Reasoning answer from the cost-optimized model of the GPT-5.6 family.

    Routes to ``gpt-5.6-luna`` (272,000-token context). Served on the Chat
    Completions endpoint only, so it does not use the Responses API.
    """

    name: str = "zerogpu_reason_gpt_luna"
    description: str = (
        "Answer a coding, chat, reasoning, RAG, summarization, or translation "
        "prompt with the cost-optimized model of OpenAI's GPT-5.6 family, "
        "which supports adjustable reasoning effort, function calling, and "
        "structured outputs over a 272K-token context. Built for "
        "cost-sensitive, high-volume workloads, though it costs more per "
        "token than both zerogpu_reason and zerogpu_reason_gpt6_luna, which "
        "has the same 272K-token context."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(model=MODEL_REASON_GPT_LUNA, text=text, system=system)

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(
            model=MODEL_REASON_GPT_LUNA, text=text, system=system
        )


class ZeroGPUReasonGPT6LunaTool(_BaseZeroGPUTool):
    """Reasoning answer from the most efficient model of the GPT-6 family.

    Routes to ``gpt-6-luna`` (272,000-token context). Served on the Chat
    Completions endpoint only, so it does not use the Responses API.
    """

    name: str = "zerogpu_reason_gpt6_luna"
    description: str = (
        "Answer a coding, chat, reasoning, RAG, summarization, or translation "
        "prompt with the most efficient model of OpenAI's GPT-6 family, built "
        "for high-volume, cost-sensitive work, with adjustable reasoning "
        "effort, function calling, and structured outputs over a 272K-token "
        "context. It costs half as much per input token and well under half "
        "as much per output token as zerogpu_reason_gpt_luna, which has the "
        "same context window."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(model=MODEL_REASON_GPT6_LUNA, text=text, system=system)

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(
            model=MODEL_REASON_GPT6_LUNA, text=text, system=system
        )


class ZeroGPUReasonGPTMiniTool(_BaseZeroGPUTool):
    """Reasoning answer from OpenAI's fast, cost-efficient GPT-4.1 model.

    Routes to ``gpt-4.1-mini`` (1,047,576-token context). Served on the Chat
    Completions endpoint only, so it does not use the Responses API.
    """

    name: str = "zerogpu_reason_gpt_mini"
    description: str = (
        "Answer a prompt that leans on instruction following and tool calling "
        "-- coding, chat, RAG, summarization, translation -- with OpenAI's "
        "GPT-4.1 mini, which supports function calling and structured outputs "
        "across a 1M-token context at low latency. It is the priciest tool "
        "here per token, so prefer zerogpu_reason_glm for large inputs that "
        "do not need it."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(model=MODEL_REASON_GPT_MINI, text=text, system=system)

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(
            model=MODEL_REASON_GPT_MINI, text=text, system=system
        )


class ZeroGPUReasonGPTNanoTool(_BaseZeroGPUTool):
    """Reasoning answer from the most cost-efficient GPT-5.4 model.

    Routes to ``gpt-5.4-nano`` (400,000-token context). Served on the Chat
    Completions endpoint only, so it does not use the Responses API.
    """

    name: str = "zerogpu_reason_gpt_nano"
    description: str = (
        "Answer a high-volume or latency-sensitive prompt -- classification, "
        "extraction, routing, sub-agent work -- with the most cost-efficient "
        "model of OpenAI's GPT-5.4 family, which supports function calling "
        "and structured outputs over a 400K-token context. The nano chat "
        "tools and the purpose-built classifiers are cheaper still for the "
        "work that fits them."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(model=MODEL_REASON_GPT_NANO, text=text, system=system)

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(
            model=MODEL_REASON_GPT_NANO, text=text, system=system
        )


class ZeroGPUModerateTool(_BaseZeroGPUTool):
    """Judge text safe or unsafe against Llama Guard's policy categories.

    Routes to ``llama-guard-4-12b`` (12B parameters, 160K-token context).
    Served on the Chat Completions endpoint only, so it does not use the
    Responses API. The returned text is the model's safe / unsafe verdict plus
    the policy categories a violation matched.
    """

    name: str = "zerogpu_moderate"
    description: str = (
        "Moderate a piece of text -- an incoming prompt or a generated reply "
        "-- with Meta's Llama Guard 4 safety model, which returns a safe / "
        "unsafe verdict together with the policy categories a violation "
        "matched. Use for chat moderation, prompt and response filtering, "
        "policy enforcement, and agent guardrails. Multilingual."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(model=MODEL_MODERATE, text=text, system=system)

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(model=MODEL_MODERATE, text=text, system=system)


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


class ZeroGPUSummarize3BTool(_BaseZeroGPUTool):
    """Condense a passage into a short summary with a smaller, cheaper model.

    Routes to ``llama-3.2-3b-instruct-fast`` (3B parameters, 131K-token
    context) over Chat Completions, whose summarization pipeline it shares
    with ``llama-3.1-8b-instruct-fast``.
    """

    name: str = "zerogpu_summarize_3b"
    description: str = (
        "Summarize / condense a passage of text into a short TL;DR using a "
        "fast 3B edge model. It costs a third as much per input token as "
        "zerogpu_summarize and a touch more per output token, so prefer it "
        "for high-volume summarization of long passages."
    )
    args_schema: type[TextInput] = TextInput

    def _run(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        return self.client.chat(
            model=MODEL_SUMMARIZE_3B, text=text, system=SUMMARIZE_INSTRUCTION
        )

    async def _arun(
        self,
        text: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await self.client.achat(
            model=MODEL_SUMMARIZE_3B, text=text, system=SUMMARIZE_INSTRUCTION
        )


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

    Routes to ``zlm-v2-iab-classify-edge-enriched``. Returns the richer
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


class ZeroGPUExtractSignalsTool(_BaseZeroGPUTool):
    """Turn free text into structured contextual signals.

    Routes to ``zlm-v1-signal-extract`` (80M parameters). Returns the parsed
    signal payload -- topics, keywords, intent, and other contextual
    attributes -- from a single inference call. Takes a passage of text rather
    than the bare domain the sibling domain classifier expects.
    """

    name: str = "zerogpu_extract_signals"
    description: str = (
        "Extract structured contextual signals -- topics, keywords, intent, "
        "and other attributes -- from a passage of text in one call. Built for "
        "content enrichment, contextual intelligence, ad targeting, agent "
        "routing, and recommendation or analytics pipelines. Use the IAB tools "
        "instead when you need formal IAB taxonomy categories."
    )
    args_schema: type[TextInput] = TextInput

    def _run(
        self,
        text: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(self.client.responses(model=MODEL_EXTRACT_SIGNALS, text=text))

    async def _arun(
        self,
        text: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(
            await self.client.aresponses(model=MODEL_EXTRACT_SIGNALS, text=text)
        )


class ZeroGPUDecideToolsTool(_BaseZeroGPUTool):
    """Pick the tool a request needs and generate its arguments.

    Routes to ``zlm-v1-tool-decisions`` (0.6B parameters, 24K-token context).
    Selection and argument generation happen in a single inference step, and
    the model returns ``NO_TOOL`` when no tool is needed. Returns the parsed
    decision payload.
    """

    name: str = "zerogpu_decide_tools"
    description: str = (
        "Decide which tool a request needs and generate the query or "
        "arguments to call it with, in one step, using a 0.6B model built for "
        "agent tool selection. Returns NO_TOOL when no tool is needed. "
        "Describe the available tools in the text or the system prompt. Far "
        "cheaper than asking a reasoning tool to route."
    )
    args_schema: type[ChatInput] = ChatInput

    def _run(
        self,
        text: str,
        system: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(
            self.client.responses(model=MODEL_DECIDE_TOOLS, text=text, system=system)
        )

    async def _arun(
        self,
        text: str,
        system: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> t.Any:
        return maybe_json(
            await self.client.aresponses(
                model=MODEL_DECIDE_TOOLS, text=text, system=system
            )
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

    Routes to ``gliner2.5-base-v1`` with ``usecase="classification"``. The
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


class ZeroGPUDecideTool(_BaseZeroGPUTool):
    """Zero-shot operational decisions against caller-supplied options.

    Routes to ``gliner2.5-decide-v1`` (486M parameters) with
    ``usecase="classification"``. The ``schema`` maps each decision to its
    options, and several decisions are answered in a single call.
    """

    name: str = "zerogpu_decide"
    description: str = (
        "Make an operational decision about text -- classify intent, route "
        "agents and tools, triage a request, decide on moderation, detect "
        "completion or human handoff -- against options supplied at request "
        'time, e.g. {"route": ["billing", "support"], "handoff": ["yes", '
        '"no"]}. Answers several decisions in one call and is the cheapest '
        "tool here per input token."
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
            self.client.responses(model=MODEL_DECIDE, text=text, metadata=metadata)
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
                model=MODEL_DECIDE, text=text, metadata=metadata
            )
        )


class ZeroGPUExtractEntitiesTool(_BaseZeroGPUTool):
    """Custom-label named-entity recognition.

    Routes to ``gliner2.5-base-v1`` with ``usecase="ner"``. Extracts spans
    for each entity type listed in ``labels``.
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

    Routes to ``gliner2.5-multi-v1`` with ``usecase="extract-pii"``.
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

    Routes to ``gliner2.5-multi-v1`` with ``usecase="redact"`` and
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

    Routes to ``gliner2.5-base-v1`` with ``usecase="json"``. The grouped
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
    ZeroGPUReasonTool,
    ZeroGPUReasonMultilingualTool,
    ZeroGPUReasonQwenTool,
    ZeroGPUReasonGLMTool,
    ZeroGPUReasonDeepseekTool,
    ZeroGPUReasonGPTLunaTool,
    ZeroGPUReasonGPT6LunaTool,
    ZeroGPUReasonGPTMiniTool,
    ZeroGPUReasonGPTNanoTool,
    ZeroGPUModerateTool,
    ZeroGPUSummarizeTool,
    ZeroGPUSummarize3BTool,
    ZeroGPUClassifyIABTool,
    ZeroGPUClassifyIABEnrichedTool,
    ZeroGPUClassifyDomainTool,
    ZeroGPUExtractSignalsTool,
    ZeroGPUDecideToolsTool,
    ZeroGPUClassifyZeroShotTool,
    ZeroGPUClassifyStructuredTool,
    ZeroGPUDecideTool,
    ZeroGPUExtractEntitiesTool,
    ZeroGPUExtractPIITool,
    ZeroGPURedactPIITool,
    ZeroGPUExtractJSONTool,
]
