<p align="center">
  <a href="https://zerogpu.ai">
    <img src="https://raw.githubusercontent.com/zerogpu/langchain-zerogpu/main/assets/logo.png" alt="ZeroGPU" width="128">
  </a>
</p>

# langchain-zerogpu

LangChain tools for [ZeroGPU](https://zerogpu.ai).

ZeroGPU is a compute-efficient inference provider for apps and agents. We run
purpose-built small and nano language models across an edge network for the
high-volume tasks you run constantly — classification, extraction, moderation,
routing, summarization — at ~10x lower latency and 50%+ lower cost than
frontier-model workflows. Auto-scaling, with zero GPU infrastructure to manage.
Plug in and you're live.

This package exposes those models as first-class LangChain
[`BaseTool`](https://python.langchain.com/docs/concepts/tools/) subclasses, so
any LangChain agent — including `create_agent` and LangGraph graphs — can offload
these repeatable NLP tasks (classification, summarization, moderation, signal /
entity / JSON extraction, PII redaction, and chat) to ZeroGPU instead of
spending frontier-model tokens — plus larger open-weight models for the calls
that do need reasoning headroom.

All calls go through the official [`zerogpu-api`](https://pypi.org/project/zerogpu-api/)
Python SDK.

## Install

```bash
pip install langchain-zerogpu
```

## Authenticate

Every request needs a ZeroGPU **API key** (starts with `zgpu-api-`). A
**project id** is optional — set one to scope requests to a specific project.
Provide them via environment variables:

```bash
export ZEROGPU_API_KEY="zgpu-api-..."
export ZEROGPU_PROJECT_ID="your-project-id"   # optional
```

…or pass them directly to any tool or the toolkit:

```python
from langchain_zerogpu import ZeroGPUSummarizeTool

tool = ZeroGPUSummarizeTool(api_key="zgpu-api-...", project_id="your-project-id")
```

The API key is stored as a `pydantic.SecretStr` and is never logged.

## The tools

| Tool class | ZeroGPU model | Purpose |
| --- | --- | --- |
| `ZeroGPUChatTool` | `LFM2.5-1.2B-Instruct` | Short single-turn chat reply |
| `ZeroGPUChatThinkingTool` | `LFM2.5-1.2B-Thinking` | Chat with a visible reasoning trace |
| `ZeroGPUReasonTool` | `gpt-oss-120b` | Heavier reasoning, 131K context |
| `ZeroGPUReasonMultilingualTool` | `qwen3-30b-a3b-fp8` | Reasoning across 100+ languages |
| `ZeroGPUReasonLongContextTool` | `glm-5.2` | Reasoning over a 262K-token context |
| `ZeroGPUReasonCodeTool` | `deepseek-v4-flash-0731` | Coding and agentic work, 1M context |
| `ZeroGPUReasonDeepseekTool` | `deepseek-v4.1-flash` | DeepSeek V4.1 Flash reasoning, 1M context |
| `ZeroGPUModerateTool` | `llama-guard-4-12b` | Safe / unsafe moderation verdict, 160K context |
| `ZeroGPUSummarizeTool` | `llama-3.1-8b-instruct-fast` | Condense a passage |
| `ZeroGPUClassifyIABTool` | `zlm-v1-iab-classify-edge` | IAB taxonomy classification |
| `ZeroGPUClassifyIABEnrichedTool` | `zlm-v2-iab-classify-edge-enriched` | IAB + topics / keywords / intent |
| `ZeroGPUClassifyDomainTool` | `zlm-v1-iab-domain-classifier` | IAB classification from a domain name |
| `ZeroGPUExtractSignalsTool` | `zlm-v1-signal-extract` | Topics / keywords / intent signals |
| `ZeroGPUClassifyZeroShotTool` | `deberta-v3-small` | Zero-shot vs. custom labels |
| `ZeroGPUClassifyStructuredTool` | `gliner2-base-v1` | Multi-axis schema classification |
| `ZeroGPUExtractEntitiesTool` | `gliner2-base-v1` | Custom-label NER |
| `ZeroGPUExtractPIITool` | `gliner-multi-pii-v1` | Extract PII entities (JSON) |
| `ZeroGPURedactPIITool` | `gliner-multi-pii-v1` | Mask PII inline with `[LABEL]` |
| `ZeroGPUExtractJSONTool` | `gliner2-base-v1` | Schema-driven JSON extraction |

## Quick start

```python
from langchain_zerogpu import ZeroGPUClassifyZeroShotTool

tool = ZeroGPUClassifyZeroShotTool()  # reads creds from the environment

print(tool.invoke({
    "text": "The new GPU smashes every benchmark we threw at it.",
    "labels": ["tech", "politics", "sports"],
}))
```

Tools work asynchronously too:

```python
result = await tool.ainvoke({"text": "...", "labels": ["a", "b"]})
```

## Bind the tools to an agent

Use the toolkit to get all nineteen tools — wired to a single shared client — and
bind them to an agent:

```python
from langchain.agents import create_agent
from langchain_zerogpu import ZeroGPUToolkit

toolkit = ZeroGPUToolkit()  # reads ZEROGPU_API_KEY / ZEROGPU_PROJECT_ID
tools = toolkit.get_tools()

agent = create_agent("anthropic:claude-sonnet-4-6", tools=tools)

agent.invoke({
    "messages": [
        {"role": "user", "content": "Redact the PII in: 'Call Jane at 555-0100.'"}
    ]
})
```

Or bind a single tool to a chat model directly:

```python
from langchain.chat_models import init_chat_model
from langchain_zerogpu import ZeroGPUExtractPIITool

llm = init_chat_model("anthropic:claude-sonnet-4-6")
llm_with_tools = llm.bind_tools([ZeroGPUExtractPIITool()])
```

## Errors

Failures surface as clear, typed exceptions instead of raw stack traces:

- `ZeroGPUAuthError` — missing / malformed credentials, `401`, or `403`.
- `ZeroGPUError` — rate limits (`429`), server errors (`5xx`), and network
  failures.

## Development

```bash
make install            # uv sync --all-groups
make lint               # ruff check + format --check
make mypy               # mypy (disallow_untyped_defs)
make test               # unit tests, sockets disabled
make integration_test   # integration tests (needs real ZeroGPU creds)
make bump               # bump the version; merging to main releases it
```

Releases are automatic — see [RELEASING.md](RELEASING.md).

## License

MIT — see [LICENSE](./LICENSE).
