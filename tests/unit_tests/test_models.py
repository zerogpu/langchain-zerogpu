"""Pin every tool to the model id published in the ZeroGPU API spec.

Model identifiers are versioned server-side (the enriched IAB classifier moved
from `zlm-v1-...` to `zlm-v2-...`), and a stale id fails only at request time
with a 403/404. Pinning them here catches the drift in CI instead.
"""

from langchain_zerogpu.tools import ALL_TOOL_CLASSES

# Model ids as listed in the `model` enum of the ZeroGPU OpenAPI spec.
EXPECTED_MODELS = {
    "zerogpu_chat": "LFM2.5-1.2B-Instruct",
    "zerogpu_chat_thinking": "LFM2.5-1.2B-Thinking",
    "zerogpu_reason": "gpt-oss-120b",
    "zerogpu_reason_multilingual": "qwen3-30b-a3b-fp8",
    "zerogpu_reason_long_context": "glm-5.2",
    "zerogpu_reason_glm": "glm-5.3-flash",
    "zerogpu_reason_deepseek": "deepseek-v4.1-flash",
    "zerogpu_reason_gpt_luna": "gpt-5.6-luna",
    "zerogpu_reason_gpt_mini": "gpt-4.1-mini",
    "zerogpu_reason_gpt_nano": "gpt-5.4-nano",
    "zerogpu_moderate": "llama-guard-4-12b",
    "zerogpu_summarize": "llama-3.1-8b-instruct-fast",
    "zerogpu_classify_iab": "zlm-v1-iab-classify-edge",
    "zerogpu_classify_iab_enriched": "zlm-v2-iab-classify-edge-enriched",
    "zerogpu_classify_domain": "zlm-v1-iab-domain-classifier",
    "zerogpu_extract_signals": "zlm-v1-signal-extract",
    "zerogpu_classify_zero_shot": "deberta-v3-small",
    "zerogpu_classify_structured": "gliner2-base-v1",
    "zerogpu_extract_entities": "gliner2-base-v1",
    "zerogpu_extract_pii": "gliner-multi-pii-v1",
    "zerogpu_redact_pii": "gliner-multi-pii-v1",
    "zerogpu_extract_json": "gliner2-base-v1",
}


def test_every_tool_has_a_pinned_model() -> None:
    tool_names = {
        tool_cls.model_fields["name"].default for tool_cls in ALL_TOOL_CLASSES
    }
    assert tool_names == set(EXPECTED_MODELS)


def test_model_constants_match_the_api_spec() -> None:
    from langchain_zerogpu import tools

    assert tools.MODEL_CHAT == EXPECTED_MODELS["zerogpu_chat"]
    assert tools.MODEL_CHAT_THINKING == EXPECTED_MODELS["zerogpu_chat_thinking"]
    assert tools.MODEL_REASON == EXPECTED_MODELS["zerogpu_reason"]
    assert (
        tools.MODEL_REASON_MULTILINGUAL
        == EXPECTED_MODELS["zerogpu_reason_multilingual"]
    )
    assert (
        tools.MODEL_REASON_LONG_CONTEXT
        == EXPECTED_MODELS["zerogpu_reason_long_context"]
    )
    assert tools.MODEL_REASON_GLM == EXPECTED_MODELS["zerogpu_reason_glm"]
    assert tools.MODEL_REASON_DEEPSEEK == EXPECTED_MODELS["zerogpu_reason_deepseek"]
    assert tools.MODEL_REASON_GPT_LUNA == EXPECTED_MODELS["zerogpu_reason_gpt_luna"]
    assert tools.MODEL_REASON_GPT_MINI == EXPECTED_MODELS["zerogpu_reason_gpt_mini"]
    assert tools.MODEL_REASON_GPT_NANO == EXPECTED_MODELS["zerogpu_reason_gpt_nano"]
    assert tools.MODEL_MODERATE == EXPECTED_MODELS["zerogpu_moderate"]
    assert tools.MODEL_SUMMARIZE == EXPECTED_MODELS["zerogpu_summarize"]
    assert tools.MODEL_IAB == EXPECTED_MODELS["zerogpu_classify_iab"]
    assert tools.MODEL_IAB_ENRICHED == EXPECTED_MODELS["zerogpu_classify_iab_enriched"]
    assert tools.MODEL_IAB_DOMAIN == EXPECTED_MODELS["zerogpu_classify_domain"]
    assert tools.MODEL_EXTRACT_SIGNALS == EXPECTED_MODELS["zerogpu_extract_signals"]
    assert tools.MODEL_ZERO_SHOT == EXPECTED_MODELS["zerogpu_classify_zero_shot"]
    assert tools.MODEL_GLINER == EXPECTED_MODELS["zerogpu_classify_structured"]
    assert tools.MODEL_PII == EXPECTED_MODELS["zerogpu_extract_pii"]
