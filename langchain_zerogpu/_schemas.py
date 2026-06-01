"""Shared Pydantic argument schemas for the ZeroGPU tools.

Each tool declares an explicit ``args_schema`` so that LangChain (and any chat
model binding the tool) can validate inputs and present a well-typed function
signature to the underlying LLM.

Two schemas expose a field literally named ``schema`` because that is the
natural, LLM-facing argument name for the schema-driven tools (and the name the
reference ZeroGPU plugin uses). ``schema`` harmlessly shadows the deprecated
:meth:`pydantic.BaseModel.schema` method, which makes pydantic emit a cosmetic
``UserWarning`` whenever such a model is built -- including LangChain's internal
rebuilds at tool-binding time. We register a single, narrowly-scoped filter for
exactly that message so the package stays quiet without touching unrelated
warnings.
"""

from __future__ import annotations

import typing as t
import warnings

from pydantic import BaseModel, Field

warnings.filterwarnings(
    "ignore",
    message=r'Field name "schema".*shadows an attribute in parent "BaseModel".*',
    category=UserWarning,
)


class TextInput(BaseModel):
    """A single plain-text passage to process."""

    text: str = Field(..., description="The input text to process.")


class ChatInput(BaseModel):
    """Input for a short, single-turn chat reply."""

    text: str = Field(..., description="The user message to respond to.")
    system: str | None = Field(
        default=None,
        description="Optional system prompt that steers the reply.",
    )


class ZeroShotInput(BaseModel):
    """Input for zero-shot classification against a flat list of labels."""

    text: str = Field(..., description="The text to classify.")
    labels: list[str] = Field(
        ...,
        description="Candidate labels to score the text against, e.g. "
        '["tech", "politics", "sports"].',
        min_length=1,
    )


class StructuredClassifyInput(BaseModel):
    """Input for multi-axis (schema-driven) classification."""

    text: str = Field(..., description="The text to classify.")
    schema: dict[str, t.Any] = Field(  # type: ignore[assignment] # noqa: D419
        ...,
        description="Classification axes mapped to their candidate labels, "
        'e.g. {"sentiment": ["positive", "negative"], '
        '"topic": ["billing", "support"]}.',
    )
    threshold: float | None = Field(
        default=None,
        description="Optional confidence threshold (0-1) for filtering labels.",
    )


class ExtractEntitiesInput(BaseModel):
    """Input for custom-label named-entity extraction."""

    text: str = Field(..., description="The text to extract entities from.")
    labels: list[str] = Field(
        ...,
        description='Entity types to extract, e.g. ["person", "company", "date"].',
        min_length=1,
    )
    threshold: float | None = Field(
        default=None,
        description="Optional confidence threshold (0-1) for filtering spans.",
    )


class ExtractJSONInput(BaseModel):
    """Input for schema-driven JSON field extraction."""

    text: str = Field(..., description="The text to extract fields from.")
    schema: dict[str, t.Any] = Field(  # type: ignore[assignment] # noqa: D419
        ...,
        description="Grouped extraction schema mapping a group name to a list "
        'of "field::type::description" specs, e.g. '
        '{"contact": ["name::str::Full name", "email::str::Email address"]}.',
    )
    threshold: float | None = Field(
        default=None,
        description="Optional confidence threshold (0-1) for filtering fields.",
    )


class ExtractPIIInput(BaseModel):
    """Input for grouped PII extraction."""

    text: str = Field(..., description="The text to scan for PII.")
    categories: list[str] | None = Field(
        default=None,
        description="Optional PII categories to restrict the scan to, e.g. "
        '["identity", "contact"]. Omit to extract all detected PII.',
    )
    threshold: float | None = Field(
        default=None,
        description="Optional confidence threshold (0-1) for filtering matches.",
    )


class RedactPIIInput(BaseModel):
    """Input for inline PII redaction."""

    text: str = Field(..., description="The text to redact PII from.")
    categories: list[str] | None = Field(
        default=None,
        description="Optional PII categories to restrict redaction to. Omit to "
        "redact all detected PII.",
    )
    threshold: float | None = Field(
        default=None,
        description="Optional confidence threshold (0-1) for filtering matches.",
    )
