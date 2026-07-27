# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Catches the package up with the ZeroGPU API spec. Four models published in the
API reference had no tool — the two large open-weight reasoning models, the
follow-up question generator, and the domain-level IAB classifier — and the
enriched IAB classifier had moved to a `v2` model id that the package was not
sending. All four are now first-class tools and the stale id is fixed, taking
the toolkit from eleven tools to fifteen.

### Added

- `ZeroGPUReasonTool` (`gpt-oss-120b`) — a 117B open-weight model with a 131K
  context window, for prompts that need more reasoning headroom than the nano
  edge models.
- `ZeroGPUReasonMultilingualTool` (`qwen3-30b-a3b-fp8`) — reasoning across
  100+ languages. This model is served on the Chat Completions endpoint only,
  so it is the one tool that does not route through the Responses API.
- `ZeroGPUFollowUpQuestionsTool` (`zlm-v1-followup-questions-edge`) — returns
  the questions a reader would naturally ask next, as a `list[str]` (the model
  emits a JSON array; newline-delimited output is accepted as a fallback).
- `ZeroGPUClassifyDomainTool` (`zlm-v1-iab-domain-classifier`) — maps a bare
  domain name (e.g. `nytimes.com`) to IAB categories, topics, keywords, and
  user intent without fetching the page. Takes a `domain` argument rather than
  `text`, via the new `DomainInput` schema.
- `ZeroGPUClient.chat()` / `ZeroGPUClient.achat()`, reinstating the Chat
  Completions path (removed in 0.2.0) because `qwen3-30b-a3b-fp8` is not served
  on `/v1/responses`. Like the Responses path, it recovers the reply from the
  response body when the SDK raises `ParsingError` on a 2xx.
- `tests/unit_tests/test_models.py`, pinning every tool to the model id listed
  in the API spec's `model` enum so a server-side rename fails in CI rather
  than at request time.

### Fixed

- `ZeroGPUClient` no longer returns a reasoning model's scratchpad in place of
  its answer. `gpt-oss-120b` prepends a `reasoning` output item whose block
  also carries a `text` field, and the old "first text block wins" extraction
  would have returned that trace; the `output_text` block is now selected by
  type, with the previous behaviour kept as a fallback for models that do not
  tag their blocks.
- `ZeroGPUClassifyIABEnrichedTool` now calls
  `zlm-v2-iab-classify-edge-enriched`, the id published in the API spec. The
  package was still sending `zlm-v1-iab-classify-edge-enriched`, which the API
  currently still serves as an alias but no longer documents.

## [0.2.3] - 2026-06-08

Bug-fix release. Every tool call failed against the current ZeroGPU API
because of a response-schema mismatch, and chat tools rejected a `system`
prompt. Both are fixed; no public API changes.

### Fixed

- Tool calls no longer fail when the API response shape drifts from the
  `zerogpu-api` SDK's model (the server returns `created_at` while the pinned
  SDK expects `created`). `ZeroGPUClient` now recovers the output text from the
  response body when the SDK raises `ParsingError` on an otherwise successful
  (2xx) response, so all tools work again out of the box.
- The `system` argument on `ZeroGPUChatTool` and `ZeroGPUChatThinkingTool` no
  longer triggers a `400 invalid_request_error`. The passage is sent as a
  plain-string `input` and the system prompt is carried as the Responses API
  `instructions` field, instead of a system/user message list that the
  endpoint rejects.

## [0.2.2] - 2026-06-03

Documentation-only release: adds the ZeroGPU logo to the README so it renders
on the PyPI project page. The library is unchanged since 0.2.1.

### Changed

- `README.md` now opens with the ZeroGPU logo (`assets/logo.png`, referenced
  by absolute URL so PyPI renders it).

## [0.2.1] - 2026-06-03

First release published to PyPI. The library is unchanged since 0.2.0 — this
release establishes the automated release pipeline: tagged versions are built,
published to PyPI via Trusted Publishing (OIDC), and turned into GitHub
releases whose notes come straight from this changelog.

### Added

- CI `changelog` job that fails pull requests changing package code without a
  matching `CHANGELOG.md` update.

### Changed

- The `Release` workflow now creates the GitHub release from the matching
  `## [X.Y.Z]` section of this changelog (titled `langchain-zerogpu X.Y.Z`,
  with an install block and the built artifacts attached) instead of
  auto-generated notes, and publishes to PyPI via Trusted Publishing.

## [0.2.0] - 2026-06-02

### Changed

- All eleven tools now route through the ZeroGPU Responses API. `ZeroGPUChatTool`
  and `ZeroGPUChatThinkingTool` were migrated off chat completions onto
  `responses.create_response`, so every tool shares one SDK surface; a system
  prompt is carried as a system/user message pair.
- `__version__` is now derived from the installed distribution metadata via
  `importlib.metadata`, making `pyproject.toml` the single source of truth for
  the version.
- Adopted the canonical ZeroGPU product description across the package metadata,
  README, and docstrings.

### Removed

- `ZeroGPUClient.chat()` / `ZeroGPUClient.achat()` and the chat-completions
  response handling, superseded by the Responses API path.

### Fixed

- Corrected the summarize model identifier to `llama-3.1-8b-instruct-fast`
  (previously the hyphenated `llama-3-1-8b-instruct-fast`).

## [0.1.0] - 2026-06-01

### Added

- Initial release of `langchain-zerogpu`.
- Eleven LangChain `BaseTool` subclasses wrapping ZeroGPU small/nano language model tasks:
  - `ZeroGPUChatTool` (`LFM2.5-1.2B-Instruct`)
  - `ZeroGPUChatThinkingTool` (`LFM2.5-1.2B-Thinking`)
  - `ZeroGPUSummarizeTool` (`llama-3.1-8b-instruct-fast`)
  - `ZeroGPUClassifyIABTool` (`zlm-v1-iab-classify-edge`)
  - `ZeroGPUClassifyIABEnrichedTool` (`zlm-v1-iab-classify-edge-enriched`)
  - `ZeroGPUClassifyZeroShotTool` (`deberta-v3-small`)
  - `ZeroGPUClassifyStructuredTool` (`gliner2-base-v1`)
  - `ZeroGPUExtractEntitiesTool` (`gliner2-base-v1`)
  - `ZeroGPUExtractPIITool` (`gliner-multi-pii-v1`)
  - `ZeroGPURedactPIITool` (`gliner-multi-pii-v1`)
  - `ZeroGPUExtractJSONTool` (`gliner2-base-v1`)
- `ZeroGPUToolkit` bundling all eleven tools behind a single shared client.
- Credential resolution from constructor arguments or the `ZEROGPU_API_KEY` /
  `ZEROGPU_PROJECT_ID` environment variables, with the API key stored as a
  `pydantic.SecretStr`.
- Synchronous and asynchronous execution for every tool via the official
  `zerogpu-api` SDK.
- Clear error messages for authentication (401), access (403), rate-limit
  (429), server (5xx), and network failures.

[Unreleased]: https://github.com/zerogpu/langchain-zerogpu/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/zerogpu/langchain-zerogpu/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/zerogpu/langchain-zerogpu/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/zerogpu/langchain-zerogpu/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/zerogpu/langchain-zerogpu/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/zerogpu/langchain-zerogpu/releases/tag/v0.1.0
