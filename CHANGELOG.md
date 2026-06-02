# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/zerogpu/langchain-zerogpu/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/zerogpu/langchain-zerogpu/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/zerogpu/langchain-zerogpu/releases/tag/v0.1.0
