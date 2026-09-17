---
name: model-sync
description: Reconcile langchain-zerogpu with the live model catalog API (https://api-dashboard.zerogpu.ai/api/models), which is the sole source of truth — correct every context window, parameter count, and cost comparison the tools state, move tools to models the API renamed, delete tools whose model it no longer returns, and add a tool for every model it returns that no tool calls, across `langchain_zerogpu/tools.py` (`MODEL_*` constants, tool docstrings and `description`s, `ALL_TOOL_CLASSES`), `langchain_zerogpu/__init__.py`, `langchain_zerogpu/toolkit.py`, `README.md`, and `tests/` — then bump the minor version (always minor, even for a removal), write the matching CHANGELOG section, cut a branch from `main`, commit, and open a PR automatically. Merging publishes to PyPI. Runs unattended — it never asks questions. Use this skill whenever the user asks to "check the LangChain tools' models", "sync langchain-zerogpu with the model catalog", "fetch models from the dashboard API and compare", "fix the context windows in the tool descriptions", "add the new models", or schedules a routine to keep the package matched to what the API serves.
---

# Model sync — langchain-zerogpu

`https://api-dashboard.zerogpu.ai/api/models` **is the sole source of truth.** Its response defines which models exist and every machine-readable fact about them — task, context window, pricing, parameter count. Where the package disagrees, the package is wrong and this skill corrects it: the model each tool calls, the tool `description`s an agent routes on, the numbers and cost comparisons in docstrings, the README tools table, the tests that pin them — and which tools exist at all.

**This skill runs unattended.** It asks nothing and waits for nothing. Every decision below is a rule with a determined answer, so a scheduled run and an interactive run do the same thing. When a rule leaves genuine slack — the wording of a rewritten clause, how to phrase a changelog bullet — pick the option most consistent with the surrounding file and note the choice in the final summary. Never end a run with an open question, a "should I…", or work deferred for a human.

**Fully in sync means both directions.** Every model the API returns is called by a tool (except the [two tasks the client cannot call](#4-add-what-is-new)), and every model a tool calls is one the API returns. The sync adds, edits, renames, and removes models and tools on its own to get there — no human decides what stays.

Work the four loops in order: **[correct](#1-correct-what-disagrees)**, **[rename](#2-follow-renames)**, **[remove](#3-remove-what-is-gone)**, **[add](#4-add-what-is-new)**. Then [verify](#5-verify), and [bump, write the changelog, branch, and open a PR against `main`](#6-bump-changelog-branch-and-open-the-pr) — every run that changes a file ends in a PR carrying a **minor** version bump and a changelog section, without being asked.

**Merging is a release.** A version bump merged to `main` is tagged and published to PyPI by `.github/workflows/release.yml` (`RELEASING.md`). PR CI fails a package change without a bump and a matching top `CHANGELOG.md` section, so the sync always writes both.

## Scope

| In scope — edit | Out of scope — never edit |
| --- | --- |
| `langchain_zerogpu/*.py` | older sections of `CHANGELOG.md` — they are history |
| `README.md` | `.github/`, `Makefile`, `RELEASING.md`, this skill |
| `tests/unit_tests/*.py`, `tests/integration_tests/*.py` | `_client.py`'s request logic — which endpoint a tool calls is not a catalog fact |
| `pyproject.toml` and `uv.lock` — through `make bump*` only | dependencies in `pyproject.toml` |
| `CHANGELOG.md` (a new top section and its link reference) | |

## Step 0 — audit

```bash
python3 .claude/skills/model-sync/scripts/audit-models.py
```

| Label | Meaning | Handled by |
| --- | --- | --- |
| `PROSE` | a docstring, tool description, or README line states a price, context window, or parameter count that is true of none of the models it is about, with `file:line` | [loop 1](#1-correct-what-disagrees) |
| `RENAME` | a `MODEL_*` id the API now serves under a longer id, plus every file to update | [loop 2](#2-follow-renames) |
| `ORPHAN` | a `MODEL_*` id the API does not return, the tools it leaves with no model (`TOOL:`), and every file to clean | [loop 3](#3-remove-what-is-gone) |
| `COUNT` | a tool count (`all seventeen tools`, `len(tools) == 17`) that disagrees with `ALL_TOOL_CLASSES` | fix the number |
| `ADD` | an API model no tool calls, with the template tool to copy | [loop 4](#4-add-what-is-new) |
| `NOTE` | a `Text Embedding` or `Text Moderation` model — the client has no endpoint for it | nothing — list them in the summary |

Flags: `--model <id>` (one model, repeatable), `--save` / `--json` (snapshot then re-run offline), `--strict` (exit 1 when anything is reported). The script only reports; every edit is by hand.

**Abort conditions.** If the fetch fails, times out, or returns zero models, change nothing, say the sync did not run, and stop. A partial or empty payload must never be treated as "the API removed everything". This is the one case where the skill does less than a full sync — it is a failure to report, not a question to ask.

Fields the script does not print:

```bash
curl -s https://api-dashboard.zerogpu.ai/api/models | python3 -m json.tool
```

## 1. Correct what disagrees

Work each `PROSE` line. Each names the file, the line, what it says, and what the API has.

- **Numbers.** Token counts in the short or long form the line already uses (`131K`, `1,048,576-token`); `262,144` becomes `262K`. Parameter counts exactly as the API writes them (`120B`, `30B`). Architecture detail next to a count — `MoE`, `13B active per token`, `100+ languages` — stays while still true.
- **Descriptions are routing.** A tool's `description` is what an agent reads to choose a tool, so a wrong number there sends work to the wrong model. Correct it first, then the class docstring above it, then the README tools table row. Keep each one's length and shape; ruff's 88-column limit applies to the wrapped string literals.
- **Derived claims.** Cost ratios, superlatives, and routing hints are not in the payload but follow from it, and the audit cannot check them. Whenever a price or context window moved, recompute every claim that depends on it from the payload's own numbers:
  - multipliers — `roughly twenty times zerogpu_reason per token` — rounded the way the text already rounds; when input and output ratios diverge widely, state both;
  - superlatives and orderings — `the most capable and the most expensive ZeroGPU model by a wide margin`, `Cheaper than zerogpu_reason_long_context` — true only if the payload still says so;
  - routing hints — `prefer zerogpu_reason unless the input genuinely does not fit in its 131K context`, `for inputs that do not fit` — true only while the context windows still order that way. When a long-context tool's window is no longer larger than the tool it tells agents to prefer, rewrite the hint around what still distinguishes them, or drop it.

  When a whole clause stops being true, rewrite the clause rather than swapping digits, keeping the sentence's voice and length. A tool's `name` and class name never change, even when a number beside them changes (`zerogpu_reason_long_context` keeps its name at 262K).

The audit only checks lines that name a model or a tool, and descriptions wrap across lines, so sweep for every old value you changed:

```bash
grep -rn "1M\|1,048,576\|131K\|twenty\|priciest\|most expensive\|most capable" langchain_zerogpu README.md tests
```

## 2. Follow renames

A `MODEL_*` id that an API id extends — `deepseek-v4-flash` in the package, `deepseek-v4-flash-0731` in the API — is the same model under a new id, provided exactly one API id extends it. The audit prints it as `RENAME` with every file under `REPLACE:`.

Replace the old id with the new one in place: the `MODEL_*` constant's value (never its name), every docstring and description naming the id, the README tools table, the `EXPECTED_MODELS` pin in `tests/unit_tests/test_models.py`, and any test comment. Keep the tool's class, name, endpoint, and wording. Then correct whatever values drifted with it ([loop 1](#1-correct-what-disagrees)). Do not keep the old id as an alias.

## 3. Remove what is gone

A `MODEL_*` id the API does not return, with no single successor, is removed from the package in full, without asking. The audit prints the tools it breaks under `TOOL:` and every file under `REMOVE:`. There is no exemption list: a model the API does not return is not a ZeroGPU model, and a tool that calls it already fails on every request.

Every tool calls exactly one model constant, so a gone model deletes every tool that uses it (`gliner2-base-v1` backs three). For each one:

1. **`langchain_zerogpu/tools.py`** — delete the class, its `MODEL_*` constant, its `ALL_TOOL_CLASSES` entry, and any helper or instruction constant only it used (the follow-up tool's `_questions`). Remove its `_schemas.py` input class only if no remaining tool uses it.
2. **`langchain_zerogpu/__init__.py`** — its import and its `__all__` entry, and its capability in the module docstring's task list if no remaining tool provides it.
3. **`README.md`** — its tools-table row, and its capability in the intro paragraph's task list if no remaining tool provides it.
4. **`tests/`** — its `EXPECTED_MODELS` entry and any `assert tools.MODEL_…` line in `test_models.py`; its import and `Test…Unit` class in `test_standard.py`; its name in `test_imports.py`; its import and `Test…Integration` class in `integration_tests/test_tools.py`; and a test file that exists only for its helper (`test_questions.py` for `_questions`) — `git rm` it.
5. **Counts** — every "all seventeen" and `== 17` the audit's `COUNT` lines name once `ALL_TOOL_CLASSES` shrinks, including test function names (`test_get_tools_returns_seventeen_tools`).
6. **Cascade.** When a removal leaves a list, sentence, or comment naming nothing, delete it rather than leaving it empty. A description of another tool that points at the deleted one (`prefer zerogpu_followup_questions`) is rewritten without it.

Removing a tool class breaks anyone who imports it; say so in the changelog bullet. The bump is still **minor**.

## 4. Add what is new

Every model the API returns gets a tool, without asking. The audit prints each model no tool calls as `ADD`, with the template to copy. Run this loop after [remove](#3-remove-what-is-gone), so a name a removal freed can be reused — a model the API replaced with a new id ends up with a tool of the same name calling the new model.

**Not callable: `Text Embedding` and `Text Moderation`.** `_client.py` only calls the Responses and Chat Completions endpoints, and those models answer only on their own. The audit prints them as `NOTE`; they get no tool, and `_client.py` is not changed to reach them.

**Every other model gets a tool** in `langchain_zerogpu/tools.py`:

1. **Template** (the audit names it). The existing tool for the same task: `Text Generation` → `ZeroGPUReasonCodeTool`, `Summarization` → `ZeroGPUSummarizeTool`, `Text Classification` → `ZeroGPUClassifyDomainTool`, `PII` → `ZeroGPUExtractPIITool`. Copy the class; keep its `args_schema`, its endpoint (`client.responses` or `client.chat`), its output handling, and the endpoint comment.
2. **Names.** `MODEL_<WHAT>` constant, `ZeroGPU<What>Tool` class, and `name` `zerogpu_<what>`, following the existing tools (`MODEL_REASON_CODE` / `ZeroGPUReasonCodeTool` / `zerogpu_reason_code`). `<what>` is the task verb plus what distinguishes the model, from its id: `zlm-v1-signal-extract` → `zerogpu_extract_signals`; a new chat model → `zerogpu_reason_<family>` (`deepseek-v4.1-flash` → `zerogpu_reason_deepseek`). If a name is taken, append the distinguishing part of the id.
3. **`description` and docstring.** The `description` is what an agent routes on: write it in the template's shape and length, from the payload alone — what `pricing.description` says the model is for, `pricing.use_cases`, parameter count, context window. The docstring names the model id with the same facts. A comparison with another tool only if it follows from the payload's own prices. Wrap at ruff's 88 columns.
4. **Registry.** Add the class to `ALL_TOOL_CLASSES` next to its template, and to `__init__.py`'s import and `__all__`; add its capability to the module docstring's task list if no tool names it yet.
5. **`README.md`** — a tools-table row in the template's shape, and its capability in the intro's task list if new.
6. **`tests/`** — its `EXPECTED_MODELS` entry in `test_models.py`; a `Test…Unit` class in `test_standard.py` and a `Test…Integration` class in `integration_tests/test_tools.py`, each copied from the template's; its name in `test_imports.py`.
7. **Counts** — every "all seventeen", `== 17`, and test function name the audit's `COUNT` lines name once `ALL_TOOL_CLASSES` grows.

## Rules

### Endpoints

Never change whether an existing tool calls `client.responses` or `client.chat`. A renamed model keeps its tool's endpoint, a new tool keeps its template's, and the payload's sample bodies are not evidence of routability either way — the comments explaining why a tool uses Chat Completions stay.

### Never invent

API-sourced facts only: id, task, `maxTokens`, input/output price, parameter count, `pricing.use_cases`. Architecture details (`MoE`, `13B active per token`), language counts, and use-case phrasing may stay while still true, or come from `pricing.description` — never generated. Comparisons that follow from the payload's own prices are allowed. Never invent an argument or an example output. A new tool is built only as [loop 4](#4-add-what-is-new) describes.

## 5. Verify

Verification gates the PR: nothing is pushed until all of it passes.

```bash
python3 .claude/skills/model-sync/scripts/audit-models.py --strict   # expect: 0 finding(s) (NOTE lines are fine)
make install     # uv sync --all-groups — install uv first with `pip install uv` if it is missing
make lint        # ruff check + format --check; run `make format` and re-check if only formatting failed
make mypy
make test        # unit tests, sockets disabled
uv run python -c "from langchain_zerogpu import ZeroGPUToolkit; print(len(ZeroGPUToolkit(api_key='zgpu-api-x').get_tools()))"
```

Integration tests need live ZeroGPU credentials and are not part of the gate.

Confirm every renamed or removed id and class is gone, and nothing outside scope changed:

```bash
grep -rn "<old-id>\|<DeletedToolClass>" langchain_zerogpu tests README.md
git status --short | grep -vE ' (langchain_zerogpu/|tests/|README\.md$|CHANGELOG\.md$|pyproject\.toml$|uv\.lock$)'   # expect: no output
```

After step 6's bump and changelog, run the same release check CI runs on the PR:

```bash
git fetch --no-tags origin main
OLD=$(git show origin/main:pyproject.toml | sed -n 's/^version = "\(.*\)"$/\1/p' | head -1)
NEW=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -1)
TOP=$(awk '/^## \[/{print; exit}' CHANGELOG.md | sed 's/^## \[\([^]]*\)\].*/\1/')
uv lock --check
echo "main $OLD -> PR $NEW, CHANGELOG top: $TOP"   # expect: NEW above OLD, TOP equal to NEW
```

If a check fails, fix the cause and re-run it. If it still fails, commit nothing, open no PR, and report the failure with the command output — a broken release is worse than a stale number. CI runs lint, mypy, and unit tests on Python 3.10–3.13 and the release check on the PR.

## 6. Bump, changelog, branch, and open the PR

Once verification passes, ship it. No questions, no waiting.

**Nothing changed?** If the audit reported 0 findings and no file was modified, bump nothing, write no changelog, create no branch and no PR. Report "already in sync" and stop.

```bash
# 1. a fresh branch cut from up-to-date main — never commit on main
git fetch origin
BRANCH="model-sync/$(date -u +%Y-%m-%d-%H%M)"
git switch --create "$BRANCH" origin/main
```

Cutting from `origin/main` makes the branch unique per run and bases the bump on the version `main` actually carries. If edits were made on another branch, carry them over (`git stash` before the switch, `git stash pop` after) and re-run the [verify](#5-verify) commands.

### Version

**Every run that changes a file bumps the minor version**, exactly once: `make bump-minor`. Always minor — whatever the run did, including adding a tool, deleting one, renaming a model, or correcting a single number. Never `make bump` (patch) and never `make bump-major`, even for a removal.

Each rewrites `pyproject.toml` and `uv.lock` and nothing else; commit both. Never edit the version by hand — the release refuses when `uv.lock` disagrees.

### Changelog

Add a section at the **top** of `CHANGELOG.md`, directly above the current top `## [` section. Its heading is exactly `## [<new version>] - <today, UTC, YYYY-MM-DD>` — CI and the release read the version from the first `## [` line — and its body becomes the GitHub release notes verbatim. Also add `[<new version>]: https://github.com/zerogpu/langchain-zerogpu/compare/v<old>...v<new>` at the top of the link references at the bottom of the file. Never edit an older section: its numbers were true when it shipped.

Write it in the voice of the sections below it — a short narrative paragraph, then bullets for what a user of the package sees:

```md
## [<new version>] - <YYYY-MM-DD>

Model catalog sync: <one or two sentences on what changes for someone using the package — which tools now call a different model, which descriptions were steering agents wrong, which tool is new, which tool is gone>. Tool names and outputs are unchanged apart from the notes below.

### Added

- `ZeroGPUExtractSignalsTool` (`zerogpu_extract_signals`) calls `zlm-v1-signal-extract`, which the ZeroGPU API now serves — <what it is for, from the API's description>. The toolkit now returns N+1 tools.

### Changed

- `ZeroGPUReasonCodeTool` now calls `<new-id>`. The API renamed `<old-id>`, and the old id no longer resolves.
- `ZeroGPUReasonLongContextTool`: `glm-5.2`'s context window is 262K, not 1M. Corrected in the tool description agents route on, its docstring, and the README; <the comparison that no longer held, and what it says now>.

### Removed

- `ZeroGPUFollowUpQuestionsTool` (`zerogpu_followup_questions`). `<model>` is no longer served by the ZeroGPU API, so the tool failed on every call. Remove it from your imports; the toolkit now returns N-1 tools.
```

Rules for the section:

- One bullet per user-visible change, class name first, `was → now` in the prose. Group several numbers for one tool into one bullet.
- Only `### Added`, `### Changed`, and `### Removed`, in that order. An empty one is deleted, not left as a heading. No `### Install` — the release appends one.
- Do not mention embedding or moderation models, test edits, or the audit script.

```bash
# 2. stage only what the sync touched — never `git add -A`
git add langchain_zerogpu/... README.md tests/... CHANGELOG.md pyproject.toml uv.lock
git status --short          # confirm nothing unrelated is staged
```

A repo that was dirty before the run stays dirty: unrelated work is not the sync's to commit. A deleted test file is staged with `git rm`, so the removal lands in the commit.

```bash
# 3. commit
git commit -m "$(cat <<'EOF'
chore: sync models with dashboard API (v<new version>)

<one line per change, e.g.:>
- ZeroGPUReasonCodeTool: deepseek-v4-flash -> deepseek-v4-flash-0731 (constant, docs, README, pin)
- ZeroGPUReasonLongContextTool: glm-5.2 context 1M -> 262K; routing hint rewritten (description, README)
- ZeroGPUReasonTool: gpt-oss-120b parameters 117B -> 120B (docstring)
- remove ZeroGPUFollowUpQuestionsTool: zlm-v1-followup-questions-edge no longer served
- add ZeroGPUExtractSignalsTool: zlm-v1-signal-extract now served (tool, exports, README, tests)
- version 0.2.4 -> 0.3.0; CHANGELOG section added

Source: https://api-dashboard.zerogpu.ai/api/models

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
)"

# 4. push and open the PR against main
git push -u origin "$BRANCH"
gh pr create --base main --head "$BRANCH" \
  --title "chore: sync models with dashboard API (v<new version>)" \
  --body "$(cat <<'EOF'
Automated model-catalog sync. The dashboard API is the source of truth; every value below was taken from it. **Merging publishes `langchain-zerogpu` v<new version> to PyPI** — the release tags it and uses the CHANGELOG section below as the release notes.

## Version
<old> → <new> (minor — every sync is a minor bump)

## Corrected
| Tool | Model | Field | Was | Now |
| --- | --- | --- | --- | --- |

## Added
| Tool | Model | Template |
| --- | --- | --- |

## Renamed
## Removed
<tool class — model gone; what an importer must change>

## Not changed
- Embedding and moderation models (no client endpoint): <ids>

## CHANGELOG
<the new section, verbatim>

## Verification
- `audit-models.py --strict` — 0 findings
- `make lint`, `make mypy`, `make test` pass
- version above `main`'s, `uv.lock` in sync, and the top CHANGELOG section matches it

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Rules for this step:

- **Never** commit to `main`, push a tag, run `uv publish` or `twine upload`, force-push, merge the PR, or delete a branch. Opening the PR is the whole job; review and the release are someone else's.
- Fill both templates with the run's actual changes. An empty section is deleted, not left as a heading.
- Each run gets its own timestamped branch. Check for an earlier sync PR still open, and if there is one, say "supersedes #N" in the new PR's body and leave the old one alone:
  ```bash
  gh pr list --state open --json number,headRefName \
    --jq '.[] | select(.headRefName | startswith("model-sync/")) | "#\(.number) \(.headRefName)"'
  ```
- If the push or `gh pr create` fails — no auth, no network, protected branch — the commit still stands on the branch. Report the exact error and the branch name so it can be pushed later. Do not retry in a loop, and do not fall back to committing on `main`.

## 7. Report

One pass, no questions: values corrected, claims rewritten, models renamed, tools deleted, tools created (with the names and template chosen), the minor version bump, embedding and moderation models left without a tool, any claim that could not be sourced, and the PR URL (or the branch name and the exact error if the PR could not be opened).
