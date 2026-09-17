#!/usr/bin/env python3
"""Audit langchain-zerogpu against the live model catalog API.

The API (https://api-dashboard.zerogpu.ai/api/models) is the source of truth for every
machine-readable fact about a model: pricing, maxTokens, task, parameters.

Reports:
  * PROSE   — a docstring, tool description, or README line stating a context window,
              parameter count, or price that is true of none of the models it is about
  * RENAME  — a MODEL_* id the API now serves under a longer id, plus every file to update
  * ORPHAN  — a MODEL_* id the API does not return, the tools it breaks, and every file
              to clean
  * COUNT   — a tool count ("all seventeen tools", `len(tools) == 17`) that disagrees
              with ALL_TOOL_CLASSES
  * ADD     — an API model no tool calls, with the template tool a new tool copies
  * NOTE    — an embedding or moderation model no tool calls (the client has no endpoint for it)

Read-only. Every finding is located for the caller to edit by hand.

A line is "about" the models it names, the models of any tool it names (by class name
or `zerogpu_*` tool name), and — inside a tool class in tools.py — that tool's own
model. Cost ratios ("roughly twenty times") and superlatives ("the priciest") are not
checked; sweep for them by hand whenever a price or context window moves.

Usage:
  python3 .claude/skills/model-sync/scripts/audit-models.py
  python3 .claude/skills/model-sync/scripts/audit-models.py --model glm-5.2
  python3 .claude/skills/model-sync/scripts/audit-models.py --save models.json
  python3 .claude/skills/model-sync/scripts/audit-models.py --json models.json   # offline
  python3 .claude/skills/model-sync/scripts/audit-models.py --strict             # CI/loop
"""

import argparse
import json
import os
import re
import sys
import urllib.request

API_URL = "https://api-dashboard.zerogpu.ai/api/models"

TOOLS = "langchain_zerogpu/tools.py"
README = "README.md"
CHANGELOG = "CHANGELOG.md"
# Files whose prose states model facts. Tests are searched for renames and removals only.
PROSE_FILES = [
    TOOLS,
    "langchain_zerogpu/_client.py",
    "langchain_zerogpu/__init__.py",
    "langchain_zerogpu/toolkit.py",
    README,
]
SEARCH_PATHS = ["langchain_zerogpu", "tests", README]

NUMBER_WORDS = (
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two "
    "twenty-three twenty-four twenty-five"
).split()


# --------------------------------------------------------------------------- helpers


def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    if not os.path.exists(os.path.join(root, TOOLS)):
        root = os.getcwd()
    return root


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "langchain-zerogpu-audit"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def read_lines(root, rel):
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            return f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []


CONSTANT = re.compile(r'^(MODEL_\w+) = "([^"]+)"')
CLASS = re.compile(r"^class (ZeroGPU\w+Tool)\(")
TOOL_NAME = re.compile(r'^\s+name: str = "(zerogpu_\w+)"')
USES = re.compile(r"model=(MODEL_\w+)")
REGISTRY = re.compile(r"^ALL_TOOL_CLASSES\b")


def parse_tools(root):
    """(constants {MODEL_X: id}, tools {class: {"name", "models", "start"}}, registry size)."""
    lines = read_lines(root, TOOLS)
    constants, tools, current = {}, {}, None
    registry, in_registry = 0, False
    for n, line in enumerate(lines):
        m = CONSTANT.match(line)
        if m:
            constants[m.group(1)] = m.group(2)
        m = CLASS.match(line)
        if m:
            current = m.group(1)
            tools[current] = {"name": None, "models": set(), "start": n}
            continue
        if line and not line[0].isspace() and not line.startswith(("class ", ")", "#")):
            current = None
        if current:
            m = TOOL_NAME.match(line)
            if m:
                tools[current]["name"] = m.group(1)
            tools[current]["models"] |= set(USES.findall(line))
        if REGISTRY.match(line):
            in_registry = True
            continue
        if in_registry:
            if line.startswith("]"):
                in_registry = False
            elif re.match(r"^\s+ZeroGPU\w+Tool,", line):
                registry += 1
    for info in tools.values():
        info["models"] = {constants[c] for c in info["models"] if c in constants}
    tools = {k: v for k, v in tools.items() if v["models"]}
    return constants, tools, registry


def class_at(tools, line_no):
    """The tool class whose body contains this tools.py line, if any."""
    best = None
    for cls, info in tools.items():
        if info["start"] <= line_no and (best is None or info["start"] > tools[best]["start"]):
            best = cls
    return best


def mentions(line, ident):
    """True when the line names this exact id — not a longer id that starts with it."""
    return re.search(r"(?<![\w.-])" + re.escape(ident) + r"(?![\w-]|\.\w)", line) is not None


def human_forms(n):
    """Ways a token count is legitimately written in prose:
    1048576 -> 1,048,576 / 1024K / 1048K / 1M;  131072 -> 131,072 / 128K / 131K."""
    forms = {f"{n:,}", str(n)}
    if n % 1_048_576 == 0:
        forms.add(f"{n // 1_048_576}M")
    if n % 1024 == 0:
        forms.add(f"{n // 1024}K")
    if n >= 1000:
        forms.add(f"{round(n / 1000)}K")
    if n >= 1_000_000:
        forms.add(f"{round(n / 1_000_000)}M")
    return forms


def money(v):
    return f"${v:.2f}" if round(v, 2) == v else f"${v:g}"


# Context-window claims only: not "per 1M tokens" prices or "78 tokens in / 41 out" usage.
TOKENS = re.compile(
    r"(?<!per )\b(\d[\d,]*|\d+(?:\.\d+)?[KM])[-\s](?:token|context)(?!s? (?:in|out)\b)"
)
PARAMS = re.compile(r"\b(\d+(?:\.\d+)?[BM])\s+(?:MoE|param|model)")
PRICE_PAIR = re.compile(r"\\?\$(\d+(?:\.\d+)?)\s*/\s*\\?\$(\d+(?:\.\d+)?)")


def stale_numbers(line, facts):
    """Claims on the line that are true of none of the models it is about."""
    out = []

    def untrue(values, ok):
        return values and not any(ok(v) for v in values)

    for a, b in PRICE_PAIR.findall(line):
        pairs = [(f["in"], f["out"]) for f in facts if f["in"] is not None]
        if untrue(pairs, lambda p: abs(p[0] - float(a)) < 1e-9 and abs(p[1] - float(b)) < 1e-9):
            api = " or ".join(sorted({f"{money(i)} / {money(o)}" for i, o in pairs}))
            out.append(f"says ${a} / ${b} per 1M — API has {api}")
    for raw in TOKENS.findall(line):
        raw = raw.rstrip(",")
        windows = [f["maxTokens"] for f in facts if f["maxTokens"]]
        if untrue(windows, lambda n: raw in human_forms(n)):
            api = " or ".join(f"{n:,}" for n in sorted(set(windows)))
            out.append(f"says '{raw}' tokens — API maxTokens is {api}")
    for raw in PARAMS.findall(line):
        params = [f["params"] for f in facts if f["params"]]
        if untrue(params, lambda p: p == raw.upper()):
            out.append(f"says '{raw}' parameters — API says {' or '.join(sorted(set(params)))}")
    return out


def searchable_files(root):
    """Every .py / .md file a model id can appear in. CHANGELOG is history, never edited."""
    out = []
    for base in SEARCH_PATHS:
        path = os.path.join(root, base)
        if os.path.isfile(path):
            out.append(base)
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for fn in sorted(filenames):
                if fn.endswith((".py", ".md")):
                    out.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return out


def locations(root, mid, users, rename):
    hits = []
    for rel in searchable_files(root):
        count = sum(1 for line in read_lines(root, rel) if mentions(line, mid))
        if not count:
            continue
        if rel == TOOLS:
            note = (
                "(change the MODEL_* value and every docstring / description naming it)"
                if rename
                else f"(delete the tool class(es) {', '.join(users)}, their constant, "
                "ALL_TOOL_CLASSES entries, and helpers only they use)"
            )
        elif rel == README:
            note = (
                "(replace it in the tools table and every mention)"
                if rename
                else "(delete the tools-table row(s) and every mention)"
            )
        elif rel.startswith("tests/"):
            note = (
                "(replace the pinned id)"
                if rename
                else "(delete the tool's test classes and pins; swap sample-data uses for a surviving model)"
            )
        else:
            note = "(replace the mention)" if rename else "(rewrite or delete the mention)"
        hits.append(
            f"{'REPLACE' if rename else 'REMOVE'}: {rel} ({count} line{'s' if count != 1 else ''})  {note}"
        )
    if not rename:
        listed = {h.split(" ")[1] for h in hits}
        for rel in searchable_files(root):
            if rel in listed or rel == TOOLS:
                continue
            found = [c for c in users if any(mentions(line, c) for line in read_lines(root, rel))]
            if found:
                hits.append(f"REMOVE: {rel}  (drop {', '.join(found)})")
    return hits


# "all seventeen tools", "bundles all seventeen behind", "returns_seventeen_tools".
_WORDS = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
COUNT_WORDS = re.compile(r"\b(?:all (" + _WORDS + r")|(" + _WORDS + r") (?:ZeroGPU )?tool)\b", re.I)
COUNT_INT = re.compile(r"len\((?:tools|tool_names|ALL_TOOL_CLASSES)\)(?: == len\(ALL_TOOL_CLASSES\))? == (\d+)")


def stale_counts(root, size):
    """Tool counts that disagree with ALL_TOOL_CLASSES."""
    out = []
    for rel in searchable_files(root):
        for n, line in enumerate(read_lines(root, rel)):
            said = [
                NUMBER_WORDS.index((a or b).lower()) + 1
                for a, b in COUNT_WORDS.findall(line.replace("_", " "))
            ]
            said += [int(v) for v in COUNT_INT.findall(line)]
            for v in said:
                if v != size:
                    out.append(
                        f"COUNT: {rel}:{n + 1}: says {v} tools — ALL_TOOL_CLASSES has {size}: {line.strip()[:100]}"
                    )
    return out


# --------------------------------------------------------------------------- main


# Tasks `_client.py` cannot call (their models answer only on their own endpoints), and
# the tool a new tool for each other task is copied from — see SKILL.md loop 4.
NO_ENDPOINT = {"Text Embedding", "Text Moderation"}
TEMPLATES = {
    "Text Generation": "ZeroGPUReasonCodeTool",
    "Summarization": "ZeroGPUSummarizeTool",
    "Text Classification": "ZeroGPUClassifyDomainTool",
    "PII": "ZeroGPUExtractPIITool",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="read the model list from this file instead of the API")
    ap.add_argument("--save", help="write the fetched payload here")
    ap.add_argument("--model", action="append", help="limit the report to these model ids")
    ap.add_argument("--strict", action="store_true", help="exit 1 when anything is reported")
    args = ap.parse_args()

    root = repo_root()

    if args.json:
        with open(args.json, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = fetch(API_URL)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    models = payload.get("models") or []
    if not models:
        print("ERROR: the API returned no models — do not touch the package.", file=sys.stderr)
        return 2

    api = {m["modelId"]: m for m in models}
    constants, tools, registry = parse_tools(root)
    package_ids = set(constants.values())

    successors = {}
    for oid in sorted(package_ids - set(api)):
        cands = [a for a in api if a.startswith(oid + "-") and a not in package_ids]
        successors[oid] = cands[0] if len(cands) == 1 else None

    def facts(mid):
        m = api.get(mid) or api.get(successors.get(mid) or "")
        if not m:
            return None
        p = m.get("pricing") or {}
        return {
            "in": p.get("input_per_1m_tokens"),
            "out": p.get("output_per_1m_tokens"),
            "maxTokens": m.get("maxTokens"),
            "params": str(m.get("parameters") or "").replace(" ", "").upper() or None,
        }

    refs = {}  # identifier naming a tool -> its models
    for cls, info in tools.items():
        refs[cls] = info["models"]
        if info["name"]:
            refs[info["name"]] = info["models"]

    wanted = set(args.model or [])

    print(f"API: {len(models)} models — {', '.join(api)}")
    print(f"Package: {len(tools)} tools calling {len(package_ids)} models — {', '.join(sorted(package_ids))}\n")

    findings = 0

    # --- prose: stale numbers in docstrings, descriptions, and the README -------------
    prose = []
    for rel in PROSE_FILES:
        lines = read_lines(root, rel)
        for n, line in enumerate(lines):
            about = {mid for mid in package_ids | set(api) if mentions(line, mid)}
            # A wrapped description names a tool on one line and its number on the next.
            context = line if n == 0 or rel != TOOLS else lines[n - 1] + " " + line
            for ident, mids in refs.items():
                if mentions(context, ident):
                    about |= mids
            if rel == TOOLS:
                own = class_at(tools, n)
                if own:
                    about |= tools[own]["models"]
            if wanted and not (about & wanted or {successors.get(w) for w in about} & wanted):
                continue
            fs = [f for f in map(facts, sorted(about)) if f]
            for claim in stale_numbers(line, fs):
                prose.append(f"PROSE: {rel}:{n + 1}: {claim}: {line.strip()[:120]}")
    prose = list(dict.fromkeys(prose))
    for p in prose:
        print(p)
    findings += len(prose)
    if prose:
        print()

    # --- renames and removals -----------------------------------------------------------
    for oid, new in successors.items():
        if wanted and oid not in wanted and new not in wanted:
            continue
        users = [cls for cls, info in tools.items() if oid in info["models"]]
        names = ", ".join(f"{c} ({tools[c]['name']})" for c in users)
        if new:
            print(f"RENAME: {oid} -> {new} — the API serves it under the new id (tools: {names}).")
        else:
            print(f"ORPHAN: {oid} is called by the package but the API does not return it (tools: {names}).")
            for cls in users:
                print(f"   TOOL: {cls} ({tools[cls]['name']}) has no model left — delete the tool")
        for loc in locations(root, oid, users, bool(new)):
            print(f"   {loc}")
        print()
        findings += 1

    # --- tool counts --------------------------------------------------------------------
    if not wanted:
        counts = list(dict.fromkeys(stale_counts(root, registry)))
        for c in counts:
            print(c)
        findings += len(counts)
        if counts:
            print()

    # --- API models no tool calls -------------------------------------------------------
    called = package_ids | {s for s in successors.values() if s}
    for mid, m in api.items():
        if mid not in called and (not wanted or mid in wanted):
            task = m.get("taskDisplayName") or "?"
            if task in NO_ENDPOINT:
                print(f"NOTE: {mid} ({task}) — no tool calls it; the client has no endpoint for this task")
                continue
            template = TEMPLATES.get(task, TEMPLATES["Text Generation"])
            print(f"ADD: {mid} ({task}) — no tool calls it; copy {template}")
            findings += 1

    print(f"\n{findings} finding(s).")
    return 1 if (args.strict and findings) else 0


if __name__ == "__main__":
    sys.exit(main())
