"""Fact-check judge (optional evaluation; paid; never part of a production run). Copy of the fixed judge.py
(enterprise_agent/results/experiments/judge_finish_gemini_nano_20261123/judge.py, = e2e_stream_final_gemini_20261121/judge.py with the
cap-creep fix) with path changes only:
  - reads <out>/<slug>/verified.json (the pipeline's output) instead of a prepared verified_<label>/<slug>/final.json;
  - the registry record comes from the --companies file (any format storage.inputs reads);
  - cache, budget and results live in <out>/eval/ (judge_cache/, budget_judge_<label>.json, judge_<label>.json);
  - find_quote (evaluate/quotes.py) and the company reader (storage/inputs.py) come from this package (identical logic).
Judge logic, prompts, model (Claude Sonnet 5.5 on OpenRouter, effort medium), web tools and the verdict rule are unchanged.
usage (from Provinces/): python -m on_the_fly_pipeline evaluate judge --out OUT --companies FILE --cap-usd 5 [--label run] [--dry-run]

Original description:
JUDGE FINISH (2026-11-23): unchanged copy of ../e2e_stream_final_gemini_20261121/judge.py except this note, run on the VERIFIED output of Gemini (E10)
and GPT-5.4 Nano (E11) (read only) to finish their fact-checks, which stopped at the cap; outputs stay in this folder; judge_cache seeded from all earlier caches.
E2E LATENCY TEST, QWEN3.5-9B EXTRACTION (2026-11-20): copy of ../e2e_latency_haiku_20261120/judge.py; one fix after the run (backup
internal/judge_before_capfix.py): the +$0.05 cap headroom for a formatting call is now removed after that call (it used to stay, so the
cap crept from $5.00 to $7.50 and the judge spent $5.68). (evaluation only, not timed); its judge_cache
is seeded with a copy of that run's cache, so entities judged there with the same request are not paid again.
(Earlier note:) MERGED SINGLE-STAGE TEST, CLAUDE HAIKU 5.5 EXTRACTION (2026-11-20): unchanged copy of ../qwen36_local_2stage_23co_noreason_20261119/judge.py, run on the verifier-v3 output.
(Earlier note:) (2026-11-16, 60-company run) Changes to this copy: the page judge (L1) reads the FULL page (up to about 150k tokens) instead of 3,000
characters around the quote; the registry record (legal form, registry number) is read through company_pipeline's loader; the budget
reserve before a call is 0.5 (web) or scales with the page (L1); plus the earlier fetch-cap and formatting-call fixes. It runs on the
VERIFIED output. Budget: the cap check also counts the reserves of calls already in progress (parallel workers had overshot it).
Fact-check judge for extracted people and organizations (Claude Sonnet 5.5 through OpenRouter, anthropic/claude-sonnet-5.5, paid; run with
Provinces/.venv/bin/python). 2026-11-16: switched from the Anthropic SDK to OpenRouter (no Anthropic key on this machine); web search and
web fetch are OpenRouter server tools with the native Anthropic engine. The SDK version is kept in internal/judge_anthropic_sdk_version.py. Evaluation code only: it reads the registry record and the saved source pages; nothing here is sent to an extractor.

For every entity of a run's final.json files (one judgement per entity, on its first two mentions):
  L0 source check (code, free)   the quote is on the saved page (company pipeline find_quote), the name is in the quote or near it.
  L1 page judge (no web)         is it a real person/organization name; does the page support this relation to the target, in the right
                                 direction; is the page about our registered company or a namesake; the relation the page does support.
  L2 web fact-check              web_search (at most 5) + web_fetch (at most 2): is the relation confirmed by an independent source,
                                 contradicted, or not found; is the company in those sources ours. Skipped when L1 says "not a name".
  verdict (code)                 confirmed | minor | unverified | incorrect | wrong_entity | not_entity (the audits' key), from the layers.

Requests are cached on disk (judge_cache/<sha>.json, key = model + prompt + request); a rerun never pays twice. A hard --cap-usd stops new
calls when spent + the next call's reserve would pass it. Spend = usage.cost reported by OpenRouter. 2026-11-16 fix: web_fetch max_content_tokens 20000 (an uncapped fetch cost $2.08 in one call), and the formatting call after a paid
web check is never refused by the cap (a refused one threw away a paid result). Prices (OpenRouter models API,
2026-11-16): $2 / $10 per M input / output tokens, $0.01 per web search.
usage: Provinces/.venv/bin/python -B judge.py --run out --companies companies_sample20.txt --label s20 --cap-usd 5 [--dry-run] [--layers L1,L2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

H = Path(__file__).resolve().parent
PROV = H.parents[1]
sys.path.insert(0, str(PROV))
from on_the_fly_pipeline.evaluate.quotes import find_quote          # noqa: E402

MODEL = "anthropic/claude-sonnet-5.5"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
PRICE_IN, PRICE_OUT, PRICE_CACHE_READ, PRICE_SEARCH = 2.0, 10.0, 0.10, 0.01
CACHE = Path("judge_cache")             # set in main(): <out>/eval/judge_cache
WINDOW = 1500                      # characters on each side of the quote shown to the page judge
VERSION = "judge-v1"

RELS = ["director", "officer", "shareholder_or_owner", "employee", "parent", "subsidiary", "affiliate", "predecessor_or_former_name",
        "successor", "counsel_or_auditor", "other"]

L1_SCHEMA = {"type": "object", "additionalProperties": False,
             "required": ["is_name", "same_company", "page_supports", "supported_relation", "reason", "quote"],
             "properties": {"is_name": {"type": "string", "enum": ["yes", "no"]},
                            "same_company": {"type": "string", "enum": ["yes", "no", "unclear"]},
                            "page_supports": {"type": "string", "enum": ["yes", "partly", "no"]},
                            "supported_relation": {"type": "string", "enum": RELS + ["none"]},
                            "reason": {"type": "string"}, "quote": {"type": "string"}}}
L2_SCHEMA = {"type": "object", "additionalProperties": False,
             "required": ["online", "same_company", "sources", "reason"],
             "properties": {"online": {"type": "string", "enum": ["confirmed", "contradicted", "not_found"]},
                            "same_company": {"type": "string", "enum": ["yes", "no", "unclear"]},
                            "sources": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                                                   "required": ["url", "quote"],
                                                                   "properties": {"url": {"type": "string"}, "quote": {"type": "string"}}}},
                            "reason": {"type": "string"}}}

RECORD = """TARGET COMPANY (registry record)
legal name: {name}
province: {province}
legal form: {form}
registry number: {rid}"""

L1_PROMPT = """You are checking one item that an automatic extractor took from a web page. The extractor claims that the person or
organization below is connected to the target company in the stated way. Judge ONLY from the page text given.

{record}

EXTRACTED ITEM
name: {ename} ({etype})
claimed relation to the target: {rel}{role}
evidence quote given by the extractor: "{evidence}"
source: {url}

FULL PAGE TEXT (the quote {found} on the page)
<<<
{window}
>>>

Answer:
- is_name: is "{ename}" a real name of a person or organization (not a job title, product, placeholder, generic phrase or a date)?
- same_company: does the page describe our registered company (same name and consistent province or city), or a different
  company with the same or a similar name? "unclear" when the page gives nothing either way.
- page_supports: does the page state this relation between the item and the target? "yes" if stated as claimed; "partly" if the
  item is linked to the target but the relation is wrong or only loosely stated (for example a former role given as current is
  still "yes"; owner vs director is "partly"); "no" if the page does not link them, links the item to another organization on the
  page, or states the relation in the opposite direction (seller given as parent, buyer given as subsidiary).
- supported_relation: the relation the page does support (one of the listed values), or "none".
- reason: one or two sentences. quote: the shortest verbatim passage from the page that decides it (empty if none).
Dates and whether a role is current are not judged."""

L2_PROMPT = """Fact-check one claim about a Canadian company using web search. Search for independent sources (registries, court or
regulatory filings, company websites, press releases, news). Use at most 5 searches.

{record}

CLAIM: {ename} ({etype}) is the target company's {rel}{role}.
The claim was extracted from: {url}
Quote on that page: "{evidence}"

Answer:
- online: "confirmed" if a source other than the page above states the claim (or the same relation in other words); "contradicted"
  if a reliable source states something incompatible (a different company, the opposite direction of a transaction, a different
  role that rules this one out); "not_found" if you find nothing either way.
- same_company: do the sources you found describe our registered company (same name and consistent province or city)?
  "unclear" if they do not say.
- sources: up to 3 sources you relied on, each with its URL and a short verbatim quote.
- reason: one or two sentences.
Dates and whether a role is current are not judged. Do not count the page above as confirmation."""

LOCK = threading.Lock()


class Budget:
    def __init__(self, cap: float, path: Path):
        self.cap, self.path = cap, path
        self.spent = json.loads(path.read_text())["spent_usd"] if path.is_file() else 0.0
        self.inflight = 0.0                # reserves of calls already started (parallel workers)

    def check(self, reserve: float) -> None:
        """Refuse a call if spent + the reserves of calls in progress + this reserve would pass the cap; else hold the reserve."""
        with LOCK:
            if self.spent + self.inflight + reserve > self.cap:
                raise RuntimeError(f"budget cap ${self.cap:.2f} reached (spent ${self.spent:.4f}, in progress ${self.inflight:.2f})")
            self.inflight += reserve

    def release(self, reserve: float) -> None:
        with LOCK:
            self.inflight = max(0.0, self.inflight - reserve)

    def add(self, usd: float) -> None:
        with LOCK:
            self.spent += usd
            self.path.write_text(json.dumps({"spent_usd": round(self.spent, 6), "cap_usd": self.cap, "updated": time.time()}))


def cost(usage: dict) -> float:
    st = usage.get("server_tool_use") or {}
    return (usage.get("input_tokens", 0) * PRICE_IN + usage.get("output_tokens", 0) * PRICE_OUT
            + usage.get("cache_read_input_tokens", 0) * PRICE_CACHE_READ + usage.get("cache_creation_input_tokens", 0) * PRICE_IN * 1.25) / 1e6 \
        + (st.get("web_search_requests", 0) or 0) * PRICE_SEARCH


def load_registry(companies_file: str) -> dict:
    """slug -> {entity_form, id_value} through company_pipeline's loader (companies.jsonl has no slug field; the earlier lookup by
    "slug" always missed and sent "unknown")."""
    from on_the_fly_pipeline.storage import open_companies as load_companies
    return {c.slug: {"entity_form": c.entity_form, "id_value": (c.company_id or "").partition(":")[2]}
            for c in load_companies(companies_file)}


def company_record(c: dict, reg: dict) -> str:
    return RECORD.format(name=c["name"], province=c["province"], form=reg.get("entity_form") or "unknown", rid=reg.get("id_value") or "unknown")


def page_text(run: Path, slug: str, url: str) -> str:
    for f in (run / slug / "pages").glob("*.json"):
        p = json.loads(f.read_text(encoding="utf-8"))
        if url in (p.get("url"), p.get("final_url"), p.get("canonical_url")):
            return p.get("content") or ""
    return ""


def l0_check(ent: dict, m: dict, text: str) -> dict:
    ev = (m.get("evidence") or "").strip()
    pos = find_quote(text, ev) if ev and text else None
    name = ent["name"]
    last = name.split()[-1].lower() if name.split() else ""
    near = text[max(0, (pos or 0) - 400):(pos or 0) + len(ev) + 400].lower() if pos is not None else ""
    return {"has_url": bool(m.get("source_url")), "page_saved": bool(text), "quote_on_page": pos is not None,
            "name_in_quote": bool(last) and last in ev.lower(), "name_near_quote": bool(last) and last in near, "pos": pos}


MAX_PAGE_CHARS = 540_000            # about 150k tokens, the verifier's limit


def full_page(text: str) -> str:
    """The whole page (2026-11-16: the judge reads the full page, not 3,000 characters around the quote); longer pages are cut."""
    return text if len(text) <= MAX_PAGE_CHARS else text[:MAX_PAGE_CHARS] + "\n[... page cut at about 150k tokens ...]"


def window(text: str, pos: int | None, ev: str) -> str:
    if pos is None:
        return text[:2 * WINDOW]
    return text[max(0, pos - WINDOW):pos + len(ev) + WINDOW]


def call(client, budget: Budget, prompt: str, schema: dict, tools: list | None, effort: str, dry: bool) -> dict:
    """One OpenRouter chat completion (anthropic/claude-sonnet-5.5) with a strict JSON schema; tools = OpenRouter server tools
    (openrouter:web_search / openrouter:web_fetch, native Anthropic engine), which the model calls by itself on OpenRouter's side.
    Cost = the usage.cost OpenRouter reports (tokens + searches); cached on disk when the reply is valid JSON."""
    key = hashlib.sha256(json.dumps([VERSION, MODEL, prompt, schema, tools, effort], sort_keys=True).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.is_file():
        return {**json.loads(f.read_text()), "cached": True}
    if dry:
        return {"dry": True, "prompt_chars": len(prompt)}
    reserve = 0.5 if tools else len(prompt) / 3.0 * PRICE_IN / 1e6 + 0.03     # web: fetch capped at 20k tokens; page: scales with the full page
    budget.check(reserve)
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16000,
            "reasoning": {"effort": effort}, "usage": {"include": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": schema}}}
    if tools:
        body["tools"] = tools
        body["max_tool_calls"] = 8
    t0 = time.monotonic()
    try:
        r = client.post(OPENROUTER, json=body, timeout=(30, 900))
    finally:
        budget.release(reserve)
    secs = round(time.monotonic() - t0, 2)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    if d.get("error"):
        raise RuntimeError(f"API error: {json.dumps(d['error'])[:300]}")
    usage = d.get("usage") or {}
    usd = float(usage.get("cost") or 0.0) or cost({"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0),
                                                    "server_tool_use": {"web_search_requests": (usage.get("server_tool_use_details") or {}).get("web_search_requests", 0)}})
    budget.add(usd)
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    text = msg.get("content") or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            parsed = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            parsed = None
    if parsed is None and tools and text.strip():      # OpenRouter does not enforce the schema with server tools: one formatting call, no tools
        with LOCK:                                      # the formatting call belongs to a call already paid: never refuse it
            budget.cap += 0.05                          # temporary headroom for this one call only (2026-11-20 fix: it used to stay,
        try:                                            # so 50 formatting calls let the cap creep from $5.00 to $7.50)
            fmt = call(client, budget, "Convert this fact-check answer into the JSON format. Copy URLs and quotes exactly; do not add "
                       "anything that is not in the answer.\n\nANSWER:\n" + text, schema, None, "low", dry)
        finally:
            with LOCK:
                budget.cap -= 0.05
        parsed = fmt.get("parsed")
        usd += fmt.get("usd") or 0.0
    cites = [a.get("url_citation", {}).get("url") for a in msg.get("annotations") or [] if a.get("type") == "url_citation"]
    rec = {"finish": ch.get("finish_reason"), "parsed": parsed, "text": text, "usage": usage, "usd": round(usd, 6),
           "seconds": secs, "citations": cites, "web_searches": (usage.get("server_tool_use_details") or {}).get("web_search_requests"),
           "model": d.get("model"), "provider": d.get("provider")}
    if parsed is not None:
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    else:                                               # kept for inspection, not reused
        (CACHE.parent / "judge_unparsed").mkdir(parents=True, exist_ok=True)
        (CACHE.parent / "judge_unparsed" / f"{key[:16]}.json").write_text(json.dumps({**rec, "raw": d}, ensure_ascii=False, indent=1))
    return rec


def verdict(l0: dict, l1: dict | None, l2: dict | None) -> str:
    if not l1:
        return "error"
    if l1["is_name"] == "no":
        return "not_entity"
    if l1["same_company"] == "no" or (l2 and l2["same_company"] == "no" and l2["online"] != "confirmed"):
        return "wrong_entity"
    online = (l2 or {}).get("online", "not_found")
    if online == "contradicted":
        return "incorrect"
    if l1["page_supports"] == "yes":
        return "confirmed"
    if l1["page_supports"] == "partly":
        return "minor"
    return "confirmed" if online == "confirmed" else "incorrect" if l0["quote_on_page"] else "unverified"


def judge_entity(client, budget, run: Path, comp: dict, reg: dict, typ: str, ent: dict, layers: set, dry: bool) -> dict:
    m = ent["mentions"][0]
    text = page_text(run, comp["slug"], m.get("source_url") or "")
    l0 = l0_check(ent, m, text)
    role = f" (role: {m['role']})" if m.get("role") else ""
    rel = (m.get("relationship") or "").replace("_", " ")
    rec = company_record(comp, reg)
    p1 = L1_PROMPT.format(record=rec, ename=ent["name"], etype=typ, rel=rel, role=role, evidence=m.get("evidence") or "",
                          url=m.get("source_url"), found="was found" if l0["quote_on_page"] else "was NOT found verbatim",
                          window=full_page(text))
    r1 = call(client, budget, p1, L1_SCHEMA, None, "medium", dry) if "L1" in layers else None
    l1 = (r1 or {}).get("parsed")
    r2 = None
    if "L2" in layers and not (l1 and l1["is_name"] == "no"):
        p2 = L2_PROMPT.format(record=rec, ename=ent["name"], etype=typ, rel=rel, role=role, url=m.get("source_url"),
                              evidence=m.get("evidence") or "")
        tools = [{"type": "openrouter:web_search", "parameters": {"engine": "native", "max_uses": 5}},
                 {"type": "openrouter:web_fetch", "parameters": {"engine": "native", "max_uses": 2, "max_content_tokens": 20000}}]
        r2 = call(client, budget, p2, L2_SCHEMA, tools, "medium", dry)
    l2 = (r2 or {}).get("parsed")
    return {"slug": comp["slug"], "company": comp["name"], "name": ent["name"], "type": typ, "relationship": m.get("relationship"),
            "role": m.get("role"), "evidence": m.get("evidence"), "source_url": m.get("source_url"), "l0": {k: v for k, v in l0.items() if k != "pos"},
            "l1": l1, "l2": l2, "verdict": verdict(l0, l1, l2) if not dry else None,
            "calls": {"l1": {k: (r1 or {}).get(k) for k in ("usd", "usage", "seconds", "cached", "prompt_chars", "finish")},
                      "l2": {k: (r2 or {}).get(k) for k in ("usd", "usage", "seconds", "cached", "prompt_chars", "finish", "web_searches", "citations")}}}


def main(argv: list[str] | None = None) -> int:
    global CACHE
    ap = argparse.ArgumentParser(description="Fact-check the verified entities of a pipeline output folder (paid: Claude Sonnet 5.5).")
    ap.add_argument("--out", required=True, help="the pipeline's output folder (<slug>/verified.json and <slug>/pages)")
    ap.add_argument("--companies", required=True, help="the companies file given to the pipeline run (registry records)")
    ap.add_argument("--slugs", default=None, help="optional file with the slugs to judge (default: every company folder in --out)")
    ap.add_argument("--file", default="verified.json", help="which entity file to judge (default verified.json)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--cap-usd", type=float, required=True)
    ap.add_argument("--layers", default="L1,L2")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cache", default=None, help="judge cache folder (default <out>/eval/judge_cache); seed it to avoid paying twice")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.out).resolve()
    ev = run / "eval"
    CACHE = Path(a.cache).resolve() if a.cache else ev / "judge_cache"
    slugs = Path(a.slugs).read_text().split() if a.slugs else sorted(p.name for p in run.iterdir() if (p / a.file).is_file())
    regs = load_registry(a.companies)
    jobs = []
    for s in slugs:
        f = run / s / a.file
        if not f.is_file():
            continue
        fj = json.loads(f.read_text())
        comp = {"slug": s, "name": fj["company"]["name"], "province": fj["company"]["province"]}
        for typ, key in (("person", "people"), ("organization", "organizations")):
            for e in fj.get(key, []):
                jobs.append((comp, regs.get(s, {}), typ, e))
    client = None
    budget = Budget(a.cap_usd, ev / f"budget_judge_{a.label}.json")
    if not a.dry_run:
        client = make_client()
    layers = set(a.layers.split(","))
    t0 = time.monotonic()
    rows, errors = [], []

    def one(j):
        try:
            return judge_entity(client, budget, run, *j, layers, a.dry_run)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{j[0]['slug']} {j[3]['name']}: {type(exc).__name__}: {exc}"[:300])
            return None
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows = [r for r in pool.map(one, jobs) if r]
    out = {"label": a.label, "run": str(run), "entities": len(jobs), "judged": len(rows), "errors": errors, "spent_usd": round(budget.spent, 4),
           "wall_s": round(time.monotonic() - t0, 1), "dry_run": a.dry_run, "rows": rows}
    ev.mkdir(parents=True, exist_ok=True)
    (ev / f"judge_{a.label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


def make_client():
    """requests session with the OpenRouter key (on_the_fly_pipeline.credentials; never printed)."""
    import requests
    from on_the_fly_pipeline.credentials import openrouter_key
    key = openrouter_key()
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return s


if __name__ == "__main__":
    raise SystemExit(main())
