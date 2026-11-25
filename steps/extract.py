"""Extraction (E6): one request per selected page; a page longer than [extraction].batch_tokens is split into pieces.

Each request = the extraction prompt + the company header + one page block (shared.extract.batch_user_message, the validated page format with
a page number). The model returns strict JSON {"e": [{n, t, rel, r, ev, pg}]}: name, p|o, relation (shared.extract.RELATION_TYPES), role,
verbatim evidence, page number. The page number is mapped back to the page's exact URL in code; an item citing any other page number gets
no URL and is listed in source_url_problems (it is kept in extract.json but cannot become a claim).

Attempts (as E6 / E9): at most two. A failed attempt is an HTTP error, a network error, a reply not ending with "stop", or invalid JSON.
For a profile with length_retry_max_tokens (local Qwen), a reply cut off at max_tokens is retried with that larger cap.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..shared.config import resolve_path
from ..shared.models import Company
from ..shared.llm import BudgetExceeded, ChatClient

# ---------------------------------------------------------------- selected pages (was core/pages.py)
# From company_pipeline's extract.py page_records, without the list-page cap and identity-gate branches (never used here).

def page_records(out_root: Path, company: Company, kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Selected pages, rank = the result number (r007 -> 7), as in the validated runs."""
    pages = []
    for k in kept:
        p = json.loads((out_root / company.slug / "pages" / f"{k['result_id']}.json").read_text(encoding="utf-8"))
        qids = []
        for f in p.get("found_by") or []:
            if f.get("query_id") and f["query_id"] not in qids:
                qids.append(f["query_id"])
        pages.append({"rank": int(k["result_id"].lstrip("r")), "url": p["url"], "title": p.get("title") or "", "query_ids": qids,
                      "content_source": "raw_content" if p.get("content_source") == "page" else p.get("content_source"),
                      "content": p.get("content") or ""})
    return sorted(pages, key=lambda p: p["rank"])


# ---------------------------------------------------------------- page pieces and request messages (was core/chunking.py)
# From company_pipeline's extract.py, minus the identity-gate page notes and the unused structure-aware splitter (removed 2026-11-25):
# estimate_tokens (chars / 2.8), page_block (the validated page format with a page number), batch_user_message, split_page /
# pack_chunks / pack_batches.

def estimate_tokens(text: str) -> int:
    """Conservative token estimate used for batching (over-counts vs ~4 chars/token)."""
    return max(1, int(len(text) / 2.8))


def page_block(page: dict[str, Any]) -> str:
    rank = page.get("rank") or "?"
    chunk_i, n_chunks = int(page.get("chunk_index") or 1), int(page.get("n_chunks") or 1)
    label = f"[PAGE {rank}]" if n_chunks == 1 else f"[PAGE {rank} CHUNK {chunk_i}/{n_chunks}]"
    lines = [label, f"URL   : {page.get('url') or ''}", f"Title : {(page.get('title') or '').strip()}",
             f"Queries: {', '.join(page.get('query_ids') or [])}", f"Content source: {page.get('content_source') or ''}"]
    if n_chunks > 1:
        lines.append("This is one chunk of a longer page. Cite the same URL for all chunks.")
    lines.extend(["---", (page.get("content") or "").strip(), ""])
    return "\n".join(lines)


def batch_user_message(ident: dict[str, str], pages: list[dict[str, Any]], *, batch_i: int, n_batches: int, total_pages: int, prompt: str) -> str:
    ranks = sorted({int(p.get("rank") or 0) for p in pages if p.get("rank")})
    header = "\n".join([
        "=== TARGET COMPANY IDENTITY ===", f"company_name : {ident['company_name']}", f"province     : {ident['province']}",
        f"country      : {ident['country']}", "", "=== PRE-RETRIEVED WEB PAGES (ONE BATCH) ===", f"Batch {batch_i} of {n_batches}",
        f"URL ranks in this batch: {', '.join(str(r) for r in ranks)}", f"Chunks in this batch: {len(pages)}",
        f"Total frozen ranked URLs: {total_pages}", "Each block is the full available extracted content for that URL,",
        "or one chunk of it if the page was split to fit this batch.", "Snippet fallback appears only when raw page content was unavailable.",
        "Cite the URL on every fact. Do not use information not in these pages.", ""])
    return prompt + "\n\n" + header + "\n" + "\n".join(page_block(p) for p in pages)


def _solo_tokens(page: dict[str, Any], *, ident: dict[str, str], prompt: str, total_pages: int) -> int:
    return estimate_tokens(batch_user_message(ident, [page], batch_i=1, n_batches=1, total_pages=total_pages, prompt=prompt))


def _max_content_chars(page: dict[str, Any], *, ident: dict[str, str], prompt: str, target_tokens: int, total_pages: int) -> int:
    content = page.get("content") or ""
    lo, hi, best = 1, len(content), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = {**page, "content": content[:mid], "chunk_index": 1, "n_chunks": 2}
        if _solo_tokens(trial, ident=ident, prompt=prompt, total_pages=total_pages) <= target_tokens:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return max(1, best)


def split_page(page: dict[str, Any], *, ident: dict[str, str], prompt: str, target_tokens: int, total_pages: int) -> list[dict[str, Any]]:
    """One page, or consecutive chunks of it that each fit in a request alone (cut at a fixed character position: the validated split)."""
    content = page.get("content") or ""
    whole = {**page, "chunk_index": 1, "n_chunks": 1, "content_offset": 0}
    if not content or _solo_tokens(whole, ident=ident, prompt=prompt, total_pages=total_pages) <= target_tokens:
        return [whole]
    step = _max_content_chars(page, ident=ident, prompt=prompt, target_tokens=target_tokens, total_pages=total_pages)
    spans = [(i, min(i + step, len(content))) for i in range(0, len(content), step)]
    out = []
    for idx, (a, b) in enumerate(spans, start=1):
        piece = content[a:b]
        out.append({**page, "content": piece, "chunk_index": idx, "n_chunks": len(spans), "content_offset": a, "chunk_chars": len(piece)})
    return out


def pack_chunks(chunks: list[dict[str, Any]], *, ident: dict[str, str], prompt: str, target_tokens: int, total_pages: int) -> list[list[dict[str, Any]]]:
    """Chunks in order, packed so each batch's message is at most target_tokens (estimated); a chunk over it goes alone."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for chunk in chunks:
        trial = current + [chunk]
        if current and estimate_tokens(batch_user_message(ident, trial, batch_i=1, n_batches=1, total_pages=total_pages, prompt=prompt)) > target_tokens:
            batches.append(current)
            current = [chunk]
        else:
            current = trial
    if current:
        batches.append(current)
    return batches


def pack_batches(pages: list[dict[str, Any]], *, ident: dict[str, str], prompt: str, target_tokens: int) -> list[list[dict[str, Any]]]:
    """Pages (and chunks of long pages) in order, packed so each request's message is at most target_tokens (estimated). The pipeline
    calls it with one page at a time, so every request holds one page or one piece of one."""
    chunks = [c for p in pages for c in split_page(p, ident=ident, prompt=prompt, target_tokens=target_tokens, total_pages=len(pages))]
    return pack_chunks(chunks, ident=ident, prompt=prompt, target_tokens=target_tokens, total_pages=len(pages))


# ---------------------------------------------------------------- extraction requests
RELATION_TYPES = ["director", "officer", "shareholder_or_owner", "employee", "parent", "subsidiary", "affiliate", "predecessor_or_former_name",
                  "successor", "counsel_or_auditor", "other"]
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["e"], "properties": {"e": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["n", "t", "rel", "r", "ev", "pg"],
    "properties": {"n": {"type": "string"}, "t": {"type": "string", "enum": ["p", "o"]}, "rel": {"type": "string", "enum": RELATION_TYPES},
                   "r": {"type": ["string", "null"]}, "ev": {"type": "string"}, "pg": {"type": "integer"}}}}}}
TYPES = {"p": "person", "o": "organization"}


class Prompts:
    def __init__(self, ecfg: dict[str, Any]):
        self.extract = resolve_path(ecfg["prompt_file"]).read_text(encoding="utf-8").strip()
        self.sizing = resolve_path(ecfg["chunk_sizing_prompt_file"]).read_text(encoding="utf-8").strip()
        self.batch_tokens = int(ecfg["batch_tokens"])
        self.country = ecfg["country"]


def company_header(c, country: str) -> dict[str, str]:
    """The company header of every request; country = the country profile's [config.extraction].country ("Canada" for CA, as E6)."""
    return {"company_name": c.name.strip(), "province": c.province.strip(), "country": country}


def page_pieces(page: dict, ident: dict, prompts: Prompts) -> list:
    """The pieces of one page (one request each); sized exactly as E6 did."""
    return pack_batches([page], ident=ident, prompt=prompts.sizing, target_tokens=prompts.batch_tokens)


def piece_message(ident: dict, piece, n: int, total_pages: int, prompts: Prompts) -> str:
    return batch_user_message(ident, piece, batch_i=n, n_batches=n, total_pages=total_pages, prompt=prompts.extract)


def cache_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def extract_piece(client: ChatClient, budget, message: str, prompts: Prompts, cache_dir: Path | None = None) -> dict:
    """One extraction request with its retry. Returns {ok, text, seconds, attempts, cached, error?}.
    cache_dir: successful replies are stored under the SHA-256 of the full first-attempt request body, so a resumed run never pays twice
    for a piece that already got an answer (a hit costs $0 and reserves nothing)."""
    p = client.profile
    content: Any = message
    if p.prompt_cache:
        assert message.startswith(prompts.extract)
        content = [{"type": "text", "text": prompts.extract, "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": message[len(prompts.extract):]}]
    f = cache_dir / f"{cache_key(client.body(content, 'entities', SCHEMA))}.json" if cache_dir is not None else None
    if f is not None and f.is_file():
        c = json.loads(f.read_text())
        return {"ok": True, "text": c["text"], "seconds": 0.0, "cached": True,
                "attempts": [{"error": "", "seconds": 0.0, "usage": c.get("usage") or {}, "max_tokens": c.get("max_tokens"), "cost_usd": 0.0,
                              "cached": True}]}
    read_timeout = 600 if p.paid else 900
    attempts, secs = [], 0.0
    for k in range(2):
        mt = p.max_tokens
        if k == 1 and p.length_retry_max_tokens and attempts[-1]["error"] == "finish_reason length":
            mt = p.length_retry_max_tokens
        reserve = ((len(message.encode("utf-8")) + 1000) * p.price_in / 1e6 * 1.5 + mt * p.price_out / 1e6) if p.paid else 0.0
        try:
            budget.reserve(reserve)
        except BudgetExceeded as exc:
            attempts.append({"error": f"budget: {exc}", "seconds": 0.0, "usage": {}})
            break
        r, cost, err = None, 0.0, ""
        try:
            r = client.post(client.body(content, "entities", SCHEMA, max_tokens=mt), read_timeout)
            err = r["transport_error"]
            if not err:
                if r["status"] != 200:
                    err = f"HTTP {r['status']}: {str(r['data'].get('error'))[:150]}" if p.paid else f"HTTP {r['status']}: {str(r['data'])[:150]}"
                elif r["finish"] != "stop":
                    err = f"finish_reason {r['finish']}"
                else:
                    try:
                        json.loads(r["text"])
                    except json.JSONDecodeError as exc:
                        err = f"invalid JSON: {exc}"
            cost = float((r["usage"] or {}).get("cost") or 0) if p.paid else 0.0
        finally:                                     # one atomic step: release the reserve and record the real cost
            budget.settle(reserve, cost, {"usage": (r or {}).get("usage"), "seconds": (r or {}).get("seconds"), "error": err,
                                          "provider": (r or {}).get("provider")})
        secs += r["seconds"]
        attempts.append({"error": err, "seconds": r["seconds"], "usage": r["usage"], "max_tokens": mt, "cost_usd": cost})
        if not err:
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps({"text": r["text"], "usage": r["usage"], "max_tokens": mt}, ensure_ascii=False))
            return {"ok": True, "text": r["text"], "seconds": round(secs, 2), "attempts": attempts, "cached": False}
    return {"ok": False, "text": "", "seconds": round(secs, 2), "attempts": attempts, "cached": False, "error": attempts[-1]["error"]}


def items_from_reply(res: dict, rank: int, url_of_rank: dict[int, str], batch: int) -> tuple[list[dict], list[dict]]:
    """(items, problems): the reply's items with their exact source URL; an item citing another page number gets source_url None."""
    items, problems = [], []
    if not res["ok"]:
        return items, problems
    for it in json.loads(res["text"])["e"]:
        url = url_of_rank.get(it["pg"]) if it["pg"] == rank else None
        if url is None:
            problems.append({"batch": batch, "name": it["n"], "page": it["pg"]})
        items.append({"name": it["n"], "type": TYPES[it["t"]], "role": it["r"], "relationship": it["rel"], "status": "unknown", "date": None,
                      "link": "stated", "evidence": it["ev"], "page": it["pg"], "source_url": url, "batch": batch})
    return items, problems


def write_extract_json(d: Path, c, profile_name: str, model: str, status: str, items: list, problems: list, batches: list) -> None:
    ex = {"company": c.__dict__, "profile": profile_name, "model": model, "status": status, "page_verdicts": [], "entities": items,
          "source_url_problems": problems, "batches": batches}
    (d / "extract.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1))
