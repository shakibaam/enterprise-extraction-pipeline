"""Verifier (E6, "V5"): one yes/no question per claim; all claims citing one page go in one call with the whole page; no web.

Question (prompts/verify.txt): does this page state that <name> has the claimed relationship with THIS registered company (not a namesake,
not another party on the page), in the claimed direction? A claim is "confirmed" when the answer is yes AND the name passes name_ok()
(a person needs two name words besides titles and no digits; no dates in any name); otherwise "unverified".

Copied from E6 without behaviour change: vfy_simple.verify_page + vfy_nw.call/_send + verifier_v3 (RECORD, page_text, name_ok, parse_json,
load_pages). The answer cache key is E6's, sha256(["vfy_nw", model, effort, message, schema]), so an E6 verify cache answers the same calls.
Unlike extraction, a failed verifier call (HTTP or API error) raises; the pipeline records it as a verify error and the page's claims stay
unconfirmed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..shared.config import resolve_path
from ..shared.llm import ChatClient

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["claims"], "properties": {"claims": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "supported"],
    "properties": {"id": {"type": "integer"}, "supported": {"type": "string", "enum": ["yes", "no"]}}}}}}
TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "hon", "rev", "canon", "the", "jr", "sr", "kc", "qc", "cpa", "ca", "phd", "md"}


def load_prompt(vcfg: dict[str, Any]) -> str:
    return resolve_path(vcfg["prompt_file"]).read_text(encoding="utf-8")


def load_record_template(vcfg: dict[str, Any]) -> str:
    """The registry-record block of the verifier prompt (country profile: [config.verification].registry_record_file)."""
    return resolve_path(vcfg["registry_record_file"]).read_text(encoding="utf-8")


def registry_record(c, template: str) -> str:
    _, _, rid = (c.company_id or "").partition(":")
    return template.format(name=c.name, province=c.province, form=c.entity_form or "unknown", rid=rid or "unknown")


def load_pages(company_dir: Path) -> dict[str, dict]:
    """The company's saved pages, keyed by url, final_url and canonical_url."""
    pages = {}
    for f in (company_dir / "pages").glob("*.json"):
        p = json.loads(f.read_text(encoding="utf-8"))
        for u in {p.get("url"), p.get("final_url"), p.get("canonical_url")}:
            if u:
                pages[u] = p
    return pages


def page_text(content: str, names: list[str], max_chars: int = 540_000) -> str:
    """The whole page, or for a page over max_chars its start plus the text around each name."""
    if len(content) <= max_chars:
        return content
    keep = [(0, 20000)]
    for n in names:
        for m in re.finditer(re.escape(n.split()[-1]), content):
            keep.append((max(0, m.start() - 4000), m.end() + 4000))
    keep.sort()
    out, end = [], -1
    for a, b in keep:
        if a > end:
            out.append(("\n[...]\n" if out else "") + content[a:b])
        else:
            out.append(content[end:b])
        end = max(end, b)
        if sum(len(x) for x in out) > max_chars:
            break
    return "".join(out)


def name_ok(name: str, typ: str) -> bool:
    if re.search(r"\b(19|20)\d\d\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d", name, re.I):
        return False
    if typ == "person":
        words = [w for w in re.split(r"[\s.,]+", name.lower()) if w and w not in TITLES]
        return len([w for w in words if len(w) > 1]) >= 2 and not re.search(r"\d", name)
    return True


def parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return None


def page_message(prompt: str, rec_txt: str, url: str, page: dict, claims: list[dict], max_chars: int) -> str:
    lines = [f"{c['id']}. {c['name']} ({c['type']}): claimed {c['rel']}{' (role: ' + c['role'] + ')' if c['role'] else ''}; "
             f"quote: \"{c['evidence']}\"" for c in claims]
    return prompt.format(record=rec_txt, url=url, title=page.get("title") or "",
                         page=page_text(page.get("content") or "", [c["name"] for c in claims], max_chars), claims="\n".join(lines))


def cache_key(client: ChatClient, message: str) -> str:
    p = client.profile
    return hashlib.sha256(json.dumps(["vfy_nw", p.model, p.effort, message, SCHEMA], sort_keys=True).encode()).hexdigest()


def call(client: ChatClient, budget, cache_dir: Path, message: str) -> dict:
    """One verifier call, answered from the disk cache when possible."""
    p = client.profile
    f = cache_dir / f"{cache_key(client, message)}.json"
    if f.is_file():
        return {**json.loads(f.read_text()), "cached": True}
    reserve = len(message) / 3.0 * p.price_in / 1e6 + 0.02 if p.paid else 0.0
    budget.reserve(reserve)
    r, usd = None, 0.0
    try:
        r = client.post(client.body(message, "verdict", SCHEMA), 900)
        usage = r["usage"] or {}
        if p.paid and r["status"] == 200:
            usd = float(usage.get("cost") or 0.0) or (usage.get("prompt_tokens", 0) * p.price_in + usage.get("completion_tokens", 0) * p.price_out) / 1e6
    finally:                                         # one atomic step: release the reserve and record the real cost
        budget.settle(reserve, usd, {"usage": (r or {}).get("usage"), "seconds": (r or {}).get("seconds"), "provider": (r or {}).get("provider")})
    if r["transport_error"]:
        raise RuntimeError(f"network error: {r['transport_error']}")
    if r["status"] != 200:
        raise RuntimeError(f"HTTP {r['status']}: {json.dumps(r['data'])[:300]}")
    if r["data"].get("error"):
        raise RuntimeError(f"API error: {json.dumps(r['data']['error'])[:300]}")
    parsed = parse_json(r["text"])
    rec = {"finish": r["finish"], "parsed": parsed, "text": r["text"], "usage": usage, "usd": round(usd, 6), "seconds": r["seconds"],
           "provider": r["provider"], "model": p.model, "effort": p.effort, "cache_instr": False}
    if parsed is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def verify_page(client: ChatClient, budget, cache_dir: Path, prompt: str, url: str, claims: list[dict], rec_txt: str,
                pages: dict[str, dict], max_chars: int = 540_000) -> tuple[list[dict], dict]:
    """Check all claims citing one page. Returns (decision rows, call record); rows have the shape of E6's decisions.json."""
    p = pages.get(url) or {}
    content = p.get("content") or ""
    msg = page_message(prompt, rec_txt, url, p, claims, max_chars)
    ra = call(client, budget, cache_dir, msg) if content else {"parsed": None, "usd": 0, "error": "page not saved", "seconds": 0}
    by_id = {x["id"]: x for x in (ra.get("parsed") or {}).get("claims", [])}
    out = []
    for c in claims:
        a = by_id.get(c["id"])
        ok = bool(a) and a["supported"] == "yes" and name_ok(c["name"], c["type"])
        out.append({**c, "url": url, "page_chars": len(content), "pass_a": a, "pass_a2": None, "pass_b": None,
                    "final": "confirmed" if ok else "unverified",
                    "cost": {"a_share": round((ra.get("usd") or 0) / max(len(claims), 1), 6), "b": None, "a_cached": ra.get("cached"),
                             "a_prompt_chars": len(msg), "a_seconds": ra.get("seconds"), "provider": ra.get("provider"),
                             "usage": ra.get("usage") if c is claims[0] else None}})
    return out, ra
