"""Select component (select_pages.py: a file named select.py would shadow the standard-library module): choose the pages that go to extraction.

Input:  OUT/<slug>/rerank.json and OUT/<slug>/pages/<result_id>.json.
Output: OUT/<slug>/select.json     kept pages (with URL, score, why the name matched) and dropped pages (with the reason)
        OUT/select_summary.json    run summary

A page is kept when both hold (same rule as the validated pipeline, enterprise_variant_d/runs/pipeline_granite9b_bc80_20261112):
  1. its rerank score >= the threshold of the reranker profile that scored it ([rerank.profiles.<profile>].threshold in the current config;
     the profile name is read from rerank.json, so scores and threshold always belong to the same model. Granite: 0.8);
  2. it names the company, by either test:
     - "words": every word of the company's short name (legal ending such as Ltd./Inc. removed; one-letter words dropped unless digits)
       is a word of the page title or text, or the first distinctive word (not and/the, over 2 letters) is in the website's host name
       (mentions() of enterprise_variant_d/runs/bc30_arctic_cos028_pagefilter_20261110/page_filter.py);
     - "glued": the short name with spaces and punctuation removed is in the title + text with spaces and punctuation removed (catches
       registry notices that glue the incorporation number to the name; glued() of
       enterprise_variant_d/runs/onestage9b_qwen3rr_namecheck_bc80_20261111/make_pages.py).
  3. it is not on a foreign website ([select.foreign_sites], shared/foreign_pages.py: a non-Canadian country domain, a foreign registry or
     directory, OpenCorporates outside Canada), since 2026-11-13: same-name companies elsewhere were the main source of unrelated names.
  4. it passes the location check ([select.location], shared/foreign_pages.py, since 2026-11-13): not a page that places its business
     outside Canada (or, for a sole proprietorship, only in another province), judged over the pages that pass 1 to 3.
Dropped reasons: foreign_site (here, or already in rerank, which skips such pages), foreign_location, other_province, below_threshold, name_not_found (here, or already in
rerank, which skips such pages), not_scored (no content or a page error in rerank). The name tests are in pipeline/names.py.
CPU only; fast. Errors per company: missing_rerank (run rerank first), rerank_failed (rerank.json has an error; rerun rerank),
unknown_profile (rerank.json names a profile that is not in the config).
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..shared.models import Company
from ..shared.names import NameMatcher, short_name  # noqa: F401  (short_name re-exported for callers and tests)
from ..shared.foreign_pages import LocationCheck
from ..shared.foreign_pages import SiteFilter
from ..storage import utc_now, write_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- name tests: pipeline/names.py

# ---------------------------------------------------------------- one company

def select_company(out_root: Path, company: Company, thresholds: dict[str, float], sites: SiteFilter | None = None,
                   location: LocationCheck | None = None, host_rule: str = "first_word") -> dict[str, Any]:
    rp = out_root / company.slug / "rerank.json"
    if not rp.is_file():
        return {"status": "error", "error": "missing_rerank", "error_message": f"{rp} not found: run rerank first"}
    rr = json.loads(rp.read_text(encoding="utf-8"))
    if rr.get("status") != "ok":
        return {"status": "error", "error": "rerank_failed", "error_message": f"rerank.json: {rr.get('error')} {rr.get('error_message', '')}"[:300]}
    profile = rr.get("profile", "")
    if profile not in thresholds:
        return {"status": "error", "error": "unknown_profile", "error_message": f"rerank.json profile {profile!r} is not in the config"}
    threshold = thresholds[profile]
    matcher = NameMatcher(company.name, host_rule)
    kept, dropped, kept_pages = [], [], {}
    for row in rr["pages"]:
        base = {"result_id": row["result_id"], "url": row["url"], "score": row["score"]}
        if row["score"] is None:
            dropped.append({**base, "reason": "not_scored"})
            continue
        page = json.loads((out_root / company.slug / "pages" / f"{row['result_id']}.json").read_text(encoding="utf-8"))
        if sites is not None and sites.is_foreign(page):
            dropped.append({**base, "reason": "foreign_site"})
            continue
        if row["score"] < threshold:
            dropped.append({**base, "reason": "below_threshold"})
            continue
        how = matcher.match(page.get("title") or "", page.get("content") or "", page.get("url") or row["url"])
        if how:
            kept.append({**base, "name_match": how, "content_source": page.get("content_source"), "content_chars": page.get("content_chars")})
            kept_pages[row["url"]] = page
        else:
            dropped.append({**base, "reason": "name_not_found"})
    if location is not None and kept:                            # location check over the pages that would be kept (shared/foreign_pages.py)
        loc = location.drops(kept_pages, company.province, company.entity_form)
        dropped += [{**{k: p[k] for k in ("result_id", "url", "score")}, "reason": loc[p["url"]]} for p in kept if p["url"] in loc]
        kept = [p for p in kept if p["url"] not in loc]
    for rid in rr.get("skipped_no_content", []) + [e["result_id"] for e in rr.get("page_errors", [])]:
        dropped.append({"result_id": rid, "url": None, "score": None, "reason": "not_scored"})
    for rid in rr.get("skipped_no_name", []):                  # rerank's name test (the same as below): not scored
        dropped.append({"result_id": rid, "url": None, "score": None, "reason": "name_not_found"})
    for rid in rr.get("skipped_foreign", []):                  # rerank's foreign-site test (the same as above): not scored
        dropped.append({"result_id": rid, "url": None, "score": None, "reason": "foreign_site"})
    kept.sort(key=lambda p: p["result_id"])
    dropped.sort(key=lambda p: p["result_id"])
    return {"status": "ok", "profile": profile, "threshold": threshold, "kept": kept, "dropped": dropped, "rerank_fingerprint": rr.get("fingerprint")}


# ---------------------------------------------------------------- run

def profile_thresholds(cfg: dict[str, Any]) -> dict[str, float]:
    return {name: float(p["threshold"]) for name, p in cfg["rerank"].get("profiles", {}).items()}


def select_one(out_root: Path, c: Company, thresholds: dict[str, float], sites: SiteFilter | None = None,
               location: LocationCheck | None = None, host_rule: str = "first_word") -> dict[str, Any]:
    """Select one company's pages and write select.json; returns the result (status ok or error)."""
    res = select_company(out_root, c, thresholds, sites, location, host_rule)
    write_json(out_root / c.slug / "select.json", {"company": c.__dict__, **res, "written_at": utc_now()})
    return res
