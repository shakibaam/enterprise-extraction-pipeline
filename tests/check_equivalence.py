"""Free equivalence check against the adopted E6 run (no model calls; any uncached verifier call fails the check instead of being paid).

A. Extraction messages: for the 83 companies, the messages this package builds from the saved pages (country profile CA) are compared byte
   for byte with the rebuild of the same requests in tp_dp_benchmark_20261121/requests.json (made with the E9 runner's own functions; same
   prompt and page format as E6; pieces numbered in page order). Must be identical.
B. Verification + clean-up replay: E6's claims (decisions.json, same ids and page grouping) go through this package's verifier, answered
   only from E6's verifier cache. E6's raw items (extract.json) get their claim ids (decisions.json "ei" = the item's index) and go through
   this package's clean-up. Checks:
     - every claim decision equals E6's;
     - final.json equals E6's apart from the two new fields (mention "claim_ids", company "country");
     - with E6's OLD filter (source URL + evidence or name), verified.json equals E6's: nothing else changed;
     - with the NEW filter (claim ids; review fix 2026-11-24), the differences from E6 are listed: the mentions the old rule let through
       without a confirmed claim of their own. These are expected, not failures.
usage (from Provinces/): .venv/bin/python -B -m on_the_fly_pipeline.tests.check_equivalence [--work DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from on_the_fly_pipeline.shared.config import PROVINCES_DIR
from on_the_fly_pipeline.steps.extract import page_records
from on_the_fly_pipeline.shared.config import resolve_config
from on_the_fly_pipeline.shared.llm import ChatClient, load_profile
from on_the_fly_pipeline.steps import extract as X
from on_the_fly_pipeline.steps import verify as V
from on_the_fly_pipeline.steps.finalize import confirmed_only, finalize_one
from on_the_fly_pipeline.pipeline import SavedRetrieval
from on_the_fly_pipeline.storage import open_companies

EXP = PROVINCES_DIR / "enterprise_agent/results/experiments"
E6 = EXP / "e2e_stream_final_terra_20261120"
SRC_RUN = PROVINCES_DIR / "enterprise_variant_d/runs/recall_fixes_20261115"
CLAIM_KEYS = ("id", "slug", "key", "ei", "mi", "name", "type", "rel", "role", "evidence")
VOLATILE = {"written_at", "finalized_at", "claim_ids", "country"}


class CacheOnly(ChatClient):
    def __init__(self, profile):
        self.profile = profile


class NoMoney:
    def reserve(self, amount):
        raise RuntimeError("uncached verifier call (would be paid)")

    def settle(self, reserved, cost, record=None):
        pass


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in VOLATILE}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def old_filter(final: dict, claims: list[dict]) -> dict:
    """E6's confirmed-only filter (run_stream.py 379-392), for the comparison only."""
    conf_q = {(x["url"], (x.get("evidence") or "").strip()) for x in claims if x.get("final") == "confirmed"}
    conf_n = {(x["url"], x["name"].lower().strip()) for x in claims if x.get("final") == "confirmed"}
    out = {**final}
    for key in ("people", "organizations"):
        keep = []
        for e in final.get(key, []):
            ms = [m for m in e["mentions"] if (m.get("source_url"), (m.get("evidence") or "").strip()) in conf_q
                  or (m.get("source_url"), e["name"].lower().strip()) in conf_n]
            if ms:
                keep.append({**e, "mentions": ms, "relationships": sorted({m.get("relationship") for m in ms if m.get("relationship")}),
                             "roles": sorted({m.get("role") for m in ms if m.get("role")})})
        out[key] = keep
    return out


def mentions(v: dict) -> set:
    return {(e["name"], m.get("source_url"), m.get("relationship"), (m.get("evidence") or "")[:80])
            for k in ("people", "organizations") for e in v.get(k, []) for m in e["mentions"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    cfg = resolve_config(None, "CA")
    prompts = X.Prompts(cfg["extraction"])
    comps = {c.slug: c for c in open_companies(str(SRC_RUN / "companies.jsonl"))}
    slugs = (E6 / "companies_83.txt").read_text().split()

    # A. messages
    ref = json.loads((EXP / "tp_dp_benchmark_20261121/requests.json").read_text())
    bad_a, n_msgs = [], 0
    for s in slugs:
        c = comps[s]
        sel = json.loads((SRC_RUN / "out_AE" / s / "select.json").read_text())
        kept = sel.get("kept", []) if sel.get("status") == "ok" else []
        ident = X.company_header(c, prompts.country)
        epages = {p["url"]: p for p in page_records(SRC_RUN / "out_AE", c, kept)}
        mine, n = [], 0
        for k in kept:
            page = epages.get(k["url"])
            if page is None:
                continue
            for piece in X.page_pieces(page, ident, prompts):
                n += 1
                mine.append(X.piece_message(ident, piece, n, len(kept), prompts))
        theirs = [r["message"] for r in ref[s]]
        n_msgs += len(mine)
        if mine != theirs:
            bad_a.append((s, len(mine), len(theirs), sum(x != y for x, y in zip(mine, theirs))))
    print(f"A. extraction messages: {n_msgs} built for {len(slugs)} companies; companies differing from the reference: {len(bad_a)} {bad_a[:5]}")

    # B. verify + clean-up replay
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="otf_equiv_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    client = CacheOnly(load_profile("verifiers", "terra_none"))
    prompt, record = V.load_prompt(cfg["verification"]), V.load_record_template(cfg["verification"])
    retr = SavedRetrieval(SRC_RUN / "out_AE", work)
    n_claims = n_dec_diff = n_uncached = 0
    bad_final, bad_old, new_diff = [], [], []
    for s in slugs:
        c = comps[s]
        retr.run(c)
        e6d = E6 / "out_E6" / s
        dec = json.loads((e6d / "decisions.json").read_text()) if (e6d / "decisions.json").is_file() else []
        ex = json.loads((e6d / "extract.json").read_text())
        for x in dec:                                            # claim ids onto E6's raw items (as the pipeline now does)
            ex["entities"][x["ei"]]["claim_id"] = x["id"]
        (work / s / "extract.json").write_text(json.dumps(ex, ensure_ascii=False))
        by_url: dict[str, list] = {}
        for x in dec:
            by_url.setdefault(x["url"], []).append({k: x[k] for k in CLAIM_KEYS})
        pages = V.load_pages(work / s)
        rec = V.registry_record(c, record)
        rows = []
        for u, cl in by_url.items():
            try:
                r, _ = V.verify_page(client, NoMoney(), E6 / "verify_cache_E6", prompt, u, cl, rec, pages)
            except RuntimeError:
                n_uncached += 1
                r = [{**x, "url": u, "final": "UNCACHED"} for x in cl]
            rows += r
        want = {(x["url"], x["id"]): x["final"] for x in dec}
        got = {(x["url"], x["id"]): x["final"] for x in rows}
        n_claims += len(want)
        n_dec_diff += sum(want[k] != got.get(k) for k in want)
        final = finalize_one(work, c, cfg["finalize"])
        e6f = json.loads((e6d / "final.json").read_text()) if (e6d / "final.json").is_file() else {}
        e6v = json.loads((e6d / "verified.json").read_text()) if (e6d / "verified.json").is_file() else {}
        if strip(final) != strip(e6f):
            bad_final.append(s)
        if strip(old_filter(final, rows)) != strip(e6v):
            bad_old.append(s)
        new = confirmed_only(final, rows)
        gone = mentions(e6v) - mentions(new)
        if gone or mentions(new) - mentions(e6v):
            new_diff.append({"slug": s, "removed": sorted(gone), "added": sorted(mentions(new) - mentions(e6v)),
                             "entities_e6": sum(len(e6v.get(k, [])) for k in ("people", "organizations")),
                             "entities_new": sum(len(new.get(k, [])) for k in ("people", "organizations"))})
    print(f"B. verifier replay: {n_claims} claims, decisions differing from E6: {n_dec_diff}, uncached pages: {n_uncached}")
    print(f"   final.json differing from E6 (apart from claim_ids, country): {len(bad_final)} {bad_final[:5]}")
    print(f"   verified.json with E6's OLD filter differing from E6: {len(bad_old)} {bad_old[:5]}")
    rm = sum(len(d["removed"]) for d in new_diff)
    ents = sum(d["entities_e6"] - d["entities_new"] for d in new_diff)
    print(f"   NEW claim-id filter vs E6: {rm} mentions removed, {sum(len(d['added']) for d in new_diff)} added, "
          f"{ents} entities fewer, in {len(new_diff)} companies (mentions E6 kept without a confirmed claim of their own):")
    for d in new_diff:
        for m in d["removed"]:
            print(f"     - {d['slug']}: {m[0]} | {m[2]} | {m[1]}")
    (work / "new_filter_differences.json").write_text(json.dumps(new_diff, ensure_ascii=False, indent=1))
    print(f"   work folder: {work}")
    ok = not bad_a and not n_dec_diff and not n_uncached and not bad_final and not bad_old and not any(d["added"] for d in new_diff)
    print("EQUIVALENT (apart from the listed claim-id filter removals)" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
