"""The streaming pipeline: for each company, retrieval -> extraction -> verification -> clean-up, with extraction and verification overlapping.

Per company (several companies at once, --company-workers; the request pools are shared by all companies):
  1. retrieval (steps/retrieve.py): live search -> fetch -> rerank -> select, or the saved pages of an earlier run (--from-saved).
  2. extraction (steps/extract.py): every piece of every selected page is sent at once to the extractor pool (64 in flight).
  3. verification (steps/verify.py): as soon as ALL pieces of a page are back, that page's claims go to the verifier pool (32 in flight),
     while other pages are still being extracted. This streaming is E6's; it was measured equal to running the phases one after another
     on a single company, and is kept because it never waits for the slowest page before verifying the others.
  4. clean-up (steps/finalize.py): finalize -> final.json; the confirmed-only filter -> verified.json (the output).
No identity check (E6 dropped it: no precision gain, 1.5 s slower, and it blocked pages with 32 gold entities).

Outputs (written through the output sink, storage.py; today JSON files) in OUT/<slug>/: search.json fetch.json pages/ rerank.json select.json (retrieval), extract.json (raw items, every request and
attempt), decisions.json (every claim with the verifier's answer and cost), final.json (after clean-up), verified.json (OUTPUT: confirmed
mentions only, each with its source URL), timing.json (seconds per stage, cost per stage).
Run-level, in OUT/: run_summary.json, run_companies.jsonl, ledger_extract.jsonl, ledger_verify.jsonl, verify_cache/.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import statistics as st
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from . import __version__
from .shared.foreign_pages import LocationCheck, SiteFilter
from .shared.llm import ChatClient, FreeBudget, PaidBudget, Profile
from .steps import extract as X
from .steps import verify as V
from .steps.extract import page_records
from .steps.fetch import Fetcher
from .steps.finalize import finalize_and_filter
from .steps.rerank import Reranker
from .steps.search import Searcher
from .steps.select import profile_thresholds, select_one
from .storage import ResultSink, open_sink

log = logging.getLogger("on_the_fly_pipeline")


# ---------------------------------------------------------------- run state: new run or resume (was runstate.py)
# New run or resume: the run fingerprint and the output folder rules.
#
# A run is identified by a fingerprint (SHA-256) over everything that decides its results:
#   companies   the companies source file's content hash and the slug list
#   config      the resolved configuration (default.toml + the country profile)
#   extractor / verifier   the full model profiles
#   prompts     the content of every prompt file used
#   retrieval   live (provider) or --from-saved (its path)
#   version     the package version
# Rules (cli.py `run`):
#   - a NEW run needs an output folder that does not exist or is empty; the manifest (OUT/run_manifest.json) is written first;
#   - --resume continues a run only when the fingerprint matches; otherwise it stops and names the parts that changed;
#   - a non-empty folder without --resume is refused (no silent mixing of two runs).
# On resume: companies with OUT/<slug>/done.json (written last, after the company's outputs) are skipped; the others run again, but the
# extraction reply cache (OUT/extract_cache/) and the verifier cache (OUT/verify_cache/) answer every request that already got a reply, the
# paid ledgers keep counting against the caps, and live search/fetch reuse their saved per-query and per-page files.

MANIFEST = "run_manifest.json"
DONE = "done.json"


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(*, companies_source: str, slugs: list[str], cfg: dict, extractor: dict, verifier: dict, prompts: dict[str, str],
                retrieval: dict) -> dict[str, Any]:
    src = Path(companies_source)
    parts = {"companies": _h({"source": file_hash(src) if src.is_file() else companies_source, "slugs": slugs}),
             "config": _h(cfg), "extractor": _h(extractor), "verifier": _h(verifier), "prompts": _h(prompts),
             "retrieval": _h(retrieval), "version": __version__}
    return {"fingerprint": _h(parts), "parts": parts}


def prepare_out(out: Path, fp: dict[str, Any], resume: bool, details: dict[str, Any]) -> None:
    """Apply the new-run / resume rules; write the manifest for a new run."""
    m = out / MANIFEST
    used = out.is_dir() and any(out.iterdir())
    if not used:
        if resume:
            raise SystemExit(f"--resume: {out} has no run to resume")
        out.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps({**fp, "created": time.strftime("%Y-%m-%d %H:%M:%S"), **details}, indent=1, default=str))
        return
    if not resume:
        raise SystemExit(f"{out} is not empty: use a new folder for a new run, or --resume to continue the run in it")
    if not m.is_file():
        raise SystemExit(f"--resume: {out} has no {MANIFEST}; it was not written by this pipeline (use a new folder)")
    old = json.loads(m.read_text())
    if old.get("fingerprint") != fp["fingerprint"]:
        changed = sorted(k for k in fp["parts"] if old.get("parts", {}).get(k) != fp["parts"][k])
        raise SystemExit(f"--resume refused: this command differs from the run in {out} in: {', '.join(changed)}. "
                         "Use a new output folder for a different run.")


def is_done(out: Path, slug: str, fp: str) -> dict | None:
    f = out / slug / DONE
    if not f.is_file():
        return None
    d = json.loads(f.read_text())
    return d if d.get("fingerprint") == fp else None


def mark_done(out: Path, slug: str, fp: str, row: dict) -> None:
    (out / slug / DONE).write_text(json.dumps({"fingerprint": fp, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "row": row}, default=str))


# ---------------------------------------------------------------- retrieval: the search, fetch, rerank and select steps for one company
# Retrieval (was steps/retrieve.py):
# Retrieval: search -> fetch -> rerank -> select, per company (the step modules steps/search.py, fetch.py, rerank.py, select.py), or the saved pages of an earlier run.
# 
# Live (default): the copied company_pipeline components, shared by all companies:
#   search   3 queries x 10 results ("{name}" {province} corporate registry | address contact | public records), Serper or Keenable,
#            hard cap on paid calls (--max-search-calls), rate limiter, results merged by canonical URL (ids r001...)  -> search.json
#   fetch    every hit in parallel (64 workers, 4 per website), HTML -> text, PDFs not parsed, search snippets kept when a page fails
#                                                                                                         -> fetch.json, pages/rNNN.json
#   rerank   Granite cross-encoder on the local vLLM /classify servers (ports 8020/8021); page score = best 8k-token chunk; pages that
#            do not name the company or sit on foreign sites are not scored                                 -> rerank.json
#   select   keep pages with score >= 0.82 that name the company, minus foreign sites and pages placed outside Canada  -> select.json
# Saved (--from-saved DIR): copy DIR/<slug>/{select,rerank,search}.json and link DIR/<slug>/pages, exactly as E6 did (no identity.json:
# E6 skipped the identity check). Used to reproduce E6-E14 on the same pages.


class LiveRetrieval:
    def __init__(self, cfg: dict[str, Any], out_root: Path, provider: str, max_search_calls: int):
        self.cfg, self.out_root = cfg, out_root
        self.searcher = Searcher(cfg, out_root, provider, max_search_calls)
        self.fetcher = Fetcher(cfg, out_root)
        self.reranker = Reranker(cfg, out_root)
        self.thresholds = profile_thresholds(cfg)
        self.sites, self.location = SiteFilter.from_config(cfg), LocationCheck.from_config(cfg)
        self.host_rule = cfg["select"].get("host_rule", "first_word")

    def check(self) -> None:
        """Raises if the reranker servers are not ready (start them with python -m on_the_fly_pipeline serve rerank)."""
        self.reranker.check()

    @property
    def stopped_reason(self) -> str:
        return self.searcher.stopped_reason or self.reranker.stopped_reason

    def run(self, c) -> tuple[list[dict], dict[str, Any]]:
        """Returns (kept pages from select.json, info with per-stage seconds and statuses)."""
        info: dict[str, Any] = {"seconds": {}}
        t = time.monotonic()
        srow = self.searcher.search_company(c)
        info["seconds"]["search"] = round(time.monotonic() - t, 3)
        info["search_status"] = srow["status"]
        if srow["status"] == "error":
            return [], info
        t = time.monotonic()
        frow = self.fetcher.fetch_company(c)
        info["seconds"]["fetch"] = round(time.monotonic() - t, 3)
        info["fetch_status"] = frow["status"]
        if frow["status"] != "ok":
            return [], info
        t = time.monotonic()
        info["rerank_status"] = self.reranker.rerank_one(c)
        info["seconds"]["rerank"] = round(time.monotonic() - t, 3)
        if info["rerank_status"] in ("error", "not_run"):
            return [], info
        t = time.monotonic()
        sel = select_one(self.out_root, c, self.thresholds, self.sites, self.location, self.host_rule)
        info["seconds"]["select"] = round(time.monotonic() - t, 3)
        info["select_status"] = sel["status"]
        return (sel.get("kept", []) if sel["status"] == "ok" else []), info

    def summary(self, n: int, wall: float) -> dict[str, Any]:
        return {"search_calls": self.searcher.budget.used, "search": self.searcher.summary(n, wall), "fetch": self.fetcher.summary(n, wall),
                "rerank": self.reranker.summary(n, wall)}

    def close(self) -> None:
        self.searcher.close()
        self.fetcher.close()


class SavedRetrieval:
    def __init__(self, saved_root: Path, out_root: Path):
        self.saved, self.out_root = saved_root, out_root

    def check(self) -> None:
        if not self.saved.is_dir():
            raise SystemExit(f"--from-saved: {self.saved} is not a folder")

    stopped_reason = ""

    def run(self, c) -> tuple[list[dict], dict[str, Any]]:
        """Always re-copies the source's select/rerank/search files (2026-11-24 fix: existing destination files were kept while the
        selection was read from the source, which could mix two runs) and refuses a pages/ link that points somewhere else."""
        src, d = self.saved / c.slug, self.out_root / c.slug
        d.mkdir(parents=True, exist_ok=True)
        for f in ("select.json", "rerank.json", "search.json"):
            if (src / f).is_file():
                shutil.copy2(src / f, d / f)
            elif (d / f).exists():
                (d / f).unlink()
        if (src / "pages").exists():
            want = (src / "pages").resolve()
            if (d / "pages").is_symlink() or (d / "pages").exists():
                if (d / "pages").resolve() != want:
                    raise RuntimeError(f"{d / 'pages'} points to {(d / 'pages').resolve()}, not to the saved pages {want}")
            else:
                (d / "pages").symlink_to(want)
        sel = json.loads((src / "select.json").read_text()) if (src / "select.json").is_file() else {}
        return (sel.get("kept", []) if sel.get("status") == "ok" else []), {"seconds": {}, "source": str(src)}

    def summary(self, n: int, wall: float) -> dict[str, Any]:
        return {"from_saved": str(self.saved)}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- the pipeline
class Pipeline:
    def __init__(self, cfg: dict[str, Any], out: Path, retrieval, extractor: Profile, verifier: Profile,
                 cap_extract_usd: float | None, cap_verify_usd: float | None, verify_cache: Path | None = None, sink: ResultSink | None = None,
                 fingerprint: str = ""):
        self.cfg, self.out, self.retrieval = cfg, out, retrieval
        self.sink = sink or open_sink("json", out)
        self.ecfg, self.vcfg, self.pcfg = cfg["extraction"], cfg["verification"], cfg["pipeline"]
        self.prompts = X.Prompts(self.ecfg)
        self.vprompt = V.load_prompt(self.vcfg)
        self.vrecord = V.load_record_template(self.vcfg)
        self.max_page_chars = int(self.vcfg.get("max_page_chars", 540_000))
        self.skip_rel = self.vcfg.get("skip_relation", "other")
        self.xclient = ChatClient(extractor, pool_size=int(self.pcfg["extract_workers"]))
        self.vclient = ChatClient(verifier, pool_size=int(self.pcfg["verify_workers"]))
        self.xbudget = PaidBudget("extraction", cap_extract_usd, out / "ledger_extract.jsonl") if extractor.paid else FreeBudget()
        self.vbudget = PaidBudget("verification", cap_verify_usd, out / "ledger_verify.jsonl", allow_when_idle=True) \
            if verifier.paid else FreeBudget()
        self.vcache = verify_cache or out / "verify_cache"
        self.xcache = out / "extract_cache"
        self.fingerprint = fingerprint
        self.skipped: list[str] = []
        self.xpool = ThreadPoolExecutor(int(self.pcfg["extract_workers"]))
        self.vpool = ThreadPoolExecutor(int(self.pcfg["verify_workers"]))
        self.fcfg = cfg["finalize"]
        self._lock = threading.Lock()
        self.stopped_reason = ""
        self.rows: list[dict] = []

    # ------------------------------------------------------------------ one company
    def run_company(self, c) -> dict[str, Any]:
        t0 = time.monotonic()
        now = lambda: round(time.monotonic() - t0, 3)                       # noqa: E731
        d = self.out / c.slug
        d.mkdir(parents=True, exist_ok=True)
        kept, rinfo = self.retrieval.run(c)
        T: dict[str, float] = {"retrieval_done": now()}
        tr = T["retrieval_done"]
        ident = X.company_header(c, self.prompts.country)
        items, problems, claims, results, vcalls = [], [], [], [], []
        vstats, pt = Counter(), {}
        if kept:
            epages = {p["url"]: p for p in page_records(self.out, c, kept)}
            url_of_rank = {p["rank"]: p["url"] for p in epages.values()}
            vpages = V.load_pages(d)
            rec_txt = V.registry_record(c, self.vrecord)
            pending, remaining, by_url, cid, n_req = {}, {}, {}, 0, 0
            for k in kept:                                                   # every piece of every page at once
                page = epages.get(k["url"])
                if page is None:
                    continue
                pieces = X.page_pieces(page, ident, self.prompts)
                remaining[page["rank"]] = len(pieces)
                for piece in pieces:
                    n_req += 1
                    msg = X.piece_message(ident, piece, n_req, len(kept), self.prompts)
                    pending[self.xpool.submit(X.extract_piece, self.xclient, self.xbudget, msg, self.prompts, self.xcache)] = ("extract", (n_req, page["rank"]))
            while pending:
                done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                for f in done:
                    kind, info = pending.pop(f)
                    if kind == "extract":
                        i, rank = info
                        res = f.result()
                        results.append({"batch": i, "page": rank, "ok": res["ok"], "seconds": res["seconds"], "attempts": res["attempts"],
                                        "error": res.get("error"), "cached": bool(res.get("cached"))})
                        new, probs = X.items_from_reply(res, rank, url_of_rank, i)
                        problems += probs
                        for it in new:
                            items.append(it)
                            if it["source_url"] and it["relationship"] != self.skip_rel:
                                cid += 1
                                it["claim_id"] = cid                  # carried through clean-up; verified.json keeps only confirmed ids
                                by_url.setdefault(it["source_url"], []).append({
                                    "id": cid, "slug": c.slug, "key": "raw", "ei": len(items) - 1, "mi": 0, "name": it["name"], "type": it["type"],
                                    "rel": it["relationship"].replace("_", " "), "role": it["role"] or "", "evidence": it["evidence"] or ""})
                        remaining[rank] -= 1
                        u = url_of_rank[rank]
                        if remaining[rank] == 0:                             # the page is done: verify its claims now
                            pt.setdefault(u, {})["extracted"] = now()
                            if by_url.get(u):
                                pending[self.vpool.submit(V.verify_page, self.vclient, self.vbudget, self.vcache, self.vprompt, u, by_url[u],
                                                          rec_txt, vpages, self.max_page_chars)] = ("verify", u)
                    else:
                        try:
                            rows, ra = f.result()
                            claims.extend(rows)
                            vcalls.append({"url": info, "seconds": ra.get("seconds"), "usd": ra.get("usd") or 0, "cached": bool(ra.get("cached"))})
                        except Exception as exc:  # noqa: BLE001  (budget stop or API error: the page's claims stay unconfirmed)
                            vstats["verify_errors"] += 1
                            problems.append({"verify_error": f"{type(exc).__name__}: {exc}"[:200], "url": info})
                        pt.setdefault(info, {})["verified"] = now()
        T["extraction_done"] = max([v["extracted"] for v in pt.values() if "extracted" in v] or [tr])
        T["verification_done"] = now()
        status = "ok" if kept else "no_pages"
        X.write_extract_json(d, c, self.xclient.profile.name, self.xclient.profile.model, status, items, problems, results)
        final, verified = finalize_and_filter(self.out, c, self.fcfg, claims)
        T["total"] = now()
        ext_usd = sum(a.get("cost_usd") or 0 for r in results for a in r["attempts"])
        ver_usd = sum(v["usd"] for v in vcalls if not v["cached"])
        tm = {"slug": c.slug, "extractor": self.xclient.profile.name, "verifier": self.vclient.profile.name,
              "pages_selected": len(kept), "batches": len(results), "raw_items": len(items), "claims": len(claims),
              "confirmed": sum(x.get("final") == "confirmed" for x in claims),
              "final_entities": sum(len(final.get(k, [])) for k in ("people", "organizations")),
              "verified_entities": sum(len(verified.get(k, [])) for k in ("people", "organizations")),
              "failed_batches": sum(not r["ok"] for r in results), "retries": sum(len(r["attempts"]) - 1 for r in results),
              "cached_requests": sum(r["cached"] for r in results),
              "source_url_problems": sum(1 for p in problems if "name" in p), **vstats,
              # t: seconds from the company's start; t_after_retrieval: from the end of retrieval (= what E6-E14 reported)
              "t": T, "t_after_retrieval": {k: round(v - tr, 3) for k, v in T.items() if k != "retrieval_done"},
              "retrieval": rinfo, "batch_seconds": [r["seconds"] for r in results],
              "verifier_calls": vcalls, "cost_usd": {"extraction": round(ext_usd, 6), "verification": round(ver_usd, 6),
                                                     "total": round(ext_usd + ver_usd, 6)},
              "pages_t": pt}
        self.sink.write_company(c, verified, final, claims, tm)
        mark_done(self.out, c.slug, self.fingerprint, self._row(tm))     # last write: the company is complete
        return tm

    # ------------------------------------------------------------------ run
    def _admit(self) -> bool:
        reason = self.retrieval.stopped_reason or \
            (f"extraction cap ${self.xbudget.cap:.2f} reached" if self.xbudget.exhausted else "") or \
            (f"verification cap ${self.vbudget.cap:.2f} reached" if self.vbudget.exhausted else "")
        if reason and not self.stopped_reason:
            self.stopped_reason = reason
            log.warning("no new companies: %s", reason)
        return not self.stopped_reason

    @staticmethod
    def _row(tm: dict) -> dict:
        return {k: v for k, v in tm.items() if k not in ("pages_t", "batch_seconds", "verifier_calls")}

    def _one(self, c) -> dict | None:
        done = is_done(self.out, c.slug, self.fingerprint)
        if done is not None:                                                 # resume: finished in an earlier process
            with self._lock:
                self.rows.append(done["row"])
                self.skipped.append(c.slug)
            return done["row"]
        if not self._admit():
            return None
        try:
            tm = self.run_company(c)
        except Exception as exc:  # noqa: BLE001  (one company's failure must not stop the others)
            log.exception("company %s failed", c.slug)
            tm = {"slug": c.slug, "error": f"{type(exc).__name__}: {exc}"[:300]}
        with self._lock:
            self.rows.append(tm)
            self.sink.log_company(self._row(tm))
        log.info("%s: %s pages, %s requests, %s claims -> %s verified, %.2f s (after retrieval %.2f s)%s", c.slug, tm.get("pages_selected"),
                 tm.get("batches"), tm.get("claims"), tm.get("verified_entities"), (tm.get("t") or {}).get("total", -1),
                 (tm.get("t_after_retrieval") or {}).get("total", -1), f"  ERROR {tm['error']}" if "error" in tm else "")
        return tm

    def run(self, companies: list, company_workers: int) -> dict[str, Any]:
        self.retrieval.check()
        t0 = time.monotonic()
        with ThreadPoolExecutor(company_workers) as cw:
            list(cw.map(self._one, companies))
        wall = round(time.monotonic() - t0, 1)
        self.xpool.shutdown()
        self.vpool.shutdown()
        rows = [r for r in self.rows if "t" in r]
        med = lambda xs: round(st.median(xs), 2) if xs else None            # noqa: E731
        wc = [r for r in rows if r["claims"]]
        summ = {
            "companies": len(companies), "finished": len(rows), "resumed_skipped": len(self.skipped), "errors": [r for r in self.rows if "error" in r],
            "not_run": len(companies) - len(self.rows), "stopped_reason": self.stopped_reason, "wall_s": wall,
            "extractor": self.xclient.profile.name, "verifier": self.vclient.profile.name,
            "requests": sum(r["batches"] for r in rows), "failed_requests": sum(r["failed_batches"] for r in rows),
            "retries": sum(r["retries"] for r in rows), "verify_errors": sum(r.get("verify_errors", 0) for r in rows),
            "claims": sum(r["claims"] for r in rows), "confirmed": sum(r["confirmed"] for r in rows),
            "verified_entities": sum(r["verified_entities"] for r in rows),
            "spent_usd": {"extraction": round(self.xbudget.spent, 4), "verification": round(self.vbudget.spent, 4)},
            "cost_per_company_usd": {k: round(sum(r["cost_usd"][k] for r in rows) / len(rows), 4) if rows else None
                                     for k in ("extraction", "verification", "total")},
            "median_s": {
                "all_companies": {k: med([r["t"][k] for r in rows]) for k in ("retrieval_done", "extraction_done", "verification_done", "total")},
                "after_retrieval": {k: med([r["t_after_retrieval"][k] for r in rows]) for k in ("extraction_done", "verification_done", "total")},
                "after_retrieval_companies_with_claims": {k: med([r["t_after_retrieval"][k] for r in wc])
                                                          for k in ("extraction_done", "verification_done", "total")},
                "retrieval_stages": {s: med([r["retrieval"]["seconds"][s] for r in rows if s in r["retrieval"]["seconds"]])
                                     for s in ("search", "fetch", "rerank", "select")}},
            "retrieval": self.retrieval.summary(len(companies), wall)}
        self.sink.close(summ)
        self.retrieval.close()
        self.xclient.close()
        self.vclient.close()
        return summ
