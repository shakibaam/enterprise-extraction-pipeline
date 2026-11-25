"""Free check of the retrieval half on the GPUs: rerank + select on saved search and fetch results, compared with the saved run.

For the companies of the 83-company set whose search.json and fetch.json were saved (the new180 sample, retrieved with this same config),
the saved pages are reranked by this package (Granite on local vLLM) and selected; the kept pages and scores are compared with the saved
rerank.json / select.json. Starts the reranker servers (GPU, about 0.09 of each GPU) and stops them at the end. No paid calls.
usage (from Provinces/): /home/shakiba/vllm-env/bin/python -B -m on_the_fly_pipeline.tests.check_rerank --work DIR
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from on_the_fly_pipeline.storage import open_companies as load_companies
from on_the_fly_pipeline.shared.config import PROVINCES_DIR
from on_the_fly_pipeline.shared.config import resolve_config
from on_the_fly_pipeline.shared.foreign_pages import LocationCheck
from on_the_fly_pipeline.steps.rerank import Reranker
from on_the_fly_pipeline.steps.select import profile_thresholds, select_one
from on_the_fly_pipeline.shared.foreign_pages import SiteFilter
from on_the_fly_pipeline.serve import start, stop

SRC = PROVINCES_DIR / "enterprise_variant_d/runs/recall_fixes_20261115"
E6 = PROVINCES_DIR / "enterprise_agent/results/experiments/e2e_stream_final_terra_20261120"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--no-servers", action="store_true", help="reranker servers already running")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    cfg = resolve_config(None, "CA")
    comps = {c.slug: c for c in load_companies(str(SRC / "companies.jsonl"))}
    slugs = [s for s in (E6 / "companies_83.txt").read_text().split()
             if (SRC / "out_AE" / s / "search.json").is_file() and (SRC / "out_AE" / s / "fetch.json").is_file()]
    for s in slugs:
        d = work / s
        d.mkdir(exist_ok=True)
        for f in ("search.json", "fetch.json"):
            shutil.copy2(SRC / "out_AE" / s / f, d / f)
        if not (d / "pages").exists():
            (d / "pages").symlink_to((SRC / "out_AE" / s / "pages").resolve())
    if not a.no_servers and start("rerank", None, [0, 1], None) != 0:
        print("reranker servers could not start")
        return 1
    try:
        rr = Reranker(cfg, work)
        rr.check()
        th, sites, loc = profile_thresholds(cfg), SiteFilter.from_config(cfg), LocationCheck.from_config(cfg)
        same_kept, score_diff, secs, rows = 0, [], [], []
        for s in slugs:
            t = time.monotonic()
            status = rr.rerank_one(comps[s])
            sel = select_one(work, comps[s], th, sites, loc, cfg["select"].get("host_rule", "first_word"))
            secs.append(time.monotonic() - t)
            old_sel = json.loads((SRC / "out_AE" / s / "select.json").read_text())
            old_r = {p["result_id"]: p.get("score") for p in json.loads((SRC / "out_AE" / s / "rerank.json").read_text())["pages"]}
            new_r = {p["result_id"]: p.get("score") for p in json.loads((work / s / "rerank.json").read_text())["pages"]}
            k_old = sorted(x["result_id"] for x in old_sel.get("kept", []))
            k_new = sorted(x["result_id"] for x in sel.get("kept", []))
            same_kept += k_old == k_new
            score_diff += [abs((new_r.get(i) or 0) - (old_r.get(i) or 0)) for i in old_r if old_r[i] is not None]
            rows.append({"slug": s, "status": status, "kept_old": k_old, "kept_new": k_new, "same": k_old == k_new})
        print(f"companies {len(slugs)}: same kept pages {same_kept}; max |score difference| {max(score_diff or [0]):.4f}; "
              f"rerank+select seconds median {sorted(secs)[len(secs) // 2]:.2f}, max {max(secs):.1f}")
        for r in rows:
            if not r["same"]:
                print("  differs:", r)
        (work / "check_rerank.json").write_text(json.dumps(rows, indent=1))
        return 0 if same_kept == len(slugs) else 1
    finally:
        if not a.no_servers:
            stop("rerank")


if __name__ == "__main__":
    raise SystemExit(main())
