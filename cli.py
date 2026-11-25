"""Command line. Run from Provinces/ with the vLLM environment's Python (the reranker needs transformers):

  PY=/home/shakiba/vllm-env/bin/python
  $PY -m on_the_fly_pipeline profiles                                   list extractor and verifier profiles
  $PY -m on_the_fly_pipeline check-balance --need 5                     both OpenRouter limits (key and account); exit 1 if below --need
  $PY -m on_the_fly_pipeline serve rerank | serve llm --profile qwen9b_local [--stop]
  $PY -m on_the_fly_pipeline run --companies FILE --out DIR [--max-search-calls N] [--cap-extract-usd X] [--cap-verify-usd Y]
        [--slugs FILE] [--extractor haiku_medium] [--verifier terra_none] [--provider serper|keenable]
        [--from-saved DIR] [--company-workers 8] [--verify-cache DIR] [--config FILE] [--input-format auto] [--output-format json]
        [--country CA] [--resume]
  $PY -m on_the_fly_pipeline evaluate recall --out DIR [--slugs FILE]
  $PY -m on_the_fly_pipeline evaluate judge  --out DIR --companies FILE --cap-usd Z [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .shared.llm import load_models, load_profile


def cmd_profiles(_a) -> int:
    m = load_models()
    for role in ("extractors", "verifiers"):
        print(role)
        for name, p in m.get(role, {}).items():
            print(f"  {name:20} {p['endpoint']:10} {p['model']:32} reasoning={p.get('reasoning')}")
    return 0


def balance() -> tuple[float | None, float]:
    import requests
    from .shared.config import openrouter_key
    h = {"Authorization": "Bearer " + openrouter_key()}
    k = requests.get("https://openrouter.ai/api/v1/key", headers=h, timeout=20).json()["data"]
    c = requests.get("https://openrouter.ai/api/v1/credits", headers=h, timeout=20).json()["data"]
    return k.get("limit_remaining"), round(c["total_credits"] - c["total_usage"], 4)


def cmd_check_balance(a) -> int:
    key_left, acc_left = balance()
    print(f"key left ${key_left}  account left ${acc_left}")
    ok = (key_left is None or key_left >= a.need) and acc_left >= a.need
    return 0 if ok else 1


def cmd_run(a) -> int:
    from .shared.config import resolve_config
    from .shared.config import resolve_path
    from .pipeline import Pipeline
    from .pipeline import fingerprint, prepare_out
    from .pipeline import LiveRetrieval, SavedRetrieval
    from .storage import open_companies, open_sink
    cfg = resolve_config(a.config, a.country)
    country = cfg["country"]
    if country["status"] != "validated":
        logging.warning("country profile %s is %s: untested settings (see config/countries/%s.toml)", country["code"], country["status"],
                        country["code"].lower())
    xprof = load_profile("extractors", a.extractor or cfg["extraction"]["extractor"])
    vprof = load_profile("verifiers", a.verifier or cfg["verification"]["verifier"])
    # No spending caps by default (production, 2026-11-24): cost is always recorded (ledgers, timing.json, run_summary.json); the
    # --cap-* / --max-search-calls flags are optional hard caps for experiments.
    out = Path(a.out)
    io = cfg.get("io", {})
    out_fmt = a.output_format or io.get("output_format", "json")
    open_sink(out_fmt, out)                                                   # fails early for a planned format
    comps = open_companies(a.companies, a.input_format or io.get("input_format", "auto"))
    if a.slugs:
        want = Path(a.slugs).read_text().split()
        by = {c.slug: c for c in comps}
        missing = [s for s in want if s not in by]
        if missing:
            sys.exit(f"{len(missing)} slugs not in {a.companies}: {missing[:5]}")
        comps = [by[s] for s in want]
    other = [f"{c.slug} ({c.country})" for c in comps if c.country != country["code"]]
    if other:                                                                 # one country per run
        sys.exit(f"{len(other)} companies are not {country['code']} (this run's --country): {other[:5]}. Run each country separately.")
    provider = a.provider or cfg["search"]["provider"]
    prompt_files = [cfg["extraction"]["prompt_file"], cfg["extraction"]["chunk_sizing_prompt_file"], cfg["verification"]["prompt_file"],
                    cfg["verification"]["registry_record_file"]]
    fp = fingerprint(companies_source=a.companies, slugs=[c.slug for c in comps], cfg=cfg, extractor=xprof.__dict__,
                     verifier=vprof.__dict__, prompts={f: resolve_path(f).read_text(encoding="utf-8") for f in prompt_files},
                     retrieval={"from_saved": str(Path(a.from_saved).resolve()) if a.from_saved else None,
                                "provider": None if a.from_saved else provider, "output_format": out_fmt})
    prepare_out(out, fp, a.resume, {"args": vars(a), "country": country, "extractor": xprof.name, "verifier": vprof.name})
    sink = open_sink(out_fmt, out)
    retrieval = SavedRetrieval(Path(a.from_saved), out) if a.from_saved else LiveRetrieval(cfg, out, provider, a.max_search_calls)
    (out / "run_args.json").write_text(json.dumps({**vars(a), "extractor": xprof.name, "verifier": vprof.name, "country": country,
                                                   "extractor_profile": xprof.__dict__, "verifier_profile": vprof.__dict__}, indent=1, default=str))
    p = Pipeline(cfg, out, retrieval, xprof, vprof, a.cap_extract_usd, a.cap_verify_usd,
                 Path(a.verify_cache) if a.verify_cache else None, sink, fp["fingerprint"])
    s = p.run(comps, a.company_workers or int(cfg["pipeline"]["company_workers"]))
    print(json.dumps({k: v for k, v in s.items() if k not in ("retrieval", "errors")}, indent=1))
    if s["errors"]:
        print(f"{len(s['errors'])} companies failed: see run_summary.json")
    return 2 if s["stopped_reason"] else (1 if s["errors"] or s["failed_requests"] else 0)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("urllib3", "transformers", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m on_the_fly_pipeline", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("profiles", help="list model profiles")
    b = sub.add_parser("check-balance", help="OpenRouter key and account balance")
    b.add_argument("--need", type=float, default=0.0)
    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("--companies", required=True, help="companies source: a JSON Lines, CSV or TSV file (see storage.py)")
    r.add_argument("--input-format", default=None, help="auto (default, by extension) | jsonl | csv | tsv; planned: json, database")
    r.add_argument("--output-format", default=None, help="json (default); planned: jsonl, csv, database")
    r.add_argument("--slugs", default=None, help="optional file of slugs to run, in order")
    r.add_argument("--out", required=True)
    r.add_argument("--config", default=None)
    r.add_argument("--extractor", default=None)
    r.add_argument("--verifier", default=None)
    r.add_argument("--provider", default=None, choices=["serper", "keenable"])
    r.add_argument("--max-search-calls", type=int, default=0, help="optional cap on paid search calls (default 0 = no cap)")
    r.add_argument("--from-saved", default=None, help="use the saved pages of an earlier run (DIR/<slug>/select.json, pages/)")
    r.add_argument("--cap-extract-usd", type=float, default=None, help="optional hard cap on extraction spend (default: none)")
    r.add_argument("--cap-verify-usd", type=float, default=None, help="optional hard cap on verifier spend (default: none)")
    r.add_argument("--company-workers", type=int, default=0)
    r.add_argument("--verify-cache", default=None, help="verifier answer cache (default OUT/verify_cache)")
    r.add_argument("--country", default="CA", help="country profile (config/countries/<code>.toml); one country per run. Default CA")
    r.add_argument("--resume", action="store_true", help="continue the run in --out (only if nothing that decides results changed)")
    sv = sub.add_parser("serve", help="start/stop vLLM servers", add_help=False)
    ev = sub.add_parser("evaluate", help="optional evaluation (recall: free; judge: paid)", add_help=False)
    a, rest = ap.parse_known_args(argv)
    if a.cmd == "serve":
        from .serve import main as serve_main
        return serve_main(rest)
    if a.cmd == "evaluate":
        if not rest or rest[0] not in ("recall", "judge"):
            print("usage: python -m on_the_fly_pipeline evaluate recall|judge ...")
            return 2
        if rest[0] == "recall":
            from .evaluate.recall import main as m
        else:
            from .evaluate.judge import main as m
        return m(rest[1:])
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    return {"profiles": cmd_profiles, "check-balance": cmd_check_balance, "run": cmd_run}[a.cmd](a)
