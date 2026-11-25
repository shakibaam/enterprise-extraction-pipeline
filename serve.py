"""Start (or stop) vLLM servers: the Granite reranker (one server per GPU) or a local extraction LLM (one server over its GPUs).
Copy of company_pipeline/scripts/serve_vllm.py (2026-11-24; was scripts/serve_vllm.py, a top-level module since 2026-11-24). Changes: the LLM comes from a profile in config/models.toml
([extractors.<name>.serve]) instead of [extract.profiles]; component "llm" replaces "extract"; logs go to on_the_fly_pipeline/logs/.
usage (from Provinces/):
  python -m on_the_fly_pipeline serve rerank                      [--gpus 0 1]
  python -m on_the_fly_pipeline serve llm --profile qwen9b_local  [--gpus 0 1]
  python -m on_the_fly_pipeline serve rerank --stop               (stops only the servers this launcher started)
The GPUs are shared with other users: if any GPU lacks the free memory the servers need, nothing is started. The reranker needs about
0.09 of each GPU; Qwen 9B (TP=2) 0.80, so both fit together.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .shared.config import load_config
from .shared.llm import load_models

VLLM = "/home/shakiba/vllm-env/bin/vllm"
LOGS = Path(__file__).resolve().parent / "logs"


def gpu_mib(gpu: int) -> tuple[int, int]:
    """(free, total) MiB of one GPU."""
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1])


def plan(cfg: dict, component: str, profile_name: str | None, gpus: list[int]) -> tuple[dict, str, list[dict]]:
    """(profile, model, servers) where each server is {gpus, port}."""
    if component == "rerank":
        section = cfg["rerank"]
        name = profile_name or section["profile"]
        profile = section["profiles"][name]
    else:
        name = profile_name or cfg["extraction"]["extractor"]
        profile = load_models()["extractors"].get(name)
        if profile is None:
            raise SystemExit(f"unknown extractor profile {name!r}")
    if "serve" not in profile:
        raise SystemExit(f"profile {name!r} has no [serve] table: it is not served by vLLM (an API model?)")
    per = int(profile["serve"].get("gpus_per_server", 1))
    if len(gpus) % per:
        raise SystemExit(f"{len(gpus)} GPUs can't be split into servers of {per}")
    groups = [gpus[i:i + per] for i in range(0, len(gpus), per)]
    if component == "rerank":
        servers = [{"gpus": g, "port": int(section["base_port"]) + g[0]} for g in groups]
    else:
        first = urlsplit(profile.get("base_url", "")).port or 8000
        servers = [{"gpus": g, "port": first + i} for i, g in enumerate(groups)]
    return profile, name, servers


def start(component: str, profile_name: str | None, gpus: list[int], config: str | None) -> int:
    cfg = load_config(config)
    profile, name, servers = plan(cfg, component, profile_name, gpus)
    serve = profile["serve"]
    mem = float(serve.get("gpu_memory_utilization", 0.9))
    for g in gpus:
        free, total = gpu_mib(g)
        if free < mem * total + 500:
            print(f"GPU {g}: {free} MiB free, the {name} server needs about {int(mem * total + 500)} MiB: nothing started")
            return 1
    LOGS.mkdir(exist_ok=True)
    env = dict(os.environ, **{k: str(v) for k, v in serve.get("env", {}).items()})
    started = []
    for s in servers:
        cmd = [VLLM, "serve", serve.get("model", profile["model"]), "--port", str(s["port"]), "--gpu-memory-utilization", str(mem), *serve["args"]]
        log = open(LOGS / f"{component}_gpu{'_'.join(map(str, s['gpus']))}.log", "a")
        proc = subprocess.Popen(cmd, env=dict(env, CUDA_VISIBLE_DEVICES=",".join(map(str, s["gpus"]))), stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        started.append({**s, "pid": proc.pid, "profile": name, "model": profile["model"], "log": log.name})
        print(f"GPUs {s['gpus']}: pid {proc.pid}, port {s['port']}, {profile['model']}")
    (LOGS / f"{component}_pids.json").write_text(json.dumps(started, indent=1))
    for s in started:
        for _ in range(360):                                        # up to 30 minutes (first start compiles kernels)
            try:
                if profile["model"] in [m["id"] for m in requests.get(f"http://localhost:{s['port']}/v1/models", timeout=5).json()["data"]]:
                    print(f"port {s['port']} ready")
                    break
            except Exception:  # noqa: BLE001 - not up yet
                pass
            if subprocess.run(["kill", "-0", str(s["pid"])], capture_output=True).returncode:
                print(f"server on GPUs {s['gpus']} exited: see {s['log']}")
                return 1
            time.sleep(5)
        else:
            print(f"server on GPUs {s['gpus']} not ready after 30 minutes: see {s['log']}")
            return 1
    return 0


def stop(component: str) -> int:
    path = LOGS / f"{component}_pids.json"
    if not path.is_file():
        print(f"no {component} servers started by this script")
        return 0
    for s in json.loads(path.read_text()):
        try:
            os.killpg(s["pid"], signal.SIGTERM)      # its own process group (start_new_session): the server and its workers only
            print(f"stopped {component} server on GPUs {s['gpus']} (pid {s['pid']})")
        except ProcessLookupError:
            print(f"{component} server on GPUs {s['gpus']} (pid {s['pid']}) was not running")
    path.unlink()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Start or stop the vLLM servers (Granite reranker, or a local LLM profile).")
    ap.add_argument("component", choices=["rerank", "llm"])
    ap.add_argument("--profile", default=None, help="rerank: [rerank.profiles] name; llm: an extractor profile with a [serve] table")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--config", default=None)
    ap.add_argument("--stop", action="store_true")
    a = ap.parse_args(argv)
    return stop(a.component) if a.stop else start(a.component, a.profile, a.gpus, a.config)


if __name__ == "__main__":
    sys.exit(main())
