"""Model access for the extract and verify steps (one module since 2026-11-24; was llm/client.py + llm/budget.py).

PROFILES AND CLIENT
One chat client for every model: OpenRouter (paid) and local vLLM servers (free), both OpenAI-compatible.

A Profile (config/models.toml) holds everything that differed between the E6-E14 runner copies: endpoint, model, reasoning field,
temperature, output caps, extra request fields (provider routing, Qwen's thinking switch), prompt caching and reserve prices. The request
body is built from the profile in the same field order and with the same fields as those runners, so an E6 request is byte-identical.

Keys: the OpenRouter key is read only through credentials.openrouter_key() (environment, then the env file) and only kept in the session header;
it is never printed or logged.

COST LEDGER (and optional caps)
Hard spending cap for one paid stage (extraction or verification), shared by all threads.

Same rule as E6's PaidBudget (run_stream.py) and the verifier's in-flight reserves (vfy_nw.call):
  - before a call, reserve an upper estimate of its cost;
  - if the money already SPENT plus this reserve passes the cap, stop (RuntimeError): the hard cap;
  - if spent + reserves of calls in flight + this reserve passes the cap, wait for running calls to settle (parallel calls cannot overshoot);
  - allow_when_idle (verifier rule): a call may go when nothing is in flight even if reserves alone would pass the cap (a very long page has a
    large reserve; refusing it would stop the run although the money is there).
Spend = usage.cost reported by OpenRouter for each reply. Every settled call is appended to a JSONL ledger; a new process with the same
ledger starts from the ledger's total (so a resumed run cannot spend the cap twice).
settle() releases the reserve AND adds the real cost under one lock (2026-11-24 fix: releasing first and adding later let another thread
reserve money that was about to be spent; a parallel reproduction reached $1.20 against a $1 cap). Remaining limit: a reply that costs
more than its reserve (the reserve is an upper estimate: max output tokens, 1.5x input) can pass the cap by at most that difference.
"""
from __future__ import annotations

import json
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_FILE = Path(__file__).resolve().parents[1] / "config" / "models.toml"


@dataclass(frozen=True)
class Profile:
    name: str
    role: str                                   # "extractors" | "verifiers"
    endpoint: str                               # "openrouter" | "local"
    model: str
    max_tokens: int
    base_url: str = ""
    reasoning: dict | None = None
    temperature: float | None = None
    length_retry_max_tokens: int | None = None
    extra_body: dict = field(default_factory=dict)
    prompt_cache: bool = False
    price_in: float = 0.0
    price_out: float = 0.0
    serve: dict | None = None

    @property
    def paid(self) -> bool:
        return self.endpoint == "openrouter"

    @property
    def url(self) -> str:
        return OPENROUTER_URL if self.endpoint == "openrouter" else self.base_url.rstrip("/") + "/chat/completions"

    @property
    def effort(self) -> str:
        """The reasoning setting as one string (verifier cache key; E6 used the effort string)."""
        r = self.reasoning or {}
        return str(r.get("effort") if "effort" in r else ("off" if r.get("enabled") is False else "default"))


def load_models(path: Path | None = None) -> dict[str, Any]:
    with open(path or MODELS_FILE, "rb") as fh:
        return tomllib.load(fh)


def load_profile(role: str, name: str, path: Path | None = None) -> Profile:
    """role: "extractors" or "verifiers"."""
    table = load_models(path).get(role, {})
    if name not in table:
        raise SystemExit(f"unknown {role[:-1]} profile {name!r}; choices: {', '.join(sorted(table))}")
    p = dict(table[name])
    if p.get("endpoint") not in ("openrouter", "local"):
        raise SystemExit(f"profile {name!r}: endpoint must be 'openrouter' or 'local'")
    if p["endpoint"] == "local" and not p.get("base_url"):
        raise SystemExit(f"profile {name!r}: a local profile needs base_url")
    return Profile(name=name, role=role, **p)


class ChatClient:
    """Thread-safe: one requests.Session with a connection pool, shared by all worker threads."""

    def __init__(self, profile: Profile, pool_size: int = 64):
        self.profile = profile
        self.session = requests.Session()
        self.session.mount("http://", requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size))
        self.session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size))
        if profile.paid:
            from .config import openrouter_key
            key = openrouter_key()
            self.session.headers.update({"Authorization": f"Bearer {key}"})

    def body(self, content: Any, schema_name: str, schema: dict, max_tokens: int | None = None, usage: bool = True) -> dict:
        p = self.profile
        b: dict[str, Any] = {"model": p.model, "messages": [{"role": "user", "content": content}]}
        if p.temperature is not None:
            b["temperature"] = p.temperature
        b["max_tokens"] = max_tokens or p.max_tokens
        if p.reasoning is not None:
            b["reasoning"] = dict(p.reasoning)
        if p.paid and usage:
            b["usage"] = {"include": True}
        b.update(json.loads(json.dumps(p.extra_body)))
        b["response_format"] = {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}}
        return b

    def post(self, body: dict, read_timeout: float) -> dict:
        """One HTTP call. Returns {status, data, text, finish, usage, provider, seconds, transport_error}; never raises on HTTP errors."""
        t0 = time.monotonic()
        out: dict[str, Any] = {"status": None, "data": {}, "text": "", "finish": None, "usage": {}, "provider": None, "transport_error": ""}
        try:
            r = self.session.post(self.profile.url, json=body, timeout=(30 if self.profile.paid else 10, read_timeout))
            out["status"] = r.status_code
            try:
                d = r.json()
            except ValueError:
                d = {"error": r.text[:200]}
            out["data"] = d
            out["usage"], out["provider"] = d.get("usage") or {}, d.get("provider")
            ch = (d.get("choices") or [{}])[0]
            out["text"] = (ch.get("message") or {}).get("content") or ""
            out["finish"] = ch.get("finish_reason")
        except requests.RequestException as exc:
            out["transport_error"] = type(exc).__name__
        out["seconds"] = round(time.monotonic() - t0, 2)
        return out

    def close(self) -> None:
        self.session.close()


# No cap (production default since 2026-11-24): PaidBudget(name, None, ledger) only records spend; reserve() never waits or refuses.


class BudgetExceeded(RuntimeError):
    """No further paid call may be made: a spend cap (this module) or the search call cap (steps/search.py) was reached."""
    pass


class PaidBudget:
    def __init__(self, name: str, cap_usd: float | None, ledger: Path | None, allow_when_idle: bool = False):
        self.name, self.ledger, self.allow_when_idle = name, ledger, allow_when_idle
        self.cap = float(cap_usd) if cap_usd else None                 # None or 0 = no cap: spend is only recorded
        self.spent, self.inflight, self.n_inflight = 0.0, 0.0, 0
        self._lock = threading.Lock()
        if ledger is not None and ledger.is_file():
            self.spent = sum(float(json.loads(x).get("cost_usd") or 0) for x in ledger.read_text().splitlines() if x.strip())

    def reserve(self, amount: float) -> float:
        if self.cap is None:
            with self._lock:
                self.inflight += amount
                self.n_inflight += 1
            return amount
        while True:
            with self._lock:
                if self.spent + amount > self.cap:
                    raise BudgetExceeded(f"{self.name} cap ${self.cap:.2f} reached (spent ${self.spent:.4f})")
                if self.spent + self.inflight + amount <= self.cap or (self.allow_when_idle and self.n_inflight == 0):
                    self.inflight += amount
                    self.n_inflight += 1
                    return amount
            time.sleep(0.2)

    def settle(self, reserved: float, cost: float, record: dict | None = None) -> None:
        """Release a reserve and record the call's real cost atomically (cost 0 when the call failed before any cost was known)."""
        with self._lock:
            self.inflight -= reserved
            self.n_inflight -= 1
            self.spent += cost
            if self.ledger is not None and (cost or record is not None):
                self.ledger.parent.mkdir(parents=True, exist_ok=True)
                with self.ledger.open("a") as fh:
                    fh.write(json.dumps({"cost_usd": cost, **(record or {})}) + "\n")

    @property
    def exhausted(self) -> bool:
        return self.cap is not None and self.spent >= self.cap


class FreeBudget:
    """Local models: nothing to reserve or record."""
    name, cap, spent = "free", None, 0.0

    def reserve(self, amount: float) -> float:
        return 0.0

    def settle(self, reserved: float, cost: float, record: dict | None = None) -> None:
        pass

    exhausted = False
