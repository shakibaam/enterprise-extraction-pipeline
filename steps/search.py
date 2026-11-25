"""Search component: build the queries for each company, call the search API (Serper or Keenable), save the raw responses and one
merged, de-duplicated list of result URLs per company.

Output per company (under <out>/<slug>/):
    search/<provider>/<query_id>.json   raw API response + request metadata + error category (no API key)
    search.json                         merged hits, one per canonical URL, with the queries and ranks that returned it
and <out>/search_summary.json for the run (calls, error report, incomplete companies).

Speed: all queries of all companies go into one thread pool (query_workers), so a company's queries run in parallel; the provider's
rate limit is the real cap. The first call of a run is sent alone (probe), so a rejected key or missing credits stop the run after one
call rather than after every call already in flight. On HTTP 429 the limiter doubles the gap between calls (up to max_interval_s), so an unknown plan limit
is found automatically. A company's search.json is written as soon as its last query finishes.

Errors (every one is categorized, saved with the raw response and counted in the run's error report):
    run-stopping  auth (401/403: key rejected), credits (402 or a credits message): no further calls are made; the run ends with a
                  clear message. A Keenable response reporting paid usage also stops the run, but its results are kept.
    retried       rate_limited (429), server_error (5xx), network_error: up to max_attempts tries.
    not retried   bad_request (other 4xx), bad_response (200 but not a valid result): the same request would fail again.
Resume: a query whose raw response was saved as successful is not sent again. Paid calls: every HTTP attempt counts against a hard cap
(max_calls); when it is reached no further calls are made and the affected companies are marked incomplete.

Command line (from Provinces/):
    .venv/bin/python -m company_pipeline.pipeline.search --companies companies.csv --out OUT_DIR --max-calls N [--provider keenable]
Exit code: 0 all complete, 1 some companies incomplete, 2 the run was stopped (auth, credits or call cap).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import requests

from ..shared.config import read_secret
from ..shared.models import Company, Query, SearchHit
from ..shared.llm import BudgetExceeded
from ..shared.names import search_name
from ..storage import utc_now, write_json

log = logging.getLogger(__name__)
PROVIDERS = ("serper", "keenable")
RUN_STOPPING = {"auth", "credits"}
RETRYABLE = {"rate_limited", "server_error", "network_error"}


# --------------------------------------------------------------------------- helpers

def canonical_url(url: str) -> str:
    """Normalize a URL for de-duplication: lower-case scheme and host, no trailing slash, no fragment (same rule as
    enterprise_agent.tavily.canonical_url, used by the earlier runs)."""
    parts = urlsplit(url.strip())
    return urlunsplit(((parts.scheme or "https").lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def build_queries(company: Company, templates: list[dict[str, str]]) -> list[Query]:
    """Fill the query templates for one company. The name inside the quotes is search_name(): the first part of a bilingual name, with
    double quotes removed (they would end the quoted phrase early). The company name must stay inside quotes."""
    queries = []
    qname = search_name(company.name)
    for t in templates:
        text = t["template"].format(name=qname, province=company.province.strip(), region=company.province.strip())   # {region}: country profiles
        if f'"{qname}"' not in text:
            raise ValueError(f"query {t['id']} does not quote the company name: {text!r}")
        queries.append(Query(query_id=t["id"], category=t["category"], text=text))
    if len({q.query_id for q in queries}) != len(queries):
        raise ValueError("duplicate query ids in the configuration")
    return queries


class CallBudget:
    """Thread-safe hard cap on API calls (every HTTP attempt counts). `stop()` blocks all further calls (auth or credit errors)."""

    def __init__(self, max_calls: int):
        if max_calls < 0:
            raise ValueError("max_calls must be 0 (no cap) or positive")
        self.max_calls, self.used, self.stopped_reason = max_calls, 0, ""        # 0 = no cap (production default, 2026-11-24)
        self._lock = threading.Lock()

    def take(self) -> None:
        with self._lock:
            if self.stopped_reason:
                raise BudgetExceeded(self.stopped_reason)
            if self.max_calls and self.used >= self.max_calls:
                self.stopped_reason = f"call cap reached ({self.max_calls} calls)"
                raise BudgetExceeded(self.stopped_reason)
            self.used += 1

    def stop(self, reason: str) -> None:
        with self._lock:
            if not self.stopped_reason:
                self.stopped_reason = reason
                log.error("search stopped: %s", reason)


class RateLimiter:
    """At most one call every `interval` seconds across all threads. `slow_down()` doubles the interval (up to `max_interval`)
    after an HTTP 429, so the run settles below an unknown plan limit."""

    def __init__(self, min_interval_s: float, max_interval_s: float):
        self.interval, self.max_interval, self.slowdowns = min_interval_s, max(max_interval_s, min_interval_s), 0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            time.sleep(start - now)

    def slow_down(self) -> None:
        with self._lock:
            new = min(self.max_interval, max(self.interval * 2, 0.01))
            if new > self.interval:
                self.interval, self.slowdowns = new, self.slowdowns + 1
                log.warning("HTTP 429: slowing search calls to one every %.3f s", new)


def classify(status: int | None, body: dict[str, Any], ok: bool) -> str:
    """Error category of one HTTP attempt ("" when it succeeded)."""
    if ok:
        return ""
    message = str(body.get("error") or body.get("message") or "").lower()
    if status is None:
        return "network_error"
    if status in (401, 403):
        return "auth"
    if status == 402 or ("credit" in message and status < 500):
        return "credits"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server_error"
    if status >= 400:
        return "bad_request"
    return "bad_response"


# --------------------------------------------------------------------------- providers

class SearchProvider:
    """Common request logic. Subclasses define the request body, the headers and how results are parsed."""

    name = ""

    def __init__(self, cfg: dict[str, Any], api_key: str, budget: CallBudget, limiter: RateLimiter,
                 post: Callable[..., requests.Response] = requests.post):
        self.cfg, self._key, self.budget, self.limiter, self._post = cfg, api_key, budget, limiter, post
        self.results_per_query = int(cfg["results_per_query"])
        self.max_attempts, self.timeout_s = int(cfg["max_attempts"]), float(cfg["timeout_s"])

    # subclass hooks
    def endpoint(self) -> str: raise NotImplementedError
    def headers(self) -> dict[str, str]: raise NotImplementedError
    def payload(self, query: Query) -> dict[str, Any]: raise NotImplementedError
    def is_ok(self, status: int | None, body: dict[str, Any]) -> bool: raise NotImplementedError
    def parse(self, body: dict[str, Any], query: Query) -> list[SearchHit]: raise NotImplementedError

    def stop_signal(self, status: int | None, body: dict[str, Any]) -> str:
        """Provider-specific reason to stop the run found in a successful response ("" if none). The result itself is kept."""
        return ""

    def search(self, query: Query) -> dict[str, Any]:
        """Send one query with retries. Returns the response body plus metadata, including `_error` (category, "" if ok)
        and `_error_message`. Stops the whole run on auth or credit errors. Raises BudgetExceeded if no call may be made."""
        payload = self.payload(query)
        status, body, attempt, latency, category, history = None, {}, 0, 0.0, "", []
        for attempt in range(1, self.max_attempts + 1):
            self.budget.take()
            self.limiter.wait()
            t0, retry_after = time.monotonic(), None
            try:
                resp = self._post(self.endpoint(), json=payload, headers=self.headers(), timeout=self.timeout_s)
                status, retry_after = resp.status_code, resp.headers.get("Retry-After")
                try:
                    body = resp.json()
                except ValueError:
                    body = {"error": resp.text[:300]}
                if not isinstance(body, dict):
                    body = {"error": "response is not a JSON object"}
            except requests.RequestException as exc:
                status, body = None, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            latency = round(time.monotonic() - t0, 3)
            ok = self.is_ok(status, body)
            category = classify(status, body, ok)
            signal = "" if category else self.stop_signal(status, body)
            history.append({"attempt": attempt, "status": status, "error": category, **({"stop_signal": signal} if signal else {})})
            if signal:   # a valid result, kept, but no further calls may be made
                self.budget.stop(f"{self.name}: {signal}")
                break
            if category in RUN_STOPPING:
                self.budget.stop(f"{self.name}: {'API key rejected' if category == 'auth' else 'no credits'} "
                                 f"(HTTP {status}): {str(body.get('error') or body.get('message') or '')[:200]}")
                break
            if category == "rate_limited":
                self.limiter.slow_down()
            if category not in RETRYABLE:
                break
            if attempt < self.max_attempts:
                time.sleep(float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else 1.5 * attempt)
        message = "" if not category else str(body.get("error") or body.get("message") or f"HTTP {status}")[:300]
        return {**body, "_provider": self.name, "_http_status": status, "_ok": category == "", "_error": category,
                "_error_message": message, "_attempts": attempt, "_attempt_history": history, "_latency_s": latency,
                "_query_id": query.query_id, "_query_text": query.text, "_request": payload, "_saved_at": utc_now()}


class SerperProvider(SearchProvider):
    """Serper (Google results). Same request as the earlier runs: gl=ca, hl=en, num=10, first page only."""

    name = "serper"

    def endpoint(self) -> str:
        return self.cfg["serper"]["endpoint"]

    def headers(self) -> dict[str, str]:
        return {"X-API-KEY": self._key, "Content-Type": "application/json"}

    def payload(self, query: Query) -> dict[str, Any]:
        s = self.cfg["serper"]
        return {"q": query.text, "gl": s["country"], "hl": s["language"], "num": self.results_per_query, "page": 1}

    def is_ok(self, status: int | None, body: dict[str, Any]) -> bool:
        return status == 200 and not body.get("error") and isinstance(body.get("organic", []), list)

    def parse(self, body: dict[str, Any], query: Query) -> list[SearchHit]:
        hits = []
        for item in body.get("organic") or []:
            url = str(item.get("link") or "").strip() if isinstance(item, dict) else ""
            if not url:
                continue
            hits.append(SearchHit(url=url, canonical_url=canonical_url(url), title=item.get("title") or "", snippet=item.get("snippet") or "",
                                  query_id=query.query_id, rank=int(item.get("position") or len(hits) + 1),
                                  extra={k: item[k] for k in ("date",) if k in item}))
        return hits[: self.results_per_query]


class KeenableProvider(SearchProvider):
    """Keenable search. Same request as the earlier runs: max_results=10, snippet_max_length=180, default mode; no pagination,
    no country or language parameter. A response that reports paid usage is kept, but the run stops making calls."""

    name = "keenable"

    def endpoint(self) -> str:
        return self.cfg["keenable"]["endpoint"]

    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self._key, "Content-Type": "application/json"}

    def payload(self, query: Query) -> dict[str, Any]:
        return {"query": query.text, "max_results": self.results_per_query, "snippet_max_length": int(self.cfg["keenable"]["snippet_max_length"])}

    def is_ok(self, status: int | None, body: dict[str, Any]) -> bool:
        return status == 200 and isinstance(body.get("results"), list)

    def stop_signal(self, status: int | None, body: dict[str, Any]) -> str:
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return "response reports paid usage (free credits used up)" if usage.get("paid") else ""

    def parse(self, body: dict[str, Any], query: Query) -> list[SearchHit]:
        hits = []
        for item in body.get("results") or []:
            url = str(item.get("url") or "").strip() if isinstance(item, dict) else ""
            if not url:
                continue
            hits.append(SearchHit(url=url, canonical_url=canonical_url(url), title=item.get("title") or "", snippet=item.get("snippet") or "",
                                  query_id=query.query_id, rank=len(hits) + 1,
                                  extra={k: item[k] for k in ("published_at", "acquired_at") if item.get(k)}))
        return hits[: self.results_per_query]


def make_provider(name: str, cfg: dict[str, Any], budget: CallBudget, post: Callable[..., requests.Response] = requests.post) -> SearchProvider:
    """Create the provider named in the config or on the command line. The API key is read from the .env file."""
    if name not in PROVIDERS:
        raise ValueError(f"unknown search provider {name!r}; choose one of {PROVIDERS}")
    s = cfg["search"]
    key = read_secret(cfg, s[name]["api_key_name"])
    cls = SerperProvider if name == "serper" else KeenableProvider
    limiter = RateLimiter(float(s[name]["min_interval_s"]), float(s["max_interval_s"]))
    return cls(s, key, budget, limiter, post=post)


# --------------------------------------------------------------------------- per company

def merge_hits(hits_by_query: list[list[SearchHit]]) -> list[dict[str, Any]]:
    """One entry per canonical URL. Order: best rank across queries; ties go to the query (in config order) that gave that best rank.
    Keeps every (query, rank) that returned the URL. Each entry gets a stable id r001, r002, ... in that order."""
    merged: dict[str, dict[str, Any]] = {}
    for qi, hits in enumerate(hits_by_query):
        for h in hits:
            m = merged.get(h.canonical_url)
            if m is None:
                merged[h.canonical_url] = m = {"url": h.url, "canonical_url": h.canonical_url, "title": h.title, "snippet": h.snippet,
                                               "best_rank": h.rank, "best_query_index": qi, "found_by": []}
            m["found_by"].append({"query_id": h.query_id, "rank": h.rank, "snippet": h.snippet, **h.extra})
            if h.rank < m["best_rank"]:
                m.update(best_rank=h.rank, best_query_index=qi, title=h.title or m["title"], snippet=h.snippet or m["snippet"])
    out = sorted(merged.values(), key=lambda m: (m["best_rank"], m["best_query_index"]))
    for i, m in enumerate(out, start=1):
        m["result_id"] = f"r{i:03d}"
        m.pop("best_query_index")
    return out


def raw_path(out_root: Path, company: Company, provider: str, query: Query) -> Path:
    return out_root / company.slug / "search" / provider / f"{query.query_id}.json"


def load_saved(path: Path, query: Query) -> dict[str, Any] | None:
    """A saved successful response for this exact query text, or None."""
    if not path.is_file():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    return body if body.get("_ok") and body.get("_query_text") == query.text else None


def run_query(provider: SearchProvider, query: Query, path: Path) -> dict[str, Any]:
    """Send one query and save its raw response (also when it failed, for diagnosis). If no call may be made, nothing is saved."""
    try:
        body = provider.search(query)
    except BudgetExceeded as exc:
        return {"_ok": False, "_error": "not_sent", "_error_message": str(exc), "_query_id": query.query_id, "_query_text": query.text}
    write_json(path, body)
    return body


def company_summary(company: Company, provider: SearchProvider, queries: list[Query], bodies: dict[str, dict[str, Any]],
                     out_root: Path) -> dict[str, Any]:
    """Merge the hits of one company's successful queries and write search.json. Returns the company's summary."""
    failed = {q.query_id: {"error": bodies[q.query_id].get("_error"), "message": bodies[q.query_id].get("_error_message", "")[:200],
                           "http_status": bodies[q.query_id].get("_http_status")}
              for q in queries if not bodies[q.query_id].get("_ok")}
    hits = merge_hits([provider.parse(bodies[q.query_id], q) for q in queries if bodies[q.query_id].get("_ok")])
    status = "complete" if not failed else "incomplete"
    write_json(out_root / company.slug / "search.json", {
        "company": {"name": company.name, "province": company.province, "slug": company.slug, "company_id": company.company_id},
        "provider": provider.name, "status": status, "failed_queries": failed,
        "queries": [{"query_id": q.query_id, "category": q.category, "text": q.text} for q in queries],
        "hits": hits, "written_at": utc_now()})
    return {"slug": company.slug, "status": status, "urls": len(hits), "failed_queries": failed}


class Searcher:
    """Searches one company at a time (search_company); many threads may call it at once (the streaming pipeline does).
    Shared by all companies: the paid-call budget, the provider's rate limiter, one pool of query workers, and the probe: the first call
    of the run is sent alone, so a rejected key or missing credits stop the run after one call instead of after every call in flight."""

    def __init__(self, cfg: dict[str, Any], out_root: Path, provider_name: str, max_calls: int,
                 post: Callable[..., requests.Response] = requests.post):
        self.cfg, self.out_root, self.max_calls = cfg["search"], out_root, max_calls
        self.budget = CallBudget(max_calls)
        self.provider = make_provider(provider_name, cfg, self.budget, post=post)
        self.pool = ThreadPoolExecutor(int(self.cfg["query_workers"]))
        self.reused = 0
        self.results: list[dict[str, Any]] = []
        self._probe_lock, self._probe_done, self._lock = threading.Lock(), False, threading.Lock()

    @property
    def stopped_reason(self) -> str:
        return self.budget.stopped_reason

    def _run(self, q: Query, path: Path) -> dict[str, Any]:
        try:
            return run_query(self.provider, q, path)
        except Exception as exc:  # noqa: BLE001 - unexpected error in one call: recorded, the run continues
            log.exception("search call failed for %s", q.text)
            return {"_ok": False, "_error": "internal_error", "_error_message": f"{type(exc).__name__}: {str(exc)[:300]}"}

    def search_company(self, c: Company) -> dict[str, Any]:
        """Search one company (its queries in parallel), write search.json; returns its summary row ("status": complete | incomplete | error)."""
        try:
            queries = build_queries(c, self.cfg["queries"])
        except Exception as exc:  # noqa: BLE001 - a bad company row is recorded, the run continues
            return self._record({"slug": c.slug, "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        bodies: dict[str, dict[str, Any]] = {}
        todo = []
        for q in queries:
            path = raw_path(self.out_root, c, self.provider.name, q)
            saved = load_saved(path, q)
            if saved is not None:
                bodies[q.query_id] = saved
                with self._lock:
                    self.reused += 1
            else:
                todo.append((q, path))
        if todo and not self._probe_done:
            with self._probe_lock:
                if not self._probe_done:
                    q, path = todo.pop(0)
                    bodies[q.query_id] = self._run(q, path)
                    self._probe_done = True
        futures = {self.pool.submit(self._run, q, path): q.query_id for q, path in todo}
        for fut in as_completed(futures):
            bodies[futures[fut]] = fut.result()
        try:
            return self._record(company_summary(c, self.provider, queries, bodies, self.out_root))
        except Exception as exc:  # noqa: BLE001
            log.exception("could not write search.json for %s", c.slug)
            return self._record({"slug": c.slug, "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}"})

    def _record(self, row: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.results.append(row)
        return row

    def summary(self, n_companies: int, wall_s: float) -> dict[str, Any]:
        """Run summary (also written to search_summary.json)."""
        with self._lock:
            results = list(self.results)
        errors = Counter(f["error"] for r in results for f in (r.get("failed_queries") or {}).values())
        errors.update("company_error" for r in results if r["status"] == "error")
        examples = [{"slug": r["slug"], "query_id": q, **f} for r in results for q, f in (r.get("failed_queries") or {}).items()][:20]
        summary = {"provider": self.provider.name, "companies": n_companies, "complete": sum(r["status"] == "complete" for r in results),
                   "incomplete": sorted(r["slug"] for r in results if r["status"] != "complete"),
                   "api_calls": self.budget.used, "max_calls": self.max_calls, "queries_reused": self.reused,
                   "stopped_reason": self.budget.stopped_reason, "errors": dict(errors), "error_examples": examples,
                   "rate_limit_slowdowns": self.provider.limiter.slowdowns, "final_interval_s": self.provider.limiter.interval,
                   "wall_s": round(wall_s, 1), "written_at": utc_now(), "per_company": sorted(results, key=lambda r: r["slug"])}
        write_json(self.out_root / "search_summary.json", summary)
        return summary

    def close(self) -> None:
        self.pool.shutdown(wait=True)
