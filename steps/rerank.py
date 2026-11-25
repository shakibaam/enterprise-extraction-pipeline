"""Rerank component: score every fetched page of a company with a reranker model served by vLLM.

Input:  OUT/<slug>/fetch.json and OUT/<slug>/pages/<result_id>.json (written by fetch.py).
Output: OUT/<slug>/rerank.json     one score per page (0..1) with its URL, chunk count and tokens
        OUT/rerank_summary.json    run summary: companies, pages, tokens, time (tokenizing vs scoring), errors

Which reranker: [rerank].profile names one of the [rerank.profiles.<name>] sections in the config. A profile holds everything that depends
on the model: model and tokenizer names, input format, token limit, which output is the score, the query, the document header, the
threshold used by select, and the vLLM server settings (python -m on_the_fly_pipeline serve rerank). To change the reranker, add a profile and set
[rerank].profile; no code change is needed unless the model needs a new input format (add it to ENCODERS).

Input formats (ENCODERS):
  - "pair":     cross-encoder pair, tokenizer(query, document) with the model's own special tokens (Granite, other BERT-style rerankers);
  - "template": one text built from the profile's `template` with {query} and {document} (prompt-style rerankers such as Qwen3-Reranker).
Score: probs[score_index] of vLLM's POST /classify answer (a single-output model gives one sigmoid probability: index 0).

Requests (with the "granite" profile: the same as the validated pipeline, enterprise_variant_d/runs/pipeline_granite9b_bc80_20261112):
  - document: the profile's document_header ("Title: <title>\\nURL: <url>\\n\\n") + the whole page text, up to max_tokens with the query;
  - a longer page is cut into consecutive chunks that each fit (query and header repeated); page score = its best chunk;
  - nothing is truncated silently: every request is checked against max_tokens before it is sent.
Pages without content (content_source "none") are not scored. With [rerank].name_filter (default on), pages that don't name the company
(pipeline/names.py, the same tests select applies) are not scored either: select would drop them anyway, so the pages kept are the same and
the GPU scores far fewer tokens (BC80: 9% of pages name the company, 36% of the tokens). Choosing pages (threshold, name test) is done by
select_pages.py, with the threshold of the profile recorded in rerank.json. With [select.foreign_sites].enabled, pages on foreign websites
(shared/foreign_pages.py) are not scored either, for the same reason (select drops them; skipped_foreign).

Servers: one or more vLLM servers (python -m on_the_fly_pipeline serve rerank, one per GPU). Companies alternate between servers; each company's requests go
in one POST /classify (split into posts of at most max_inputs_per_post). Tokenizing runs on the CPU in `workers` threads.

Errors (per company, counted in rerank_summary.json):
  retried (up to max_attempts, on the next server): server_unreachable (connection error), timeout, server_error (5xx)
  not retried: bad_request (4xx, e.g. a request over the server's length), bad_response (wrong number of scores), page_error (a page that can't
               be chunked, e.g. a title longer than the token budget; the page is skipped and listed, the company still completes),
               missing_fetch (no fetch.json: run fetch first)
  run-stopping: no server answers at start, or server_failure_limit companies in a row fail with server_unreachable/timeout.
Rerunning the same command reuses a company's rerank.json when the profile settings and the fetched pages are unchanged.

Needs `transformers` for the tokenizer: run with the vLLM environment (/home/shakiba/vllm-env/bin/python).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from ..shared.models import Company
from ..shared.names import NameMatcher
from ..shared.foreign_pages import SiteFilter
from ..storage import utc_now, write_json

log = logging.getLogger(__name__)

RETRYABLE = {"server_unreachable", "timeout", "server_error"}
SERVER_DOWN = {"server_unreachable", "timeout"}


class RunStopped(RuntimeError):
    """The reranker servers are not usable; no further company can be scored."""


class CompanyError(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category, self.message = category, message


# ---------------------------------------------------------------- reranker profile

@dataclass(frozen=True)
class RerankProfile:
    """Everything that depends on the reranker model (one [rerank.profiles.<name>] section)."""
    name: str
    model: str
    tokenizer: str
    input_format: str
    max_tokens: int
    score_index: int
    threshold: float
    query: str
    document_header: str
    template: str = ""
    add_special_tokens: bool = True


def load_rerank_profile(cfg: dict[str, Any], name: str | None = None) -> RerankProfile:
    rcfg = cfg["rerank"]
    name = name or rcfg["profile"]
    profiles = rcfg.get("profiles", {})
    if name not in profiles:
        raise ValueError(f"reranker profile {name!r} not in the config (have: {', '.join(profiles) or 'none'})")
    p = profiles[name]
    fmt = p.get("input_format", "pair")
    if fmt not in ENCODERS:
        raise ValueError(f"profile {name!r}: unknown input_format {fmt!r} (have: {', '.join(ENCODERS)})")
    if fmt == "template" and not ("{query}" in p.get("template", "") and "{document}" in p.get("template", "")):
        raise ValueError(f"profile {name!r}: input_format 'template' needs a template with {{query}} and {{document}}")
    return RerankProfile(name=name, model=p["model"], tokenizer=p.get("tokenizer") or p["model"], input_format=fmt, max_tokens=int(p["max_tokens"]),
                   score_index=int(p.get("score_index", 0)), threshold=float(p["threshold"]), query=p["query"],
                   document_header=p.get("document_header", ""), template=p.get("template", ""),
                   add_special_tokens=bool(p.get("add_special_tokens", True)))


# ---------------------------------------------------------------- input formats

def encode_pair(tok: Any, profile: RerankProfile, query: str, document: str) -> list[int]:
    return tok(query, document)["input_ids"]


def encode_template(tok: Any, profile: RerankProfile, query: str, document: str) -> list[int]:
    text = profile.template.replace("{query}", query).replace("{document}", document)
    return tok(text, add_special_tokens=profile.add_special_tokens)["input_ids"]


ENCODERS: dict[str, Callable[[Any, RerankProfile, str, str], list[int]]] = {"pair": encode_pair, "template": encode_template}


# ---------------------------------------------------------------- requests (with the granite profile: same as the validated pipeline)

def page_requests(tok: Any, profile: RerankProfile, query: str, title: str, url: str, content: str) -> list[list[int]]:
    """Token-id inputs for one page: the whole page if it fits in max_tokens with the query, else consecutive chunks that each fit."""
    encode = ENCODERS[profile.input_format]
    limit = profile.max_tokens
    head = profile.document_header.replace("{title}", title).replace("{url}", url)
    ids = encode(tok, profile, query, head + content)
    if len(ids) <= limit:
        return [ids]
    body = tok(content, add_special_tokens=False)["input_ids"]
    budget = limit - len(encode(tok, profile, query, head)) - 16
    if budget < 64:
        raise ValueError(f"query and header leave only {budget} tokens for the page")
    out, s = [], 0
    while s < len(body):
        n = budget
        while True:
            ids = encode(tok, profile, query, head + tok.decode(body[s:s + n]))
            if len(ids) <= limit:
                break
            n -= len(ids) - limit           # re-tokenizing decoded text can add tokens: shrink until it fits
            if n <= 0:
                raise ValueError("a chunk can't be made to fit the token limit")
        out.append(ids)
        s += n
    return out


def load_tokenizer(model: str) -> Callable[[], Any]:
    """One tokenizer per thread (fast tokenizers are not safe to share across threads)."""
    from transformers import AutoTokenizer
    local = threading.local()

    def get() -> Any:
        if not hasattr(local, "t"):
            local.t = AutoTokenizer.from_pretrained(model)
            local.t.model_max_length = 10**9   # the default (1,124) only triggers a warning; requests are checked against max_tokens
        return local.t
    return get


# ---------------------------------------------------------------- servers

class Servers:
    def __init__(self, urls: list[str], model: str, score_index: int, timeout_s: float, post: Callable[..., Any], get: Callable[..., Any]):
        if not urls:
            raise ValueError("[rerank].servers is empty")
        self.urls, self.model, self.timeout_s, self.post, self.get = [u.rstrip("/") for u in urls], model, timeout_s, post, get
        self.score_index = score_index
        self._rr = itertools.count()
        self._lock = threading.Lock()

    def check(self) -> None:
        """Every server must answer and serve the configured model."""
        bad = []
        for u in self.urls:
            try:
                r = self.get(f"{u}/v1/models", timeout=10)
                ids = [m.get("id") for m in r.json().get("data", [])]
                if r.status_code != 200 or self.model not in ids:
                    bad.append(f"{u} serves {ids or 'nothing'} (HTTP {r.status_code})")
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{u} unreachable ({type(exc).__name__})")
        if bad:
            raise RunStopped("reranker servers not ready: " + "; ".join(bad) + " (start them with python -m on_the_fly_pipeline serve rerank)")

    def next_index(self) -> int:
        with self._lock:
            return next(self._rr) % len(self.urls)

    def classify(self, inputs: list[list[int]], server_index: int) -> list[float]:
        """probs[score_index] of each input, in input order. Raises CompanyError with a category."""
        url = f"{self.urls[server_index]}/classify"
        try:
            r = self.post(url, json={"model": self.model, "input": inputs}, timeout=self.timeout_s)
        except requests.exceptions.Timeout as exc:
            raise CompanyError("timeout", f"{url}: {type(exc).__name__}") from exc
        except requests.exceptions.ConnectionError as exc:
            raise CompanyError("server_unreachable", f"{url}: {type(exc).__name__}") from exc
        if r.status_code >= 500:
            raise CompanyError("server_error", f"{url}: HTTP {r.status_code} {r.text[:200]}")
        if r.status_code >= 400:
            raise CompanyError("bad_request", f"{url}: HTTP {r.status_code} {r.text[:300]}")
        try:
            data = r.json()["data"]
            scores = [None] * len(inputs)
            for d in data:
                scores[d["index"]] = float(d["probs"][self.score_index])
        except Exception as exc:  # noqa: BLE001
            raise CompanyError("bad_response", f"{url}: unreadable answer ({type(exc).__name__}: {exc})"[:300]) from exc
        if len(data) != len(inputs) or any(s is None for s in scores):
            raise CompanyError("bad_response", f"{url}: {len(data)} scores for {len(inputs)} inputs")
        return scores


# ---------------------------------------------------------------- one company

def fingerprint(profile: RerankProfile, pages: list[dict[str, Any]], name_filter: bool, sites: SiteFilter | None = None,
                host_rule: str = "first_word") -> str:
    settings = {k: v for k, v in asdict(profile).items() if k != "threshold"}     # the threshold does not change the scores
    settings["name_filter"] = name_filter
    if host_rule != "first_word":
        settings["host_rule"] = host_rule
    if sites is not None and sites.enabled:
        settings["foreign_sites"] = sites.settings()
    key = [settings, [(p["result_id"], p.get("fetched_at"), p.get("content_chars")) for p in pages]]
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def load_pages(out_root: Path, company: Company) -> list[dict[str, Any]]:
    fp = out_root / company.slug / "fetch.json"
    if not fp.is_file():
        raise CompanyError("missing_fetch", f"{fp} not found: run fetch first")
    index = json.loads(fp.read_text(encoding="utf-8"))
    return [json.loads((out_root / company.slug / "pages" / f"{row['result_id']}.json").read_text(encoding="utf-8")) for row in index["pages"]]


def rerank_company(company: Company, pages: list[dict[str, Any]], rcfg: dict[str, Any], profile: RerankProfile, servers: Servers,
                   tok: Callable[[], Any], sites: SiteFilter | None = None, host_rule: str = "first_word") -> dict[str, Any]:
    t0 = time.monotonic()
    query = profile.query.format(name=company.name, province=company.province, country=rcfg["country"])
    limit = profile.max_tokens
    inputs: list[list[int]] = []
    owner: list[int] = []            # page row of each input
    chunk_no: list[int] = []         # chunk number of each input within its page
    rows, page_errors, skipped, no_name, foreign = [], [], [], [], []
    matcher = NameMatcher(company.name, host_rule) if rcfg.get("name_filter", True) else None
    for p in pages:
        if not p.get("content"):
            skipped.append(p["result_id"])
            continue
        if sites is not None and sites.is_foreign(p):
            foreign.append(p["result_id"])
            continue
        if matcher and not matcher.match(p.get("title") or "", p["content"], p["url"]):
            no_name.append(p["result_id"])
            continue
        try:
            reqs = page_requests(tok(), profile, query, p.get("title") or "", p["url"], p["content"])
        except ValueError as exc:
            page_errors.append({"result_id": p["result_id"], "url": p["url"], "message": str(exc)})
            continue
        assert all(len(r) <= limit for r in reqs), "request over the token limit"
        rows.append({"result_id": p["result_id"], "url": p["url"], "content_source": p.get("content_source"), "score": None,
                     "best_chunk": None, "chunks": len(reqs), "tokens": sum(map(len, reqs))})
        owner += [len(rows) - 1] * len(reqs)
        chunk_no += list(range(len(reqs)))
        inputs += reqs
    t_tok = time.monotonic()

    scores: list[float] = []
    attempts = 0
    if inputs:
        first, step = servers.next_index(), int(rcfg["max_inputs_per_post"])
        for attempt in range(int(rcfg["max_attempts"])):
            attempts = attempt + 1
            try:
                scores = []
                for s in range(0, len(inputs), step):
                    scores += servers.classify(inputs[s:s + step], (first + attempt) % len(servers.urls))
                break
            except CompanyError as exc:
                if exc.category not in RETRYABLE or attempts == int(rcfg["max_attempts"]):
                    raise
                log.warning("%s: %s, retrying (%s)", company.slug, exc.category, exc.message)
                time.sleep(float(rcfg["retry_wait_s"]) * attempts)
    for i, sc in enumerate(scores):
        row = rows[owner[i]]
        if row["score"] is None or sc > row["score"]:
            row["score"], row["best_chunk"] = round(sc, 6), chunk_no[i]
    return {"status": "ok", "pages": rows, "skipped_no_content": skipped, "skipped_no_name": no_name, "skipped_foreign": foreign, "page_errors": page_errors, "inputs": len(inputs),
            "tokens": sum(map(len, inputs)), "attempts": attempts, "tokenize_s": round(t_tok - t0, 3), "score_s": round(time.monotonic() - t_tok, 3)}


# ---------------------------------------------------------------- run

class Reranker:
    """Reranks one company at a time (rerank_one); many threads may call it at once (the streaming pipeline does).
    Shared by all companies: the servers, the profile, the per-thread tokenizers, the run counters, and the stop switch
    (server_failure_limit companies in a row that can't reach the servers stop the run)."""

    def __init__(self, cfg: dict[str, Any], out_root: Path, tokenizer: Callable[[], Any] | None = None, post: Callable[..., Any] = requests.post,
                 get: Callable[..., Any] = requests.get, profile_name: str | None = None):
        self.rcfg, self.out_root = cfg["rerank"], out_root
        self.profile = load_rerank_profile(cfg, profile_name)
        self.servers = Servers(list(self.rcfg["servers"]), self.profile.model, self.profile.score_index, float(self.rcfg["timeout_s"]), post, get)
        self._tokenizer = tokenizer
        self.tok: Callable[[], Any] | None = None
        self.name_filter = bool(self.rcfg.get("name_filter", True))
        self.sites = SiteFilter.from_config(cfg)
        self.host_rule = cfg.get("select", {}).get("host_rule", "first_word")
        self._lock = threading.Lock()
        self._down_streak = 0
        self.stopped_reason = ""
        self.s: dict[str, Any] = {"ok": 0, "reused": 0, "failed": [], "not_run": [], "errors": Counter(), "error_examples": [], "pages_scored": 0,
                                  "pages_without_name": 0, "pages_foreign": 0, "page_errors": 0, "inputs": 0, "tokens": 0, "tokenize_s": 0.0, "score_s": 0.0}

    def check(self) -> None:
        """Raises RunStopped if the servers are not ready; loads the tokenizer."""
        try:
            self.servers.check()
        except RunStopped as exc:
            self.stopped_reason = str(exc)
            raise
        self.tok = self._tokenizer or load_tokenizer(self.profile.tokenizer)

    def rerank_one(self, c: Company) -> str:
        """Score one company's pages and write rerank.json. Returns ok | reused | error | not_run."""
        if self.stopped_reason:
            with self._lock:
                self.s["not_run"].append(c.slug)
            return "not_run"
        path = self.out_root / c.slug / "rerank.json"
        fp = ""
        try:
            pages = load_pages(self.out_root, c)
            fp = fingerprint(self.profile, pages, self.name_filter, self.sites, self.host_rule)
            old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if old.get("status") == "ok" and old.get("fingerprint") == fp:
                with self._lock:
                    self.s["reused"] += 1; self.s["ok"] += 1
                return "reused"
            res = rerank_company(c, pages, self.rcfg, self.profile, self.servers, self.tok, self.sites, self.host_rule)
        except CompanyError as exc:
            res = {"status": "error", "error": exc.category, "error_message": exc.message}
        write_json(path, {"company": c.__dict__, "profile": self.profile.name, "model": self.profile.model, "threshold": self.profile.threshold,
                          "max_tokens": self.profile.max_tokens, "fingerprint": fp, **res, "written_at": utc_now()})
        with self._lock:
            s = self.s
            if res["status"] == "ok":
                self._down_streak = 0
                s["ok"] += 1
                s["pages_scored"] += len(res["pages"]); s["page_errors"] += len(res["page_errors"])
                s["pages_without_name"] += len(res["skipped_no_name"])
                s["pages_foreign"] += len(res.get("skipped_foreign", []))
                for k in ("inputs", "tokens", "tokenize_s", "score_s"):
                    s[k] += res[k]
                for pe in res["page_errors"]:
                    s["errors"]["page_error"] += 1
                    if len(s["error_examples"]) < 20:
                        s["error_examples"].append({"company": c.slug, "error": "page_error", **pe})
                return "ok"
            s["failed"].append(c.slug)
            s["errors"][res["error"]] += 1
            if len(s["error_examples"]) < 20:
                s["error_examples"].append({"company": c.slug, "error": res["error"], "message": res["error_message"]})
            self._down_streak = self._down_streak + 1 if res["error"] in SERVER_DOWN else 0
            if self._down_streak >= int(self.rcfg["server_failure_limit"]) and not self.stopped_reason:
                self.stopped_reason = (f"{self._down_streak} companies in a row could not reach the reranker servers "
                                       f"(last: {res['error_message']})")
            return "error"

    def summary(self, n_companies: int, wall_s: float, not_run: list[str] | None = None) -> dict[str, Any]:
        """Run summary (also written to rerank_summary.json)."""
        with self._lock:
            s = {"profile": self.profile.name, "model": self.profile.model, "companies": n_companies, **{k: v for k, v in self.s.items()}}
            s["failed"], s["not_run"] = list(s["failed"]), list(s["not_run"]) + list(not_run or [])
        s["errors"] = dict(s["errors"])
        s["stopped_reason"] = self.stopped_reason
        s["tokenize_s"], s["score_s"] = round(s["tokenize_s"], 2), round(s["score_s"], 2)
        s["wall_s"] = round(wall_s, 2)
        s["written_at"] = utc_now()
        write_json(self.out_root / "rerank_summary.json", s)
        return s
