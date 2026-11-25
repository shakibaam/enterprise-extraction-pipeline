"""Fetch component: download the text of every URL that search found for a company.

Input:  OUT/<slug>/search.json (written by search.py).
Output: OUT/<slug>/pages/<result_id>.json   one file per URL (text, source URL, status, error category)
        OUT/<slug>/fetch.json                 per-company list of pages and counts
        OUT/fetch_summary.json                run summary: page counts, error counts with examples, hosts skipped

Text extraction is the same as in the earlier runs (enterprise_agent/runners/run_serpapi_hylife_top100.fetch_page_text,
used for the BC80 pages the reranker threshold was tuned on): HTML and text only, script/style/noscript removed, one line per
block element, blank lines dropped, first 1.5 MB of the body, pages under 80 characters count as failed.
PDFs (content type application/pdf, or a .pdf URL served as a generic binary, or a body starting with %PDF-): with [fetch].parse_pdfs =
false (the default since 2026-11-13, as in the earlier runs) they are not downloaded or parsed and their search snippets are kept
(unsupported_type). Parsing all PDFs made the BC80 end-to-end run 4.8 times slower in extraction: 14 kept PDFs (public lists that name the
company) held 71% of the kept text (enterprise_variant_d/runs/company_pipeline_e2e_bc80_20261113). With parse_pdfs = true they are downloaded up to
max_pdf_bytes and parsed with pypdfium2 in a separate process (pipeline/pdf_text.py; at most pdf_workers at once, pdf_timeout_s each),
so a PDF that hangs or crashes the parser can't stop the fetcher. Their text is cleaned like HTML text (blank lines dropped) and the
same 80-character minimum applies (a scanned PDF without a text layer counts as too_short).
When a page can't be used, its search snippets are kept as the content (content_source = "snippet"), so the URL is not lost.

Speed: all URLs of all companies share one pool of `workers` threads. Each website gets at most `per_host_concurrency`
requests at a time, spaced by `per_host_min_interval_s`. A website that times out or refuses connections
`host_failure_limit` times in a row is skipped for the rest of the run (error `host_down`), so one dead website does not
hold workers for 30 s per URL.

Errors (fetch_error on each page; counted in fetch_summary.json):
  retried (up to max_attempts): timeout, connection_error, rate_limited (429), server_error (5xx)
  not retried: blocked (401/403/451), not_found (404/410), client_error (other 4xx), ssl_error, too_many_redirects,
               unsupported_type (images, Word/Excel files, XML sitemaps and feeds, other non-text non-PDF files), too_short
               (< 80 chars of text),
               pdf_too_large (over max_pdf_bytes; a cut PDF can't be read), pdf_error (the parser failed), pdf_timeout, bad_url,
               host_down (skipped, see above), other (unexpected error; message kept)
Rerunning the same command reuses saved pages and refetches only pages that ended with a retried error or host_down.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import requests

from ..shared.config import resolve_path
from ..shared.models import Company
from ..shared.pdf_text import PdfError, pdf_to_text
from ..storage import utc_now, write_json

log = logging.getLogger(__name__)

RETRYABLE = {"timeout", "connection_error", "rate_limited", "server_error"}
REFETCH_ON_RESUME = RETRYABLE | {"host_down", "internal_error"}
HOST_FAILURES = {"timeout", "connection_error"}          # count towards marking a website as down
SNIPPET_SEPARATOR = "\n\n---\n\n"


# ---------------------------------------------------------------- HTTP layer

@dataclass
class RawResponse:
    status: int
    content_type: str
    body: bytes = b""            # empty when the content type is neither text nor PDF (not downloaded)
    encoding: str | None = None
    final_url: str = ""
    truncated: bool = False
    retry_after: str | None = None


class DeadlineExceeded(Exception):
    """The whole download took longer than max_seconds_per_page."""


def is_text_type(content_type: str) -> bool:
    ct = content_type.lower()
    return "html" in ct or ("text" in ct and "xml" not in ct)


def is_xml(content_type: str, url: str, body: bytes = b"") -> bool:
    """XML that is not a web page: sitemaps, feeds, data files. They hold links or data, no readable content (two BBB sitemaps of
    ~320k tokens each, served as text/xml, made 43 of one company's 45 LLM batches). By content type (xml but not xhtml), by a .xml
    address (also sitemap*.xml.gz), or by a body that starts as XML and has no <html."""
    ct = content_type.lower()
    if "xml" in ct and "html" not in ct:
        return True
    path = urlsplit(url).path.lower()
    if path.endswith((".xml", ".xml.gz")):
        return True
    head = body[:2048].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    return head.startswith(b"<?xml") and b"<html" not in head


GENERIC_BINARY = ("application/octet-stream", "binary/octet-stream", "application/x-download", "application/download", "application/force-download")


def is_pdf_type(content_type: str, url: str) -> bool:
    """A PDF by its content type, or a .pdf URL served with a generic binary (or no) content type."""
    ct = content_type.lower()
    if "pdf" in ct:
        return True
    return urlsplit(url).path.lower().endswith(".pdf") and (not ct.strip() or any(g in ct for g in GENERIC_BINARY))


def is_pdf_bytes(body: bytes) -> bool:
    return body[:1024].lstrip().startswith(b"%PDF-")


_sessions = threading.local()


def requests_get(url: str, *, fcfg: dict[str, Any]) -> RawResponse:
    """GET with streaming. HTML/text: up to max_bytes within max_seconds_per_page. PDF: up to max_pdf_bytes within pdf_max_seconds.
    Other files: the body is not downloaded."""
    s = getattr(_sessions, "s", None)
    if s is None:
        s = _sessions.s = requests.Session()
        s.max_redirects = int(fcfg["max_redirects"])
        s.headers["User-Agent"] = fcfg["user_agent"]
    t0 = time.monotonic()
    with s.get(url, stream=True, allow_redirects=True, timeout=(float(fcfg["connect_timeout_s"]), float(fcfg["read_timeout_s"]))) as r:
        ct = r.headers.get("Content-Type", "")
        out = RawResponse(status=r.status_code, content_type=ct, encoding=r.encoding, final_url=r.url,
                          retry_after=r.headers.get("Retry-After"))
        pdf = is_pdf_type(ct, r.url)
        if r.status_code >= 400 or is_xml(ct, r.url) or not (pdf or is_text_type(ct)) or (pdf and not fcfg.get("parse_pdfs", False)):
            return out
        limit = int(fcfg["max_pdf_bytes"] if pdf else fcfg["max_bytes"])
        deadline = float(fcfg["pdf_max_seconds"] if pdf else fcfg["max_seconds_per_page"])
        chunks, size = [], 0
        for chunk in r.iter_content(65536):
            chunks.append(chunk)
            size += len(chunk)
            if size >= limit:
                out.truncated = True
                break
            if time.monotonic() - t0 > deadline:
                raise DeadlineExceeded(f"download took more than {deadline} s")
        out.body = b"".join(chunks)[:limit]
        return out


def classify_exception(exc: Exception) -> str:
    # Order matters: SSLError and ConnectTimeout are also ConnectionErrors.
    if isinstance(exc, requests.exceptions.SSLError):
        return "ssl_error"
    if isinstance(exc, (requests.exceptions.Timeout, DeadlineExceeded)):
        return "timeout"
    if isinstance(exc, requests.exceptions.TooManyRedirects):
        return "too_many_redirects"
    if isinstance(exc, (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema, requests.exceptions.InvalidSchema)):
        return "bad_url"
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError)):
        return "connection_error"
    return "other"


def classify_status(status: int) -> str:
    if status < 400:
        return ""
    if status in (401, 403, 451):
        return "blocked"
    if status in (404, 410):
        return "not_found"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server_error"
    return "client_error"


# ---------------------------------------------------------------- text extraction (same as the earlier runs)

class _TextExtractor(HTMLParser):
    """table_cells ([fetch].table_cells, change C, off by default): table cells joined with " | " and each row on its own line (block
    tags inside a cell add a space, not a line break); off = the validated conversion, which joins cells with no separator."""

    def __init__(self, table_cells: bool = False) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.table_cells = table_cells
        self.cell = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip += 1
        if self.table_cells and tag in {"td", "th"}:
            self.cell += 1
        elif self.table_cells and tag == "tr":
            self.parts.append("\n")
            self.cell = 0

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip:
            self.skip -= 1
        if self.table_cells and tag in {"td", "th"}:
            self.parts.append(CELL_SEP)
            self.cell = max(0, self.cell - 1)
        elif self.table_cells and self.cell and tag in {"p", "div", "br", "li", "h1", "h2", "h3"}:
            self.parts.append(" ")
        elif tag in {"p", "div", "br", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip and data.strip():
            self.parts.append(data)


CELL_SEP = " | "


def _clean_line(line: str, table_cells: bool) -> str:
    line = line.strip()
    if table_cells and "|" in line:                                   # drop empty cells and the separators at the row's ends
        line = CELL_SEP.join(c.strip() for c in line.split("|") if c.strip())
    return line


def html_to_text(body: bytes, encoding: str | None, table_cells: bool = False) -> str:
    parser = _TextExtractor(table_cells)
    parser.feed(body.decode(encoding or "utf-8", errors="replace"))
    lines = (_clean_line(line, table_cells) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def merged_snippets(hit: dict[str, Any]) -> str:
    """Distinct snippets of every query that returned the URL, best rank first."""
    found = sorted(hit.get("found_by") or [], key=lambda f: f.get("rank", 0))
    texts = [f.get("snippet") or "" for f in found]
    if not any(texts):                       # search.json written before found_by carried snippets
        texts = [hit.get("snippet") or ""]
    seen: list[str] = []
    for t in (t.strip() for t in texts):
        if t and t not in seen:
            seen.append(t)
    return SNIPPET_SEPARATOR.join(seen)


# ---------------------------------------------------------------- per-website limits

class HostGate:
    """Per website: at most `concurrency` requests at once, `min_interval_s` between request starts, and a switch that
    marks the website as down after `failure_limit` timeouts/connection errors in a row."""

    def __init__(self, concurrency: int, min_interval_s: float, failure_limit: int):
        self.concurrency, self.min_interval_s, self.failure_limit = concurrency, min_interval_s, failure_limit
        self._lock = threading.Lock()
        self._sem: dict[str, threading.BoundedSemaphore] = {}
        self._next: dict[str, float] = {}
        self._fails: Counter[str] = Counter()
        self.down: set[str] = set()

    def acquire(self, host: str) -> None:
        with self._lock:
            sem = self._sem.setdefault(host, threading.BoundedSemaphore(self.concurrency))
        sem.acquire()
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next.get(host, now))
            self._next[host] = start + self.min_interval_s
        if start > now:
            time.sleep(start - now)

    def release(self, host: str) -> None:
        self._sem[host].release()

    def is_down(self, host: str) -> bool:
        with self._lock:
            return host in self.down

    def record(self, host: str, error: str) -> None:
        with self._lock:
            if error in HOST_FAILURES:
                self._fails[host] += 1
                if self._fails[host] >= self.failure_limit and host not in self.down:
                    self.down.add(host)
                    log.warning("website %s marked down after %d timeouts/connection errors in a row", host, self._fails[host])
            else:
                self._fails[host] = 0


# ---------------------------------------------------------------- one page

def fetch_page(hit: dict[str, Any], fcfg: dict[str, Any], gate: HostGate, get: Callable[..., RawResponse],
               parse_pdf: Callable[[bytes], dict[str, Any]]) -> dict[str, Any]:
    """Fetch one URL with retries and build its page record (snippet fallback when the page can't be used)."""
    url = hit["url"]
    host = (urlsplit(url).hostname or "").lower()
    t0 = time.monotonic()
    error, message, status, resp, attempts, history = "", "", None, None, 0, []
    if urlsplit(url).scheme not in ("http", "https") or not host:
        error, message = "bad_url", "not an http(s) URL"
    else:
        for attempt in range(1, int(fcfg["max_attempts"]) + 1):
            if gate.is_down(host):
                error, message = "host_down", f"skipped: {host} timed out or refused connections repeatedly in this run"
                break
            attempts = attempt
            gate.acquire(host)
            try:
                resp = get(url, fcfg=fcfg)
                status = resp.status
                error = classify_status(resp.status)
                message = f"HTTP {resp.status}" if error else ""
            except Exception as exc:  # noqa: BLE001 - every failure is categorized and reported
                resp, status, error, message = None, None, classify_exception(exc), f"{type(exc).__name__}: {exc}"[:300]
            finally:
                gate.release(host)
            gate.record(host, error)
            history.append({"attempt": attempt, "http_status": status, "error": error})
            if error not in RETRYABLE or attempt == int(fcfg["max_attempts"]):
                break
            wait = 1.5 * attempt
            if resp is not None and resp.retry_after and resp.retry_after.isdigit():
                wait = min(float(resp.retry_after), float(fcfg["max_retry_wait_s"]))
            time.sleep(wait)

    text, doc_type, pdf_pages = "", "", None
    if not error and resp is not None:
        is_pdf = is_pdf_bytes(resp.body) or is_pdf_type(resp.content_type, resp.final_url or url)
        if is_pdf and not fcfg.get("parse_pdfs", False):
            doc_type = "pdf"
            error, message = "unsupported_type", "PDF not parsed ([fetch].parse_pdfs = false)"
        elif is_pdf and resp.body:
            doc_type = "pdf"
            if resp.truncated:
                error, message = "pdf_too_large", f"over {fcfg['max_pdf_bytes']} bytes"
            elif not is_pdf_bytes(resp.body):
                error, message = "pdf_error", "served as a PDF but the body is not a PDF"
            else:
                try:
                    parsed = parse_pdf(resp.body)
                    text, pdf_pages = parsed["text"], {"pages": parsed["pages"], "pages_read": parsed["pages_read"]}
                except PdfError as exc:
                    error, message = exc.category, exc.message
        elif is_xml(resp.content_type, resp.final_url or url, resp.body):
            error, message = "unsupported_type", f"XML file, not a web page (sitemap, feed or data): content-type {resp.content_type or '(none)'}"
        elif is_text_type(resp.content_type):
            doc_type = "html"
            text = html_to_text(resp.body, resp.encoding, bool(fcfg.get("table_cells", False)))
        else:
            error, message = "unsupported_type", f"content-type {resp.content_type or '(none)'}"
        if not error and len(text) < int(fcfg["min_text_chars"]):
            error, message, text = "too_short", f"{len(text)} chars of text", ""

    if text:
        content, source = text, "page"
    elif fcfg["snippet_fallback"]:
        content = merged_snippets(hit)
        source = "snippet" if content else "none"
    else:
        content, source = "", "none"
    return {
        "result_id": hit["result_id"], "url": url, "canonical_url": hit["canonical_url"],
        "final_url": resp.final_url if resp is not None else "", "title": hit.get("title") or "",
        "best_rank": hit.get("best_rank"), "found_by": hit.get("found_by") or [],
        "fetch_ok": not error, "fetch_error": error, "fetch_error_message": message, "http_status": status,
        "content_type": resp.content_type if resp is not None else "", "document_type": doc_type, "pdf_pages": pdf_pages,
        "truncated": bool(resp and resp.truncated),
        "content": content, "content_source": source, "content_chars": len(content),
        "attempts": attempts, "attempt_history": history, "latency_s": round(time.monotonic() - t0, 3), "fetched_at": utc_now(),
    }


# ---------------------------------------------------------------- run

def page_path(out_root: Path, company: Company, result_id: str) -> Path:
    return out_root / company.slug / "pages" / f"{result_id}.json"


def load_saved_page(path: Path, hit: dict[str, Any]) -> dict[str, Any] | None:
    """A saved page for this exact URL that does not need refetching, or None."""
    if not path.is_file():
        return None
    try:
        page = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if page.get("url") != hit["url"] or page.get("fetch_error") in REFETCH_ON_RESUME:
        return None
    return page


def page_index_row(p: dict[str, Any]) -> dict[str, Any]:
    return {k: p[k] for k in ("result_id", "url", "fetch_ok", "fetch_error", "content_source", "content_chars", "latency_s")}


def company_summary(out_root: Path, company: Company, search_status: str, pages: list[dict[str, Any]]) -> dict[str, Any]:
    pages = sorted(pages, key=lambda p: p["result_id"])
    counts = {"urls": len(pages), "page": sum(p["content_source"] == "page" for p in pages),
              "snippet": sum(p["content_source"] == "snippet" for p in pages), "none": sum(p["content_source"] == "none" for p in pages)}
    write_json(out_root / company.slug / "fetch.json", {"company": company.__dict__, "search_status": search_status, "counts": counts,
                                                        "pages": [page_index_row(p) for p in pages], "written_at": utc_now()})
    return counts


def make_pdf_parser(fcfg: dict[str, Any]) -> Callable[[bytes], dict[str, Any]]:
    """pdf_to_text in a child process, at most pdf_workers at once (parsing is CPU-bound). The child runs with [fetch].pdf_python (the
    environment that has pypdfium2), so the fetcher itself can run in another environment (the streaming pipeline runs in the vLLM one)."""
    slots = threading.BoundedSemaphore(int(fcfg["pdf_workers"]))
    python = str(resolve_path(fcfg["pdf_python"])) if fcfg.get("pdf_python") else None

    def parse(data: bytes) -> dict[str, Any]:
        with slots:
            return pdf_to_text(data, float(fcfg["pdf_timeout_s"]), int(fcfg["pdf_max_pages"]), python=python)
    return parse


class Fetcher:
    """Fetches one company at a time (fetch_company); many threads may call it at once (the streaming pipeline does).
    Shared by all companies: one pool of page workers, the per-website limits (HostGate) and the PDF parser slots."""

    def __init__(self, cfg: dict[str, Any], out_root: Path, get: Callable[..., RawResponse] = requests_get,
                 parse_pdf: Callable[[bytes], dict[str, Any]] | None = None):
        self.fcfg, self.out_root, self.get = cfg["fetch"], out_root, get
        self.parse_pdf = parse_pdf or make_pdf_parser(self.fcfg)
        self.gate = HostGate(int(self.fcfg["per_host_concurrency"]), float(self.fcfg["per_host_min_interval_s"]), int(self.fcfg["host_failure_limit"]))
        self.pool = ThreadPoolExecutor(max_workers=int(self.fcfg["workers"]))
        self._lock = threading.Lock()
        self.errors: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []
        self.per_company: dict[str, Any] = {}
        self.missing_search: list[str] = []
        self.reused = self.fetched_now = self.pdfs = 0

    def _fetch(self, hit: dict[str, Any], path: Path) -> dict[str, Any]:
        try:
            page = fetch_page(hit, self.fcfg, self.gate, self.get, self.parse_pdf)
        except Exception as exc:  # noqa: BLE001 - a bug in our code must not lose the other pages
            log.exception("internal error on %s", hit["url"])
            page = {"result_id": hit["result_id"], "url": hit["url"], "canonical_url": hit["canonical_url"], "title": hit.get("title") or "",
                    "fetch_ok": False, "fetch_error": "internal_error", "fetch_error_message": f"{type(exc).__name__}: {exc}"[:300],
                    "content": "", "content_source": "none", "content_chars": 0, "latency_s": 0.0}
        write_json(path, page)
        return page

    def fetch_company(self, c: Company) -> dict[str, Any]:
        """Fetch every URL in the company's search.json (pages in parallel), write fetch.json; returns its counts, or
        {"status": "missing_search"}."""
        sp = self.out_root / c.slug / "search.json"
        if not sp.is_file():
            with self._lock:
                self.missing_search.append(c.slug)
            return {"status": "missing_search"}
        search = json.loads(sp.read_text(encoding="utf-8"))
        pages, futures, reused = [], [], 0
        for hit in search.get("hits") or []:
            path = page_path(self.out_root, c, hit["result_id"])
            saved = load_saved_page(path, hit)
            if saved is not None:
                pages.append(saved)
                reused += 1
            else:
                futures.append(self.pool.submit(self._fetch, hit, path))
        pages += [f.result() for f in futures]
        counts = company_summary(self.out_root, c, search.get("status", ""), pages)
        with self._lock:
            self.reused += reused
            self.fetched_now += len(futures)
            self.pdfs += sum(1 for p in pages if p.get("document_type") == "pdf" and p["content_source"] == "page")
            for p in pages:
                if p["fetch_error"]:
                    self.errors[p["fetch_error"]] += 1
                    if len(self.examples) < 30:
                        self.examples.append({"company": c.slug, "url": p["url"], "error": p["fetch_error"], "message": p.get("fetch_error_message", "")})
            self.per_company[c.slug] = counts
        return {"status": "ok", **counts}

    def summary(self, n_companies: int, wall_s: float) -> dict[str, Any]:
        """Run summary (also written to fetch_summary.json)."""
        with self._lock:
            totals: Counter = Counter()
            for counts in self.per_company.values():
                totals.update(counts)
            summary = {
                "companies": n_companies, "companies_fetched": len(self.per_company), "missing_search": list(self.missing_search),
                "urls": totals["urls"], "content_page": totals["page"], "content_snippet": totals["snippet"], "content_none": totals["none"],
                "pdfs_parsed": self.pdfs, "pages_reused": self.reused, "pages_fetched_now": self.fetched_now, "errors": dict(self.errors),
                "error_examples": list(self.examples), "hosts_marked_down": sorted(self.gate.down), "wall_s": round(wall_s, 2),
                "per_company": dict(self.per_company), "written_at": utc_now()}
        write_json(self.out_root / "fetch_summary.json", summary)
        return summary

    def close(self) -> None:
        self.pool.shutdown(wait=True)
