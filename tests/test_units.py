"""Unit tests (free, no network). Run from Provinces/: .venv/bin/python -m pytest -q on_the_fly_pipeline/tests/test_units.py"""
from __future__ import annotations

import json
import threading
import time

import pytest

from on_the_fly_pipeline.shared.llm import BudgetExceeded, ChatClient, PaidBudget, load_profile
from on_the_fly_pipeline.steps import extract as X
from on_the_fly_pipeline.steps import verify as V
from on_the_fly_pipeline.steps.finalize import confirmed_only

SCHEMA_RF = {"type": "json_schema", "json_schema": {"name": "entities", "strict": True, "schema": X.SCHEMA}}
ROUTING = {"allow_fallbacks": True, "require_parameters": True}


class _NoKeyClient(ChatClient):
    """Builds bodies without loading a key (no network)."""
    def __init__(self, profile):
        self.profile = profile


# ---------------------------------------------------------------- request bodies = the E6-E14 runners'
@pytest.mark.parametrize("name,model,reasoning", [
    ("haiku_medium", "anthropic/claude-haiku-5.5", {"effort": "medium"}),           # E6 run_stream.haiku()
    ("haiku_off", "anthropic/claude-haiku-5.5", {"enabled": False}),                # E7
    ("haiku_low", "anthropic/claude-haiku-5.5", {"effort": "low"}),                 # E8
    ("gemini_flash_lite", "google/gemini-3.1-flash-lite", {"effort": "medium"}),    # E10
    ("gpt54_nano", "openai/gpt-5.4-nano", {"effort": "medium"}),                    # E11
    ("gpt6_luna", "openai/gpt-6-luna", {"effort": "medium"}),                       # E12
    ("terra_none", "openai/gpt-6.1-terra", {"effort": "none"}),                     # E13
    ("sonnet_low", "anthropic/claude-sonnet-5.5", {"effort": "low"}),               # E14
])
def test_openrouter_extractor_body(name, model, reasoning):
    c = _NoKeyClient(load_profile("extractors", name))
    content = [{"type": "text", "text": "P", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "rest"}]
    want = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 16000, "reasoning": reasoning,
            "usage": {"include": True}, "provider": ROUTING, "response_format": SCHEMA_RF}
    got = c.body(content, "entities", X.SCHEMA)
    assert json.dumps(got) == json.dumps(want)                  # same fields, same order
    assert c.profile.prompt_cache


def test_local_qwen_body():                                     # E9 run_stream.qwen35b()
    c = _NoKeyClient(load_profile("extractors", "qwen9b_local"))
    want = {"model": "Qwen/Qwen3.5-9B", "messages": [{"role": "user", "content": "m"}], "temperature": 0, "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": False}, "response_format": SCHEMA_RF}
    assert json.dumps(c.body("m", "entities", X.SCHEMA)) == json.dumps(want)
    assert c.profile.url == "http://localhost:8010/v1/chat/completions" and c.profile.length_retry_max_tokens == 12000


def test_verifier_body_and_effort():                            # E6 vfy_nw._send with VCFG (Terra, effort none)
    c = _NoKeyClient(load_profile("verifiers", "terra_none"))
    want = {"model": "openai/gpt-6.1-terra", "messages": [{"role": "user", "content": "m"}], "max_tokens": 16000,
            "reasoning": {"effort": "none"}, "usage": {"include": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": V.SCHEMA}}}
    assert json.dumps(c.body("m", "verdict", V.SCHEMA)) == json.dumps(want)
    assert c.profile.effort == "none"


# ---------------------------------------------------------------- page number -> exact URL
def test_items_map_page_number_to_url():
    res = {"ok": True, "text": json.dumps({"e": [
        {"n": "Jane Doe", "t": "p", "rel": "director", "r": None, "ev": "Jane Doe, director", "pg": 3},
        {"n": "Other Co", "t": "o", "rel": "parent", "r": None, "ev": "x", "pg": 4}]})}
    items, problems = X.items_from_reply(res, 3, {3: "https://a.ca/p", 4: "https://b.ca/q"}, batch=7)
    assert items[0]["source_url"] == "https://a.ca/p" and items[0]["type"] == "person" and items[0]["batch"] == 7
    assert items[1]["source_url"] is None                       # cites a page that was not in this request
    assert problems == [{"batch": 7, "name": "Other Co", "page": 4}]


def test_failed_reply_gives_no_items():
    assert X.items_from_reply({"ok": False, "text": ""}, 1, {1: "u"}, 1) == ([], [])


# ---------------------------------------------------------------- verifier helpers
def test_name_ok():
    assert V.name_ok("Jane Doe", "person") and not V.name_ok("Dr. Smith", "person") and not V.name_ok("John 2nd", "person")
    assert not V.name_ok("Acme Ltd. March 3", "organization") and V.name_ok("Acme Ltd.", "organization")


def test_page_text_long_page_keeps_names():
    content = "A" * 600_000 + " Jane Doe here " + "B" * 10
    t = V.page_text(content, ["Jane Doe"], max_chars=540_000)
    assert "Jane Doe" in t and len(t) < 600_000
    assert V.page_text("short", ["x"]) == "short"


def test_verifier_message_format():
    claims = [{"id": 1, "name": "Jane Doe", "type": "person", "rel": "director", "role": "", "evidence": "Jane Doe, director"},
              {"id": 2, "name": "Acme", "type": "organization", "rel": "parent", "role": "owner", "evidence": "owned by Acme"}]
    msg = V.page_message("{record}|{url}|{title}|{page}|{claims}", "REC", "https://a.ca", {"title": "T", "content": "C"}, claims, 540_000)
    assert msg == ('REC|https://a.ca|T|C|1. Jane Doe (person): claimed director; quote: "Jane Doe, director"\n'
                   '2. Acme (organization): claimed parent (role: owner); quote: "owned by Acme"')


# ---------------------------------------------------------------- confirmed-only filter (claim ids)
def test_confirmed_only_by_claim_id():
    """Review issue 1: Jane Doe's director claim is confirmed, her owner claim (same page, same evidence) is not -> director only."""
    final = {"company": {}, "people": [
        {"name": "Jane Doe", "mentions": [
            {"source_url": "u1", "evidence": "Jane Doe, director and owner", "relationship": "director", "role": None, "claim_ids": [1]},
            {"source_url": "u1", "evidence": "Jane Doe, director and owner", "relationship": "shareholder_or_owner", "role": None, "claim_ids": [2]}]},
        {"name": "Bob Roe", "mentions": [{"source_url": "u1", "evidence": "Bob", "relationship": "employee", "claim_ids": []}]}],
        "organizations": []}
    claims = [{"id": 1, "url": "u1", "evidence": "Jane Doe, director and owner", "name": "Jane Doe", "final": "confirmed"},
              {"id": 2, "url": "u1", "evidence": "Jane Doe, director and owner", "name": "Jane Doe", "final": "unverified"}]
    out = confirmed_only(final, claims)
    assert [e["name"] for e in out["people"]] == ["Jane Doe"]
    assert out["people"][0]["relationships"] == ["director"] and len(out["people"][0]["mentions"]) == 1


def test_claim_ids_survive_merge():
    from on_the_fly_pipeline.steps.finalize import merge_entities
    items = [{"name": "Jane Doe", "type": "person", "relationship": "director", "evidence": "e", "source_url": "u", "claim_id": 3},
             {"name": "Jane Doe", "type": "person", "relationship": "director", "evidence": "e", "source_url": "u", "claim_id": 7},
             {"name": "Jane Doe", "type": "person", "relationship": "officer", "evidence": "e", "source_url": "u"}]
    m = merge_entities(items, {})["person"][0]["mentions"]
    assert [x["claim_ids"] for x in m] == [[3, 7], []]          # equal mentions merged, ids combined; no claim -> no id


# ---------------------------------------------------------------- budget
def test_budget_hard_cap_and_ledger_resume(tmp_path):
    led = tmp_path / "ledger.jsonl"
    b = PaidBudget("x", 1.0, led)
    r = b.reserve(0.4)
    b.settle(r, 0.7, {"k": 1})
    with pytest.raises(BudgetExceeded):
        b.reserve(0.31)                                          # 0.7 spent + 0.31 > 1.0
    assert PaidBudget("x", 1.0, led).spent == pytest.approx(0.7)  # a new process starts from the ledger


def test_budget_waits_for_inflight():
    b = PaidBudget("x", 1.0, None)
    r = b.reserve(0.8)
    t0 = time.monotonic()
    threading.Timer(0.5, lambda: b.settle(r, 0.1)).start()
    b.reserve(0.5)                                               # must wait until the first call settles
    assert time.monotonic() - t0 >= 0.4


def test_budget_parallel_never_exceeds_cap():
    """Review issue 2: many threads whose real cost equals their reserve; the old release-then-add order reached $1.20 on $1."""
    b = PaidBudget("x", 1.0, None)

    def one():
        try:
            r = b.reserve(0.1)
        except BudgetExceeded:
            return
        time.sleep(0.01)
        b.settle(r, 0.1)
    ts = [threading.Thread(target=one) for _ in range(60)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert b.spent <= 1.0 + 1e-9 and b.inflight == pytest.approx(0) and b.n_inflight == 0


def test_budget_allow_when_idle():
    b = PaidBudget("x", 1.0, None, allow_when_idle=True)
    b.settle(b.reserve(0.0), 0.6)
    r = b.reserve(0.3)                                           # 0.6 + 0.3 <= 1: fine
    b.settle(r, 0.0)
    with pytest.raises(BudgetExceeded):
        b.reserve(0.5)                                           # spent + reserve > cap: always refused


# ---------------------------------------------------------------- verifier cache: a cached call is never paid
def test_verifier_cache_hit(tmp_path):
    c = _NoKeyClient(load_profile("verifiers", "terra_none"))
    msg = "hello"
    (tmp_path / f"{V.cache_key(c, msg)}.json").write_text(json.dumps({"parsed": {"claims": []}, "usd": 0.01}))

    class NoMoney:
        def reserve(self, a):
            raise AssertionError("a cached call must not reserve money")
    r = V.call(c, NoMoney(), tmp_path, msg)
    assert r["cached"] and r["parsed"] == {"claims": []}


# ---------------------------------------------------------------- storage: input and output formats
from on_the_fly_pipeline.shared.models import Company                          # noqa: E402
from on_the_fly_pipeline.storage import entity_rows, open_companies, open_sink  # noqa: E402
from on_the_fly_pipeline.storage import detect                         # noqa: E402
from on_the_fly_pipeline.storage import ENTITY_FIELDS                  # noqa: E402


def test_input_formats(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("name,province,slug\nAcme Ltd.,British Columbia,acme\n")
    assert [c.slug for c in open_companies(str(f))] == ["acme"]
    j = tmp_path / "c.jsonl"
    j.write_text('{"name": "Acme Ltd.", "province": "Manitoba", "slug": "acme"}\n')
    assert open_companies(str(j))[0].province == "Manitoba"
    assert detect("x.tsv") == "tsv" and detect("postgresql://h/db") == "database" and detect("x.json") == "json"
    with pytest.raises(SystemExit, match="planned"):
        open_companies("sqlite:///c.db")
    with pytest.raises(SystemExit, match="planned"):
        open_companies(str(tmp_path / "c.json"))


def test_output_formats(tmp_path):
    with pytest.raises(SystemExit, match="planned"):
        open_sink("csv", tmp_path)
    with pytest.raises(SystemExit, match="unknown"):
        open_sink("xml", tmp_path)
    c = Company(name="Acme Ltd.", province="Manitoba", slug="acme", company_id="ca.mb:1")
    sink = open_sink("json", tmp_path)
    ver = {"people": [{"name": "Jane Doe", "other_spellings": ["J. Doe"], "mentions": [
        {"source_url": "https://a.ca", "evidence": "Jane Doe, director", "relationship": "director", "role": None, "page_title": "T"}]}],
        "organizations": []}
    sink.write_company(c, ver, {}, [{"id": 1}], {"slug": "acme"})
    sink.close({"companies": 1})
    assert json.loads((tmp_path / "acme" / "verified.json").read_text()) == ver
    assert (tmp_path / "acme" / "decisions.json").is_file() and (tmp_path / "run_summary.json").is_file()
    rows = entity_rows(c, ver)
    assert len(rows) == 1 and list(rows[0]) == ENTITY_FIELDS and rows[0]["source_url"] == "https://a.ca"
    assert rows[0]["other_spellings"] == "J. Doe" and rows[0]["entity_type"] == "person"


# ---------------------------------------------------------------- resume (review issue 3)
from on_the_fly_pipeline.pipeline import fingerprint, is_done, mark_done, prepare_out   # noqa: E402


def _fp(tmp_path, model="m1"):
    src = tmp_path / "c.csv"
    if not src.exists():
        src.write_text("name,province\nAcme Ltd.,Manitoba\n")
    return fingerprint(companies_source=str(src), slugs=["acme"], cfg={"a": 1}, extractor={"model": model}, verifier={"model": "v"},
                       prompts={"p": "text"}, retrieval={"provider": "serper"})


def test_resume_rules(tmp_path):
    out = tmp_path / "out"
    fp = _fp(tmp_path)
    prepare_out(out, fp, resume=False, details={})              # new run: writes the manifest
    (out / "acme").mkdir()
    with pytest.raises(SystemExit, match="not empty"):
        prepare_out(out, fp, resume=False, details={})          # non-empty folder without --resume
    prepare_out(out, fp, resume=True, details={})               # same fingerprint: fine
    with pytest.raises(SystemExit, match="extractor"):
        prepare_out(out, _fp(tmp_path, model="m2"), resume=True, details={})   # names what changed
    with pytest.raises(SystemExit, match="no run to resume"):
        prepare_out(tmp_path / "empty", fp, resume=True, details={})


def test_done_marker(tmp_path):
    (tmp_path / "acme").mkdir()
    mark_done(tmp_path, "acme", "F1", {"slug": "acme"})
    assert is_done(tmp_path, "acme", "F1")["row"] == {"slug": "acme"}
    assert is_done(tmp_path, "acme", "F2") is None                # another run's marker does not count


def test_extraction_cache_hit_never_pays(tmp_path):
    prof = load_profile("extractors", "haiku_medium")
    c = _NoKeyClient(prof)

    class P:
        extract = "PROMPT"
    msg = "PROMPT\n\nrest"
    content = [{"type": "text", "text": "PROMPT", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "\n\nrest"}]
    (tmp_path / f"{X.cache_key(c.body(content, 'entities', X.SCHEMA))}.json").write_text(json.dumps({"text": '{"e": []}', "usage": {}}))

    class NoMoney:
        def reserve(self, a):
            raise AssertionError("a cached reply must not reserve money")
    r = X.extract_piece(c, NoMoney(), msg, P(), tmp_path)
    assert r["ok"] and r["cached"] and r["attempts"][0]["cost_usd"] == 0.0


def test_saved_retrieval_overwrites_and_checks_pages(tmp_path):
    from on_the_fly_pipeline.pipeline import SavedRetrieval
    src, out = tmp_path / "src", tmp_path / "out"
    (src / "acme" / "pages").mkdir(parents=True)
    (src / "acme" / "select.json").write_text(json.dumps({"status": "ok", "kept": [{"result_id": "r001", "url": "u"}]}))
    (out / "acme").mkdir(parents=True)
    (out / "acme" / "select.json").write_text(json.dumps({"status": "ok", "kept": []}))        # stale file from another run
    c = Company(name="Acme Ltd.", province="Manitoba", slug="acme")
    kept, _ = SavedRetrieval(src, out).run(c)
    assert kept and json.loads((out / "acme" / "select.json").read_text())["kept"]       # re-copied from the source
    other = tmp_path / "other_pages"
    other.mkdir()
    (out / "acme" / "pages").unlink()
    (out / "acme" / "pages").symlink_to(other)
    with pytest.raises(RuntimeError, match="points to"):
        SavedRetrieval(src, out).run(c)


# ---------------------------------------------------------------- countries
from on_the_fly_pipeline.shared.config import all_jurisdictions, available, resolve_config   # noqa: E402


def test_country_profiles():
    assert {"CA", "US"} <= set(available())
    ca, us = resolve_config(None, "CA"), resolve_config(None, "US")
    assert ca["country"]["status"] == "validated" and us["country"]["status"] == "draft"
    assert ca["search"]["serper"]["country"] == "ca" and us["search"]["serper"]["country"] == "us"
    assert ca["search"]["queries"][0]["template"] == '"{name}" {province} corporate registry'
    assert ca["select"]["location"]["enabled"] and not us["select"]["location"]["enabled"]
    assert ca["extraction"]["country"] == "Canada" and ".ca" in us["select"]["foreign_sites"]["country_domains"]
    j = all_jurisdictions()
    assert j["CA-BC"] == ("CA", "British Columbia") and j["US-DE"] == ("US", "Delaware")


def test_site_filter_home_country():
    from on_the_fly_pipeline.shared.foreign_pages import SiteFilter
    ca = SiteFilter(resolve_config(None, "CA")["select"]["foreign_sites"])
    us = SiteFilter(resolve_config(None, "US")["select"]["foreign_sites"])
    assert ca.foreign_url("https://opencorporates.com/companies/us_de/123") == "opencorporates"
    assert ca.foreign_url("https://opencorporates.com/companies/ca_bc/123") == ""
    assert us.foreign_url("https://opencorporates.com/companies/us_de/123") == ""
    assert us.foreign_url("https://opencorporates.com/companies/ca_bc/123") == "opencorporates"
    assert us.foreign_url("https://example.ca/x") == "domain" and ca.foreign_url("https://example.ca/x") == ""


def test_location_locale_required():
    from on_the_fly_pipeline.shared.foreign_pages import LocationCheck
    with pytest.raises(SystemExit, match="no locale"):
        LocationCheck({"enabled": True, "locale": "us"})
    assert LocationCheck({"enabled": False, "locale": "us"}).drops({"u": {"content": "x"}}, "Delaware") == {}


def test_corptrace_country_from_jurisdiction(tmp_path):
    f = tmp_path / "c.jsonl"
    f.write_text(json.dumps({"legal_name": "Acme Inc.", "jurisdiction": "US-DE", "reg_key": "us.de:1", "id_value": "1"}) + "\n" +
                 json.dumps({"legal_name": "Acme Ltd.", "jurisdiction": "CA-BC", "reg_key": "ca.bc:A1", "id_value": "A1"}) + "\n")
    cs = open_companies(str(f))
    assert [(c.country, c.province, c.slug) for c in cs] == [("US", "Delaware", "us_de_1"), ("CA", "British Columbia", "ca_bc_a1")]


def test_run_refuses_other_country(tmp_path):
    from on_the_fly_pipeline.cli import main
    f = tmp_path / "c.csv"
    f.write_text("name,province,country\nAcme Inc.,Delaware,US\n")
    with pytest.raises(SystemExit, match="not CA"):
        main(["run", "--companies", str(f), "--out", str(tmp_path / "o"), "--from-saved", str(tmp_path), "--cap-extract-usd", "1",
              "--cap-verify-usd", "1"])


def test_explicit_input_format_overrides_extension(tmp_path):
    f = tmp_path / "companies.txt"                               # no telling extension
    f.write_text("name\tprovince\nAcme Ltd.\tManitoba\n")
    assert open_companies(str(f), "tsv")[0].province == "Manitoba"
    g = tmp_path / "companies.dat"
    g.write_text('{"name": "Acme Ltd.", "province": "Manitoba"}\n')
    assert open_companies(str(g), "jsonl")[0].name == "Acme Ltd."


# ---------------------------------------------------------------- credentials
def test_credentials_env_then_file(tmp_path, monkeypatch):
    from on_the_fly_pipeline.shared import config as C
    env = tmp_path / ".env"
    env.write_text("# comment\nexport FOO_KEY='from-file'\nBAR_KEY=\n")
    monkeypatch.delenv("FOO_KEY", raising=False)
    assert C.get("FOO_KEY", env) == "from-file"
    monkeypatch.setenv("FOO_KEY", "from-env")
    assert C.get("FOO_KEY", env) == "from-env"                  # environment first
    assert C.get("BAR_KEY", env) is None and C.get("MISSING", env) is None
    for n in C.OPENROUTER_NAMES:
        monkeypatch.delenv(n, raising=False)
    with pytest.raises(SystemExit) as e:
        C.openrouter_key(env)
    assert "from-file" not in str(e.value) and "from-env" not in str(e.value)   # never shows a value


# ---------------------------------------------------------------- no caps (production default, 2026-11-24)
def test_budget_without_cap_only_records(tmp_path):
    b = PaidBudget("x", None, tmp_path / "led.jsonl")
    for _ in range(5):
        b.settle(b.reserve(10.0), 3.0, {"k": 1})                  # far above any cap: never waits or refuses
    assert b.spent == pytest.approx(15.0) and not b.exhausted and b.cap is None
    assert len((tmp_path / "led.jsonl").read_text().splitlines()) == 5   # every call still recorded


def test_search_calls_without_cap():
    from on_the_fly_pipeline.steps.search import CallBudget
    cb = CallBudget(0)
    for _ in range(1000):
        cb.take()
    assert cb.used == 1000 and not cb.stopped_reason


def test_unknown_extension_asks_for_format(tmp_path):
    f = tmp_path / "companies.txt"
    f.write_text("name,province\nAcme Ltd.,Manitoba\n")
    with pytest.raises(SystemExit, match="--input-format"):
        open_companies(str(f))                                   # was silently read as CSV
    assert open_companies(str(f), "csv")[0].name == "Acme Ltd."


def test_sink_logs_company_rows(tmp_path):
    sink = open_sink("json", tmp_path)
    sink.log_company({"slug": "a", "claims": 1})
    sink.log_company({"slug": "b", "error": "x"})
    assert [json.loads(x)["slug"] for x in (tmp_path / "run_companies.jsonl").read_text().splitlines()] == ["a", "b"]
