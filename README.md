# on_the_fly_pipeline

A frozen, reusable pipeline that finds the **people and organizations connected to a registered company**, with a source URL and a verbatim quote for every mention. It was frozen on 2026-11-24 from the adopted **E6** setup; this is version 1.1.0.

**Countries:**
- **Canada** is validated: every measured result used it.
- **United States** is ready as a **draft profile that has not been tested**.
- Other countries can be added with one settings file (section 5).

```
company ─► search ─► fetch ─► rerank ─► select ─► extract (any model) ─► verify (any model) ─► clean-up ─► verified.json
           Serper    pages    Granite   score≥0.82  one request per       one yes/no question     merge,
           or        → text   (local    + name      page piece; page      per claim, whole page   drop junk,
           Keenable           GPU)      + location  number → exact URL   as context, no web      keep confirmed claims
```

**Settings live in files, not code:**
- The extractor and the verifier are **profiles** in [config/models.toml](config/models.toml).
- Everything that depends on the country is a **country profile** in [config/countries/](config/countries/).
- File formats are handled in [storage.py](storage.py).

---

## 1. Quick start

**Getting the code.** The repository root *is* the Python package, so clone it into a folder named `on_the_fly_pipeline` and run from that folder's parent:

```bash
git clone https://github.com/shakibaam/enterprise-extraction-pipeline.git on_the_fly_pipeline
pip install -r on_the_fly_pipeline/requirements.txt
python -m on_the_fly_pipeline --help
```

API keys are read by name from the environment, or from the env file named in `config/default.toml` (`[secrets]`). None are stored in the repository.


Run everything from `Provinces/` with the vLLM environment's Python; the reranker needs `transformers`.

```bash
cd /home/shakiba/Reviewrly_RA/Enterprise/Provinces
PY=/home/shakiba/vllm-env/bin/python

$PY -m on_the_fly_pipeline profiles                      # list the extractor and verifier profiles
$PY -m on_the_fly_pipeline check-balance --need 5        # both OpenRouter limits (key and account)
nvidia-smi                                               # the GPUs are shared: check first
$PY -m on_the_fly_pipeline serve rerank                  # Granite reranker, about 0.09 of each GPU

# live run: search → … → verified.json   (defaults: --country CA, Serper, haiku_medium, terra_none)
$PY -m on_the_fly_pipeline run --companies companies.jsonl --out runs/my_run

# continue an interrupted run (only if nothing that decides results changed; see section 6)
$PY -m on_the_fly_pipeline run <the same arguments> --resume

# another extractor or verifier
$PY -m on_the_fly_pipeline run ... --extractor sonnet_low --verifier terra_none

# local Qwen 9B extractor (free); start its server first
$PY -m on_the_fly_pipeline serve llm --profile qwen9b_local
$PY -m on_the_fly_pipeline run ... --extractor qwen9b_local

# US companies (DRAFT profile, untested)
$PY -m on_the_fly_pipeline run --country US --companies us_companies.csv ...

# reproduce the E6–E14 comparison on the SAME saved pages (no search, fetch or rerank)
$PY -m on_the_fly_pipeline run --companies enterprise_variant_d/runs/recall_fixes_20261115/companies.jsonl \
    --slugs enterprise_agent/results/experiments/e2e_stream_final_terra_20261120/companies_83.txt \
    --from-saved enterprise_variant_d/runs/recall_fixes_20261115/out_AE --out runs/repro

# optional evaluation (never part of production)
$PY -m on_the_fly_pipeline evaluate recall --out runs/my_run                          # free; the only code that reads gold
$PY -m on_the_fly_pipeline evaluate judge --out runs/my_run --companies companies.jsonl --cap-usd 5 --dry-run   # paid

$PY -m on_the_fly_pipeline serve rerank --stop          # stop the servers this launcher started
```

**Companies:** JSON Lines, CSV or TSV (section 4).
- Each company needs a name and a province (or state).
- The registry key and legal form are used when present.
- `--slugs FILE` restricts the run to a list, in that order.

**Cost: no caps in production (since 2026-11-24); every cost is still recorded.**
- A run never stops because of cost.
- The cost of every paid call is written to `ledger_extract.jsonl` and `ledger_verify.jsonl`, per company in `timing.json` (`cost_usd`), and per run in `run_summary.json`.
- **Optional caps**, for experiments: `--max-search-calls`, `--cap-extract-usd` and `--cap-verify-usd`. When a cap is reached, companies already in progress finish with what they have, and no new company starts.
- The judge (evaluation only) still needs `--cap-usd`.

---

## 2. The steps

| # | Step | What it does | Model / service | Typical time |
|---|---|---|---|---|
| 1 | **Search** | 3 queries × 10 results from the country profile (CA: `"{name}" {province} corporate registry`, `… address contact`, `… public records`). Hits are merged by canonical URL (ids r001…). | Serper (default) or Keenable; paid per call | Serper ≈ 1 s. Keenable was rate-limited, with a median of about 500 s per company at 128 companies in flight. |
| 2 | **Fetch** | Downloads every hit in parallel (64 workers, 4 per website) and converts HTML to text. PDFs are not parsed; their search snippet is kept instead. A failed page falls back to its snippets. | none | median 9.8 s |
| 3 | **Rerank** | Granite cross-encoder scores each page against a question about the company's people and owners. The page score is the best 8k-token chunk. Pages that don't name the company, or that sit on foreign sites, are not scored. | `ibm-granite/granite-embedding-reranker-english-r2`, local vLLM `/classify` (ports 8020/8021) | median 0.22–0.45 s |
| 4 | **Select** | Keeps pages with score ≥ 0.82 that name the company. Then drops foreign sites, and pages placed outside the country (or, for sole proprietorships, in another province). | none | ≈ 0 s |
| 5 | **Extract** | One request per selected page. A page over 6,000 tokens is split into pieces, and every piece is sent at once (64 in flight). Strict JSON `{"e":[{n,t,rel,r,ev,pg}]}`: name, person/organization, one of 11 relations, role, verbatim evidence, page number. Code maps the page number back to the exact URL. Every item sent to the verifier gets a **claim id**. | **extractor profile** (default `haiku_medium`) | median 4.2 s (Haiku medium) |
| 6 | **Verify** | Starts as soon as all pieces of a page are back, while other pages are still being extracted. All claims citing the page go in one call with the whole page: "does this page state that X has this relationship with THIS registered company, in this direction?" Confirmed = "yes" and the name passes a code check. | **verifier profile** (default `terra_none`) | ≈ 1.3 s per call. Median added after extraction: 1.3 s for companies with claims, 0 s overall. |
| 7 | **Clean-up** | Drops items without a URL, the company's own name, non-names (e-mails, addresses, roles), relation "other", and entities linked only to another person. Merges spellings; each mention keeps its claim ids. Then keeps only mentions with a **confirmed claim id**. | none | ≈ 0 s |

There is **no identity check** ("is this page about our company?"). E6 dropped it: it made no difference to precision, cost 1.5 s, and blocked pages that held 32 gold entities. The verifier's question covers namesakes instead.

**Streaming.** Verification overlaps extraction. On one test company this matched running the phases one after another, and it never waits for the slowest page before verifying the others.

**The confirmed-only filter uses claim ids (fixed 2026-11-24).**
- **Before:** E6 matched a mention to a confirmed claim by source URL plus evidence, or source URL plus name. A confirmed relation could therefore carry through an *unverified* relation of the same person on the same page.
- **Now:** a mention is kept only if one of its own claims was confirmed.
- **Effect on E6's 83 companies:** 1 entity and 1 mention fewer; everything else is identical (section 10):
  - **Everett Lord** (person, "officer") is removed. The verifier rejected his own claim; he survived in E6 only because the confirmed claim of a *different* entity, Everett Lord Holdings Limited, quoted the same sentence. He is a gold entity and the fact-check rated him "minor", so on E6's data recall goes from 78 to 77 gold, strict precision from 90.3% to 91.2% (93/102), and lenient precision stays at 99.0%.
  - **Tian Kusumoto** loses his "counsel" mention. He stays in the output as a confirmed director.
- The filter now keeps exactly what the verifier confirmed. A rejected claim is dropped even when it was right; the verifier, not the filter, decides.

---

## 3. Model profiles

All results below come from the same 83 companies, the same saved Keenable pages and the GPT Terra verifier (2026-11-20 to 11-23; before the claim-id fix):
- precision is from the Sonnet fact-check;
- recall is gold found out of 132, on the 76-company set;
- cost is per company.

| Extractor profile | Median latency | Median latency, companies with claims | Cost per company | Strict precision | Lenient precision | Recall |
|---|---|---|---|---|---|---|
| **`haiku_medium`** (default) | 4.6 s | 6.7 s | $0.0170 | 90.3% | 99.0% | 78 (59%) |
| `haiku_off` | 2.7 s | 3.2 s | $0.0219 | 68.2% | 95.5% | 38 (29%) |
| `haiku_low` | 3.9 s | 5.9 s | $0.0173 | 80.7% | 96.5% | 37 (28%) |
| `qwen9b_local` | 3.1 s | 5.5 s | $0.0038* | 93.4% | 96.1% | 62 (47%) |
| `gemini_flash_lite` | 6.4 s | 7.1 s | $0.0457 | 78.0% | 100% | 54 (41%) |
| `gpt54_nano` | 7.2 s | 11.5 s | $0.0289 | 79.7% | 96.2% | 51 (39%) |
| `gpt6_luna` | 6.4 s | 9.2 s | $0.0146 | 94.4% | 100% | 56 (42%) |
| `terra_none` | 3.3 s | 5.7 s | $0.1049 | 100% (53/97 judged) | 100% | 73 (55%) |
| `sonnet_low` | 2.8 s | 3.9 s | $0.1453 | 95.8% (72/108 judged) | 100% | 80 (61%) |

\* Qwen 9B runs on the local GPUs, so it has no API cost for extraction. The figure is the verifier only.

Latency here is measured after retrieval (search, fetch, rerank and select are not included).

**What the comparison showed:**
- After the verifier, lenient precision is 95–100% for every extractor.
- The extractor mostly decides **recall, latency and cost**.
- Verification adds about 1.1–1.8 s and $0.003–0.011 per company, whichever extractor is used.

**Verifier profiles:**
- `terra_none` (default): GPT-6.1 Terra, no reasoning.
- `sonnet_medium`: the first V5 verifier. It reached the same precision but was slower and about 4× the cost.

**Adding a model:** add an `[extractors.<name>]` or `[verifiers.<name>]` table, using the fields described at the top of `models.toml`. A local model needs a `[.serve]` table so that `serve llm` can start it.

**Model quirks:**
- Claude 5.5 models don't accept `temperature`.
- Sonnet 5.5 rejects reasoning off or none.
- Qwen runs with thinking off, and a reply cut off at 4,096 tokens is retried with 12,000.

---

## 4. Input and output formats

**Today the pipeline reads companies from JSON Lines, CSV or TSV files, and writes its results as JSON files.** All file-format code sits in one module, [storage.py](storage.py), so a CSV export or a database can be added later without touching the steps.

| | Format | Status | How it is chosen |
|---|---|---|---|
| **Input** (companies) | JSON Lines: corptrace records (`legal_name`, `jurisdiction`, `reg_key`, `id_value`) or plain objects | **implemented** | `--input-format auto` (default: by extension `.jsonl/.ndjson`, `.csv`, `.tsv/.tab`; any other extension stops and asks for the format), or explicitly `jsonl`, `csv`, `tsv`, which then wins over the extension. Also settable as `[io].input_format`. |
| | CSV, TSV: columns `name` (or `company_name`), `province` (or `region`), optional `slug`, `company_id`, `entity_form`, `country` (ISO code, default `CA`) | **implemented** | |
| | One JSON array (`.json`) | planned | stops with a message until written |
| | Database (`postgresql://…`, `sqlite:///…`) | planned | |
| **Output** (results) | JSON files: `OUT/<slug>/verified.json`, `final.json`, `decisions.json`, `timing.json`, `OUT/run_summary.json` | **implemented** (default) | `--output-format json`, or `[io].output_format` |
| | JSON Lines: `OUT/entities.jsonl`, one line per verified mention | planned | stops with a message until written |
| | CSV: `OUT/entities.csv`, one row per verified mention | planned | |
| | Database table(s) | planned | |

**Country of a company.** corptrace records get it from `jurisdiction` through the country profiles' `[jurisdictions]` tables: `CA-BC` → CA, British Columbia; `US-DE` → US, Delaware. Plain rows can give a `country` column, which defaults to CA.

**Required company fields:**
- Name and province (or state).
- Optional: the registry key (`company_id`, e.g. `ca.bc.registration_number:A0000029`) and the legal form. They give the verifier a better registry record.
- Slugs must be unique; they name the output folders.

**The flat output record.** Every future CSV, JSON Lines or database output must write the same fields, which `storage.entity_rows()` already produces: one row per verified mention.

`company_slug, company_name, province, company_id, entity_name, entity_type, other_spellings, relationship, role, evidence, source_url, page_title`

**Adding a format:**
- **Input:** write a reader in `storage.py` that returns `Company` records, add it to `READERS`, and move the format from `INPUT_PLANNED` to `INPUT_IMPLEMENTED`.
- **Output:** write a class with `write_company(company, verified, final, decisions, timing)`, `log_company(row)` and `close(summary)` in `storage.py`, using `entity_rows()`, and register it in `OUTPUT_IMPLEMENTED`.

**Working files stay JSON whatever the output format:** `search.json`, `fetch.json`, `pages/`, `rerank.json`, `select.json`, `extract.json`, `final.json`, `done.json` and the caches. They are the pipeline's scratch space, they make reruns and resumes free, and the evaluation commands read them.

---

## 5. Countries

**One country per run:** `--country CA` is the default. Companies of any other country are refused with a list, so mixed lists are run separately.

A country profile, `config/countries/<code>.toml`, holds everything that depends on the country. Its `[config.*]` tables are merged over [config/default.toml](config/default.toml):

| What | Canada (`ca.toml`, **validated**) | United States (`us.toml`, **DRAFT, untested**) |
|---|---|---|
| Search queries | `"{name}" {province} corporate registry` / `address contact` / `public records` | `"{name}" {region} secretary of state business entity` / `address contact` / `public records` |
| Google country (Serper `gl`) | `ca` | `us` |
| Reranker query country | Canada | United States |
| Foreign sites: dropped before reranking | non-Canadian country domains (`.us`, `.uk`, …), US/UK/NZ registries (sunbiz, bizapedia, …), foreign URL paths, OpenCorporates outside `ca` | non-US domains (**`.ca`** included), Canadian registries (opengovca, canadacompanyregistry, ic.gc.ca, …), foreign URL paths, OpenCorporates outside `us` |
| Location check | **on** (locale `ca`): Canadian provinces and postal codes against US states, US addresses, UK postcodes | **off**: no US locale written yet |
| Prompts | `prompts/ca/`, E6's prompts | `prompts/us/`: the same text with "Canadian" → "US", and province explained as the state |
| Jurisdictions | `CA-AB` … `CA-YT` → province | `US-AL` … `US-WY`, `US-DC` → state |

**The Canada profile reproduces the earlier behaviour exactly.** The resolved CA settings equal the pre-profile config value for value; the only differences are the new `locale` and `opencorporates_home` keys and the moved prompt paths. The checks in section 9 confirm it.

**Before trusting a US run:**
1. Look at the search results for a sample of US companies.
2. Review the foreign-site lists.
3. Write a US location locale in `shared/foreign_pages.py`, and switch the check on.
4. Measure precision and recall with the judge.

**Adding a country:**
1. Copy `ca.toml` to `<code>.toml`.
2. Edit every `[config.*]` value and the `[jurisdictions]` table.
3. Add `prompts/<code>/`.
4. Either set `[config.select.location].enabled = false`, or register a location locale (`LOCALES` in `shared/foreign_pages.py`).
5. Keep `status = "draft"` until it has been measured. A draft profile logs a warning at the start of every run.

**Known country limit:** the name matcher's list of generic words (`shared/names.py`) still includes "canada" for every country, so it is optional in a host-name match. That is harmless for Canada, the only validated country.

---

## 6. Reruns and resume

Every run writes `OUT/run_manifest.json` with a **fingerprint**: a SHA-256 over everything that decides results.

**What the fingerprint covers:**
- the companies file and the slug list;
- the resolved config, including the country profile;
- both model profiles;
- the content of every prompt file;
- the retrieval mode (live provider, or `--from-saved` path) and the output format;
- the package version.

**Rules:**
- **New run:** `--out` must not exist, or be empty.
- **Non-empty `--out` without `--resume`:** refused. Use a new folder.
- **`--resume`:** continues only if the fingerprint matches. Otherwise it refuses and names the parts that changed (e.g. `extractor, prompts`). Use a new folder for a different run.

**What a resumed run reuses:**
- **Finished companies** are skipped. A company is finished when `OUT/<slug>/done.json` exists, written last after its outputs.
- **Unfinished companies run again, but already-paid answers are free:**
  - successful extraction replies (`OUT/extract_cache/`, keyed by the full request);
  - verifier answers (`OUT/verify_cache/`);
  - live search and fetch results, saved per query and per page;
  - the reranker's `rerank.json`, when its inputs are unchanged.
- **The paid ledgers** (`ledger_extract.jsonl`, `ledger_verify.jsonl`) keep recording every paid call (and count against the optional caps, when given).
- **`--from-saved`** re-copies `select`, `rerank` and `search` from the source every time. It refuses a `pages/` link that points elsewhere, so two runs are never mixed.

---

## 7. Output files (JSON, the current format)

`OUT/<slug>/`:

| File | Content |
|---|---|
| `search.json`, `fetch.json`, `pages/rNNN.json`, `rerank.json`, `select.json` | Retrieval: hits, download outcome, page text (`content`), scores, kept and dropped pages with reasons |
| `extract.json` | Every raw item with its source URL and claim id; every request with its attempts, errors, tokens, cost and `cached` flag; `source_url_problems` |
| `decisions.json` | Every claim (`id` = claim id) with the verifier's answer (`final`: confirmed / unverified) and its cost and time |
| `final.json` | After clean-up, before the confirmed-only filter; each mention lists its `claim_ids` |
| **`verified.json`** | **The output.** People and organizations, each mention with `source_url`, `evidence`, `relationship`, `role` and `claim_ids` |
| `timing.json` | `retrieval.seconds` (search, fetch, rerank, select); `t` (from the company's start) and `t_after_retrieval` (`extraction_done`, `verification_done`, `total`: what E6–E14 reported); `cost_usd` (extraction, verification, total); `cached_requests`; per-page times; each verifier call |
| `done.json` | Written last: the company is complete (used by `--resume`) |

`OUT/`:
- `run_manifest.json`: the fingerprint.
- `run_args.json`: the exact arguments, profiles and country.
- `run_summary.json`: medians, spend per step, requests, failures, retries, verifier errors, companies skipped on resume, the stop reason, and the retrieval summaries.
- `run_companies.jsonl`: one line per finished company.
- `ledger_extract.jsonl`, `ledger_verify.jsonl`: every paid call, with its cost.
- `extract_cache/`, `verify_cache/`: replies already paid for.

---

## 8. Folder layout and where the code came from

```
on_the_fly_pipeline/
  README.md, requirements.txt   this file; the dependency list (requests; transformers for the reranker; optional pypdfium2, pytest)
  cli.py, __main__.py     command line (profiles, check-balance, serve, run, evaluate)
  pipeline.py             the streaming orchestrator (from E6 run_stream.run_company) + run state: fingerprint, new-run / --resume
                          rules, done markers (was runstate.py)
  serve.py                starts and stops the GPU servers: Granite reranker, local LLM (checks free GPU memory; logs in logs/)
  config/default.toml     country-neutral settings: retrieval, clean-up, [pipeline] [extraction] [verification] [io]
  config/countries/       ca.toml (validated), us.toml (draft)
  config/models.toml      extractor and verifier profiles (the settings of the E6–E14 runner copies)
  prompts/ca/, prompts/us/   extract.txt (E6 merged_json.txt), chunk_sizing.txt (E6 stage1_notes.txt), verify.txt (E6 V5),
                          registry_record.txt (verifier v3 RECORD); us = the draft variants
  steps/                  ONE MODULE PER PIPELINE STEP, in run order (see steps/__init__.py):
                            search.py, fetch.py, rerank.py, select.py   retrieval (from company_pipeline, standalone CLIs removed)
                            extract.py    page loading, page pieces, request messages, the extraction call (+ reply cache)
                            verify.py     the E6 V5 verifier
                            finalize.py   clean-up (incl. the canonical rules) + the claim-id confirmed-only filter
  shared/                 helpers used by several steps, not steps themselves (see shared/__init__.py):
                            config.py         config file + country profiles + API keys by name (no enterprise_agent import)
                            models.py         Company (+ country), Query, SearchHit
                            names.py          name matching and normalization
                            foreign_pages.py  foreign-site rules + location check (locale registry)
                            llm.py            model profiles, one chat client (OpenRouter and local vLLM), cost ledger with optional caps
                            pdf_text.py       PDF to text in a child process (off by default)
  storage.py              all file formats: company inputs (JSON Lines / CSV / TSV; planned JSON array, database), result sinks
                          (JSON files; planned JSON Lines, CSV, database; entity_rows()), working-file writer (write_json)
  evaluate/               judge.py (Sonnet fact-check, the fixed copy), quotes.py (its quote finder), recall.py (gold entities; free)
  tests/                  test_units.py, check_equivalence.py, check_rerank.py
```

`pipeline.py` runs the steps for each company; it also holds the two ways to get pages, live (search → fetch → rerank → select) or `--from-saved`. Removed with the old `core/` folder (2026-11-24): the old extractor, identity gate, list-page cap, company reader (now `storage.py`) and finalize runner (now `steps/finalize.py`).

Nothing outside this folder was modified. company_pipeline, enterprise_agent and the experiment folders are only read: data for the checks, and the env file that holds the keys.

---

## 9. Rules built in

- **Keys** are read only by name, through `shared/config.py`: from the environment variable, else the env file named in `[secrets].env_file` (`enterprise_agent/.env`). Keys are never printed, logged or written.
- **Cost:**
  - Without caps (the production default), costs are only recorded.
  - With an optional cap, a call reserves an upper estimate of its cost and then settles its real cost in one atomic step, so parallel calls can't spend the same headroom twice.
  - Remaining limit with a cap: a reply that costs more than its reserved estimate can pass the cap by at most that difference.
- **Source URLs:** every item keeps its `source_url`. The model returns a page number, and code maps it to the exact URL. An item citing a page outside its request gets no URL, is listed in `source_url_problems`, and can't become a claim.
- **Gold** is read only by `evaluate/recall.py`.
- **GPUs** are shared. `serve` checks free memory and starts nothing if a GPU is short. It stops only the servers it started itself.
- **Failures are recorded, not hidden:**
  - failed and retried requests (`extract.json`, `run_summary.json`);
  - verifier errors (`verify_errors`; that page's claims stay unconfirmed);
  - companies that crashed (`errors`);
  - the stop reason when a cap is reached.

---

## 10. How it was checked

| Check | Result |
|---|---|
| `tests/test_units.py` (37 tests): **no caps** (costs recorded, nothing refused), request bodies equal the E6–E14 runners', page number → URL mapping, verifier message, name check, long pages, **claim-id filter** (a confirmed director claim does not keep an unverified owner mention), claim ids through merging, **budget** (hard cap, waiting, parallel threads never above the cap, ledger resume), **resume** (fingerprint, refusals, done markers, extraction cache never pays, saved retrieval re-copies), **countries** (CA/US profiles, OpenCorporates home, location locale, jurisdictions, refusing another country), formats (explicit format wins), credentials (environment first, values never shown) | 37 / 37 pass |
| `tests/check_equivalence.py`, messages: rebuild all extraction messages of the 83 companies (CA profile) | **1,504 / 1,504 byte-identical** to the reference rebuild |
| `tests/check_equivalence.py`, replay: E6's 161 claims through this verifier, answered from E6's cache; E6's raw items through this clean-up | **161 / 161 decisions identical; 83 / 83 `final.json` identical** apart from the new `claim_ids` and `country` fields; **83 / 83 `verified.json` identical under E6's old filter**. Under the claim-id filter: **1 entity and 1 mention fewer**, nothing added (section 2): Everett Lord, gold, rated "minor" by the fact-check, so recall on this data is 78 → 77; Tian Kusumoto's counsel mention. |
| `tests/check_rerank.py`: rerank and select on the saved search and fetch results of the 62 companies retrieved with this config (Granite on the GPUs), CA profile | **62 / 62 identical kept pages**, before and after round 2. Largest score difference 0.0011 (threshold 0.82). Rerank + select took a median of 0.22 s per company, max 37 s. |
| Live end-to-end smoke test (paid: Serper + Haiku + Terra) | not run yet (needs approval) |
| Any US run | not run (draft profile) |

The unit tests and the equivalence check run with `.venv` (which has pytest). The rerank check runs with the vLLM Python and needs the GPUs.

---

## 11. Known differences and limits

- **Search provider:** every measured result used **Keenable** pages, and the default here is **Serper**. A live Serper run will return different pages, so results won't match section 3 exactly. Use `--provider keenable` for the same provider, or `--from-saved` for the same pages.
- **Claim-id filter:** it is stricter than E6's. On E6's companies it removes 1 entity (Everett Lord: gold, but his own claim was rejected by the verifier) and 1 mention. Recall there is 78 → 77; strict precision is 90.3% → 91.2%.
- **"Batch i of N" header:** pieces are numbered in page order. In E6 they followed the order in which the instant identity step finished, so a few header characters differ from a live E6 request. The page text and the prompt are identical.
- **Chunk sizing** uses `chunk_sizing.txt`, as E6 did, while the request uses `extract.txt`. Their lengths differ by one byte; this is kept so that piece boundaries match E6.
- **PDFs** are not parsed (`[fetch].parse_pdfs = false`, as in every measured run); their search snippet is used instead.
- **Retrieval latency:** E6–E14 latencies exclude retrieval. A live run reports both (`t` and `t_after_retrieval`).
- **US profile:** a draft, untested, with the location check off (section 5). The generic-word list still treats "canada" as generic for every country.
- **Judge:** use `--workers 1`, as in all measured runs. Seed `--cache` from an earlier judge cache to avoid paying twice for the same entities.

---

## 12. Plan and status

**Round 1 (2026-11-24): freeze the adopted pipeline.**
- One self-contained folder; the extractor and the verifier chosen by profile; search and the reranker included again; the judge optional.
- Decisions: search pluggable (Serper default, Keenable available); code copied in, not imported; the folder is `Provinces/on_the_fly_pipeline/`.

**Round 2 (2026-11-24): review fixes, resume, cleanup, countries.**
- Decisions: a country framework plus a draft US profile; one country per run.

| Step | Status |
|---|---|
| Round 1: layout, copy of company_pipeline's retrieval and helpers, config, model profiles, prompts, LLM client, steps, orchestrator, CLI, judge and recall, vLLM launcher | done |
| Storage layer: input/output formats in one place; JSON implemented; JSON Lines, CSV, database registered as planned | done |
| 1a. Confirmed-only filter by claim id (review issue 1) | done. On E6's data: 1 entity (gold, its own claim unverified) and 1 mention removed |
| 1b. Atomic budget settlement; admission checks both budgets (review issue 2) | done |
| 2. Resume: fingerprint, refusals, done markers, extraction reply cache, saved-retrieval fix (review issue 3) | done |
| 3. Cleanup: old extractor, identity gate, page cap and standalone CLIs removed; clean-up merged into `steps/finalize.py`; company reader merged into `storage/inputs.py`; CLI-only config options removed | done |
| 3f. Standalone: key reading in the package (now in `shared/config.py`; no enterprise_agent import), `requirements.txt` | done |
| 4. Country profiles: CA (validated, identical to before), US (draft) | done |
| Free checks after round 2: unit tests, equivalence, rerank on GPU | done (section 10) |
| No spending caps by default in production (caps optional; costs always recorded) | done (2026-11-24) |
| One folder for the steps: `steps/` holds every step (search … finalize), `shared/` only the helpers; `core/` and `steps/retrieve.py` removed | done (2026-11-24) |
| `llm/` (client.py + budget.py) merged into `shared/llm.py` | done (2026-11-24) |
| `scripts/serve_vllm.py` moved to the top-level `serve.py`; stale "scripts/serve_vllm.py" hints fixed | done (2026-11-24) |
| `shared/` 9 → 6 modules: canonical → `steps/finalize.py`; names + name_norm → `names.py`; sites + location → `foreign_pages.py`; config + countries.py + credentials.py → `shared/config.py` (one key reader) | done (2026-11-25) |
| `steps/` cleanups: unused structure-aware splitter removed from `extract.py`; `write_json`/`utc_now` moved to `storage/outputs.py` (no step imports another step); `finalize_company` in search/fetch renamed `company_summary`; reranker `Profile` → `RerankProfile`; one `BudgetExceeded` (in `shared/llm.py`) | done (2026-11-25) |
| `storage/` (inputs.py + outputs.py) merged into `storage.py`; an unknown file extension now asks for `--input-format` (was read as CSV); `run_companies.jsonl` written by the output sink | done (2026-11-25) |
| Top level: `runstate.py` merged into `pipeline.py`; `pyproject.toml` removed (one dependency list: `requirements.txt`) | done (2026-11-25) |
| Live smoke test, 1 company (paid, about $0.05: 3 Serper calls, Haiku, Terra) | waiting for approval |
| Optional: 83 companies live with Serper (about $1.5 plus about $5 for the judge), Keenable for a like-for-like comparison; a first US sample | later, with approval |
