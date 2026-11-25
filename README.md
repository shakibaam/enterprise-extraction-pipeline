# on_the_fly_pipeline

Finds the **people and organizations connected to a registered company**, with the source URL and a verbatim quote for every mention. Version 1.1.0. **Canada** is validated; the **US** profile is a draft that has not been tested.

```
company ─► search ─► fetch ─► rerank ─► select ─► extract ─► verify ─► clean-up ─► verified.json
           Serper/   pages    Granite   score ≥    any model  any model  merge, keep
           Keenable  → text   (GPU)     0.82       per page   per claim  confirmed claims
```

## Quick start

The repository root is the Python package. Clone it into a folder named `on_the_fly_pipeline`, and run commands from that folder's parent. Use the vLLM environment's Python, because the reranker needs `transformers`.

```bash
git clone <repo-url> on_the_fly_pipeline
pip install -r on_the_fly_pipeline/requirements.txt
PY=/home/shakiba/vllm-env/bin/python

$PY -m on_the_fly_pipeline profiles                          # list extractor and verifier profiles
$PY -m on_the_fly_pipeline serve rerank                      # start the Granite reranker (check nvidia-smi first; GPUs are shared)
$PY -m on_the_fly_pipeline run --companies companies.jsonl --out runs/my_run
$PY -m on_the_fly_pipeline run ... --resume                  # continue an interrupted run
$PY -m on_the_fly_pipeline serve rerank --stop
```

**Useful `run` options:**
- `--extractor` / `--verifier`: model profiles. Defaults: `haiku_medium`, `terra_none`.
- `--country`: default `CA`; one country per run.
- `--provider serper|keenable`: search provider. Default `serper`.
- `--from-saved DIR`: reuse the saved pages of an earlier run instead of searching again.
- `--slugs FILE`: run only these companies, in this order.
- `--input-format`, `--output-format`: see "Inputs and outputs".
- `--max-search-calls`, `--cap-extract-usd`, `--cap-verify-usd`: optional hard caps. Production runs have none; cost is always recorded.

**Local model:** start it with `serve llm --profile qwen9b_local`, then run with `--extractor qwen9b_local`.

**Evaluation** (optional, never part of production):
- `evaluate recall --out DIR`: free; the only code that reads gold data.
- `evaluate judge --out DIR --companies FILE --cap-usd 5`: paid; Claude Sonnet fact-check.

**Keys** are read by name from the environment, else from the env file named in `config/default.toml` (`[secrets]`). They are never printed, logged or stored here.

## How it works

| Step | Module | What it does | Output |
|---|---|---|---|
| Search | `steps/search.py` | 3 queries × 10 results from the country profile; hits merged by canonical URL | `search.json` |
| Fetch | `steps/fetch.py` | Downloads all hits in parallel, HTML → text; search snippet if a page fails; PDFs off | `fetch.json`, `pages/` |
| Rerank | `steps/rerank.py` | Granite cross-encoder on local vLLM; page score = best 8k-token chunk; skips foreign pages and pages that don't name the company | `rerank.json` |
| Select | `steps/select.py` | Keeps pages with score ≥ 0.82 that name the company and aren't placed abroad | `select.json` |
| Extract | `steps/extract.py` | One request per page (pieces of up to 6k tokens), strict JSON; the page number is mapped to the exact URL; every claim gets an id | `extract.json` |
| Verify | `steps/verify.py` | Starts as soon as a page is extracted. All claims of the page in one call with the whole page: "does it state X has this relation to THIS company?" | `decisions.json` |
| Clean-up | `steps/finalize.py` | Drops non-names, the company's own name and relation "other"; merges spellings; keeps only mentions whose own claim was confirmed | `final.json`, `verified.json` |

There is no separate identity check: the verifier's question covers namesakes. Extraction and verification overlap (streaming).

## Model profiles (`config/models.toml`)

The extractor and the verifier are chosen by name; changing a model needs no code change.

| Extractor | Model |
|---|---|
| **`haiku_medium`** (default) | Claude Haiku 5.5, reasoning medium |
| `haiku_off`, `haiku_low` | Claude Haiku 5.5, reasoning off / low |
| `qwen9b_local` | Qwen3.5-9B on the local GPUs (free); start it with `serve llm --profile qwen9b_local` |
| `gemini_flash_lite` | Gemini 3.1 Flash Lite |
| `gpt54_nano`, `gpt6_luna` | GPT-5.4 Nano, GPT-6 Luna |
| `terra_none` | GPT-6.1 Terra, no reasoning |
| `sonnet_low` | Claude Sonnet 5.5, reasoning low |

**Verifiers:** `terra_none` (default; GPT-6.1 Terra, no reasoning) and `sonnet_medium` (Claude Sonnet 5.5).

**Adding a model:** add an `[extractors.<name>]` or `[verifiers.<name>]` table. A local model also needs a `[serve]` table.

## Countries (`config/countries/`)

Each profile holds everything country-specific, merged over `config/default.toml`:
- search queries and Google country;
- the reranker's question;
- the foreign-site lists and the OpenCorporates home country;
- the location check (locale);
- the prompts (`prompts/<code>/`);
- jurisdiction codes (e.g. `CA-BC` → British Columbia).

**`ca.toml` (validated)**.

**`us.toml` (draft, untested):**
- state-based Secretary of State queries;
- Google country `us`;
- Canadian sites counted as foreign;
- location check off until a US locale is written in `shared/foreign_pages.py`.

**Adding a country:**
1. Copy `ca.toml` and edit every value.
2. Add `prompts/<code>/`.
3. Keep `status = "draft"` until it has been measured.

## Inputs and outputs (`storage.py`)

| | Implemented | Planned (registered; stops with a message) |
|---|---|---|
| **Companies in** | JSON Lines (corptrace records or plain objects), CSV, TSV: `name`, `province`, optional `slug`, `company_id`, `entity_form`, `country` | JSON array, database |
| **Results out** | JSON files, below | JSON Lines, CSV, database, all using the flat record `entity_rows()` |

**Choosing the input format:** it is taken from the file extension, or set with `--input-format`. An unknown extension stops the run and asks for the format.

**Adding a format:** one reader or one sink class in `storage.py`.

**`OUT/<slug>/`:**
- `verified.json` is **the output**: each mention with `source_url`, `evidence`, `relationship`, `role` and `claim_ids`.
- Working files, always JSON: `search.json`, `fetch.json`, `pages/`, `rerank.json`, `select.json`, `extract.json`, `decisions.json`, `final.json`.
- `timing.json`: seconds and cost per step.
- `done.json`: written last, marks the company as complete.

**`OUT/`:**
- `run_summary.json`, `run_companies.jsonl`;
- `run_manifest.json` (the fingerprint);
- cost ledgers `ledger_extract.jsonl`, `ledger_verify.jsonl`;
- answer caches `extract_cache/`, `verify_cache/`.

**Resume.** A new run needs an empty `--out`. `--resume` continues only if the fingerprint matches, and otherwise names what changed. The fingerprint covers the companies, config, country, models, prompts and retrieval mode. On resume:
- finished companies are skipped;
- requests that already got an answer are served from the caches;
- the cost ledgers keep counting.

## Layout

```
cli.py          command line (profiles, check-balance, run, serve, evaluate)
pipeline.py     per-company orchestration, live or saved retrieval, run state (fingerprint, resume, done markers)
serve.py        start/stop GPU servers (Granite reranker, local LLM); checks free GPU memory
storage.py      all file formats: company inputs, result sinks, working-file writer
steps/          one module per step: search, fetch, rerank, select, extract, verify, finalize
shared/         helpers: config (settings, country profiles, keys), models, names, foreign_pages, llm (profiles,
                client, cost ledger), pdf_text
config/         default.toml, models.toml, countries/
prompts/        ca/ and us/: extract, chunk sizing, verify, registry record
evaluate/       judge.py (Sonnet fact-check), recall.py, quotes.py
tests/          test_units.py (39 tests), check_equivalence.py and check_rerank.py (need this machine's data)
```

## Tests

- `pytest on_the_fly_pipeline/tests/test_units.py`: unit tests (no network, no GPU).
- `tests/check_equivalence.py` and `tests/check_rerank.py`: regression checks against saved runs. They need this machine's data, and the rerank check also needs the GPUs.

## Known limits

- **Search provider.** Serper is the default; Keenable is available with `--provider keenable`. Results depend on the pages each returns.
- **Optional caps.** A reply that costs more than its reserved estimate can pass a cap by at most that difference.
- **PDFs are not parsed** (only their search snippets are used).
- **US profile** is untested, and its location check is off.
- **The name matcher treats "canada" as a generic word** for every country.
