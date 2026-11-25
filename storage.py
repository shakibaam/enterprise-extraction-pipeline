"""Storage: where companies come from and where results go: the only place that knows file formats (one module since 2026-11-25;
was the storage folder: inputs.py, outputs.py).

Where companies come from and where results go: the only place that knows file formats.

Inputs:  open_companies(source, format)  -> list[Company]
Outputs: open_sink(format, out_dir)      -> ResultSink with write_company(...), log_company(...) and close()
Working files: write_json (atomic), utc_now: used by every step for its own JSON in OUT/<slug>/

Today: inputs JSON Lines / CSV / TSV files; outputs JSON files (one folder per company). Other formats (CSV export, JSON Lines, a database)
are registered as "planned": choosing one stops the run with a clear message until its class is written. The rest of the pipeline never
reads or writes these formats itself, so adding one means adding one class here and one line in the registry.
Working files (search.json, fetch.json, pages/, rerank.json, select.json, extract.json, final.json) stay JSON in the output folder whatever
the output format: they are the pipeline's own scratch space and are what makes reruns and resumes free.

COMPANY INPUTS
Company inputs (the company reader of company_pipeline/pipeline/companies.py moved here, 2026-11-24).

A source gives Company records (shared/models.py): name and province (the region: province, state, ...) are required; slug (output folder
name), company_id (registry key, e.g. "ca.bc.registration_number:A0000029"), entity_form and country (ISO code, default CA) are optional.

Implemented
  jsonl   JSON Lines: corptrace directory records (legal_name, jurisdiction, reg_key, id_value, entity_form) or plain objects
          {name|company_name, province|region, slug?, company_id?, entity_form?, country?}
  csv     comma-separated with a header row, the plain-object columns
  tsv     tab-separated, the same columns
Planned (registered, not implemented)
  json      one JSON array of the plain objects
  database  e.g. "postgresql://..." or "sqlite:///companies.db" with a query returning the same columns
"auto" picks the format from the file extension (.jsonl/.ndjson, .csv, .tsv/.tab); an explicit format is used as given, whatever the extension.

corptrace records: name = legal_name; country and province from "jurisdiction" through the country profiles' [jurisdictions] tables
(CA-BC -> CA, British Columbia; US-DE -> US, Delaware); company_id = reg_key; slug = jurisdiction + registry number (e.g.
"ca_bc_a0000029"), or jurisdiction + "item" + the item id for records without a number yet. Only these fields are read; raw_records
(source registry data) are never read, so they can't reach the extractor.
Names keep their registry spelling, with runs of spaces and line breaks collapsed to one space. Slugs must be unique.

RESULT OUTPUTS
Result outputs.

A ResultSink receives, for each finished company: the verified result (the output), the clean-up result before the confirmed-only filter,
every claim with the verifier's decision, and the timing/cost record. close() receives the run summary.

Implemented
  json    OUT/<slug>/verified.json, final.json, decisions.json, timing.json and OUT/run_summary.json (the layout of E6 and of all
          earlier runs; evaluate/ reads it).
Planned (registered, not implemented): each would write the same content in another shape; entity_rows() below is the flat record
they should use, so every format carries the same fields.
  jsonl     one line per verified entity mention (entity_rows) in OUT/entities.jsonl
  csv       the same rows in OUT/entities.csv
  database  the same rows in a table (plus companies and runs tables), e.g. PostgreSQL or SQLite
Whatever the format, the working files in OUT/<slug>/ stay JSON (see above).
"""
from __future__ import annotations

import csv
import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .shared.models import Company, slugify


# ====================================================================================================
# COMPANY INPUTS
# ====================================================================================================

INPUT_IMPLEMENTED = {"jsonl", "csv", "tsv"}
INPUT_PLANNED = {"json": "a JSON array file", "database": "a database URL and query"}
INPUT_FORMATS = sorted(INPUT_IMPLEMENTED | set(INPUT_PLANNED))


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def company_from_corptrace(r: dict[str, Any], jurisdictions: dict[str, tuple[str, str]]) -> Company:
    jur = (r.get("jurisdiction") or "").strip().upper()
    if jur not in jurisdictions:
        raise ValueError(f"unknown jurisdiction {jur!r} in {r.get('reg_key')!r} (add it to a country profile's [jurisdictions])")
    country, region = jurisdictions[jur]
    reg_key = (r.get("reg_key") or "").strip()
    if r.get("id_value"):
        slug = slugify(f"{jur}_{r['id_value']}")
    else:
        if not reg_key:
            raise ValueError("record has neither id_value nor reg_key")
        slug = slugify(f"{jur}_item_{reg_key.rsplit('/', 1)[-1].rsplit(':', 1)[-1]}")
    return Company(name=_clean(r.get("legal_name")), province=region, slug=slug, company_id=reg_key,
                   entity_form=(r.get("entity_form") or "").strip(), country=country)


def company_from_row(r: dict[str, Any]) -> Company:
    return Company(name=_clean(r.get("name") or r.get("company_name")), province=_clean(r.get("province") or r.get("region")),
                   slug=(r.get("slug") or "").strip(), company_id=str(r.get("company_id") or r.get("id") or "").strip(),
                   entity_form=(r.get("entity_form") or "").strip(), country=(_clean(r.get("country")) or "CA").upper())


def detect(source: str) -> str:
    if "://" in source:
        return "database"
    ext = Path(source).suffix.lower()
    fmt = {".jsonl": "jsonl", ".ndjson": "jsonl", ".csv": "csv", ".tsv": "tsv", ".tab": "tsv", ".json": "json"}.get(ext)
    if fmt is None:                     # 2026-11-25: was silently read as CSV
        raise SystemExit(f"cannot tell the format of {source!r} from its extension {ext or '(none)'}: give --input-format "
                         f"({', '.join(sorted(INPUT_IMPLEMENTED))})")
    return fmt


def _read_jsonl(path: Path) -> list[Company]:
    from .shared.config import all_jurisdictions
    jurisdictions = all_jurisdictions()
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                out.append(company_from_corptrace(r, jurisdictions) if "legal_name" in r and "jurisdiction" in r else company_from_row(r))
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path.name} line {n}: {exc}") from exc
    return out


def _read_delimited(path: Path, delimiter: str) -> list[Company]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=delimiter))
    out = []
    for n, r in enumerate(rows, start=2):
        try:
            out.append(company_from_row(r))
        except ValueError as exc:
            raise ValueError(f"{path.name} line {n}: {exc}") from exc
    return out


READERS = {"jsonl": _read_jsonl, "csv": lambda p: _read_delimited(p, ","), "tsv": lambda p: _read_delimited(p, "\t")}


def open_companies(source: str, fmt: str = "auto") -> list[Company]:
    fmt = detect(source) if fmt == "auto" else fmt
    if fmt in INPUT_PLANNED:
        raise SystemExit(f"input format {fmt!r} ({INPUT_PLANNED[fmt]}) is planned but not implemented yet: add it in storage.py "
                         f"(implemented: {', '.join(sorted(INPUT_IMPLEMENTED))})")
    if fmt not in INPUT_IMPLEMENTED:
        raise SystemExit(f"unknown input format {fmt!r}; choices: auto, {', '.join(INPUT_FORMATS)}")
    companies = READERS[fmt](Path(source))
    dup = [s for s, k in Counter(c.slug for c in companies).items() if k > 1]
    if dup:
        clash = [f"{c.slug}: {c.name!r} ({c.company_id or 'no id'})" for c in companies if c.slug in set(dup[:5])]
        raise ValueError(f"{len(dup)} output folder names (slugs) are used by more than one company, e.g. {clash[:6]}. "
                         "Give a unique slug column (or company_id-based slugs) for these rows.")
    return companies


# ====================================================================================================
# RESULT OUTPUTS AND WORKING FILES
# ====================================================================================================

# Working files (every step's JSON in OUT/<slug>/): written atomically, whatever the output format. Moved here from steps/search.py
# (2026-11-25) so that no step imports from another step.
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (temporary file, then rename), so an interrupted run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


# The flat output record (one row per verified mention). Future CSV / database / JSON Lines sinks must use exactly these fields.
ENTITY_FIELDS = ["company_slug", "company_name", "province", "company_id", "entity_name", "entity_type", "other_spellings",
                 "relationship", "role", "evidence", "source_url", "page_title"]


def entity_rows(company, verified: dict[str, Any]) -> list[dict[str, Any]]:
    """verified.json -> one flat row per (entity, mention); every row has a source_url."""
    rows = []
    for typ, key in (("person", "people"), ("organization", "organizations")):
        for e in verified.get(key, []):
            for m in e.get("mentions", []):
                rows.append({"company_slug": company.slug, "company_name": company.name, "province": company.province,
                             "company_id": company.company_id or "", "entity_name": e["name"], "entity_type": typ,
                             "other_spellings": "; ".join(e.get("other_spellings", [])), "relationship": m.get("relationship") or "",
                             "role": m.get("role") or "", "evidence": m.get("evidence") or "", "source_url": m.get("source_url") or "",
                             "page_title": m.get("page_title") or ""})
    return rows


class ResultSink(Protocol):
    def write_company(self, company, verified: dict, final: dict, decisions: list[dict], timing: dict) -> None: ...
    def log_company(self, row: dict) -> None: ...          # one progress row per finished (or failed) company
    def close(self, summary: dict) -> None: ...


class JsonFileSink:
    """The current output: JSON files, one folder per company."""

    def __init__(self, out: Path):
        self.out = out

    def write_company(self, company, verified: dict, final: dict, decisions: list[dict], timing: dict) -> None:
        d = self.out / company.slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "verified.json").write_text(json.dumps(verified, ensure_ascii=False, indent=1))
        (d / "decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=1))
        (d / "timing.json").write_text(json.dumps(timing, indent=1))
        # final.json is a working file written by the clean-up step itself (steps/finalize.py), as in every earlier run

    def log_company(self, row: dict) -> None:
        """OUT/run_companies.jsonl: one line per finished or failed company (the caller holds the run's lock)."""
        with (self.out / "run_companies.jsonl").open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")

    def close(self, summary: dict) -> None:
        (self.out / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))


OUTPUT_IMPLEMENTED = {"json": JsonFileSink}
OUTPUT_PLANNED = {"jsonl": "OUT/entities.jsonl, one line per verified mention", "csv": "OUT/entities.csv, one row per verified mention",
           "database": "rows in a database table"}
OUTPUT_FORMATS = sorted(set(OUTPUT_IMPLEMENTED) | set(OUTPUT_PLANNED))


def open_sink(fmt: str, out: Path) -> ResultSink:
    if fmt in OUTPUT_PLANNED:
        raise SystemExit(f"output format {fmt!r} ({OUTPUT_PLANNED[fmt]}) is planned but not implemented yet: add a sink class in "
                         f"storage.py using entity_rows() (implemented: {', '.join(sorted(OUTPUT_IMPLEMENTED))})")
    if fmt not in OUTPUT_IMPLEMENTED:
        raise SystemExit(f"unknown output format {fmt!r}; choices: {', '.join(OUTPUT_FORMATS)}")
    return OUTPUT_IMPLEMENTED[fmt](out)
