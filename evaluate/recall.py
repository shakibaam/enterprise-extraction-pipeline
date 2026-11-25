"""Recall against the recall audit's gold entities (optional evaluation; free; the ONLY code in this package that reads gold).

Gold: enterprise_variant_d/runs/recall_240_20261115/eval/recall_like_audit.html (159 entities over 240 companies; 132 in the 76-company set
used for the E6-E14 comparison). Matching is recall_fixes_20261115/sshared.py's, copied: same company slug and the same normalized name
(honorifics and credentials removed from people's names, lower case, accents, punctuation, legal endings and "the" removed); for people also
the same last name and first initial; an output entity matches with its name or any of its other spellings.
Counts only companies that are in both the gold set and the output folder (and, with --slugs, in that list).
usage (from Provinces/): python -m on_the_fly_pipeline evaluate recall --out OUT [--slugs FILE] [--file verified.json]
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from ..shared.config import PROVINCES_DIR
from ..shared.names import match_name

GOLD = PROVINCES_DIR / "enterprise_variant_d/runs/recall_240_20261115/eval/recall_like_audit.html"
LEGAL = {"ltd", "limited", "inc", "incorporated", "corp", "corporation", "co", "company", "llp", "lp", "llc", "ulc", "plc", "gp", "ltee",
         "limitee", "incorporee", "enr", "cie", "the"}


def load_d(path: Path) -> dict:
    t = path.read_text(encoding="utf-8")
    i = t.index("const D=") + len("const D=")
    return json.JSONDecoder().raw_decode(t, i)[0]


def key(name: str, typ: str) -> tuple[str, ...]:
    n = match_name(name, "person" if typ in ("p", "person") else "organization")
    plain = "".join(c for c in unicodedata.normalize("NFKD", n) if not unicodedata.combining(c)).lower()
    return tuple(w for w in re.sub(r"[^a-z0-9]+", " ", plain).split() if w not in LEGAL)


def same(a: tuple, ta: str, b: tuple, tb: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    pa, pb = ta in ("p", "person"), tb in ("p", "person")
    return pa and pb and len(a) >= 2 and len(b) >= 2 and a[-1] == b[-1] and a[0][0] == b[0][0]


def entities(f: Path) -> list[dict]:
    if not f.is_file():
        return []
    fj = json.loads(f.read_text())
    return [{"type": t, "keys": [key(n, t) for n in [e["name"]] + e.get("other_spellings", [])], "name": e["name"]}
            for t, k in (("person", "people"), ("organization", "organizations")) for e in fj.get(k, [])]


def found(ents: list[dict], n: str, k: str) -> dict | None:
    kk = key(n, k)
    for x in ents:
        if any(same(kk, k, y, x["type"]) for y in x["keys"]):
            return x
    return None


def score(out: Path, slugs: list[str] | None = None, file: str = "verified.json") -> dict:
    R = load_d(GOLD)
    done = {p.name for p in out.iterdir() if (p / file).is_file()}
    use = done & set(slugs) if slugs else done
    gold = [{"s": c["s"], "n": e["n"], "k": e["k"]} for c in R["c"] for e in c["e"] if c["s"] in use]
    ents = {s: entities(out / s / file) for s in {g["s"] for g in gold}}
    hits = [g for g in gold if found(ents[g["s"]], g["n"], g["k"])]
    return {"companies_scored": len({g["s"] for g in gold}), "companies_in_output": len(done), "gold": len(gold), "found": len(hits),
            "recall": round(len(hits) / len(gold), 4) if gold else None,
            "missed": [{"slug": g["s"], "name": g["n"]} for g in gold if g not in hits]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recall of a pipeline output folder against the recall audit's gold entities (free).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slugs", default=None, help="file with the slugs to count (e.g. the 76-company comparison set)")
    ap.add_argument("--file", default="verified.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    res = score(out, Path(a.slugs).read_text().split() if a.slugs else None, a.file)
    (out / "eval").mkdir(exist_ok=True)
    (out / "eval" / "recall.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "missed"}, indent=1))
    return 0
