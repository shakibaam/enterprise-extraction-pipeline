"""Clean-up and the confirmed-only filter (one module since 2026-11-24: the clean-up of company_pipeline/pipeline/finalize.py, moved here
without its standalone runner, and the filter that keeps only verified mentions).

Clean-up (unchanged rules; original description below): drops items without a source URL, the company's own name, near-target items,
things that are not names, relation "other", entities linked only to another extracted person; canonicalizes and merges spellings; writes
final.json.
Claim ids (2026-11-24, review fix): every raw item that was sent to the verifier carries its claim_id; each mention keeps the ids of the
items it came from ("claim_ids"; equal mentions are merged and their ids combined, also when canonicalize merges aliases). confirmed_only
keeps a mention only if one of its claim ids was confirmed. Before this fix mentions were matched by (source URL, evidence) or (source URL,
name), which let an unverified relation of a confirmed person through (e.g. a confirmed "director" claim kept the same person's unverified
"owner" mention from the same page).

Original description (company_pipeline/pipeline/finalize.py):
Finalize component: clean the extracted items and write each company's final profile.

Input:  OUT/<slug>/extract.json (and search.json, rerank.json, select.json for the run details; pages/ for page titles).
Output: OUT/<slug>/final.json        the company's final profile (schema_version 1)
        OUT/final_companies.jsonl    every company's final profile, one per line (for loading the whole run at once)
        OUT/finalize_summary.json    run summary: companies by status, people, organizations, items dropped and why

Clean-up (post-processing), in this order; every dropped item is counted by reason:
  1. near_target: drop items the model linked only as "appears near the target" (lk = c), the filter of the validated pipeline;
  2. no_source_url: drop items whose page number was not in their batch (no exact source URL; every kept item has one);
  3. own_name: drop the company's own name (same name up to case, accents, punctuation and legal endings; for a bilingual name, either part);
  3b. not_a_name (since 2026-11-13): drop items that are not names of people or organizations: e-mail and web addresses, street
     addresses, and roles, groups or documents ("Administrative Agent", "Lenders", "Board of Directors", "Credit Agreement", "Debenture");
     a name with a legal ending (Ltd, Inc, Limited Partnership, ...) is always kept. On the corptrace sample 19 of 156 names were such
     items, none related (enterprise_variant_d/runs/company_pipeline_corptrace60_20261113/internal/whatif_filters.py);
  4. merge: items with the same type and the same name (same normalization) become one entity; every mention is kept with its own role,
     relationship, status, date, link, evidence quote and source URL, so nothing is lost and every claim keeps its page.
After merging ([finalize] switches, since 2026-11-13; measured offline in enterprise_variant_d/runs/precision_rules_20261113: on the second
corptrace sample strict precision 58.7% -> 74.0% together with the location check, recall unchanged on both samples and BC80):
  5. person_linked: drop an entity when every relationship given for it names another person extracted for the company and none mentions
     the target (its name, a part of a bilingual name, or the word "target"), e.g. "employer of Adrienne Leslie-Toogood", "organization
     Lorne Kyle directs": the owner's or a team member's own career, not the company's;
  6. abstain: output no people or organizations (status "abstained") when at least abstain_bad_share of the company's pages are bad: dropped
     by select's location check, or judged "other" or "unclear" by the model. Such a company's pages are mostly about someone else.
     (A third rule, "the evidence quote must name the target", was measured and rejected: recall 86% -> 40%.)
Precision switches added 2026-11-15 (all off by default; enterprise_variant_d/runs/precision_audit_20261115):
  drop_other_unnamed (change B, with prompts/extract_rel_v1.txt): before merging, drop items of relation type "other" whose evidence quote
     does not name the target (its short name as a phrase, any part of a bilingual name) - reason "other_unnamed";
  canonicalize (change C, steps/finalize.py): before merging, drop items that are not entities - redaction characters, building
     names, role titles and defined terms made only of generic words ("Acquisition Co.", "Independent Petroleum Engineer") - reasons
     "redacted", "building", "role_or_defined_term", "address"; after merging, aliases within a type are merged into one entity, all
     mentions kept (acronyms "BOM" = "Bank of Montreal", defined terms 'Bank of Montreal (the "Bank")', short forms contained in a longer
     name, a person's surname or initial form), and aliases of the target itself are dropped as own_name.
  drop_other (2026-11-15, with relation types): drop every item of relation type "other" (reason "other"), named target or not.
  restrict_not_about (2026-11-15, with [extract.identity_gate] ask_about): items from a page the gate marked "about": "no" (the page
     is about another organization and mentions the target on the side) are kept only when their evidence names the target - reason
     "not_about_unnamed". Sources carry "about_target".
  The identity gate (change A, pipeline/identity.py) runs in extract; its decisions (identity.json) are shown per source
  ("identity_gate") and the pages it dropped are listed in "pages_dropped_by_identity_gate".
Name normalization is the one the validated evaluation used (input_format_test_20261111/pages_unrelated.py: key()).

Final profile:
  {"schema_version": 1, "company": {...}, "status": ok | partial | failed | no_pages | not_extracted | abstained,
   "people": [entity], "organizations": [entity], "sources": [page], "dropped": {reason: count}, "run": {...}, "written_at": ...}
  entity = {"name", "roles": [distinct roles], "relationships": [distinct], "status": current | former | unknown (current if any mention says
            current, else former if any says former), "mentions": [{"role", "relationship", "status", "date", "link", "evidence",
            "source_url", "page_title"}]}
  page   = {"source_url", "title", "page_is_about" (the model's page judgement), "rerank_score", "document_type"}
  run    = search provider, rerank profile/model/threshold, extract profile/model, schema version.
Companies without extract.json get status not_extracted (and are reported). CPU only; no API calls.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from ..shared.foreign_pages import REASONS as LOCATION_REASONS
from ..shared.models import Company
from ..shared.names import defined_full_name, fold, match_name, name_variants, short_name
from ..storage import utc_now, write_json


# ====================================================================================================
# CANONICAL RULES (was shared/canonical.py; used only by this step)
# ====================================================================================================
# Canonicalization of extracted entities within one company ([finalize].canonicalize, change C, since 2026-11-15). String rules only.
#
# junk_reason(name, type): items that are not entities, dropped before merging:
#   "redacted"              redaction characters (▇ █ ■ ...) or [REDACTED];
#   "address"               P.O. boxes, suites and units, Canadian postal codes (street addresses are already not_a_name);
#   "building"              a building or property name (ends in Tower, Towers, Building or Plaza), without a
#                           legal ending or an organization word (society, association, church, hospital, ...);
#   "role_or_defined_term"  every word is a role title or a generic contract word: "President", "Registered Office", "Acquisition Co.",
#                           "Independent Petroleum Engineer", "the Purchaser" (a defined term with no proper name in it).
# canonicalize(merged, target_name): after merging identical names, per type:
#   - drop aliases of the target itself (a defined term or short form of the target's name, its acronym) as own_name;
#   - merge an alias into the entity it stands for, keeping every mention and the alias as another spelling:
#       defined term     the evidence of any mention defines it after another entity's name: 'Bank of Montreal (the "Bank")', ("BMO");
#       acronym          its letters are the initials of exactly one other entity ("BOM" = Bank of Montreal; with or without of/and/the);
#       short form       its words appear, in order and adjacent, in exactly one longer name of the same type ("Forest" in "Forest
#                        Lake Holdings Ltd."; not when its words are all generic);
#       person           same surname and the shorter form's given name or initial matches exactly one fuller name ("Smith",
#                        "J. Smith" -> "John Smith").

REDACTION = re.compile(r"[▇█■□▒░▓◼⬛▆▅▉▊▋▌▍▎▏]|\[redacted\]|\*{3,}|x{5,}", re.I)
ADDRESS = re.compile(r"\bp\.?\s?o\.?\s?box\b|\b(suite|unit)\s*#?\s*\d|(?<![A-Za-z0-9])[ABCEGHJKLMNPRSTVXY]\d[A-Z][ -]?\d[A-Z]\d(?![A-Za-z0-9])", re.I)
BUILDING_END = re.compile(r"\b(tower|towers|building|buildings|plaza)\s*$", re.I)   # not Centre, Place, Court, Square: real names end so
ORG_WORDS = re.compile(r"\b(society|association|club|church|council|foundation|institute|university|college|school|hospital|clinic|"
                       r"bank|trust|fund|group|partners|services|ministry|department|agency|authority|commission|board|federation|union|"
                       r"cooperative|co-op|league|network)\b", re.I)
LEGAL = re.compile(r"\b(ltd|limited|inc|incorporated|corp|corporation|llp|lp|llc|ulc|plc|ltee|limitee|gp)\b\.?", re.I)
# Role titles and generic contract words: a name made only of these is a role or a defined term, not an entity.
GENERIC = set("""
president vice chief executive officer officers director directors secretary treasurer chair chairman chairperson chairwoman ceo cfo coo cto
manager managers general partner partners founder cofounder owner owners trustee trustees agent agents registered records office board
committee member members shareholder shareholders stockholder stockholders auditor auditors accountant accountants counsel lawyer lawyers
solicitor solicitors engineer engineers independent senior managing assistant corporate legal principal principals employee employees
acquisition purchaser purchasers vendor vendors buyer buyers seller sellers issuer borrower borrowers lender lenders guarantor guarantors
parent subsidiary subsidiaries holdco newco amalco acquireco bidco opco propco finco mergeco mergersub amalgamated resulting petroleum
operator investor investors sponsor target offeror offeree transferee transferor assignee assignor licensee licensor landlord tenant
contractor consultant registrar transfer depositary escrow administrator nominee holder holders party parties company corporation corp co
inc ltd limited partnership the a an of and for to new old reporting closing bank fund trust plan holding holdings group
enterprises ventures investments
""".split())
STOP = {"of", "and", "the", "de", "du", "des", "la", "le", "et", "for"}
DEFINED = re.compile(r"\(\s*(?:the\s+|hereinafter(?:\s+referred\s+to\s+as)?\s+(?:the\s+)?|collectively,?\s+the\s+)?[\"“”']([^\"“”']{1,60})[\"“”']\s*\)", re.I)
SUFFIX_WORDS = {"ltd", "limited", "inc", "incorporated", "corp", "corporation", "co", "company", "llp", "lp", "llc", "ulc", "plc", "ltee",
                "limitee", "gp", "the"}


def toks(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower())


def key(s: str) -> list[str]:
    """Words without legal endings and 'the' (finalize's name_key, as a list)."""
    return [t for t in toks(s) if t not in SUFFIX_WORDS]


def junk_reason(name: str, typ: str) -> str:
    """'' or why this name is not an entity (see the module docstring)."""
    if REDACTION.search(name):
        return "redacted"
    if ADDRESS.search(name) and not LEGAL.search(name):
        return "address"
    if typ == "organization" and BUILDING_END.search(name.strip(" .,")) and not LEGAL.search(name) and not ORG_WORDS.search(name):
        return "building"
    t = toks(name)
    if t and all(w in GENERIC for w in t):
        return "role_or_defined_term"
    return ""


def names_target(text: str, target_name: str) -> bool:
    """True when the text names the target: its short name (any part of a bilingual name) as a phrase, case and punctuation ignored."""
    f = fold(text or "")
    for v in name_variants(target_name):
        s = short_name(v)
        if s and re.search(r"(?<![a-z0-9])" + r"[\W_]*".join(r"(?:and|&)" if t == "and" else re.escape(t) for t in s) + r"(?![a-z0-9])", f):
            return True
    return False


def _contiguous(a: list[str], b: list[str]) -> bool:
    return 0 < len(a) < len(b) and any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))


def _initials(words: list[str], skip_stop: bool) -> str:
    return "".join(w[0] for w in words if w and not (skip_stop and w in STOP))


def _is_acronym(alias: str, full: list[str]) -> bool:
    a = re.sub(r"[^a-z0-9]", "", alias.lower())
    return len(a) >= 2 and len(full) >= 2 and a in (_initials(full, False), _initials(full, True))


def _defined_antecedent(alias: str, entities: list[dict[str, Any]], texts: list[str]) -> dict[str, Any] | None:
    """The entity whose name comes right before a definition of `alias` ( (the "alias") ) in any evidence text."""
    ka = key(alias)
    for text in texts:
        for m in DEFINED.finditer(text):
            if key(m.group(1)) != ka:
                continue
            before = fold(text[max(0, m.start() - 160):m.start()])
            best, pos = None, -1
            for e in entities:
                for n in [e["name"]] + e.get("other_spellings", []):
                    s = short_name(n)
                    if not s or key(n) == ka:
                        continue
                    for mm in re.finditer(r"(?<![a-z0-9])" + r"[\W_]*".join(re.escape(t) for t in s) + r"(?![a-z0-9])", before):
                        if mm.end() > pos:
                            best, pos = e, mm.end()
            if best is not None:
                return best
    return None


def _target_alias(name: str, target_name: str, texts: list[str]) -> bool:
    """A short form, acronym or defined term of the target's own name."""
    ka = key(name)
    if not ka or all(w in GENERIC for w in ka):
        return False
    for v in name_variants(target_name):
        kt = key(v)
        if ka == kt or _contiguous(ka, kt) or _is_acronym(name, kt):
            return True
    for text in texts:                                  # 'SANOVEST ... LTD. (the "Company")' style definitions of the target
        for m in DEFINED.finditer(text):
            if key(m.group(1)) == ka and names_target(text[max(0, m.start() - 160):m.start()], target_name):
                return True
    return False


def _merge_into(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for m in src["mentions"]:                      # equal mentions are merged; their claim ids combined (2026-11-24)
        base = {k: v for k, v in m.items() if k != "claim_ids"}
        same = next((x for x in dst["mentions"] if {k: v for k, v in x.items() if k != "claim_ids"} == base), None)
        if same is None:
            dst["mentions"].append(m)
        else:
            same["claim_ids"] = sorted(set(same.get("claim_ids") or []) | set(m.get("claim_ids") or []))
    spellings = set(dst.get("other_spellings", [])) | set(src.get("other_spellings", [])) | {src["name"]}
    dst["other_spellings"] = sorted(s for s in spellings if s != dst["name"])
    for f in ("roles", "relationships"):
        seen = {(v or "").casefold() for v in dst.get(f, [])}
        dst[f] = dst.get(f, []) + [v for v in src.get(f, []) if (v or "").casefold() not in seen]
    statuses = {dst.get("status"), src.get("status")}
    dst["status"] = next((s for s in ("current", "former", "unknown") if s in statuses), "unknown")


def _person_match(short: list[str], full: list[str]) -> bool:
    """'smith' or 'j smith' or 'john smith' (shorter) within a fuller person name with the same surname."""
    if not short or not full or short[-1] != full[-1] or short == full or len(short) > len(full):
        return False
    if len(short) == 1:
        return len(full) > 1
    if len(short) == len(full):                       # "j smith" vs "john smith": the given name as an initial
        return len(short[0]) == 1 < len(full[0]) and short[0] == full[0][0] and short[1:-1] == full[1:-1]
    return short[0] == full[0] or (len(short[0]) == 1 and short[0] == full[0][0])


def canonicalize(merged: dict[str, list[dict[str, Any]]], target_name: str, norm: bool = False) -> Counter:
    """In place (see the module docstring); returns the number of entities dropped or merged, by reason. norm ([finalize].normalize_names,
    change E): people's names compared without honorifics and credentials."""
    out: Counter = Counter()
    texts = [m.get("evidence") or "" for t in merged for e in merged[t] for m in e["mentions"]]
    for typ in ("organization", "person"):
        ents = merged.get(typ, [])
        if typ == "organization":
            keep = [e for e in ents if not _target_alias(e["name"], target_name, texts)]
            out["own_name"] += len(ents) - len(keep)
            ents = keep
        changed = True
        while changed:                                  # merge one alias at a time, shortest names first
            changed = False
            for a in sorted(ents, key=lambda e: (len(key(e["name"])), len(e["name"]))):
                others = [e for e in ents if e is not a]
                ka = key(a["name"])
                if not ka:
                    continue
                if typ == "organization":
                    target = _defined_antecedent(a["name"], others, texts)
                    reason = "alias_defined_term"
                    if target is None:
                        acr = [e for e in others if _is_acronym(a["name"], key(e["name"]))] if len(ka) == 1 else []
                        if len(acr) == 1:
                            target, reason = acr[0], "alias_acronym"
                    if target is None and not all(w in GENERIC for w in ka):
                        sup = [e for e in others if _contiguous(ka, key(e["name"]))]
                        if len(sup) == 1:
                            target, reason = sup[0], "alias_short_form"
                else:
                    if norm:
                        ka = key(match_name(a["name"], "person"))
                    sup = [e for e in others if _person_match(ka, key(match_name(e["name"], "person") if norm else e["name"]))]
                    target, reason = (sup[0], "alias_person") if len(sup) == 1 else (None, "")
                if target is not None:
                    _merge_into(target, a)
                    ents = [e for e in ents if e is not a]
                    out[reason] += 1
                    changed = True
                    break
        merged[typ] = ents
    return out


# ====================================================================================================
# CLEAN-UP
# ====================================================================================================

SCHEMA_VERSION = 1
NAME_SUFFIXES = {"ltd", "limited", "inc", "incorporated", "corp", "corporation", "co", "company", "llp", "lp", "llc", "ulc", "the", "ltee", "limitee", "incorporee", "enr", "enrg", "cie"}
STATUS_ORDER = ("current", "former", "unknown")
NOT_A_NAME = re.compile(
    r"@|https?://|\bwww\.|[a-z0-9-]\.(com|ca|org|net|info|biz)(\.[a-z]{2})?\b"                      # e-mail and web addresses
    r"|^\s*\d+[a-z]?\s+[\w .'-]*\b(road|rd|street|st|avenue|ave|drive|dr|boulevard|blvd|way|lane|place|crescent|chemin|rue)\b"   # street addresses
    r"|\b(agreements?|debentures?|guarantees?|indentures?|properties|lenders?|borrowers?|administrative agent|issuing bank|"
    r"board of directors|relevant parties|subsidiaries|natural_persons)\b", re.I)
LEGAL_ENDING = re.compile(r"\b(ltd|limited|inc|incorporated|corp|corporation|company|llp|lp|llc|ulc|plc|partnership|co|ltee|limitee)\b\.?", re.I)


def is_not_a_name(name: str) -> bool:
    """An e-mail or web address, a street address, or a role, group or document (not a name), unless it has a legal ending."""
    return bool(NOT_A_NAME.search(name)) and not LEGAL_ENDING.search(name)


def name_key(name: str) -> tuple[str, ...]:
    """Name for comparing: no accents, lower case, words only, legal endings and 'the' removed (validated evaluation's key())."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)).lower()
    return tuple(t for t in re.sub(r"[^a-z0-9]+", " ", plain).split() if t not in NAME_SUFFIXES)


def clean_items(items: list[dict[str, Any]], company_name: str, fcfg: dict[str, Any] | None = None,
                page_texts: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], Counter]:
    """Steps 1 to 3b (and the item-level parts of the 2026-11-15 switches): the items that are kept, and the count of dropped items by reason.
    With normalize_names (change E) an organization defined as a short form in its evidence or page ('Falk Enterprises Ltd. (Falko)') is
    renamed to the full name first (the short form is kept in "defined_from"), and people's names are matched without honorifics and
    credentials (shared/names.py); the output keeps the model's spelling."""
    fcfg = fcfg or {}
    norm = bool(fcfg.get("normalize_names"))
    own = {name_key(v) for v in name_variants(company_name)}           # the full name and each part of a bilingual name
    kept, dropped = [], Counter()
    for x in items:
        mname = x.get("name") or ""
        if norm and mname.strip():
            if x.get("type") == "organization":
                full = defined_full_name(mname, [x.get("evidence") or "", (page_texts or {}).get(x.get("source_url") or "", "")])
                if full:
                    x = {**x, "name": full, "defined_from": mname}
            mname = match_name(x["name"], x.get("type") or "")
        if x.get("link") == "near_target":
            dropped["near_target"] += 1
        elif not x.get("source_url"):
            dropped["no_source_url"] += 1
        elif not (x.get("name") or "").strip():
            dropped["empty_name"] += 1
        elif name_key(mname) in own:
            dropped["own_name"] += 1
        elif is_not_a_name(mname):
            dropped["not_a_name"] += 1
        elif fcfg.get("canonicalize") and junk_reason(mname, x.get("type", "")):
            dropped[junk_reason(mname, x.get("type", ""))] += 1
        elif fcfg.get("drop_other") and (x.get("relationship") or "") == "other":
            dropped["other"] += 1
        elif fcfg.get("drop_other_unnamed") and (x.get("relationship") or "") == "other" and not names_target(x.get("evidence") or "", company_name):
            dropped["other_unnamed"] += 1
        else:
            kept.append(x)
    return kept, dropped


def mention_base(m: dict[str, Any]) -> dict[str, Any]:
    """A mention without its claim ids: two mentions are duplicates when everything else is equal."""
    return {k: v for k, v in m.items() if k != "claim_ids"}


def add_mention(mentions: list[dict[str, Any]], m: dict[str, Any]) -> None:
    """Append m, or, when an equal mention is already there, add m's claim ids to it (de-duplication as before, ids kept)."""
    base = mention_base(m)
    for x in mentions:
        if mention_base(x) == base:
            x["claim_ids"] = sorted(set(x.get("claim_ids", [])) | set(m.get("claim_ids", [])))
            return
    mentions.append(m)


def merge_entities(items: list[dict[str, Any]], titles: dict[str, str], norm: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Step 4: one entity per (type, normalized name), in order of first mention; all mentions kept. norm (change E): people's names
    compared without honorifics and credentials; resolved defined terms kept as other spellings."""
    groups: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for x in items:
        kn = match_name(x["name"], x["type"]) if norm else x["name"]
        k = (x["type"], name_key(kn) or (" ".join(kn.split()).lower(),))
        g = groups.setdefault(k, {"type": x["type"], "names": Counter(), "mentions": [], "aliases": set()})
        g["names"][" ".join(x["name"].split())] += 1
        if x.get("defined_from"):
            g["aliases"].add(" ".join(x["defined_from"].split()))
        mention = {"role": x.get("role"), "relationship": x.get("relationship"), "status": x.get("status"), "date": x.get("date"),
                   "link": x.get("link"), "evidence": x.get("evidence"), "source_url": x["source_url"], "page_title": titles.get(x["source_url"], "")}
        if x.get("defined_from"):
            mention["defined_from"] = x["defined_from"]
        add_mention(g["mentions"], {**mention, "claim_ids": [x["claim_id"]] if x.get("claim_id") is not None else []})
    out: dict[str, list[dict[str, Any]]] = {"person": [], "organization": []}
    for g in groups.values():
        ms = g["mentions"]
        statuses = {m["status"] for m in ms}
        out[g["type"]].append({
            "name": g["names"].most_common(1)[0][0],                      # the most frequent spelling
            "other_spellings": sorted((set(g["names"]) | g["aliases"]) - {g["names"].most_common(1)[0][0]}),
            "roles": distinct(m["role"] for m in ms), "relationships": distinct(m["relationship"] for m in ms),
            "status": next((s for s in STATUS_ORDER if s in statuses), "unknown"), "mentions": ms})
    return out


def distinct(values: Any) -> list[str]:
    """Distinct non-empty values in first-seen order, ignoring case and extra spaces."""
    seen: dict[str, str] = {}
    for v in values:
        k = " ".join((v or "").split()).casefold()
        if k and k not in seen:
            seen[k] = v
    return list(seen.values())


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except json.JSONDecodeError:
        return None


def person_linked(relationships: list[str | None], other_people: list[str], company_name: str) -> bool:
    """True when every relationship names another extracted person and none mentions the target."""
    if not relationships:
        return False
    target = {" ".join(short_name(v)) for v in name_variants(company_name)} - {""}
    people = [fold(p) for p in other_people if p and p.strip()]
    for r in relationships:
        f = fold(r or "")
        words = " ".join(re.findall(r"[a-z0-9]+", f))
        if "target" in f or any(t in words for t in target):
            return False
        if not any(p in f for p in people):
            return False
    return True


def drop_person_linked(merged: dict[str, list[dict[str, Any]]], company_name: str) -> int:
    """Step 5 in place; returns the number of entities dropped."""
    people = [e["name"] for e in merged["person"]]
    n = 0
    for t in ("person", "organization"):
        keep = []
        for e in merged[t]:
            others = [p for p in people if p != e["name"]]
            if person_linked([m["relationship"] for m in e["mentions"]], others, company_name):
                n += 1
            else:
                keep.append(e)
        merged[t] = keep
    return n


def finalize_company(out_root: Path, company: Company, fcfg: dict[str, Any] | None = None) -> dict[str, Any]:
    fcfg = fcfg or {}
    d = out_root / company.slug
    ex, rr, sel, srch = (read_json(d / f) for f in ("extract.json", "rerank.json", "select.json", "search.json"))
    titles: dict[str, str] = {}
    sources = []
    scores = {p["result_id"]: p["score"] for p in (sel or {}).get("kept", [])}
    page_texts: dict[str, str] = {}
    for k in (sel or {}).get("kept", []):
        page = read_json(d / "pages" / f"{k['result_id']}.json") or {}
        titles[k["url"]] = page.get("title") or ""
        if fcfg.get("normalize_names"):
            page_texts[k["url"]] = page.get("content") or ""
        sources.append({"source_url": k["url"], "title": titles[k["url"]], "page_is_about": None, "rerank_score": scores.get(k["result_id"]),
                        "document_type": page.get("document_type") or None})
    gate = read_json(d / "identity.json")
    gate_dropped = []
    v2 = gate is not None and any("page_type" in g for g in gate.get("pages", []))     # page type first: read its single page decision
    if gate is not None:                                # change A ran: only the pages it passed are sources
        rule = {g["url"]: g["rule"] for g in gate.get("pages", [])}
        gkeep = {g["url"]: bool(g.get("keep")) for g in gate.get("pages", [])}
        gmode = {g["url"]: g.get("mode") for g in gate.get("pages", [])}
        about = {g["url"]: g.get("about") for g in gate.get("pages", []) if "about" in g}
        for s in sources:
            s["identity_gate"] = rule.get(s["source_url"])
            if about:
                s["about_target"] = about.get(s["source_url"])
        if v2:
            for s in sources:
                s["page_decision"] = gmode.get(s["source_url"])
            gate_dropped = [{"source_url": s["source_url"], "rule": s["identity_gate"]} for s in sources
                            if s["identity_gate"] is not None and not gkeep.get(s["source_url"])]
            sources = [s for s in sources if s.get("identity_gate") is None or gkeep.get(s["source_url"])]
        else:
            gate_dropped = [{"source_url": s["source_url"], "rule": s["identity_gate"]} for s in sources
                            if s["identity_gate"] is not None and s["identity_gate"] not in ("registry_id", "llm_yes")]
            sources = [s for s in sources if s.get("identity_gate") in (None, "registry_id", "llm_yes")]
    if ex is None:
        status, items, dropped = "not_extracted", [], Counter()
    else:
        status = ex["status"]
        verdicts = {v["url"]: v["verdict"] for v in ex.get("page_verdicts", []) if v.get("url")}
        for s in sources:
            s["page_is_about"] = verdicts.get(s["source_url"])
        items, dropped = clean_items(ex.get("entities", []), company.name, fcfg, page_texts)
        if fcfg.get("restrict_not_about") and gate is not None:
            side = {g["url"] for g in gate.get("pages", []) if g.get("keep") and g.get("about") not in (None, "yes")}
            n0 = len(items)
            items = [x for x in items if x.get("source_url") not in side or names_target(x.get("evidence") or "", company.name)]
            if n0 > len(items):
                dropped["not_about_unnamed"] += n0 - len(items)
    if fcfg.get("normalize_names"):
        n_def = sum(1 for x in items if x.get("defined_from"))
        if n_def:
            dropped["defined_term_resolved"] = n_def           # not dropped: renamed to the full name they define (counted here)
    merged = merge_entities(items, titles, bool(fcfg.get("normalize_names")))
    if fcfg.get("canonicalize", False):
        for reason, n in canonicalize(merged, company.name, bool(fcfg.get("normalize_names"))).items():
            dropped[reason] += n
    if fcfg.get("person_linked", False):
        n = drop_person_linked(merged, company.name)
        if n:
            dropped["person_linked"] += n
    share = float(fcfg.get("abstain_bad_share", 0) or 0)
    if share > 0 and ex is not None and v2:             # page type first: only "the name is something else" pages and pages about
        gp = gate.get("pages", [])                        # a company that is not the target count as bad pages
        bad = sum(1 for g in gp if g.get("page_type") == "d" or (g.get("page_type") == "a" and g.get("identity") == "no"))
        total = len(gp)
    elif share > 0 and ex is not None:
        located = [p for p in (sel or {}).get("dropped", []) if p.get("reason") in LOCATION_REASONS]
        bad = len(located) + sum(1 for s in sources if s["page_is_about"] in ("other", "unclear"))
        total = len(sources) + len(located)
    if share > 0 and ex is not None:
        if total and bad >= share * total and (merged["person"] or merged["organization"]):
            dropped["abstained"] += len(merged["person"]) + len(merged["organization"])
            merged = {"person": [], "organization": []}
            status = "abstained"
    return {
        "schema_version": SCHEMA_VERSION, "company": company.__dict__, "status": status,
        "people": merged["person"], "organizations": merged["organization"], "sources": sources, "dropped": dict(dropped),
        **({"pages_dropped_by_identity_gate": gate_dropped} if gate is not None else {}),
        "run": {"search_provider": (srch or {}).get("provider"), "rerank_profile": (rr or {}).get("profile"), "rerank_model": (rr or {}).get("model"),
                "rerank_threshold": (sel or {}).get("threshold"), "extract_profile": (ex or {}).get("profile"), "extract_model": (ex or {}).get("model")},
        "written_at": utc_now(),
    }


def finalize_one(out_root: Path, c: Company, fcfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write one company's final.json; returns the profile."""
    res = finalize_company(out_root, c, fcfg)
    write_json(out_root / c.slug / "final.json", res)
    return res


def confirmed_only(final: dict, claims: list[dict]) -> dict:
    """Keep, in each entity, the mentions with at least one confirmed claim id; drop entities left with none -> verified.json."""
    conf = {x["id"] for x in claims if x.get("final") == "confirmed"}
    out = {**final}
    for key in ("people", "organizations"):
        keep = []
        for e in final.get(key, []):
            ms = [m for m in e["mentions"] if conf & set(m.get("claim_ids") or [])]
            if ms:
                keep.append({**e, "mentions": ms, "relationships": sorted({m.get("relationship") for m in ms if m.get("relationship")}),
                             "roles": sorted({m.get("role") for m in ms if m.get("role")})})
        out[key] = keep
    return out


def finalize_and_filter(out_root: Path, c, fcfg: dict, claims: list[dict]) -> tuple[dict, dict]:
    """Run clean-up (it writes the working file final.json) and the confirmed-only filter. Returns (final, verified); the output sink
    (storage/outputs.py) writes verified and the decisions."""
    final = finalize_one(out_root, c, fcfg)
    return final, confirmed_only(final, claims)
