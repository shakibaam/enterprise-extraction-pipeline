"""Name matching and normalization (one module since 2026-11-24; was names.py + name_norm.py).

NAME MATCHING (was names.py)
Company-name tests: does a page name the company? Used by rerank (pages that don't name the company are not scored) and select.

A page names the company if one of these holds, for the full name or any part of a bilingual name:
  - "phrase": the company's short name (legal ending such as Ltd./Inc./Ltée removed) appears as a phrase in the page title or text: its
    words in order, separated only by spaces or punctuation ("0833009 B.C." also matches "0833009B.C."; "&" and "and" are the same);
    case and accents ignored;
  - "glued": the short name with spaces and punctuation removed is in the title + text with spaces and punctuation removed (catches
    registry notices that glue the incorporation number to the name; glued() of
    enterprise_variant_d/runs/onestage9b_qwen3rr_namecheck_bc80_20261111/make_pages.py);
  - "host": the name's first distinctive word (not and/the, over 2 letters) is in the website's host name (the company's own site).
    With host_rule = "words" ([select].host_rule, since 2026-11-13; default "first_word" = the rule above), the name's other words must
    also appear in the host, the URL path, the title or the text (generic words such as Services, Association, Corporation optional):
    "snow" alone matched every page of snowlake.ca for SNOW LAKE SPORTSMAN ASSOCIATION (41 unrelated names), "arc" matched archives.gnb.ca
    for L'ARC-EN-CIEL DU RESTIGOUCHE; shell.ca (Shell Lubricants, "lubricants" in the path) and asmlaw.ca (ASM Professional Law) still pass
    (enterprise_variant_d/runs/investigate_host_and_pages_20261113).
Until 2026-11-13 the first test only asked that every word of the short name appear somewhere on the page ("words", mentions() of
enterprise_variant_d/runs/bc30_arctic_cos028_pagefilter_20261110/page_filter.py). On BC80 at threshold 0.8 that let one page through that
never names the company (TransLink on Wikipedia for FLOOR SAFETY NETWORK VANCOUVER ISLAND: each word appears somewhere in 68k characters),
which brought 36 unrelated names; it was the only kept page without the name as a phrase
(enterprise_variant_d/runs/long_page_windows_bc80_20261113/REPORT.md).

NAME NORMALIZATION (was name_norm.py)
Name normalization before the clean-up rules ([finalize].normalize_names, change E, off by default; since 2026-11-15).

match_name(name, type): the name used for MATCHING only (own name, not-a-name and defined-term rules, merging, aliases, the relation
check's ENTITY); the output keeps the spelling the model gave. For people, honorifics and religious, judicial or courtesy titles in
front ("Canon", "Dr.", "Mr.", "Hon.", "Me", ...) and professional credentials and post-nominals ("CPA", "K.C.", "Q.C.", "P.Eng.",
"MBA", "AACI", "Ph.D.", ...) are removed, also credentials scraped in front of the name ("Aaci Gordon Daman"). "Jr." and "Sr." are part
of the name and kept. Organization names are not changed.

defined_full_name(name, texts, target): the full name that a short form or defined term stands for, from a definition in the evidence
or the page text: 'Falk Enterprises Ltd. (Falko)', 'J.M. Enterprises Ltd. ("Enterprises")', 'Bank of Montreal (the "Bank")',
'... Ltd. (hereinafter "Tenco")'. The full name must end in a legal ending (Ltd, Inc, Corp, LLP, ...) unless the definition is quoted.
finalize renames such an item to the full name (the short form is kept as another spelling) instead of dropping it as a role or
defined term; a short form of the target itself becomes the target's name and is dropped as own_name.
"""
from __future__ import annotations

import re
import unicodedata


# ====================================================================================================
# NAME MATCHING (was names.py)
# ====================================================================================================

# Legal endings removed from the end of the name (the "words" test; same list as fix_reference.py).
SUFFIXES = {"ltd", "limited", "inc", "incorporated", "corp", "corporation", "co", "company", "llp", "lp", "llc", "ulc", "gp", "plc",
            "ltee", "limitee", "incorporee", "enr", "enrg", "cie"}   # French legal endings (NB, MB bilingual names)
# Legal ending removed for the "glued" test (same pattern as make_pages.py).
GLUED_SUFFIX = re.compile(r"[,.\s]*(ltd|limited|inc|incorporated|corp|corporation|co|company)\.?\s*$", re.I)


# ---------------------------------------------------------------- name tests

def words(text: str) -> list[str]:
    """Lower-case ASCII words; '&' becomes 'and'."""
    t = re.findall(r"[^\W_]+|&", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower())
    return ["and" if x == "&" else x for x in t]


def short_name(name: str) -> list[str]:
    t = words(name)
    while len(t) > 1 and t[-1] in SUFFIXES:
        t = t[:-1]
    return t


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


# ---------------------------------------------------------------- bilingual and multi-part names

# A registry name can hold the same name in two languages, e.g. "SOUTHEAST FISHERIES LTD. / PECHERIES DU SUD-EST LTEE" or
# 'STUDIO "S" LTD. - STUDIO "S" LTEE'. Pages usually use only one of the parts.
PART_ENDING = re.compile(r"\b(ltd|ltee|limited|limitee|inc|incorporated|incorporee|corp|corporation|enr|enrg|co|company|cie|association)\.?\)?$", re.I)


def _significant(part: str) -> int:
    return sum(1 for t in short_name(part) if len(t) > 1 or t.isdigit())


def name_parts(name: str) -> list[str]:
    """The parts of a multi-part name, or [name] if it has one part.
    Split on "/" when every part has at least 2 significant words ("24/7 PLUMBING" and "A/C REPAIR" stay whole), and on " - " when
    both sides end with a legal ending ("STUDIO S LTD. - STUDIO S LTEE"; "SEKAI KYUSEI KYO - CHURCH OF ..." stays whole)."""
    name = " ".join(name.split())
    parts = [name]
    dash = [p.strip() for p in re.split(r"\s+-\s*|\s*-\s+", name)]      # a dash with a space on at least one side ("CAN-WEST" stays)
    if len(dash) > 1 and all(PART_ENDING.search(p) for p in dash):
        parts = dash
    out: list[str] = []
    for p in parts:
        slash = [x.strip() for x in p.split("/")]
        out += slash if len(slash) > 1 and all(_significant(x) >= 2 for x in slash) else [p]
    return out


def search_name(name: str) -> str:
    """The name to put inside the quoted search phrase: the first part of a multi-part name, with spaces collapsed and double quotes
    removed (a name like DOUBLE "D" DISTRIBUTORS INC. would otherwise end the phrase early)."""
    return " ".join(name_parts(name)[0].replace('"', " ").split())


def name_variants(name: str) -> list[str]:
    """The full name and, for a multi-part name, each part."""
    parts = name_parts(name)
    full = " ".join(name.split())
    return [full] + [p for p in parts if p != full]


# ---------------------------------------------------------------- matching

def fold(text: str) -> str:
    """Lower-case ASCII (accents removed), for matching."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


# Words a page or address need not show for the "words" host rule.
GENERIC_WORDS = {"services", "service", "association", "society", "corporation", "corp", "company", "co", "group", "holdings", "enterprises",
                 "enterprise", "limited", "ltd", "inc", "incorporated", "international", "canada", "of", "and", "the", "professional",
                 "consulting", "management", "investments", "solutions", "systems", "industries", "partners", "partnership"}
HOST_RULES = ("first_word", "words")


class _OneName:
    """The name tests for one name, prepared once."""

    def __init__(self, name: str, host_rule: str = "first_word"):
        if host_rule not in HOST_RULES:
            raise ValueError(f"host_rule must be one of {HOST_RULES}, not {host_rule!r}")
        self.host_rule = host_rule
        s = short_name(name)
        parts = [r"(?:and|&)" if t == "and" else re.escape(t) for t in s]
        self.phrase = re.compile(r"(?<![a-z0-9])" + r"[\W_]*".join(parts) + r"(?![a-z0-9])") if parts else None
        want = [t for t in s if len(t) > 1 or t.isdigit()] or s
        self.first = next((t for t in want if t not in {"and", "the"} and len(t) > 2), want[0] if want else "")
        self.glued_core = squash(GLUED_SUFFIX.sub("", name.strip()))
        self.rest = [t for t in want if t != self.first and t not in GENERIC_WORDS]

    def match(self, folded: str, squashed: str, host: str, page_words) -> str:
        if self.phrase is not None and self.phrase.search(folded):
            return "phrase"
        if self.glued_core and self.glued_core in squashed:
            return "glued"
        if self.first and self.first in host:
            if self.host_rule == "first_word" or all(t in host or t in page_words() for t in self.rest):
                return "host"
        return ""


class NameMatcher:
    """Does a page name the company? Tries the full name, then each part of a multi-part (e.g. bilingual) name."""

    def __init__(self, name: str, host_rule: str = "first_word"):
        self.variants = name_variants(name)
        self._matchers = [_OneName(v, host_rule) for v in self.variants]

    def match(self, title: str, content: str, url: str) -> str:
        """'phrase', 'glued', 'host' or '' (no match)."""
        text = f"{title or ''} {content or ''}"
        folded, squashed = fold(text), squash(fold(text))
        bare = re.sub(r"^https?://", "", url or "")
        host = squash(bare.split("/")[0])
        cache: list[set[str]] = []

        def page_words() -> set[str]:          # words of the title, text and URL path (built only if a host rule needs them)
            if not cache:
                path = fold(bare.partition("/")[2]).replace("-", " ").replace("_", " ")
                cache.append(set(re.findall(r"[a-z0-9]+", f"{folded} {path}")))
            return cache[0]

        for m in self._matchers:
            how = m.match(folded, squashed, host, page_words)
            if how:
                return how
        return ""


# ====================================================================================================
# NAME NORMALIZATION (was name_norm.py)
# ====================================================================================================

HONORIFICS = {"mr", "mrs", "ms", "miss", "mx", "dr", "doctor", "prof", "professor", "rev", "revd", "reverend", "rt", "very", "canon",
              "father", "fr", "pastor", "rabbi", "imam", "bishop", "archbishop", "sister", "brother", "hon", "honourable", "honorable",
              "the", "sir", "dame", "lord", "lady", "judge", "justice", "mme", "mlle", "me", "maitre", "maître", "senator", "capt",
              "captain", "col", "colonel", "major", "lt", "sgt"}
# Credentials and post-nominals (compared without dots, spaces or case: "K.C." = "kc", "P. Eng." = "peng").
CREDENTIALS = {"cpa", "ca", "cga", "cma", "fcpa", "fca", "fcga", "fcma", "kc", "qc", "peng", "eng", "ing", "pgeo", "pag", "mba", "phd",
               "md", "llb", "llm", "jd", "cfa", "cfp", "cim", "aaci", "papp", "mrics", "frics", "ba", "bsc", "bcomm", "bcom", "ma", "msc",
               "rn", "dds", "dmd", "icdd", "cpaca", "cpacma", "cpacga", "chrp", "chrl", "pmp", "cic", "cfe", "cpaicd", "fcia",
               "fsa", "acia", "lpn", "rpn", "frcpc", "frcsc", "ccfp", "mcfp", "obe", "cm", "oc", "aoe", "csc", "dagr", "dipagr", "bed",
               "med", "edd", "bfa", "mfa", "barrister", "solicitor", "notary", "esq"}
KEEP = {"jr", "sr", "ii", "iii", "iv"}
TOKEN = re.compile(r"[^\s,]+")
LEGAL_END = re.compile(r"\b(ltd|limited|inc|incorporated|corp|corporation|co|company|llp|lp|llc|ulc|plc|ltee|limitee|ltée|limitée|gp|"
                       r"partnership|cie)\.?$", re.I)


def _plain(tok: str) -> str:
    return re.sub(r"[^a-z]", "", tok.lower().replace("é", "e").replace("è", "e"))


def match_name(name: str, typ: str = "person") -> str:
    """The name without honorifics in front and credentials after (and credentials scraped in front), for people; else the name."""
    name = " ".join((name or "").split())
    if typ != "person" or not name:
        return name
    toks = TOKEN.findall(name)
    while len(toks) > 2 and _plain(toks[0]) in HONORIFICS:
        toks = toks[1:]
    while len(toks) > 2 and _plain(toks[0]) in CREDENTIALS and len(_plain(toks[0])) > 2:      # "Aaci Gordon Daman"; not "Ca", "Ma"
        toks = toks[1:]
    while len(toks) > 2 and _plain(toks[-1]) in CREDENTIALS and _plain(toks[-1]) not in KEEP:
        toks = toks[:-1]
    while len(toks) > 2 and re.fullmatch(r"[A-Z]\.?[A-Z]\.?[A-Z]?\.?", toks[-1]) and _plain(toks[-1]) in CREDENTIALS:
        toks = toks[:-1]
    out = " ".join(toks).strip(" ,;")
    return out or name


def _key(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if t not in {"the", "ltd", "limited", "inc", "corp", "co"}]


DEF_QUOTED = re.compile(r"\(\s*(?:the\s+|hereinafter(?:\s+(?:referred\s+to\s+as|called))?\s+(?:the\s+)?|collectively,?\s+the\s+)?"
                        r"[\"“”'‘’]([^\"“”'‘’()]{1,60})[\"“”'‘’]\s*\)", re.I)
DEF_BARE = re.compile(r"\(\s*([A-Z][\w&.'’-]*(?:\s+[A-Z][\w&.'’-]*){0,3})\s*\)")
NAME_WORD = r"(?:[A-Z0-9][\w&.'’-]*|\([A-Za-z0-9][\w.&'’ -]{0,20}\)|of|and|&|de|du|des|la|le|et|the)"
NAME_BEFORE = re.compile(r"(%s(?:[\s,]+%s)*)[\s,]*$" % (NAME_WORD, NAME_WORD))
# Where one name ends and the next begins in a list: after a closing parenthesis or a legal ending, a comma, "and" or "&".
LIST_SEP = re.compile(r"(?:\)|\b(?:ltd|limited|inc|incorporated|corp|corporation|llp|lp|llc|ulc|co|company|ltee|limitee)\.?)\s*(?:,|\band\b|&)\s+", re.I)


def _antecedent(text: str, start: int, need_legal: bool) -> str:
    """The name just before a definition's opening parenthesis."""
    m = NAME_BEFORE.search(text[max(0, start - 160):start])
    if not m:
        return ""
    words = LIST_SEP.split(m.group(1).strip(" ,"))[-1].split()
    while words and (words[0].lower() in {"of", "and", "&", "de", "du", "des", "la", "le", "et", "the"} or words[0].startswith("(")):
        words = words[1:]
    full = " ".join(words).strip(" ,")
    if not full or len(full) < 3 or not _key(full):                  # a legal ending alone ("Ltd.") is not a name
        return ""
    if need_legal and not LEGAL_END.search(full):
        return ""
    if not LEGAL_END.search(full) and len(words) > 6:
        return ""
    return full


def defined_full_name(name: str, texts: list[str]) -> str:
    """The full name that `name` is defined as an abbreviation of in one of the texts, or ''."""
    ka = _key(name)
    if not ka or len(ka) > 4:
        return ""
    for text in texts:
        if not text:
            continue
        for rx, need_legal in ((DEF_QUOTED, False), (DEF_BARE, True)):
            for m in rx.finditer(text):
                if _key(m.group(1)) != ka:
                    continue
                full = _antecedent(text, m.start(), need_legal)
                if full and _key(full) != ka:
                    return full
    return ""
