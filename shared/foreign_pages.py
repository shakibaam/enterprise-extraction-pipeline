"""Is a page about a company in another country? (one module since 2026-11-24; was sites.py + location.py). Both rules take their
settings from the country profile.

FOREIGN SITES (was sites.py): rerank does not score, and select drops, pages on websites about other countries' companies
Foreign-site test: is a page on a website about companies outside Canada? Used by rerank (such pages are not scored) and select (they
are dropped, reason "foreign_site"), when [select.foreign_sites].enabled.

Why: a company name is often shared by companies in other countries. On the 60-company corptrace sample (2026-11-13), 47 of the 91 names
our run extracted that were clearly unrelated came from pages about a same-name company elsewhere (UK Companies House, New Zealand and US
registries, US trademarks), and 0 related names did; dropping those pages raised strict precision from 41.7% to 58.0% with the same
recall (enterprise_variant_d/runs/company_pipeline_corptrace60_20261113/internal/whatif_filters.py).

A page is foreign when its URL or final URL (after redirects):
  - ends in a non-Canadian country-code domain ([select.foreign_sites].country_domains, e.g. ".uk", ".nz", ".au");
  - is on a listed foreign registry or company directory ([select.foreign_sites].hosts; a host also matches its subdomains);
  - has a foreign country as a whole path segment ([select.foreign_sites].path_countries, e.g. govintel.ai/united-kingdom/...,
    bbb.org/us/..., mapquest.com/us/...: sites covering many countries; "in" is not listed, it is LinkedIn's profile path); or
  - is an OpenCorporates company page outside Canada (/companies/<jurisdiction> not starting with "ca").
The lists are in the config so they can be extended. US federal sites such as sec.gov are not listed: SEC filings can be about Canadian
companies (Whistler Networks on BC80).

LOCATION CHECK (was location.py): select drops pages that place the business abroad (or in another province)
Location check: does a page place its business outside Canada, or (for a sole proprietorship) only in another province? Used by select
(pages dropped with reason "foreign_location" or "other_province") when [select.location].enabled, and by finalize's abstain rule.

Why: same-name companies elsewhere on generic .com sites (which the foreign-site test, shared/foreign_pages.py, can't see from the URL) were the
largest source of unrelated names on the second corptrace sample (Beacon Communications: 28 names of a US company). Measured offline on
both corptrace samples and BC80 (enterprise_variant_d/runs/location_check_20261113, ../precision_rules_20261113): sample 2 strict precision
40.9% -> 58.7%, sample 1 unchanged, recall unchanged on both; on BC80 6 kept pages dropped, all foreign, no reference entity lost.

Signals in a page's title + text:
  foreign: US addresses ("SIMPSONVILLE, SC 29681", weight 3), US state names (ambiguous ones such as Washington, Georgia, Victoria left
           out), UK postcodes (weight 2), "United States", "USA", "United Kingdom", "Companies House", "LLC", ... (weight 1);
  Canadian: Canadian postal codes (the first letter gives the province), province names and abbreviations, "Canada", "Canadian".
A page is dropped ("foreign_location") when its foreign score is >= foreign_min and it has no Canadian signal, or when the foreign score is
>= overwhelming_min and >= overwhelming_ratio times the Canadian count. With same_site, every other kept page of a website that has a dropped
page is dropped too (same reason), unless one of that website's pages names the company's province. For a sole proprietorship (it can't be registered in
another province), a page whose only Canadian signals name one other province and that has no foreign signal is dropped ("other_province");
for other companies this is not done (corporations are often registered in several provinces: applied to all, it removed related names).
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit
import re


# ====================================================================================================
# FOREIGN SITES (was sites.py): rerank does not score, and select drops, pages on websites about other countries' companies
# ====================================================================================================

_OPENCORP = re.compile(r"^/companies/([a-z]{2})(?:_[a-z]+)?(?:/|$)", re.I)


class SiteFilter:
    def __init__(self, conf: dict[str, Any] | None):
        conf = conf or {}
        self.enabled = bool(conf.get("enabled", False))
        self.domains = tuple(d.lower() if d.startswith(".") else "." + d.lower() for d in conf.get("country_domains", []))
        self.hosts = tuple(h.lower().removeprefix("www.") for h in conf.get("hosts", []))
        self.opencorporates = bool(conf.get("opencorporates_outside_canada", True))
        self.opencorporates_home = str(conf.get("opencorporates_home", "ca")).lower()[:2]   # country profile (2026-11-24); was fixed "ca"
        self.path_countries = frozenset(s.lower().strip("/") for s in conf.get("path_countries", []))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SiteFilter":
        return cls(cfg.get("select", {}).get("foreign_sites"))

    def foreign_url(self, url: str) -> str:
        """The rule that makes this URL foreign ("domain", "host", "path", "opencorporates") or ''."""
        if not url:
            return ""
        u = urlsplit(url if "://" in url else "http://" + url)
        host = (u.hostname or "").lower().removeprefix("www.")
        if not host:
            return ""
        if host.endswith(self.domains):
            return "domain"
        if any(host == h or host.endswith("." + h) for h in self.hosts):
            return "host"
        if self.path_countries and any(seg.lower() in self.path_countries for seg in u.path.split("/") if seg):
            return "path"
        if self.opencorporates and host == "opencorporates.com":
            m = _OPENCORP.match(u.path)
            if m and m.group(1).lower() != self.opencorporates_home:
                return "opencorporates"
        return ""

    def is_foreign(self, page: dict[str, Any]) -> bool:
        """True if the filter is on and the page's URL or final URL is foreign."""
        if not self.enabled:
            return False
        return any(self.foreign_url(page.get(k) or "") for k in ("url", "final_url"))

    def settings(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "domains": list(self.domains), "hosts": list(self.hosts), "opencorporates": self.opencorporates,
                "path_countries": sorted(self.path_countries)}


# ====================================================================================================
# LOCATION CHECK (was location.py): select drops pages that place the business abroad (or in another province)
# ====================================================================================================

US_STATES = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut",
             "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
             "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
             "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
             "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
             "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
             "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
             "WY": "Wyoming", "DC": "District of Columbia"}
AMBIGUOUS = {"Washington", "Georgia", "Indiana", "Maine", "Delaware", "Nevada", "Montana", "Vermont", "Virginia"}
STATE_NAMES = re.compile(r"\b(" + "|".join(re.escape(n) for n in US_STATES.values() if n not in AMBIGUOUS) + r")\b")
US_ADDRESS = re.compile(r"\b(" + "|".join(US_STATES) + r")\s+\d{5}(?:-\d{4})?\b")
FOREIGN_PHRASE = re.compile(r"\b(United States|U\.S\.A\.|USA|United Kingdom|England and Wales|Scotland|Australia|New Zealand|"
                            r"Companies House|Delaware corporation|Nevada corporation|Florida corporation|LLC)\b")
UK_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s\d[ABD-HJLNP-UW-Z]{2}\b")
CA_POSTCODE = re.compile(r"\b([ABCEGHJKLMNPRSTVXY])\d[ABCEGHJ-NPRSTV-Z][ -]?\d[ABCEGHJ-NPRSTV-Z]\d\b")
POSTAL_FIRST = {"A": "Newfoundland and Labrador", "B": "Nova Scotia", "C": "Prince Edward Island", "E": "New Brunswick", "G": "Quebec",
                "H": "Quebec", "J": "Quebec", "K": "Ontario", "L": "Ontario", "M": "Ontario", "N": "Ontario", "P": "Ontario", "R": "Manitoba",
                "S": "Saskatchewan", "T": "Alberta", "V": "British Columbia", "X": "Northwest Territories", "Y": "Yukon"}
PROVINCE_RE = {p: re.compile(rx) for p, rx in {
    "British Columbia": r"British Columbia|\bB\.C\.|\bBC\b", "Alberta": r"Alberta|\bAB\b", "Saskatchewan": r"Saskatchewan|\bSK\b",
    "Manitoba": r"Manitoba|\bMB\b", "Ontario": r"Ontario|\bON\b", "Quebec": r"Qu[eé]bec|\bQC\b",
    "New Brunswick": r"New Brunswick|Nouveau-Brunswick|\bN\.B\.|\bNB\b", "Nova Scotia": r"Nova Scotia|\bNS\b",
    "Prince Edward Island": r"Prince Edward Island|\bPEI\b", "Newfoundland and Labrador": r"Newfoundland|Labrador|\bNL\b"}.items()}
CANADA = re.compile(r"\bCanada\b|\bCanadian\b")
REASONS = ("foreign_location", "other_province")
# Location locales (2026-11-24, country profiles): the signals() function of each home country. "ca" = the rules above, unchanged.
# A new country needs its own signals() (home signals: its regions and postal codes; foreign signals) registered here.
LOCALES: dict[str, object] = {}


def signals(text: str) -> dict[str, Any]:
    t = text or ""
    prov = {p: len(rx.findall(t)) for p, rx in PROVINCE_RE.items()}
    for m in CA_POSTCODE.finditer(t):
        p = POSTAL_FIRST[m.group(1)]
        prov[p] = prov.get(p, 0) + 1
    return {"us_address": len(US_ADDRESS.findall(t)), "us_state": len(STATE_NAMES.findall(t)), "foreign_phrase": len(FOREIGN_PHRASE.findall(t)),
            "uk_postcode": len(UK_POSTCODE.findall(t)), "canada": len(CANADA.findall(t)) + sum(prov.values()), "provinces": prov}


LOCALES["ca"] = signals


def foreign_score(s: dict[str, Any]) -> int:
    return s["us_address"] * 3 + s["us_state"] + s["foreign_phrase"] + s["uk_postcode"] * 2


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


class LocationCheck:
    def __init__(self, conf: dict[str, Any] | None):
        conf = conf or {}
        self.enabled = bool(conf.get("enabled", False))
        self.foreign_min = int(conf.get("foreign_min", 2))
        self.overwhelming_min = int(conf.get("overwhelming_min", 10))
        self.overwhelming_ratio = float(conf.get("overwhelming_ratio", 10))
        self.same_site = bool(conf.get("same_site", True))
        self.sole_prop_other_province = bool(conf.get("sole_proprietorship_other_province", True))
        self.locale = str(conf.get("locale", "ca")).lower()
        if self.enabled and self.locale not in LOCALES:
            raise SystemExit(f"location check: no locale {self.locale!r} in shared/foreign_pages.py (have {sorted(LOCALES)}); "
                             "write one or set [select.location].enabled = false in the country profile")
        self.signals = LOCALES.get(self.locale, signals)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LocationCheck":
        return cls(cfg.get("select", {}).get("location"))

    def drops(self, pages: dict[str, dict[str, Any]], province: str, entity_form: str = "") -> dict[str, str]:
        """{url: reason} for the pages (url -> page record) this check drops."""
        if not self.enabled or not pages:
            return {}
        sig = {u: self.signals(f"{p.get('title') or ''}\n{p.get('content') or ''}") for u, p in pages.items()}
        out: dict[str, str] = {}
        for u, s in sig.items():
            f, c = foreign_score(s), s["canada"]
            if (f >= self.foreign_min and c == 0) or (f >= self.overwhelming_min and f >= self.overwhelming_ratio * c):
                out[u] = "foreign_location"
        if self.sole_prop_other_province and entity_form == "sole_proprietorship":
            for u, s in sig.items():
                named = {p for p, n in s["provinces"].items() if n}
                if u not in out and len(named) == 1 and province not in named and foreign_score(s) == 0:
                    out[u] = "other_province"
        if self.same_site:
            for h, r in {(_host(u), r) for u, r in out.items()}:
                same = [u for u in sig if _host(u) == h]
                if not any(sig[u]["provinces"].get(province) for u in same):
                    for u in same:
                        out.setdefault(u, r)
        return out
