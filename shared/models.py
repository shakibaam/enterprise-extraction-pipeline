"""Data types shared by the pipeline components. More types are added as components are written."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Company:
    """One company to process. `slug` names its output folder; it is derived from the name when not given."""
    name: str
    province: str
    slug: str = ""
    company_id: str = ""      # optional identifier from the input file
    entity_form: str = ""     # optional, e.g. "sole_proprietorship" (corptrace entity_form); used by the location check
    country: str = "CA"       # ISO code of the registry's country (2026-11-24): selects nothing by itself; a run refuses other countries

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", " ".join(self.name.split()))           # line breaks and runs of spaces -> one space
        object.__setattr__(self, "province", " ".join(self.province.split()))
        if not self.name or not self.province:
            raise ValueError("a company needs a name and a province")
        if not self.slug:
            object.__setattr__(self, "slug", slugify(self.name))


@dataclass(frozen=True)
class Query:
    """One search query for one company."""
    query_id: str       # e.g. "Q02"
    category: str       # e.g. "registry"
    text: str           # the final query string sent to the search API


@dataclass
class SearchHit:
    """One organic search result."""
    url: str
    canonical_url: str
    title: str
    snippet: str
    query_id: str
    rank: int                      # 1-based position within its query
    extra: dict[str, Any] = field(default_factory=dict)   # provider-specific fields (e.g. published_at)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def slugify(text: str) -> str:
    """Lower-case ASCII slug: letters and digits joined by underscores."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot build a slug from {text!r}")
    return slug
