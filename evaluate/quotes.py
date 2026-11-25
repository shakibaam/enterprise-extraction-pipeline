"""find_quote: where an evidence quote starts in a page (moved unchanged from company_pipeline/pipeline/extract.py; used by the judge)."""
from __future__ import annotations

import re


def find_quote(content: str, quote: str) -> int | None:
    """Start of an evidence quote in a page (its first 10 words, else first 5, else last 5; case and punctuation ignored), or None."""
    words = re.findall(r"\w+", quote or "")
    for ws in (words[:10], words[:5], words[-5:]):
        if len(ws) >= 3 or (ws and len(words) == len(ws)):
            m = re.compile(r"\W+".join(re.escape(w) for w in ws), re.I).search(content)
            if m:
                return m.start()
    return None
