"""The pipeline steps, one module each, in run order:

  search.py    3 queries x 10 results (Serper or Keenable); hits merged by canonical URL           -> search.json
  fetch.py     download every hit, HTML -> text, snippet fallback                                  -> fetch.json, pages/
  rerank.py    Granite cross-encoder on local vLLM; page score = best chunk                        -> rerank.json
  select.py    score >= threshold, names the company, not a foreign site, not placed abroad        -> select.json
  extract.py   one request per page piece (any extractor profile); page number -> exact URL      -> extract.json
  verify.py    one yes/no question per claim, whole page, no web (any verifier profile)            -> decisions.json
  finalize.py  clean-up (merge, drop non-names, ...) + keep only mentions with a confirmed claim  -> final.json, verified.json

search, fetch, rerank and select come from company_pipeline/pipeline/ (2026-11-24) with their standalone CLIs removed. No step imports
another step: shared helpers are in shared/, working-file writing (write_json) in storage/outputs.py; the changes are
listed in shared/__init__.py and README section 8. pipeline.py runs the steps for each company (and has the --from-saved alternative to
the first four).
"""
