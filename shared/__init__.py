"""Helpers used by several steps (not steps themselves):

  config.py         settings: the config file (paths relative to Provinces/), country profiles merged over it, API keys by name
                    (environment first, then the env file); was config.py + countries.py + credentials.py
  models.py         Company (name, province, slug, company_id, entity_form, country), Query, SearchHit
  names.py          name matching (does a page or website name the company?) and normalization (honorifics, credentials, defined terms);
                    was names.py + name_norm.py
  foreign_pages.py  is a page about a company in another country? foreign-site rules (rerank, select) and the location check with its
                    locale registry (select, clean-up); was sites.py + location.py
  llm.py            model access (extract, verify): profiles from config/models.toml, one chat client for OpenRouter and local vLLM,
                    the cost ledger with optional caps
  pdf_text.py       PDF to text in its own child process (fetch, only when [fetch].parse_pdfs = true)
The clean-up rules of the old canonical.py are now in steps/finalize.py (the only step that used them).

Origin: company_pipeline/pipeline/ (2026-11-24). Changes: foreign-site home country and location locale from the country profile,
Company.country, read_secret = the environment first; llm.py and the country/key code are new.
"""
