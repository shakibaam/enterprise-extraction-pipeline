"""on_the_fly_pipeline: frozen, reusable people-and-organization extraction for Canadian companies (adopted E6 setup, 2026-11-24).

search -> fetch -> rerank -> select -> extract (any model profile) -> verify (any model profile) -> clean-up. See README.md.
Run from Provinces/:  /home/shakiba/vllm-env/bin/python -m on_the_fly_pipeline --help
"""

__version__ = "1.1.0"   # 1.0.0 = first freeze (2026-11-24 morning); 1.1.0 = review fixes, resume, cleanup, country profiles
