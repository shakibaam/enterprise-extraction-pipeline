"""Settings: the config file, country profiles and API keys (one module since 2026-11-24; was shared/config.py, countries.py and
credentials.py).

CONFIG FILE (was shared/config.py)
Configuration and secrets.

The configuration is a TOML file (config/default.toml). API keys are read by name from the .env file given in [secrets];
they are never printed, logged or written anywhere.

COUNTRY PROFILES (was countries.py)
Country profiles (config/countries/<code>.toml): everything that depends on the registry's country, kept out of the code.

A profile has: code, name, status ("validated" | "draft"), region_label, [jurisdictions] (corptrace jurisdiction code -> region name, used
by the input reader), and [config.*] tables merged over config/default.toml:
  search      query templates ({name}, {province} or {region}), Serper country (gl) and language (hl)
  rerank      the country named in the reranker query
  select      foreign_sites (other countries' domains, registries, URL path segments, OpenCorporates home) and location (locale, rules)
  extraction  prompt files, the header's country line
  verification  prompt and registry-record files
One run = one country (--country, default CA); companies of another country are refused (cli.py).
Adding a country: copy ca.toml to <code>.toml, edit every [config.*] value, keep status = "draft" until it has been measured; a location
check needs a locale in shared/foreign_pages.py (otherwise set [config.select.location].enabled = false).

API KEYS (was credentials.py)
API keys, read by NAME only: first the environment variable, then the env file named in [secrets].env_file of config/default.toml
(default enterprise_agent/.env). Values are returned to the caller and never printed, logged or written. Replaces the runtime use of
enterprise_agent.config (2026-11-24), with the same OpenRouter lookup order: ENTERPRISE_AGENT_OPENROUTER_API_KEY, OPENROUTER_API_KEY,
then ENTERPRISE_AGENT_OPENROUTER_API_KEY_ALT.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import os
import tomllib


# ====================================================================================================
# CONFIG FILE (was shared/config.py)
# ====================================================================================================

PROVINCES_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.toml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the TOML configuration (the default file when no path is given)."""
    with open(path or DEFAULT_CONFIG, "rb") as fh:
        return tomllib.load(fh)


def resolve_path(value: str | Path) -> Path:
    """Paths in the config are relative to the Provinces/ directory unless absolute."""
    p = Path(value)
    return p if p.is_absolute() else PROVINCES_DIR / p


def read_secret(config: dict[str, Any], key_name: str) -> str:
    """The value of key_name (environment variable first, then the env file named in [secrets].env_file); used by search.
    Raises KeyError (without the value) when the key is missing or empty. Same lookup as get() below (one implementation)."""
    env_file = resolve_path(config["secrets"]["env_file"])
    v = get(key_name, env_file)
    if not v:
        raise KeyError(f"{key_name} is missing or empty in the environment and in {env_file}")
    return v


# ====================================================================================================
# COUNTRY PROFILES (was countries.py)
# ====================================================================================================

COUNTRIES_DIR = Path(__file__).resolve().parents[1] / "config" / "countries"


def available() -> list[str]:
    return sorted(p.stem.upper() for p in COUNTRIES_DIR.glob("*.toml"))


def load_country(code: str) -> dict[str, Any]:
    f = COUNTRIES_DIR / f"{code.lower()}.toml"
    if not f.is_file():
        raise SystemExit(f"no country profile for {code!r} (config/countries/{code.lower()}.toml); available: {', '.join(available())}")
    with open(f, "rb") as fh:
        prof = tomllib.load(fh)
    if prof.get("code", "").upper() != code.upper():
        raise SystemExit(f"{f.name}: code {prof.get('code')!r} does not match the file name")
    return prof


def all_jurisdictions() -> dict[str, tuple[str, str]]:
    """Every known corptrace jurisdiction code -> (country code, region name), from all profiles."""
    out = {}
    for code in available():
        for jur, region in load_country(code).get("jurisdictions", {}).items():
            out[jur.upper()] = (code, region)
    return out


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def resolve_config(config_path: str | None, country: str) -> dict[str, Any]:
    """default.toml (or --config) with the country profile merged over it; cfg["country"] holds the profile's identity."""
    prof = load_country(country)
    cfg = deep_merge(load_config(config_path), prof.get("config", {}))
    cfg["country"] = {"code": prof["code"].upper(), "name": prof["name"], "status": prof.get("status", "draft"),
                      "region_label": prof.get("region_label", "region")}
    missing = [k for k in ("queries",) if k not in cfg["search"]] + \
              [f"extraction.{k}" for k in ("prompt_file", "chunk_sizing_prompt_file", "country") if k not in cfg["extraction"]] + \
              [f"verification.{k}" for k in ("prompt_file", "registry_record_file") if k not in cfg["verification"]]
    if missing:
        raise SystemExit(f"country profile {country}: missing settings {missing}")
    return cfg


# ====================================================================================================
# API KEYS (was credentials.py)
# ====================================================================================================

OPENROUTER_NAMES = ("ENTERPRISE_AGENT_OPENROUTER_API_KEY", "OPENROUTER_API_KEY", "ENTERPRISE_AGENT_OPENROUTER_API_KEY_ALT")


def env_file() -> Path:
    return resolve_path(load_config()["secrets"]["env_file"])


def _from_file(path: Path, name: str) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == name:
            v = v.strip().strip("\"'")
            return v or None
    return None


def get(name: str, path: Path | None = None) -> str | None:
    """The key's value, or None. Environment variable first, then the env file."""
    v = (os.environ.get(name) or "").strip()
    return v or _from_file(path or env_file(), name)


def openrouter_key(path: Path | None = None) -> str:
    for n in OPENROUTER_NAMES:
        v = get(n, path)
        if v:
            return v
    raise SystemExit(f"no OpenRouter key: set one of {', '.join(OPENROUTER_NAMES)} in the environment or in {path or env_file()}")
