"""Settings from environment variables (no secrets exist in this project; all values are safe defaults).

DATA_DIR              folder with clusters.json, terms.json, ... (default: backend/data)
CORS_ORIGINS          comma-separated allowed browser origins (default: http://localhost:3000)
CORS_ORIGIN_REGEX     optional regex for extra origins, e.g. Vercel previews: https://phenopattern-.*\\.vercel\\.app
RATE_DEFAULT_PER_MIN  requests per IP per minute for data endpoints (default 120)
RATE_SEARCH_PER_MIN   autosuggest (called while typing) (default 180)
RATE_MATCH_PER_MIN    /match (default 20)
TRUST_PROXY_HEADERS   "true" to read the client IP from X-Forwarded-For (only behind a proxy you trust)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

APP_VERSION = "0.1.0"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DISCLAIMER = ("Research prototype validated only on SIMULATED data. Informational and research use only; "
              "not medical advice and not a diagnostic tool.")
LEADS_CAVEAT = ("Possible leads from published annotations. Not a diagnosis. "
                "Based on simulated data and similarity of symptom lists only.")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _int(env: Mapping[str, str], name: str, default: int, lo: int = 1, hi: int = 100_000) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        val = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if not lo <= val <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}, got {val}")
    return val


@dataclass(frozen=True)
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    cors_origin_regex: str | None = None
    rate_default_per_min: int = 120
    rate_search_per_min: int = 180
    rate_match_per_min: int = 20
    trust_proxy_headers: bool = False
    max_body_bytes: int = 4096

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        origins = _csv(env.get("CORS_ORIGINS", "")) or ("http://localhost:3000",)
        if "*" in origins:
            raise ValueError("CORS_ORIGINS must list explicit origins; '*' is not allowed")
        return cls(
            data_dir=Path(env.get("DATA_DIR") or DEFAULT_DATA_DIR),
            cors_origins=origins,
            cors_origin_regex=env.get("CORS_ORIGIN_REGEX") or None,
            rate_default_per_min=_int(env, "RATE_DEFAULT_PER_MIN", 120),
            rate_search_per_min=_int(env, "RATE_SEARCH_PER_MIN", 180),
            rate_match_per_min=_int(env, "RATE_MATCH_PER_MIN", 20),
            trust_proxy_headers=env.get("TRUST_PROXY_HEADERS", "").strip().lower() in ("1", "true", "yes"),
        )
