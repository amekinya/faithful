"""Runtime configuration. Everything secret comes from environment variables, never from code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional local .env support
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class ModelSpec:
    """A model we call on Nebius Token Factory, with list prices in USD per 1M tokens."""

    role: str
    model_id: str
    price_in: float
    price_out: float


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str = field(default_factory=lambda: _env("NEBIUS_API_KEY"))
    nebius_base_url: str = field(default_factory=lambda: _env("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"))
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))
    # Our own models on Nebius AI Cloud (one Serverless Endpoint shared with Provenance):
    # /asr = NVIDIA Parakeet TDT 0.6B v3, /score = our fine-tuned checker (NVIDIA llama-nemotron-rerank-1b-v2).
    endpoint_url: str = field(default_factory=lambda: _env("ENDPOINT_URL") or _env("MATCHER_URL"))
    endpoint_token: str = field(default_factory=lambda: _env("ENDPOINT_TOKEN") or _env("MATCHER_TOKEN"))
    checker_name: str = field(default_factory=lambda: _env("CHECKER_NAME", "faithful-checker"))
    daily_spend_cap_usd: float = field(default_factory=lambda: float(_env("DAILY_SPEND_CAP_USD", "2.0")))

    # Confirmed live with GET /v1/models on Oct 26, 2026.
    lightning: ModelSpec = field(default_factory=lambda: ModelSpec("lightning", _env("MODEL_LIGHTNING", "nvidia/Nemotron-3_5-Lightning"), 0.06, 0.24))
    super_: ModelSpec = field(default_factory=lambda: ModelSpec("super", _env("MODEL_SUPER", "nvidia/Nemotron-3.5-Super-VL"), 0.30, 0.90))
    ultra: ModelSpec = field(default_factory=lambda: ModelSpec("ultra", _env("MODEL_ULTRA", "nvidia/Nemotron-3-Ultra-550b-a55b"), 1.00, 3.00))

    @property
    def live_models(self) -> bool:
        return bool(self.nebius_api_key)

    @property
    def live_endpoint(self) -> bool:
        return bool(self.endpoint_url)


settings = Settings()
