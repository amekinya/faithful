"""Our models on a Nebius Serverless Endpoint (train/server.py): NVIDIA Parakeet speech recognition and our
fine-tuned fidelity checker (NVIDIA llama-nemotron-rerank-1b-v2). Both are optional: without the endpoint the app
falls back to the code slot diff, and says so."""
from __future__ import annotations

import base64
import time

import httpx

from app.ai.client import CallRecord, LiveModelUnavailable
from app.config import settings


def available() -> bool:
    return settings.live_endpoint


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.endpoint_token}"} if settings.endpoint_token else {}


def query(source: str, src_lang: str) -> str:
    """The checker's input format (must match scripts/build_pairs.py)."""
    return f"[{src_lang}] {source}"


def passage(rendition: str, ren_lang: str) -> str:
    return f"[{ren_lang}] {rendition}"


def score(pairs: list[tuple[str, str]], calls: list[CallRecord] | None = None, model: str | None = None, timeout: float = 30) -> list[float]:
    """-> probability each rendition is faithful (1 = faithful)."""
    if not available():
        raise LiveModelUnavailable("ENDPOINT_URL not set")
    t0 = time.perf_counter()
    out: list[float] = []
    try:
        with httpx.Client(timeout=timeout) as c:
            for i in range(0, len(pairs), 2048):
                r = c.post(settings.endpoint_url.rstrip("/") + "/score", headers=_headers(),
                           json={"model": model or settings.checker_name, "pairs": [list(p) for p in pairs[i:i + 2048]]})
                if r.status_code != 200:
                    raise LiveModelUnavailable(f"checker HTTP {r.status_code}: {r.text[:200]}")
                out += r.json()["scores"]
    except httpx.HTTPError as e:
        raise LiveModelUnavailable(f"checker unreachable: {e}") from e
    if calls is not None:
        calls.append(CallRecord("checker", f"{model or settings.checker_name} (Nebius endpoint)", time.perf_counter() - t0, len(pairs), 0, 0, 0.0,
                                "Score interpreted turns"))
    return out


def asr(audio: bytes, calls: list[CallRecord] | None = None, timeout: float = 60) -> dict:
    if not available():
        raise LiveModelUnavailable("ENDPOINT_URL not set")
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(settings.endpoint_url.rstrip("/") + "/asr", headers=_headers(), json={"audio_b64": base64.b64encode(audio).decode()})
    except httpx.HTTPError as e:
        raise LiveModelUnavailable(f"speech recognition unreachable: {e}") from e
    if r.status_code != 200:
        raise LiveModelUnavailable(f"asr HTTP {r.status_code}: {r.text[:200]}")
    d = r.json()
    if calls is not None:
        calls.append(CallRecord("asr", d.get("model", "nvidia/parakeet-tdt-0.6b-v3") + " (Nebius endpoint)", time.perf_counter() - t0, 0, 0, 0, 0.0, "Speech to text"))
    return d
