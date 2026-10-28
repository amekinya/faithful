"""Our models on a Nebius Serverless Endpoint (train/server.py): NVIDIA Parakeet speech recognition and our
fine-tuned fidelity checker (NVIDIA llama-nemotron-rerank-1b-v2). Both are optional: without the endpoint the app
falls back to the code slot diff, and says so."""
from __future__ import annotations

import base64
import time

import httpx

from app.ai.client import CallRecord, LiveModelUnavailable
from app.config import settings


_down_until = 0.0  # after a failed connection, skip the endpoint for a while instead of making every turn wait


def available() -> bool:
    return settings.live_endpoint and time.monotonic() >= _down_until


def _mark_down(e: Exception | None = None) -> None:
    """A connection failure, a timeout or a gateway error (502/503/504: a stopped endpoint) pauses the endpoint for 2 min."""
    global _down_until
    _down_until = time.monotonic() + 120


def _timeout(total: float) -> httpx.Timeout:
    return httpx.Timeout(total, connect=4.0)  # a stopped endpoint should fail fast, not after the full timeout


def _headers() -> dict:
    return {"Authorization": f"Bearer {settings.endpoint_token}"} if settings.endpoint_token else {}


def query(source: str, src_lang: str) -> str:
    """The checker's input format (must match scripts/build_pairs.py)."""
    return f"[{src_lang}] {source}"


def passage(rendition: str, ren_lang: str) -> str:
    return f"[{ren_lang}] {rendition}"


def _scores(r, n: int, name: str) -> list[float]:
    """Validate the endpoint's reply: one probability per pair, and the fine-tuned model actually answered."""
    try:
        d = r.json()
        sc = [float(x) for x in d["scores"]]
    except Exception as e:  # noqa: BLE001  (a cold-start HTML page, a proxy error, another server's format)
        raise LiveModelUnavailable(f"unexpected reply from the endpoint: {r.text[:120]!r}") from e
    if len(sc) != n or not all(0.0 <= x <= 1.0 for x in sc):
        raise LiveModelUnavailable(f"endpoint returned {len(sc)} scores for {n} pairs")
    if name != "base-reranker" and d.get("fine_tuned") is False:
        raise LiveModelUnavailable(f"endpoint answered with the base model, not {name}")
    return sc


def score(pairs: list[tuple[str, str]], calls: list[CallRecord] | None = None, model: str | None = None, timeout: float = 30) -> list[float]:
    """-> probability each rendition is faithful (1 = faithful)."""
    if not available():
        raise LiveModelUnavailable("ENDPOINT_URL not set")
    t0 = time.perf_counter()
    out: list[float] = []
    try:
        with httpx.Client(timeout=_timeout(timeout)) as c:
            for i in range(0, len(pairs), 2048):
                r = c.post(settings.endpoint_url.rstrip("/") + "/score", headers=_headers(),
                           json={"model": model or settings.checker_name, "pairs": [list(p) for p in pairs[i:i + 2048]]})
                if r.status_code != 200:
                    if r.status_code in (502, 503, 504):
                        _mark_down()
                    raise LiveModelUnavailable(f"checker HTTP {r.status_code}: {r.text[:200]}")
                out += _scores(r, len(pairs[i:i + 2048]), model or settings.checker_name)
    except httpx.HTTPError as e:
        _mark_down(e)
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
        with httpx.Client(timeout=_timeout(timeout)) as c:
            r = c.post(settings.endpoint_url.rstrip("/") + "/asr", headers=_headers(), json={"audio_b64": base64.b64encode(audio).decode()})
    except httpx.HTTPError as e:
        _mark_down(e)
        raise LiveModelUnavailable(f"speech recognition unreachable: {e}") from e
    if r.status_code != 200:
        if r.status_code in (502, 503, 504):
            _mark_down()
        raise LiveModelUnavailable(f"asr HTTP {r.status_code}: {r.text[:200]}")
    try:
        d = r.json()
        d["text"] = str(d["text"])
    except Exception as e:  # noqa: BLE001
        raise LiveModelUnavailable(f"unexpected reply from the endpoint: {r.text[:120]!r}") from e
    if calls is not None:
        calls.append(CallRecord("asr", d.get("model", "nvidia/parakeet-tdt-0.6b-v3") + " (Nebius endpoint)", time.perf_counter() - t0, 0, 0, 0, 0.0, "Speech to text"))
    return d
