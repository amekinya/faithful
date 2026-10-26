"""Faithful web app. Stateless: nothing anyone types or records is stored."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ai import endpoint
from app.ai.client import CallRecord, LiveModelUnavailable, ledger
from app.config import settings
from app.engine import gate
from app.engine.check import check_turn

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Faithful")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _read(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return default


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"live_models": settings.live_models, "checker_endpoint": endpoint.available(), "spend_remaining_today": round(ledger.remaining(), 4),
            "models": {s.role: s.model_id for s in (settings.lightning, settings.ultra)}, "checker": settings.checker_name,
            "thresholds": gate.thresholds()}


@app.get("/api/visits")
def visits():
    d = _read(ROOT / "data" / "encounters" / "results.json")
    if not d:
        d = {"visits": [{**v, "turns": [{"n": i, "speaker": t["speaker"]} for i, t in enumerate(v["turns"], 1)]}
                        for v in _read(ROOT / "data" / "encounters" / "visits.json", {"visits": []})["visits"]]}
    return [{"id": v["id"], "title": v["title"], "setting": v.get("setting", ""), "patient": v["patient"], "clinician": v["clinician"], "turns": len(v["turns"]),
             "cards": sum(1 for t in v["turns"] if t.get("check", {}).get("decision", {}).get("level") == "card")} for v in d["visits"]]


@app.get("/api/visit/{vid}")
def visit(vid: str):
    d = _read(ROOT / "data" / "encounters" / "results.json", {}) or {}
    v = next((x for x in d.get("visits", []) if x["id"] == vid), None)
    if v is None:  # not built yet: compute on the fly from the script (code + whatever models are reachable)
        src = _read(ROOT / "data" / "encounters" / "visits.json", {"visits": []})
        sv = next((x for x in src["visits"] if x["id"] == vid), None)
        if sv is None:
            raise HTTPException(404, "Unknown visit.")
        turns = []
        for i, t in enumerate(sv["turns"], 1):
            sl, rl = ("en", "es") if t["speaker"] == "clinician" else ("es", "en")
            turns.append({"n": i, "key": f"{vid}-{i:02d}", "speaker": t["speaker"], "src_lang": sl, "ren_lang": rl,
                          "script": {"source": t["source"], "rendition": t["rendition"]},
                          "heard": {"source": t["source"], "rendition": t["rendition"], "asr": False},
                          "check": check_turn(t["source"], t["rendition"], sl, rl, use_checker=False, explain=False), "planted": t.get("error")})
        v = {k: sv[k] for k in ("id", "title", "setting", "patient", "clinician")} | {"turns": turns, "review": [], "live_built": False}
    return v | {"e2e": d.get("e2e")}


@app.get("/api/eval")
def eval_results():
    return {"bench": _read(ROOT / "eval" / "results.json"), "e2e": _read(ROOT / "eval" / "e2e.json"), "training": _read(ROOT / "eval" / "training.json")}


class CheckIn(BaseModel):
    source: str = Field(..., min_length=1, max_length=1200)
    rendition: str = Field(..., min_length=1, max_length=1200)
    src_lang: str = Field("en", pattern="^(en|es)$")


@app.post("/api/check")
def check(body: CheckIn):
    rl = "es" if body.src_lang == "en" else "en"
    calls: list[CallRecord] = []
    explain = settings.live_models and ledger.remaining() > 0.05
    res = check_turn(body.source, body.rendition, body.src_lang, rl, explain=explain, calls=calls)
    return res | {"calls": [asdict(c) | {"cost_usd": round(c.cost_usd, 6), "seconds": round(c.seconds, 3)} for c in calls],
                  "live": {"explainer": explain, "checker": res["checker"]}}


@app.post("/api/asr")
async def asr(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 4_000_000:
        raise HTTPException(413, "Keep recordings under about 30 seconds.")
    if not endpoint.available():
        raise HTTPException(503, "Speech recognition runs on our Nebius GPU endpoint, which is switched off right now to save cost. Type the words instead.")
    t0 = time.perf_counter()
    try:
        d = endpoint.asr(data)
    except LiveModelUnavailable as e:
        raise HTTPException(503, f"Speech recognition isn't reachable right now ({e}). Type the words instead.") from e
    return d | {"round_trip_ms": round((time.perf_counter() - t0) * 1000, 1)}
