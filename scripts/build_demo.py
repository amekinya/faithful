"""Run the scripted visits through the whole live path and save what the demo replays.

Per turn: the speaker's audio and the interpreter's audio -> NVIDIA Parakeet (our Nebius endpoint) -> code slot diff ->
our fine-tuned checker (same endpoint) -> decision rule -> Lightning explanation for cards. After each visit, Nemotron 3
Ultra reviews the whole conversation once for anything the per-turn checks missed (quotes verified by code).
Also measures the end-to-end result against the 20 planted errors, and speech-recognition word error rate.
Writes data/encounters/results.json and eval/e2e.json.
Usage: python -m scripts.build_demo [--no-asr] [--no-review]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from app.ai import endpoint, tasks
from app.ai.client import CallRecord, LiveModelUnavailable
from app.engine.check import check_turn

ROOT = Path(__file__).resolve().parent.parent


def words(s: str) -> list[str]:
    import unicodedata

    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", s)


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = words(ref), words(hyp)
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1], len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--no-review", action="store_true")
    a = ap.parse_args()
    d = json.loads((ROOT / "data" / "encounters" / "visits.json").read_text())
    audio = json.loads((ROOT / "data" / "encounters" / "audio.json").read_text()) if (ROOT / "data" / "encounters" / "audio.json").exists() else {}
    calls: list[CallRecord] = []
    out_visits, e2e_rows = [], []
    werr = {"en": [0, 0], "es": [0, 0]}
    for v in d["visits"]:
        turns = []
        for i, t in enumerate(v["turns"], 1):
            key = f"{v['id']}-{i:02d}"
            sl, rl = ("en", "es") if t["speaker"] == "clinician" else ("es", "en")
            heard = {"source": t["source"], "rendition": t["rendition"], "asr": False, "asr_ms": None}
            if not a.no_asr and key in audio and endpoint.available():
                try:
                    t0 = time.perf_counter()
                    s = endpoint.asr((ROOT / "app" / "static" / audio[key]["src"]).read_bytes(), calls)
                    t1 = time.perf_counter()
                    r = endpoint.asr((ROOT / "app" / "static" / audio[key]["ren"]).read_bytes(), calls)
                    t2 = time.perf_counter()
                    # live, the speaker's audio is transcribed while the interpreter talks; what the room waits for is the
                    # interpreter's clip + the check
                    heard = {"source": s["text"], "rendition": r["text"], "asr": True, "asr_ms": round((t2 - t1) * 1000, 1),
                             "asr_ms_source": round((t1 - t0) * 1000, 1), "asr_model": s.get("model")}
                    for lang, ref, hyp in ((sl, t["source"], s["text"]), (rl, t["rendition"], r["text"])):
                        e, n = wer(ref, hyp)
                        werr[lang][0] += e; werr[lang][1] += n
                except LiveModelUnavailable as e:
                    print(f"  {key}: asr unavailable: {e}")
            res = check_turn(heard["source"], heard["rendition"], sl, rl, calls=calls)
            res_text = check_turn(t["source"], t["rendition"], sl, rl, explain=False, p_faithful=res["p_faithful"] if not heard["asr"] else None)
            turns.append({"n": i, "key": key, "speaker": t["speaker"], "src_lang": sl, "ren_lang": rl, "script": {"source": t["source"], "rendition": t["rendition"]},
                          "heard": heard, "audio": audio.get(key), "check": res, "check_on_script": {"decision": res_text["decision"], "findings": res_text["findings"]},
                          "planted": t.get("error")})
            e2e_rows.append({"turn": key, "error": "error" in t, "speaker": t["speaker"], "flag": res["decision"]["level"] == "card",
                             "flag_script": res_text["decision"]["level"] == "card", "ms": res["ms"]["to_decision"] + (heard["asr_ms"] or 0)})
            print(key, res["decision"]["level"], res["decision"]["title"], "| planted:", (t.get("error") or {}).get("type", "-"))
        review = []
        if not a.no_review:
            flagged = [x["n"] for x in turns if x["check"]["decision"]["level"] != "ok"]
            try:
                review = tasks.review_visit([{"n": x["n"], "speaker": x["speaker"], "src_lang": x["src_lang"], "source": x["heard"]["source"],
                                              "rendition": x["heard"]["rendition"]} for x in turns], flagged, calls)
            except LiveModelUnavailable as e:
                print("  review unavailable:", e)
        out_visits.append({k: v[k] for k in ("id", "title", "setting", "patient", "clinician")} | {"turns": turns, "review": review})

    def summ(key):
        n_err = sum(r["error"] for r in e2e_rows)
        tp = sum(r["error"] and r[key] for r in e2e_rows); fp = sum((not r["error"]) and r[key] for r in e2e_rows)
        pat = sum(r["error"] and r[key] and r["speaker"] == "patient" for r in e2e_rows)
        return {"caught": tp, "errors": n_err, "false_alarms": fp, "faithful_turns": len(e2e_rows) - n_err,
                "patient_speech_caught": f"{pat}/{sum(r['error'] and r['speaker'] == 'patient' for r in e2e_rows)}",
                "missed": [r["turn"] for r in e2e_rows if r["error"] and not r[key]],
                "false_alarm_turns": [r["turn"] for r in e2e_rows if not r["error"] and r[key]]}
    ms = sorted(r["ms"] for r in e2e_rows)
    by_model = {}
    for c in calls:
        b = by_model.setdefault(c.role, {"model": c.model_id, "calls": 0, "seconds": 0.0, "cost_usd": 0.0})
        b["calls"] += 1; b["seconds"] = round(b["seconds"] + c.seconds, 2); b["cost_usd"] = round(b["cost_usd"] + c.cost_usd, 5)
    e2e = {"from_audio": summ("flag"), "from_script_text": summ("flag_script"),
           "wer": {k: round(e / n, 3) if n else None for k, (e, n) in werr.items()},
           "ms_median_audio_to_card": ms[len(ms) // 2] if ms else None, "ms_p95": ms[int(0.95 * (len(ms) - 1))] if ms else None,
           "by_model": by_model, "cost_usd": round(sum(c.cost_usd for c in calls), 4)}
    (ROOT / "data" / "encounters" / "results.json").write_text(json.dumps({"visits": out_visits, "e2e": e2e}, ensure_ascii=False, indent=1))
    (ROOT / "eval" / "e2e.json").write_text(json.dumps(e2e, indent=1))
    print(json.dumps(e2e, indent=1))


if __name__ == "__main__":
    main()
