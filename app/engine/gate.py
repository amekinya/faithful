"""The decision: does this turn get a re-confirm card? Fixed rules, no AI.

Inputs: the code slot diff (exact words that changed) and the fine-tuned checker's score (probability the
interpretation is faithful, from our model on Nebius). Thresholds are chosen on the dev split and saved in
data/thresholds.json by eval/run_eval.py.

  card    'Re-confirm'      a HIGH-severity slot change, and the checker agrees something is off (p_error >= confirm)
  card    'Possible change' the checker is very sure (p_error >= alone) even though code found no word-level change,
                            or a MEDIUM slot change the checker agrees with
  note    'Check later'     a HIGH slot change the checker thinks is fine (often a legitimate re-wording): listed after the
                            visit, never interrupts
  ok      nothing shown
Without the checker (endpoint stopped), the slot diff alone decides: HIGH -> card, MEDIUM -> note.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT = {"confirm": 0.35, "alone": 0.85}
TH_FILE = Path(__file__).resolve().parents[2] / "data" / "thresholds.json"


def thresholds() -> dict:
    try:
        return DEFAULT | json.loads(TH_FILE.read_text())
    except Exception:  # noqa: BLE001
        return dict(DEFAULT)


def decide(findings: list[dict], p_faithful: float | None, th: dict | None = None) -> dict:
    th = th or thresholds()
    high = [f for f in findings if f["severity"] == "high"]
    med = [f for f in findings if f["severity"] == "medium"]
    if p_faithful is None:
        if high:
            return {"level": "card", "title": "Re-confirm", "why": "code", "p_error": None}
        if med:
            return {"level": "note", "title": "Check later", "why": "code", "p_error": None}
        return {"level": "ok", "title": "", "why": "", "p_error": None}
    p_err = round(1 - p_faithful, 3)
    if high and p_err >= th["confirm"]:
        return {"level": "card", "title": "Re-confirm", "why": "code+checker", "p_error": p_err}
    if p_err >= th["alone"]:
        return {"level": "card", "title": "Possible change", "why": "checker", "p_error": p_err}
    if med and p_err >= th["confirm"]:
        return {"level": "card", "title": "Possible change", "why": "code+checker", "p_error": p_err}
    if high:
        return {"level": "note", "title": "Check later", "why": "code only", "p_error": p_err}
    return {"level": "ok", "title": "", "why": "", "p_error": p_err}
