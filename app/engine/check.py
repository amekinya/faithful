"""One interpreted turn through Faithful: code slot diff -> our fine-tuned checker (Nebius) -> fixed decision rule ->
(only for cards) a one-line explanation from Nemotron 3.5 Lightning."""
from __future__ import annotations

import time

from app.ai import endpoint, tasks
from app.ai.client import CallRecord, LiveModelUnavailable
from app.config import settings
from app.engine import gate, slots


def check_turn(source: str, rendition: str, src_lang: str, ren_lang: str, *, use_checker: bool = True, explain: bool = True,
               p_faithful: float | None = None, calls: list[CallRecord] | None = None) -> dict:
    calls = calls if calls is not None else []
    t0 = time.perf_counter()
    findings = [f.as_dict() for f in slots.diff(source, rendition, src_lang, ren_lang)]
    t_code = time.perf_counter() - t0
    checker = "given" if p_faithful is not None else "off"
    t1 = time.perf_counter()
    if p_faithful is None and use_checker and endpoint.available():
        try:
            p_faithful = endpoint.score([(endpoint.query(source, src_lang), endpoint.passage(rendition, ren_lang))], calls)[0]
            checker = "on"
        except LiveModelUnavailable as e:
            checker = f"unavailable ({str(e)[:60]})"
    t_checker = time.perf_counter() - t1
    decision = gate.decide(findings, p_faithful)
    t_decided = time.perf_counter() - t0
    explanation = None
    if decision["level"] == "card" and explain and settings.live_models:
        try:
            explanation = tasks.explain(source, rendition, src_lang, ren_lang, [f for f in findings if f["severity"] != "low"], calls)
        except LiveModelUnavailable:
            explanation = None
        if explanation is not None and not explanation.get("quote_ok", False):
            explanation = None  # the model quoted words that weren't said: show the code's exact words instead
    if explanation is None and decision["level"] != "ok":
        explanation = {"say": code_line(findings), "source_quote": next((f["source"] for f in findings if f["source"]), ""), "quote_ok": True, "by": "code"}
    return {"findings": findings, "p_faithful": None if p_faithful is None else round(p_faithful, 4), "checker": checker,
            "decision": decision, "explanation": explanation,
            "ms": {"code": round(t_code * 1000, 1), "checker": round(t_checker * 1000, 1), "to_decision": round(t_decided * 1000, 1),
                   "total": round((time.perf_counter() - t0) * 1000, 1)}}


def code_line(findings: list[dict]) -> str:
    """The card's line when no model is available: the exact words, in clinical priority order (allergy, 'no', medicine,
    warning sign, dose limit, left/right, how often, numbers, units, how long), with neighbouring omissions grouped."""
    order = {k: i for i, k in enumerate(slots.PRIORITY)}
    fs = sorted((f for f in findings if f["severity"] != "low"), key=lambda f: order.get(f["kind"], 99))
    q = lambda x: f"\u201c{x}\u201d"  # noqa: E731
    parts: list[str] = []
    lost: list[str] = []

    def flush():
        if lost:
            parts.append("not interpreted: " + ", ".join(dict.fromkeys(lost)))
            lost.clear()
    for f in fs:
        if f["type"] == "omission":
            lost.append(q(f["source"]))
            continue
        flush()
        if f["kind"] == "order":
            parts.append(f["message"][0].lower() + f["message"][1:])
        elif f["type"] == "substitution":
            parts.append(f"{q(f['source'])} became {q(f['rendition'])}")
        else:
            parts.append(f"added, not said: {q(f['rendition'])}")
    flush()
    if not parts:
        return "Re-confirm this turn: the meaning may have changed."
    text = "; ".join(parts)
    return "Re-confirm. " + text[:1].upper() + text[1:] + "."
