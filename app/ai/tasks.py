"""What the Nemotron models do in Faithful. Models read and explain; code decides (app/engine/gate.py).

1. explain()      Nemotron 3.5 Lightning: one plain line for the re-confirm card, quoting the source words (code checks the quote).
2. judge_turn()   prompt-only judge, used as a baseline in the benchmark (Lightning or Ultra).
3. review_visit() Nemotron 3 Ultra: after the visit, a second read of the whole conversation for anything the turn checks missed
                  (each item must quote a real turn; code drops the rest).
"""
from __future__ import annotations

import re

from app.ai.client import CallRecord, chat_json
from app.config import settings

LANG = {"en": "English", "es": "Spanish"}


def norm(x: str) -> str:
    import unicodedata

    x = unicodedata.normalize("NFD", x.lower())
    x = "".join(c for c in x if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", x).strip()


def quote_in(quote: str, text: str) -> bool:
    q = norm(quote)
    return bool(q) and q in norm(text)


EXPLAIN_SCHEMA = {"type": "object", "properties": {
    "say": {"type": "string", "description": "one short sentence for the interpreter and clinician, in English, starting with a verb, e.g. 'Re-confirm the dose limit: no more than 4 doses a day was not interpreted.'"},
    "source_quote": {"type": "string", "description": "the exact words from the ORIGINAL that were lost or changed"}},
    "required": ["say", "source_quote"], "additionalProperties": False}


def explain(source: str, rendition: str, src_lang: str, ren_lang: str, findings: list[dict], calls: list[CallRecord] | None = None) -> dict:
    facts = "\n".join(f"- {f['message']}" for f in findings) or "- (no specific word-level change found; the checker scored the meaning as changed)"
    out = chat_json(settings.lightning, [
        {"role": "system", "content": "You help a medical interpreter and a clinician re-confirm critical content. You never speak to the patient, "
                                      "never give medical advice and never say anyone made a mistake. Be brief and concrete."},
        {"role": "user", "content": f"Original ({LANG[src_lang]}): {source}\nInterpretation ({LANG[ren_lang]}): {rendition}\n"
                                    f"Code found these differences:\n{facts}\nWrite the re-confirm line."}],
        schema=EXPLAIN_SCHEMA, purpose="Explain a re-confirm card", reasoning=False, max_tokens=300, calls=calls)
    out["quote_ok"] = quote_in(out.get("source_quote", ""), source)
    return out


JUDGE_SCHEMA = {"type": "object", "properties": {
    "faithful": {"type": "boolean", "description": "true if every critical detail (numbers, doses, units, how often, how long, allergies, left/right, medicine names, negation, warning signs) crossed over unchanged"},
    "error_type": {"type": "string", "enum": ["none", "omission", "substitution", "addition"]},
    "what": {"type": "string"}},
    "required": ["faithful", "error_type", "what"], "additionalProperties": False}

JUDGE_PROMPT = ("You check medical interpretation. Decide whether the interpretation carries every CRITICAL detail of the original unchanged: "
                "numbers, doses, units, how often, how long, dose limits, allergies, left/right, medicine names, a 'no'/'not', and warning signs. "
                "Normal interpreter choices are fine: third person ('she says'), different word order, synonyms, brand vs generic names, digits vs words, "
                "dropping fillers like 'um' or 'okay'. Flag only if a critical detail was dropped, changed or added.")


def judge_turn(source: str, rendition: str, src_lang: str, ren_lang: str, spec=None, calls: list[CallRecord] | None = None) -> dict:
    return chat_json(spec or settings.lightning, [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Original ({LANG[src_lang]}): {source}\nInterpretation ({LANG[ren_lang]}): {rendition}"}],
        schema=JUDGE_SCHEMA, purpose="Judge one interpreted turn", reasoning=False, max_tokens=400, calls=calls)


REVIEW_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
    "turn": {"type": "integer"}, "source_quote": {"type": "string", "description": "exact words from that turn's ORIGINAL"},
    "confirm": {"type": "string", "description": "what to re-confirm, one short sentence"}},
    "required": ["turn", "source_quote", "confirm"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False}


def review_visit(turns: list[dict], already: list[int], calls: list[CallRecord] | None = None) -> list[dict]:
    """turns: [{n, speaker, src_lang, source, rendition}]. Returns only items whose quote is really in that turn's original."""
    body = "\n".join(f"[{t['n']}] {t['speaker']} ({LANG[t['src_lang']]}): {t['source']}\n     interpreted: {t['rendition']}" for t in turns)
    out = chat_json(settings.ultra, [
        {"role": "system", "content": JUDGE_PROMPT + " You are reviewing a whole visit after it ended. List only critical details that did not reach the "
                                                     "other person correctly and were not already flagged. Quote the original words exactly."},
        {"role": "user", "content": f"Already flagged turns: {already or 'none'}\n\n{body}"}],
        schema=REVIEW_SCHEMA, purpose="Post-visit review", reasoning=True, max_tokens=4000, calls=calls)
    by_n = {t["n"]: t for t in turns}
    return [i | {"verified": True} for i in out.get("items", []) if i["turn"] in by_n and quote_in(i["source_quote"], by_n[i["turn"]]["source"])]
