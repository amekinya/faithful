"""Synthetic interpreted medical visits with known errors, for training and testing the fidelity checker.

1. A model from a DIFFERENT family than the Nemotron models Faithful uses writes short English-Spanish visits:
   each turn has the speaker's words and a faithful interpretation into the other language.
2. The same writer adds, per turn, one faithful re-wording (third person, other word order, synonyms) and one
   rendition with a single realistic interpreter error (omission / substitution / addition of critical content).
3. Code adds more variants with exact labels: changed numbers, swapped units, flipped left/right, dropped 'no',
   swapped medicines, dropped clauses (errors); digits<->number words, brand<->generic names, 'she says' framing (faithful).
4. Code keeps a faithful re-wording only if the slot diff finds no critical change, and an error only if it differs
   from the faithful rendition.
Writes data/synth/turns_<generator>.jsonl (one row per source/rendition pair with label, split, group).
Usage: python -m scripts.synth --n 60 --generator deepseek-ai/DeepSeek-V4.1-Flash-1017 --split trainval
"""
from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.ai.client import CallRecord, LiveModelUnavailable, _get_client, chat_json
from app.config import ModelSpec
from app.engine import slots as S

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "synth"
PREFERRED = ["deepseek-ai/DeepSeek-V4.1-Flash-1017", "Qwen/Qwen3-30B-A3B-Instruct-2507", "zai-org/GLM-5.2", "MiniMaxAI/MiniMax-M3"]
SCENES = [
    "pediatric fever discharge from the ER (acetaminophen and ibuprofen doses by weight, warning signs)",
    "ear infection, amoxicillin course for a toddler (mL dose, twice a day, 10 days, finish the course)",
    "asthma follow-up (albuterol inhaler puffs, every 4 hours as needed, when to go to the ER)",
    "allergy and medication history at intake (allergies to penicillin or sulfa, current medicines and doses)",
    "type 2 diabetes visit (metformin dose with food, insulin units at night, low-sugar symptoms)",
    "warfarin check (dose change, bleeding warning signs, no aspirin or ibuprofen)",
    "sprained ankle in urgent care (which side, ice, ibuprofen limit, return if numbness)",
    "kidney stone, flank pain (which side, pain scale, blood in urine, fever means come back)",
    "post-op wound care after appendectomy (keep dry for 2 days, no lifting more than 10 pounds for 2 weeks, fever above 38)",
    "prenatal visit (weeks pregnant, bleeding, prenatal vitamins, glucose test)",
    "chest pain triage (when it started, how long it lasts, left arm, shortness of breath, nitroglycerin)",
    "gastroenteritis in a child (ondansetron, small sips, signs of dehydration, no urine in 8 hours)",
    "head injury discharge (vomiting, confusion, wake him every few hours tonight, no sports for 1 week)",
    "high blood pressure (lisinopril once daily in the morning, dizziness, do not stop suddenly)",
    "urinary tract infection (antibiotic for 3 or 5 days, drink fluids, back pain and fever mean return)",
    "knee pain consult (right or left knee, how many weeks, physical therapy twice a week)",
]

VISIT_SCHEMA = {"type": "object", "properties": {"turns": {"type": "array", "items": {"type": "object", "properties": {
    "speaker": {"type": "string", "enum": ["clinician", "patient"]},
    "source": {"type": "string", "description": "what the speaker said, in their own language (clinician English, patient Spanish)"},
    "rendition": {"type": "string", "description": "a complete, faithful interpretation into the other language"},
    "reworded": {"type": "string", "description": "another fully faithful interpretation, worded differently (third person like 'dice que'/'she says', other word order, synonyms); same meaning, nothing dropped or added"},
    "error_type": {"type": "string", "enum": ["omission", "substitution", "addition"]},
    "with_error": {"type": "string", "description": "the interpretation with exactly ONE realistic error of error_type affecting critical content (a dose, number, time, frequency, allergy, side, medicine, a 'no', or a warning); everything else faithful"},
    "what_changed": {"type": "string"}},
    "required": ["speaker", "source", "rendition", "reworded", "error_type", "with_error", "what_changed"], "additionalProperties": False}}},
    "required": ["turns"], "additionalProperties": False}


def pick_generator(name: str | None) -> ModelSpec:
    ids = {m.id for m in _get_client().models.list().data}
    for mid in ([name] if name else []) + PREFERRED:
        if mid in ids:
            return ModelSpec("generator", mid, 0.5, 1.5)
    raise SystemExit(f"No generator model available; have: {sorted(ids)[:40]}")


def write_visit(i: int, gen: ModelSpec, seed: int) -> tuple[list[dict], list[CallRecord]]:
    rng = random.Random(seed * 1000 + i)
    scene = SCENES[i % len(SCENES)]
    who = rng.choice(["a mother speaking for her child", "an adult patient", "an older patient", "a father speaking for his child"])
    calls: list[CallRecord] = []
    prompt = (
        f"Write a realistic medical visit through a professional Spanish interpreter: {scene}. The Spanish speaker is {who}. "
        "Use invented names only. Write 12-16 turns alternating mostly between clinician (English) and patient side (Spanish; "
        "everyday Mexican or Central American Spanish, sometimes informal). Most turns should carry critical content: doses, "
        "numbers, units, how often, how long, allergies, left/right, medicine names, a 'no'/'don't', warning signs. Include a few short "
        "turns with no critical content. For each turn give: the words as said; a complete faithful interpretation; a second faithful "
        "interpretation worded differently; and a version with exactly one realistic interpreter error of the stated type (vary the "
        "types across turns: omission most often, then substitution, then addition). Errors should look like real interpreter slips: "
        "dropping 'no more than 4 a day', 'every 6 hours' becoming 'every 8 hours', 'left' becoming 'right', dropping the allergy, "
        "mg instead of mL, adding 'with food' that wasn't said. Turns with no critical content still need an error version (a small "
        "critical-looking change is fine).")
    try:
        out = chat_json(gen, [{"role": "user", "content": prompt}], schema=VISIT_SCHEMA, purpose="Synthetic interpreted visit",
                        reasoning=False, max_tokens=9000, temperature=0.8, calls=calls, enforce_cap=False)
    except (LiveModelUnavailable, Exception) as e:  # noqa: BLE001
        print(f"  visit {i} failed: {str(e)[:120]}")
        return [], calls
    group = f"{gen.model_id.split('/')[-1]}-v{i:03d}"
    rows = []
    for k, t in enumerate(out.get("turns", [])):
        if not t["source"].strip() or not t["rendition"].strip():
            continue
        sl, rl = ("en", "es") if t["speaker"] == "clinician" else ("es", "en")
        base = {"group": group, "turn": k, "scene": scene, "speaker": t["speaker"], "src_lang": sl, "ren_lang": rl, "source": t["source"]}
        rows.append(base | {"rendition": t["rendition"], "label": 0, "variant": "faithful"})
        if t["reworded"].strip() and t["reworded"].strip() != t["rendition"].strip():
            if not [f for f in S.diff(t["source"], t["reworded"], sl, rl) if f.severity == "high"]:
                rows.append(base | {"rendition": t["reworded"], "label": 0, "variant": "reworded"})
        if t["with_error"].strip() and t["with_error"].strip() != t["rendition"].strip():
            rows.append(base | {"rendition": t["with_error"], "label": 1, "variant": f"llm_{t['error_type']}", "note": t["what_changed"]})
        rows += [base | v for v in code_variants(t["rendition"], rl, rng)]
    return rows, calls


# ---------------------------------------------------------------------------------------------------------------------
# Code-made variants (exact labels)
# ---------------------------------------------------------------------------------------------------------------------
NUM_WORDS = {"en": {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight", 10: "ten", 12: "twelve"},
             "es": {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 8: "ocho", 10: "diez", 12: "doce"}}
UNIT_SWAP = {"en": [("milliliters", "milligrams"), ("mL", "mg"), ("ml", "mg"), ("teaspoon", "tablespoon"), ("tablets", "teaspoons"), ("tablet", "teaspoon"),
                    ("days", "weeks"), ("hours", "days"), ("pounds", "kilos")],
             "es": [("mililitros", "miligramos"), ("mL", "mg"), ("ml", "mg"), ("cucharadita", "cucharada"), ("pastillas", "cucharaditas"),
                    ("pastilla", "cucharadita"), ("días", "semanas"), ("horas", "días"), ("libras", "kilos")]}
SIDE_SWAP = {"en": [("left", "right"), ("right", "left")], "es": [("izquierdo", "derecho"), ("izquierda", "derecha"), ("derecho", "izquierdo"), ("derecha", "izquierda")]}
NEG_DROP = {"en": [(r"\bdo not\b", "do"), (r"\bdon't\b", "do"), (r"\bdoes not\b", "does"), (r"\bnot\b ", ""), (r"\bnever\b", "always"), (r"\bno more than\b", "")],
            "es": [(r"\bno\s+(?=\w)", ""), (r"\bnunca\b", "siempre"), (r"\bno más de\b", "")]}
DRUG_SWAP = [("ibuprofen", "acetaminophen"), ("acetaminophen", "ibuprofen"), ("Tylenol", "Motrin"), ("Motrin", "Tylenol"), ("penicillin", "amoxicillin"),
             ("ibuprofeno", "acetaminofén"), ("acetaminofén", "ibuprofeno"), ("penicilina", "amoxicilina"), ("aspirina", "ibuprofeno"), ("aspirin", "ibuprofen")]
GENERIC = [("Tylenol", "acetaminophen"), ("Motrin", "ibuprofen"), ("Advil", "ibuprofen"), ("acetaminofén", "paracetamol"), ("Tylenol", "acetaminofén")]
FRAME = {"en": ["She says ", "He says ", "The patient says "], "es": ["Dice el doctor que ", "El doctor dice que ", "Dice que "]}


def _sub_once(text: str, a: str, b: str, regex: bool = False) -> str | None:
    pat = a if regex else rf"(?<![\w]){re.escape(a)}(?![\w])"
    new, n = re.subn(pat, b, text, count=1, flags=re.I if regex else 0)
    return new if n and new != text else None


def code_variants(r: str, lang: str, rng: random.Random) -> list[dict]:
    errs, oks = [], []
    # change a number (digits or words)
    m = re.search(r"(?<![\w.])(\d+)(?![\w])", r)
    if m:
        n = int(m.group(1))
        new = rng.choice([x for x in {n + 1, n * 2, max(1, n - 1), n + 2, n * 10} if x != n])
        errs.append(("code_number", r[:m.start()] + str(new) + r[m.end():]))
        if n in NUM_WORDS[lang]:
            oks.append(("code_digits_to_words", r[:m.start()] + NUM_WORDS[lang][n] + r[m.end():]))
    else:
        for n, w in NUM_WORDS[lang].items():
            v = _sub_once(r, w, NUM_WORDS[lang][rng.choice([x for x in NUM_WORDS[lang] if x != n])])
            if v:
                errs.append(("code_number", v))
                oks.append(("code_words_to_digits", _sub_once(r, w, str(n)) or r))
                break
    for a, b in UNIT_SWAP[lang]:
        v = _sub_once(r, a, b)
        if v:
            errs.append(("code_unit", v)); break
    for a, b in SIDE_SWAP[lang]:
        v = _sub_once(r, a, b)
        if v:
            errs.append(("code_side", v)); break
    for a, b in NEG_DROP[lang]:
        v = _sub_once(r, a, b, regex=True)
        if v:
            errs.append(("code_negation", re.sub(r"\s{2,}", " ", v))); break
    for a, b in DRUG_SWAP:
        v = _sub_once(r, a, b)
        if v:
            errs.append(("code_drug", v)); break
    # drop the clause that carries an allergy / warning / limit / frequency
    parts = re.split(r"(?<=[,.;])\s+|\s+(?=(?:and|but|y|pero)\s)", r)
    if len(parts) > 1:
        crit = [i for i, p in enumerate(parts) if any(s.kind in ("allergy", "warning", "limit", "frequency", "number") for s in S.extract(p, lang))]
        if crit:
            i = rng.choice(crit)
            dropped = " ".join(p for j, p in enumerate(parts) if j != i).strip(" ,;")
            if dropped and len(dropped) < len(r) - 6:
                errs.append(("code_drop_clause", dropped))
    for a, b in GENERIC:
        v = _sub_once(r, a, b)
        if v:
            oks.append(("code_brand_generic", v)); break
    # 'She says ...' framing is normal for interpreters rendering the patient into English; only for statements.
    if lang == "en" and r and not r.rstrip().endswith("?") and not re.match(r"(?i)(she|he|the patient|the mother|the father) says", r) \
            and not re.search(r"\b(I|I'm|I've|my|me|we|our)\b", r):
        oks.append(("code_framing", rng.choice(FRAME[lang]) + r[0].lower() + r[1:]))
    rng.shuffle(errs)
    out = [{"rendition": v, "label": 1, "variant": k} for k, v in errs[:3]]
    out += [{"rendition": v, "label": 0, "variant": k} for k, v in oks[:2] if v and v != r]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--generator", default=None)
    ap.add_argument("--split", default="trainval", choices=["trainval", "test"])
    a = ap.parse_args()
    gen = pick_generator(a.generator)
    print("generator:", gen.model_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, calls = [], []
    with ThreadPoolExecutor(8) as ex:
        for rs, c in ex.map(lambda i: write_visit(i, gen, a.seed), range(a.n)):
            rows += rs
            calls += c
    groups = sorted({r["group"] for r in rows})
    rng = random.Random(a.seed)
    rng.shuffle(groups)
    dev = set(groups[: max(1, len(groups) // 6)])
    for r in rows:
        r["generator"] = gen.model_id
        r["split"] = "test" if a.split == "test" else ("dev" if r["group"] in dev else "train")
    out = OUT_DIR / f"turns_{a.split}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    by = {}
    for r in rows:
        by[r["variant"]] = by.get(r["variant"], 0) + 1
    print(f"{len(groups)} visits, {len(rows)} pairs ({sum(r['label'] for r in rows)} with errors) -> {out}")
    print(by)
    print(f"{len(calls)} calls, ${sum(c.cost_usd for c in calls):.3f}")


if __name__ == "__main__":
    main()
