"""Synthetic interpreted medical visits with known errors, for training and testing the fidelity checker.

1. A model from a DIFFERENT family than the Nemotron models Faithful uses writes short English-Spanish visits:
   each turn has the speaker's words and a faithful interpretation into the other language.
2. The same writer adds, per turn, one faithful re-wording (third person, other word order, synonyms) and one
   rendition with a single realistic interpreter error (omission / substitution / addition of critical content).
3. Training visits only: code adds more variants with exact labels: changed numbers, swapped units, flipped left/right,
   dropped 'no', swapped medicines, dropped clauses (errors); digits<->number words, brand<->generic names, 'she says'
   framing (faithful). The TEST split gets none of these, because they are built from the same word lists our code
   checks with, and would flatter the code baseline.
4. Training visits only: code keeps a faithful re-wording only if the slot diff finds no critical change. Test visits
   are never filtered by the system under test.
Writes data/synth/turns_<generator>.jsonl (one row per source/rendition pair with label, split, group).
Usage: python -m scripts.synth --n 60 --generator deepseek-ai/DeepSeek-V4.1-Flash-1017 --split trainval
       python -m scripts.synth --rederive     (rebuild the code-made variants of existing visits after a code change; no model calls)
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
# A tiny language check, independent of the system under test: generators sometimes "interpret" into the same
# language. Such rows are dropped from both splits.
_ES = set("el la los las de que y en un una es por con para se su al lo como pero sus le ya este sí porque esta entre cuando muy "
          "sobre también me hasta hay donde desde todo nos durante todos les ni ese eso esto antes yo otro él tanto esa estos mucho "
          "nada usted ustedes cada día días horas veces tiene tome dele le está están tengo duele dolor mi bien gracias doctora doctor".split())
_EN = set("the of and to in is you that it he was for on are as with his they at be this have from or had by but not what all were "
          "we when your can there an each which she do how their if will up about out then them these so some her would like him into "
          "has more no could my than been who its now did get may every hours day days take give i'm it's don't doctor thank pain".split())


def lang_of(text: str) -> str:
    w = re.findall(r"[a-záéíóúñü']+", text.lower())
    e, s = sum(x in _EN for x in w), sum(x in _ES for x in w)
    return "en" if e > s else "es" if s > e else "?"


def in_lang(text: str, lang: str) -> bool:
    return lang_of(text) in (lang, "?")


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
    "rendition": {"type": "string", "description": "a complete, faithful interpretation into the OTHER language: Spanish for a clinician turn, English for a patient turn"},
    "reworded": {"type": "string", "description": "another fully faithful interpretation, worded differently (third person like 'dice que'/'she says', other word order, synonyms); same meaning, nothing dropped or added"},
    "error_type": {"type": "string", "enum": ["omission", "substitution", "addition"]},
    "with_error": {"type": "string", "description": "the interpretation with exactly ONE realistic error of error_type affecting critical content (a dose, number, time, frequency, allergy, side, medicine, a 'no', or a warning); everything else faithful"},
    "what_changed": {"type": "string"}},
    "required": ["speaker", "source", "rendition", "reworded", "error_type", "with_error", "what_changed"], "additionalProperties": False}}},
    "required": ["turns"], "additionalProperties": False}


def pick_generator(name: str | None) -> ModelSpec:
    ids = {m.id for m in _get_client().models.list().data}
    if name and name not in ids:  # never fall back silently: train and test writers must differ
        raise SystemExit(f"Generator {name} isn't available; have: {sorted(ids)[:40]}")
    for mid in ([name] if name else []) + PREFERRED:
        if mid in ids:
            return ModelSpec("generator", mid, 0.5, 1.5)
    raise SystemExit(f"No generator model available; have: {sorted(ids)[:40]}")


def write_visit(i: int, gen: ModelSpec, seed: int, split: str = "trainval") -> tuple[list[dict], list[CallRecord]]:
    rng = random.Random(seed * 1000 + i)
    scene = SCENES[i % len(SCENES)]
    who = rng.choice(["a mother speaking for her child", "an adult patient", "an older patient", "a father speaking for his child"])
    calls: list[CallRecord] = []
    prompt = (
        f"Write a realistic medical visit through a professional Spanish interpreter: {scene}. The Spanish speaker is {who}. "
        "Use invented names only. Write 12-16 turns alternating mostly between clinician (English) and patient side (Spanish; "
        "the interpreter turns clinician English INTO SPANISH and patient Spanish INTO ENGLISH; "
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
    turns = out.get("turns", [])
    wrong = 0
    for k, t in enumerate(turns):
        if not t["source"].strip() or not t["rendition"].strip():
            continue
        sl, rl = ("en", "es") if t["speaker"] == "clinician" else ("es", "en")
        if not in_lang(t["source"], sl) or not in_lang(t["rendition"], rl):
            wrong += 1
            continue
        for key in ("reworded", "with_error"):
            if t[key].strip() and not in_lang(t[key], rl):
                t[key] = ""
        base = {"group": group, "turn": k, "scene": scene, "speaker": t["speaker"], "src_lang": sl, "ren_lang": rl, "source": t["source"]}
        rows.append(base | {"rendition": t["rendition"], "label": 0, "variant": "faithful"})
        if t["reworded"].strip() and t["reworded"].strip() != t["rendition"].strip():
            # Training data drops re-wordings the slot diff disagrees with (cleaner labels). The TEST split keeps them all,
            # so the code baseline isn't scored on data it filtered.
            if split == "test" or not [f for f in S.diff(t["source"], t["reworded"], sl, rl) if f.severity == "high"]:
                rows.append(base | {"rendition": t["reworded"], "label": 0, "variant": "reworded"})
        if t["with_error"].strip() and t["with_error"].strip() != t["rendition"].strip():
            rows.append(base | {"rendition": t["with_error"], "label": 1, "variant": f"llm_{t['error_type']}", "note": t["what_changed"]})
        if split != "test":
            rows += [base | v for v in code_variants(t["rendition"], rl, rng)]
    if wrong:
        print(f"  visit {i}: {wrong} of {len(turns)} turns dropped (interpretation in the wrong language)")
    if turns and wrong > len(turns) * 0.6:
        print(f"  visit {i} dropped: too many turns in the wrong language")
        return [], calls
    return rows, calls


def rederive() -> None:
    """Re-make the code variants from the visits already written (deterministic per turn), without new model calls."""
    import zlib
    for split in ("trainval", "test"):
        p = OUT_DIR / f"turns_{split}.jsonl"
        if not p.exists():
            continue
        old = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        keep = [r for r in old if not r["variant"].startswith("code_") and in_lang(r["source"], r["src_lang"]) and in_lang(r["rendition"], r["ren_lang"])]
        rows = []
        for r in keep:
            rows.append(r)
            if split != "test" and r["variant"] == "faithful":
                rng = random.Random(zlib.crc32(f"{r['group']}#{r['turn']}".encode()))
                base = {k: r[k] for k in r if k not in ("rendition", "label", "variant", "note")}
                rows += [base | v for v in code_variants(r["rendition"], r["ren_lang"], rng)]
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        by: dict[str, int] = {}
        for r in rows:
            by[r["variant"]] = by.get(r["variant"], 0) + 1
        print(f"{split}: {len(old)} -> {len(rows)} pairs", by)


# ---------------------------------------------------------------------------------------------------------------------
# Code-made variants (exact labels)
# ---------------------------------------------------------------------------------------------------------------------
NUM_WORDS = {"en": {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight", 10: "ten", 12: "twelve"},
             "es": {1: "una", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis", 8: "ocho", 10: "diez", 12: "doce"}}
UNIT_SWAP = {"en": [("milliliters", "milligrams"), ("mL", "mg"), ("ml", "mg"), ("teaspoon", "tablespoon"), ("tablets", "teaspoons"), ("tablet", "teaspoon"),
                    ("days", "weeks"), ("hours", "days"), ("pounds", "kilos")],
             "es": [("mililitros", "miligramos"), ("mL", "mg"), ("ml", "mg"), ("cucharadita", "cucharada"), ("pastillas", "cucharaditas"),
                    ("pastilla", "cucharadita"), ("días", "semanas"), ("horas", "días"), ("libras", "kilos")]}
# Sides are swapped only where they name a side (a body part, "the left one", "on the right"), never the verb "left"
# ("she left the clinic") or "right" meaning correct; Spanish never in "derecho a" (a right to).
SIDE_PATTERNS = {
    "en": [r"\b(left|right)(?=\s+(?:\w+\s+)?(?:side|arm|leg|ear|eye|hand|foot|knee|hip|shoulder|kidney|lung|breast|ankle|wrist|elbow|"
           r"finger|toe|thumb|thigh|flank|cheek|heel|calf|one|lower|upper)\b)", r"(?<=\b(?:on|to) the )(left|right)\b"],
    "es": [r"\b(izquierd[oa]|derech[oa])\b(?! a (?:la|las|los|el|un|una|recibir|tener)\b)"],
}
SIDE_OTHER = {"left": "right", "right": "left", "izquierdo": "derecho", "izquierda": "derecha", "derecho": "izquierdo", "derecha": "izquierda"}
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
    m = re.search(r"(?<![\w.,])(\d+)(?![\w.,]\d|[\w])", r)  # a whole number, not part of 2.5 or 1,000
    if m:
        n = int(m.group(1))
        new = rng.choice([x for x in {n + 1, n * 2, max(1, n - 1), n + 2, n * 10} if x != n])
        errs.append(("code_number", r[:m.start()] + str(new) + r[m.end():]))
        if n in NUM_WORDS[lang]:
            oks.append(("code_digits_to_words", r[:m.start()] + NUM_WORDS[lang][n] + r[m.end():]))
    else:
        for n, w in NUM_WORDS[lang].items():
            # "one"/"una" are often an article or pronoun ("una pregunta", "the left one"): only a quantity before a noun it counts
            pat = rf"(?<![\w]){re.escape(w)}(?![\w])" + (rf"(?=\s+(?:{'|'.join(sorted(S.NOUNS, key=len, reverse=True))})\b)" if n == 1 else "")
            if not re.search(pat, r):
                continue
            other = NUM_WORDS[lang][rng.choice([x for x in NUM_WORDS[lang] if x != n])]
            errs.append(("code_number", re.sub(pat, other, r, count=1)))
            oks.append(("code_words_to_digits", re.sub(pat, str(n), r, count=1)))
            break
    for a, b in UNIT_SWAP[lang]:
        v = _sub_once(r, a, b)
        if v:
            errs.append(("code_unit", v)); break
    for pat in SIDE_PATTERNS[lang]:
        m = re.search(pat, r, flags=re.I)
        if m:
            w = m.group(1)
            o = SIDE_OTHER[w.lower()]
            o = o.capitalize() if w[0].isupper() else o
            errs.append(("code_side", r[:m.start(1)] + o + r[m.end(1):])); break
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
    ap.add_argument("--rederive", action="store_true")
    ap.add_argument("--append", action="store_true", help="add these visits to the split's file instead of replacing it")
    a = ap.parse_args()
    if a.rederive:
        rederive()
        return
    gen = pick_generator(a.generator)
    print("generator:", gen.model_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, calls = [], []
    with ThreadPoolExecutor(8) as ex:
        for rs, c in ex.map(lambda i: write_visit(i, gen, a.seed, a.split), range(a.n)):
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
    if a.split == "trainval" and a.append:
        raise SystemExit("--append is for the test split (the dev split is drawn per run)")
    if a.append and out.exists():
        old = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        old = [r for r in old if r.get("generator") != gen.model_id]  # re-running a writer replaces its own visits
        print(f"keeping {len(old)} rows from other writers")
        rows = old + rows
    if not rows:
        raise SystemExit("no visits written; the file was left unchanged")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    by = {}
    for r in rows:
        by[r["variant"]] = by.get(r["variant"], 0) + 1
    print(f"{len(groups)} visits, {len(rows)} pairs ({sum(r['label'] for r in rows)} with errors) -> {out}")
    print(by)
    print(f"{len(calls)} calls, ${sum(c.cost_usd for c in calls):.3f}")


if __name__ == "__main__":
    main()
