"""Does Faithful catch critical interpreting errors without crying wolf? Same turns, several methods.

Test sets
  synth    held-out visits written by a different model than the training visits (exact labels by construction)
  aces     ACES challenge set, English<->Spanish, the critical phenomena only (numbers, units, dates, antonyms, omissions,
           additions, names): general-domain, human-designed, never seen in training (out-of-distribution). Downloaded at run time.
  visits   our 6 scripted visits (20 planted errors, JMIR 2026 design). The code rules were written while looking at these,
           so the code-only score here is optimistic.
Methods
  code          slot diff only (a HIGH-severity change = flag)
  lightning     Nemotron 3.5 Lightning, prompt only, one call per turn
  ultra         Nemotron 3 Ultra, prompt only, one call per turn
  base_ranker   NVIDIA llama-nemotron-rerank-1b-v2 before fine-tuning (threshold chosen on dev)
  checker       our fine-tuned checker alone (threshold chosen on dev)
  faithful      code slot diff + our checker + the fixed decision rule (thresholds chosen on dev)
Metrics: errors caught (recall), false alarms on faithful turns, precision, F1, median/p95 time per turn, cost per 1,000 turns.
Fairness: with --n, one random sample per set is drawn once and every method is scored on it. Rows a model failed to answer
are excluded from that method's counts and reported as "failed". Every threshold (base ranker, checker alone, the decision
rule) is chosen on the synthetic dev split under the same constraint: best F1 with false alarms <= 5% on dev. Running a set
replaces all of its earlier results, so no row in the table comes from an older run.
Usage: python -m eval.run_eval --sets synth,aces,visits --methods code,lightning,ultra,base_ranker,checker,faithful [--n 400]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.ai import endpoint, tasks
from app.ai.client import CallRecord, LiveModelUnavailable
from app.config import settings
from app.engine import gate, slots

ROOT = Path(__file__).resolve().parent.parent
ACES_URL = "https://huggingface.co/datasets/nikitam/ACES/resolve/main/challenge_set.jsonl"
ACES_KEEP = ("hallucination-number", "hallucination-unit", "hallucination-date-time", "antonym-replacement", "omission", "addition",
             "xnli-omission-contradiction", "xnli-addition-contradiction", "hallucination-named-entity")
ACES_DROP = ("xnli-omission-neutral", "xnli-addition-neutral")
# ACES "unit conversion" pairs count a correct conversion (12 km -> 7.5 miles) as faithful; in a clinic a converted dose
# should be re-confirmed, so those pairs measure a different policy and are left out.
ACES_DROP_SUB = ("conversion",)


def load_synth(split: str) -> list[dict]:
    p = ROOT / "data" / "synth" / ("turns_test.jsonl" if split == "test" else "turns_trainval.jsonl")
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return [r | {"error": bool(r["label"])} for r in rows if r["split"] == split]


def load_aces() -> list[dict]:
    import urllib.request

    cache = ROOT / "eval" / "_aces.jsonl"
    if not cache.exists():
        cache.write_bytes(urllib.request.urlopen(ACES_URL, timeout=300).read())
    out, seen = [], set()
    for line in cache.read_text().splitlines():
        r = json.loads(line)
        ph = r["phenomena"]
        if r.get("langpair") not in ("en-es", "es-en") or not ph.startswith(ACES_KEEP) or ph in ACES_DROP or any(x in ph for x in ACES_DROP_SUB):
            continue
        sl, rl = r["langpair"].split("-")
        base = {"source": r["source"], "src_lang": sl, "ren_lang": rl, "group": ph, "variant": ph}
        for ren, err in ((r["good-translation"], False), (r["incorrect-translation"], True)):
            key = (r["source"].strip(), ren.strip(), sl, err)
            if key in seen:  # ACES repeats many good translations across phenomena
                continue
            seen.add(key)
            out.append(base | {"rendition": ren, "error": err})
    return out


def load_visits() -> list[dict]:
    d = json.loads((ROOT / "data" / "encounters" / "visits.json").read_text())
    out = []
    for v in d["visits"]:
        for i, t in enumerate(v["turns"], 1):
            sl, rl = ("en", "es") if t["speaker"] == "clinician" else ("es", "en")
            out.append({"source": t["source"], "rendition": t["rendition"], "src_lang": sl, "ren_lang": rl, "group": v["id"],
                        "variant": (t.get("error") or {}).get("type", "faithful"), "error": "error" in t, "speaker": t["speaker"], "turn": f"{v['id']}-{i}"})
    return out


# ---------------------------------------------------------------------------------------------------------------------
def run_code(rows):
    return [{"flag": any(f.severity == "high" for f in slots.diff(r["source"], r["rendition"], r["src_lang"], r["ren_lang"])), "ms": 0.0} for r in rows]


def run_prompt(rows, spec, calls):
    def one(r):
        local: list[CallRecord] = []
        t0 = time.perf_counter()
        try:
            j = tasks.judge_turn(r["source"], r["rendition"], r["src_lang"], r["ren_lang"], spec=spec, calls=local)
            flag = not j["faithful"]
        except LiveModelUnavailable:
            flag = None
        return {"flag": flag, "ms": (time.perf_counter() - t0) * 1000}, local
    out = []
    with ThreadPoolExecutor(8) as ex:
        for res, local in ex.map(one, rows):
            out.append(res)
            calls.extend(local)
    return out


def score_rows(rows, model, calls) -> list[float]:
    pairs = [(endpoint.query(r["source"], r["src_lang"]), endpoint.passage(r["rendition"], r["ren_lang"])) for r in rows]
    return endpoint.score(pairs, calls, model=model, timeout=600)


def per_turn_ms(model: str, rows: list[dict], k: int = 25) -> list[float]:
    """Latency the way the live app sees it: one turn per request."""
    ms = []
    for r in rows[:k]:
        t0 = time.perf_counter()
        try:
            score_rows([r], model, [])
        except LiveModelUnavailable:
            break
        ms.append((time.perf_counter() - t0) * 1000)
    return ms


MAX_DEV_FA = 0.05  # every method's threshold: best F1 on dev among settings with <= 5% false alarms on dev


def _key(errs: list[bool], flags: list[bool]) -> tuple[bool, float]:
    tp = sum(1 for e, x in zip(errs, flags) if e and x); fp = sum(1 for e, x in zip(errs, flags) if not e and x)
    fn = sum(1 for e, x in zip(errs, flags) if e and not x)
    n_ok = max(sum(1 for e in errs if not e), 1)
    return fp / n_ok <= MAX_DEV_FA, 2 * tp / max(2 * tp + fp + fn, 1)


def best_threshold(errs: list[bool], p_err: list[float]) -> tuple[float, bool]:
    """-> (threshold, whether it meets the dev false-alarm limit)."""
    t = max((t / 100 for t in range(5, 100)), key=lambda t: _key(errs, [p >= t for p in p_err]))
    return t, _key(errs, [p >= t for p in p_err])[0]


def tune_gate(rows, p_faith) -> tuple[dict, bool]:
    """Pick the decision rule's two thresholds on dev, under the same rule as best_threshold."""
    fnd = [[f.as_dict() for f in slots.diff(r["source"], r["rendition"], r["src_lang"], r["ren_lang"])] for r in rows]
    errs = [r["error"] for r in rows]
    best, best_key = dict(gate.DEFAULT), (False, -1.0)
    for c in range(5, 80, 5):
        for a in range(50, 100, 5):
            th = {"confirm": c / 100, "alone": a / 100}
            key = _key(errs, [gate.decide(f, p, th)["level"] == "card" for f, p in zip(fnd, p_faith)])
            if key > best_key:
                best_key, best = key, th
    return best, best_key[0]


def metrics(rows, preds, seconds, cost, ms_list=None) -> dict:
    """Rows the model failed to answer are left out of every count and reported as "failed"."""
    done = [(r, p) for r, p in zip(rows, preds) if p["flag"] is not None]
    n_err = sum(1 for r, _ in done if r["error"]); n_ok = len(done) - n_err
    tp = sum(1 for r, p in done if r["error"] and p["flag"]); fp = sum(1 for r, p in done if not r["error"] and p["flag"])
    fn = sum(1 for r, p in done if r["error"] and not p["flag"])
    ms = ms_list if ms_list is not None else [p["ms"] for _, p in done if p["ms"]]
    q = sorted(ms)
    return {"turns": len(done), "errors": n_err, "caught": tp, "recall": round(tp / max(n_err, 1), 3), "false_alarms": fp,
            "false_alarm_rate": round(fp / max(n_ok, 1), 3), "precision": round(tp / max(tp + fp, 1), 3),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 3), "failed": len(rows) - len(done),
            "ms_median": round(statistics.median(q), 1) if q else None, "ms_p95": round(q[int(0.95 * (len(q) - 1))], 1) if q else None,
            "cost_per_1000": round(cost / max(len(rows), 1) * 1000, 3), "seconds": round(seconds, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="synth,aces,visits")
    ap.add_argument("--methods", default="code,lightning,ultra,base_ranker,checker,faithful")
    ap.add_argument("--n", type=int, default=0, help="cap rows per set: one random sample (fixed seed) shared by every method")
    a = ap.parse_args()
    methods = a.methods.split(",")
    sets = {"synth": lambda: load_synth("test"), "aces": load_aces, "visits": load_visits}
    prev = json.loads((ROOT / "eval" / "results.json").read_text()) if (ROOT / "eval" / "results.json").exists() else {}
    results = prev.get("results", {})
    meta = prev.get("meta", {})

    th = {}
    need_ranker = {"base_ranker", "checker", "faithful"} & set(methods)
    if need_ranker:
        if not endpoint.available():
            print("ENDPOINT_URL not set: skipping", sorted(need_ranker))
            methods = [m for m in methods if m not in need_ranker]
        else:
            dev = load_synth("dev")
            if not dev:
                raise SystemExit("no dev split in data/synth/turns_trainval.jsonl: thresholds can't be chosen")
            p_dev_base = score_rows(dev, "base-reranker", [])
            p_dev = score_rows(dev, settings.checker_name, [])
            errs = [r["error"] for r in dev]
            (tb, ob), (tc, oc) = best_threshold(errs, [1 - p for p in p_dev_base]), best_threshold(errs, [1 - p for p in p_dev])
            tg, og = tune_gate(dev, p_dev)
            th = {"base_ranker": tb, "checker": tc, "gate": tg, "dev_rows": len(dev), "max_dev_false_alarms": MAX_DEV_FA,
                  "met_limit": {"base_ranker": ob, "checker": oc, "gate": og}}
            (ROOT / "data" / "thresholds.json").write_text(json.dumps(tg | {"chosen_on": f"synthetic dev split, {len(dev)} turns",
                                                                          "generated": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
            print("thresholds from dev:", th)

    for sname in a.sets.split(","):
        rows = sets[sname]()
        if not rows:
            print(f"{sname}: no rows")
            continue
        results[sname] = {}  # a set that runs replaces all of its earlier rows
        sub = rows
        if a.n and len(rows) > a.n:
            sub = random.Random(3).sample(rows, a.n)
        meta[sname] = {"rows": len(sub), "of": len(rows), "generated": time.strftime("%Y-%m-%d %H:%M")}
        for m in methods:
            calls: list[CallRecord] = []
            t0 = time.perf_counter()
            ms_list = None
            if m == "code":
                preds = run_code(sub)
                t_each = (time.perf_counter() - t0) * 1000 / max(len(sub), 1)
                ms_list = [t_each] * len(sub)
            elif m in ("lightning", "ultra"):
                if not settings.live_models:
                    print(f"{sname} {m}: NEBIUS_API_KEY not set, skipped")
                    continue
                preds = run_prompt(sub, settings.lightning if m == "lightning" else settings.ultra, calls)
            else:
                model = "base-reranker" if m == "base_ranker" else settings.checker_name
                p = score_rows(sub, model, calls)
                if m == "faithful":
                    fnd = [[f.as_dict() for f in slots.diff(r["source"], r["rendition"], r["src_lang"], r["ren_lang"])] for r in sub]
                    preds = [{"flag": gate.decide(f, x, th["gate"])["level"] == "card", "ms": 0} for f, x in zip(fnd, p)]
                else:
                    preds = [{"flag": (1 - x) >= th[m], "ms": 0} for x in p]
                ms_list = per_turn_ms(model, sub)
            cost = sum(c.cost_usd for c in calls)
            res = metrics(sub, preds, time.perf_counter() - t0, cost, ms_list)
            if sname == "visits":
                res["missed"] = [r["turn"] for r, p in zip(sub, preds) if r["error"] and not p["flag"]]
                res["false_alarm_turns"] = [r["turn"] for r, p in zip(sub, preds) if not r["error"] and p["flag"]]
                res["patient_speech_caught"] = f"{sum(1 for r, p in zip(sub, preds) if r['error'] and p['flag'] and r['speaker'] == 'patient')}/" \
                                               f"{sum(1 for r in sub if r['error'] and r['speaker'] == 'patient')}"
            results[sname][m] = res
            print(sname, m, res)

    names = {"code": "Code slot diff only (no model)", "lightning": "Nemotron 3.5 Lightning, prompt only", "ultra": "Nemotron 3 Ultra, prompt only",
             "base_ranker": "NVIDIA rerank-1b, before fine-tuning", "checker": "Our fine-tuned checker alone (Nebius)",
             "faithful": "Faithful: code + our checker + decision rule"}
    set_names = {"synth": "Held-out synthetic visits (different writer model)", "aces": "ACES English-Spanish challenge set (out-of-distribution)",
                 "visits": "Our 6 scripted visits (20 planted errors)"}
    out = {"results": results, "meta": meta, "thresholds": th or prev.get("thresholds", {}), "names": names, "set_names": set_names,
           "generated": time.strftime("%Y-%m-%d %H:%M")}
    (ROOT / "eval" / "results.json").write_text(json.dumps(out, indent=1))
    md = ["# Faithful benchmark", ""]
    for sname, by in results.items():
        mt = meta.get(sname, {})
        md += [f"## {set_names.get(sname, sname)}", "",
               f"{mt.get('rows', '?')} turns" + (f" (random sample of {mt['of']})" if mt.get("of", 0) > mt.get("rows", 0) else "") + f", run {mt.get('generated', '?')}.", "",
               "| Method | Errors caught | False alarms | Precision | F1 | Median time / turn | Cost / 1,000 turns | Failed |",
               "|---|---|---|---|---|---|---|---|"]
        for m, r in by.items():
            md.append(f"| {names.get(m, m)} | {r['caught']}/{r['errors']} ({r['recall']:.0%}) | {r['false_alarms']} ({r['false_alarm_rate']:.1%}) | "
                      f"{r['precision']:.0%} | {r['f1']:.2f} | {r['ms_median'] if r['ms_median'] is not None else '-'} ms | ${r['cost_per_1000']:.2f} | {r['failed']} |")
        md.append("")
    (ROOT / "eval" / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
