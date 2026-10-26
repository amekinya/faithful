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


def load_synth(split: str) -> list[dict]:
    p = ROOT / "data" / "synth" / ("turns_test.jsonl" if split == "test" else "turns_trainval.jsonl")
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return [r | {"error": bool(r["label"])} for r in rows if r["split"] == split]


def load_aces() -> list[dict]:
    import urllib.request

    cache = ROOT / "eval" / "_aces.jsonl"
    if not cache.exists():
        cache.write_bytes(urllib.request.urlopen(ACES_URL, timeout=300).read())
    out = []
    for line in cache.read_text().splitlines():
        r = json.loads(line)
        if r.get("langpair") not in ("en-es", "es-en") or not r["phenomena"].startswith(ACES_KEEP) or r["phenomena"] in ACES_DROP:
            continue
        sl, rl = r["langpair"].split("-")
        base = {"source": r["source"], "src_lang": sl, "ren_lang": rl, "group": r["phenomena"], "variant": r["phenomena"]}
        out.append(base | {"rendition": r["good-translation"], "error": False})
        out.append(base | {"rendition": r["incorrect-translation"], "error": True})
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


def best_threshold(errs: list[bool], p_err: list[float]) -> float:
    def f1(t):
        tp = sum(1 for e, p in zip(errs, p_err) if e and p >= t); fp = sum(1 for e, p in zip(errs, p_err) if not e and p >= t)
        fn = sum(1 for e, p in zip(errs, p_err) if e and p < t)
        return 2 * tp / max(2 * tp + fp + fn, 1)
    return max((t / 100 for t in range(5, 96)), key=f1)


def tune_gate(rows, p_faith) -> dict:
    """Pick the decision rule's two thresholds on dev: best F1 among settings with false alarms <= 5%."""
    fnd = [[f.as_dict() for f in slots.diff(r["source"], r["rendition"], r["src_lang"], r["ren_lang"])] for r in rows]
    best, best_key = dict(gate.DEFAULT), (-1, -1)
    n_ok = max(sum(1 for r in rows if not r["error"]), 1)
    for c in range(5, 80, 5):
        for a in range(50, 100, 5):
            th = {"confirm": c / 100, "alone": a / 100}
            fl = [gate.decide(f, p, th)["level"] == "card" for f, p in zip(fnd, p_faith)]
            tp = sum(1 for r, x in zip(rows, fl) if r["error"] and x); fp = sum(1 for r, x in zip(rows, fl) if not r["error"] and x)
            fn = sum(1 for r, x in zip(rows, fl) if r["error"] and not x)
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            key = (fp / n_ok <= 0.05, f1)
            if key > best_key:
                best_key, best = key, th
    return best


def metrics(rows, preds, seconds, cost, ms_list=None) -> dict:
    n_err = sum(1 for r in rows if r["error"]); n_ok = len(rows) - n_err
    done = [(r, p) for r, p in zip(rows, preds) if p["flag"] is not None]
    tp = sum(1 for r, p in done if r["error"] and p["flag"]); fp = sum(1 for r, p in done if not r["error"] and p["flag"])
    fn = sum(1 for r, p in done if r["error"] and not p["flag"])
    ms = ms_list if ms_list is not None else [p["ms"] for _, p in done if p["ms"]]
    q = sorted(ms)
    return {"turns": len(rows), "errors": n_err, "caught": tp, "recall": round(tp / max(n_err, 1), 3), "false_alarms": fp,
            "false_alarm_rate": round(fp / max(n_ok, 1), 3), "precision": round(tp / max(tp + fp, 1), 3),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 3), "failed": len(rows) - len(done),
            "ms_median": round(statistics.median(q), 1) if q else None, "ms_p95": round(q[int(0.95 * (len(q) - 1))], 1) if q else None,
            "cost_per_1000": round(cost / max(len(rows), 1) * 1000, 3), "seconds": round(seconds, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="synth,aces,visits")
    ap.add_argument("--methods", default="code,lightning,ultra,base_ranker,checker,faithful")
    ap.add_argument("--n", type=int, default=0, help="cap rows per set (random sample, fixed seed) for prompt baselines")
    a = ap.parse_args()
    methods = a.methods.split(",")
    rng = random.Random(3)
    sets = {"synth": lambda: load_synth("test"), "aces": load_aces, "visits": load_visits}
    prev = json.loads((ROOT / "eval" / "results.json").read_text()) if (ROOT / "eval" / "results.json").exists() else {}
    results = prev.get("results", {})

    th = {}
    need_ranker = {"base_ranker", "checker", "faithful"} & set(methods)
    if need_ranker:
        if not endpoint.available():
            print("ENDPOINT_URL not set: skipping", sorted(need_ranker))
            methods = [m for m in methods if m not in need_ranker]
        else:
            dev = load_synth("dev")
            p_dev_base = score_rows(dev, "base-reranker", [])
            p_dev = score_rows(dev, settings.checker_name, [])
            errs = [r["error"] for r in dev]
            th = {"base_ranker": best_threshold(errs, [1 - p for p in p_dev_base]), "checker": best_threshold(errs, [1 - p for p in p_dev])}
            th["gate"] = tune_gate(dev, p_dev)
            (ROOT / "data" / "thresholds.json").write_text(json.dumps(th["gate"], indent=1))
            print("thresholds from dev:", th)

    for sname in a.sets.split(","):
        rows = sets[sname]()
        if not rows:
            print(f"{sname}: no rows")
            continue
        results.setdefault(sname, {})
        for m in methods:
            sub = rows
            if a.n and m in ("lightning", "ultra") and len(rows) > a.n:
                sub = rng.sample(rows, a.n)
            calls: list[CallRecord] = []
            t0 = time.perf_counter()
            ms_list = None
            if m == "code":
                preds = run_code(sub)
                t_each = (time.perf_counter() - t0) * 1000 / max(len(sub), 1)
                ms_list = [t_each] * len(sub)
            elif m in ("lightning", "ultra"):
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
    out = {"results": results, "thresholds": th or prev.get("thresholds", {}), "names": names, "set_names": set_names,
           "generated": time.strftime("%Y-%m-%d %H:%M")}
    (ROOT / "eval" / "results.json").write_text(json.dumps(out, indent=1))
    md = ["# Faithful benchmark", ""]
    for sname, by in results.items():
        md += [f"## {set_names.get(sname, sname)}", "", "| Method | Errors caught | False alarms | Precision | F1 | Median time / turn | Cost / 1,000 turns |",
               "|---|---|---|---|---|---|---|"]
        for m, r in by.items():
            md.append(f"| {names.get(m, m)} | {r['caught']}/{r['errors']} ({r['recall']:.0%}) | {r['false_alarms']} ({r['false_alarm_rate']:.1%}) | "
                      f"{r['precision']:.0%} | {r['f1']:.2f} | {r['ms_median'] if r['ms_median'] is not None else '-'} ms | ${r['cost_per_1000']:.2f} |")
        md.append("")
    (ROOT / "eval" / "RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
