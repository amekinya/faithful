"""Synthetic turns -> training pairs for the checker (label 1 = faithful, 0 = critical error), split by visit.
Usage: python -m scripts.build_pairs   (reads data/synth/turns_trainval.jsonl and turns_test.jsonl)"""
from __future__ import annotations

import json
from pathlib import Path

from app.ai.endpoint import passage, query

ROOT = Path(__file__).resolve().parent.parent
SYN = ROOT / "data" / "synth"


def main() -> None:
    rows = []
    for name in ("turns_trainval.jsonl", "turns_test.jsonl"):
        p = SYN / name
        if p.exists():
            rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    out = [{"query": query(r["source"], r["src_lang"]), "passage": passage(r["rendition"], r["ren_lang"]), "label": 1 - int(r["label"]),
            "split": r["split"], "group": r["group"], "variant": r["variant"]} for r in rows]
    (SYN / "pairs.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n")
    by = {}
    for x in out:
        k = (x["split"], "faithful" if x["label"] else "error")
        by[k] = by.get(k, 0) + 1
    print(f"{len(out)} pairs:", {f"{a}/{b}": n for (a, b), n in sorted(by.items())})


if __name__ == "__main__":
    main()
