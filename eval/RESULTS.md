# Faithful benchmark

## Held-out synthetic visits (different writer model)

400 turns (random sample of 1449), run 2026-10-28 01:51.

| Method | Errors caught | False alarms | Precision | F1 | Median time / turn | Cost / 1,000 turns | Failed |
|---|---|---|---|---|---|---|---|
| Code slot diff only (no model) | 64/128 (50%) | 26 (9.6%) | 71% | 0.59 | 1.7 ms | $0.00 | 0 |
| Nemotron 3.5 Lightning, prompt only | 65/128 (51%) | 4 (1.5%) | 94% | 0.66 | 869.2 ms | $0.03 | 0 |
| Nemotron 3 Ultra, prompt only | 101/128 (79%) | 2 (0.7%) | 98% | 0.87 | 897.3 ms | $0.36 | 0 |
| NVIDIA rerank-1b, before fine-tuning | 26/128 (20%) | 19 (7.0%) | 58% | 0.30 | 464.9 ms | $0.00 | 0 |
| Our fine-tuned checker alone (Nebius) | 105/128 (82%) | 9 (3.3%) | 92% | 0.87 | 461.5 ms | $0.00 | 0 |
| Faithful: code + our checker + decision rule | 106/128 (83%) | 25 (9.2%) | 81% | 0.82 | 462.6 ms | $0.00 | 0 |

## ACES English-Spanish challenge set (out-of-distribution)

400 turns (random sample of 1853), run 2026-10-28 01:53.

| Method | Errors caught | False alarms | Precision | F1 | Median time / turn | Cost / 1,000 turns | Failed |
|---|---|---|---|---|---|---|---|
| Code slot diff only (no model) | 110/224 (49%) | 9 (5.1%) | 92% | 0.64 | 1.5 ms | $0.00 | 0 |
| Nemotron 3.5 Lightning, prompt only | 140/224 (62%) | 12 (6.8%) | 92% | 0.74 | 929.1 ms | $0.03 | 0 |
| Nemotron 3 Ultra, prompt only | 206/224 (92%) | 23 (13.1%) | 90% | 0.91 | 889.7 ms | $0.38 | 0 |
| NVIDIA rerank-1b, before fine-tuning | 29/224 (13%) | 2 (1.1%) | 94% | 0.23 | 746.3 ms | $0.00 | 0 |
| Our fine-tuned checker alone (Nebius) | 115/224 (51%) | 1 (0.6%) | 99% | 0.68 | 464.8 ms | $0.00 | 0 |
| Faithful: code + our checker + decision rule | 129/224 (58%) | 5 (2.8%) | 96% | 0.72 | 464.8 ms | $0.00 | 0 |

## Our 6 scripted visits (20 planted errors)

54 turns, run 2026-10-28 01:56.

| Method | Errors caught | False alarms | Precision | F1 | Median time / turn | Cost / 1,000 turns | Failed |
|---|---|---|---|---|---|---|---|
| Code slot diff only (no model) | 17/20 (85%) | 0 (0.0%) | 100% | 0.92 | 1.2 ms | $0.00 | 0 |
| Nemotron 3.5 Lightning, prompt only | 17/20 (85%) | 1 (2.9%) | 94% | 0.90 | 734.6 ms | $0.03 | 0 |
| Nemotron 3 Ultra, prompt only | 19/20 (95%) | 0 (0.0%) | 100% | 0.97 | 891.1 ms | $0.33 | 0 |
| NVIDIA rerank-1b, before fine-tuning | 12/20 (60%) | 9 (26.5%) | 57% | 0.58 | 464.1 ms | $0.00 | 0 |
| Our fine-tuned checker alone (Nebius) | 19/20 (95%) | 2 (5.9%) | 90% | 0.93 | 465.9 ms | $0.00 | 0 |
| Faithful: code + our checker + decision rule | 20/20 (100%) | 2 (5.9%) | 91% | 0.95 | 470.5 ms | $0.00 | 0 |

