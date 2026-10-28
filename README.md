# Faithful

**A second ear for medical interpreters.** Faithful listens to both sides of an interpreted visit and, right after each turn, asks the interpreter to re-confirm any dose, number, allergy, left or right, medicine, warning sign or “no” that didn’t make it across the language gap. Spanish and English.

> A mother says *“ya le di dos dosis de Tylenol hoy… y es alérgico a la penicilina.”* The interpreter says *“Yes, I gave him Tylenol today.”* The doctor never hears about the two earlier doses or the penicillin allergy, and nobody in the room can tell. Faithful puts a red **Re-confirm** card in front of the interpreter before the doctor writes the prescription.

Live demo: _link after deploy_ · Built for the Nebius × NVIDIA Global AI Hackathon (Best Apps and Agents).

## Why this matters

- About 27 million people in the US have limited English proficiency ([KFF, 2025](https://kff.org/racial-equity-and-health-policy/designating-english-as-the-official-language-of-the-united-states-could-impact-millions-with-limited-english-proficiency/)).
- Interpreting errors are common. In 57 pediatric emergency visits, 18% of errors could have had clinical consequences. The rate was 12% with professional interpreters and 22% with ad hoc ones ([Flores et al., Ann Emerg Med 2012](https://www.physiciansweekly.com/post/emergency-department-interpreter-errors)). Half of US primary-care physicians regularly rely on ad hoc interpreters (JAMA Netw Open 2026).
- Errors now spread further. When researchers planted 20 interpreter errors in scripted visits, two ambient AI scribes copied **55–60%** of them into the visit note, and **80–100%** of the errors in the patient’s own speech ([Rabotin et al., JMIR Med Inform 2026](https://medinform.jmir.org/2026/1/e88734/PDF)).
- Existing tools sit next to this problem but don't solve it. Some help interpreters hear terms and numbers live; others grade calls after they end. We found none that compares what was said with what was interpreted, turn by turn, while the patient is still in the room.

## What it does

- **Re-confirm cards, live.** After each turn, Faithful shows a card only when critical content was dropped, changed or added. The card points at the exact words.
- **Both directions.** It checks the clinician into Spanish and the patient into English. The patient direction matters most, because clinicians can't check it themselves and scribes copy it.
- **Confirm before they leave.** At the end there is a short checklist of every card. It also includes anything Nemotron 3 Ultra found when it re-read the whole visit afterwards.
- **Interpreter-first by design.** Cards go to the interpreter and the clinician only. Faithful never speaks to the patient, never scores interpreters and stores nothing.

## How it works: models listen and read, code decides

| Step | What runs | Where |
|---|---|---|
| 1. Hear both sides | **NVIDIA Parakeet TDT 0.6B v3**, multilingual speech recognition | Our **Nebius Serverless Endpoint** (one L40S) |
| 2. Read the critical content | Plain code: numbers (digits and Spanish or English number words), units, how often, how long, dose limits, allergies, left/right, medicine names (brand ↔ generic), warning signs, negation | The app |
| 3. Weigh the meaning | **Our fine-tuned checker**: NVIDIA `llama-nemotron-rerank-1b-v2`, fine-tuned to score whether an interpretation still means the same thing | Trained on a **Nebius Serverless Job**, served on the same endpoint |
| 4. Decide | A fixed rule combines steps 2 and 3. The outcome is a re-confirm card, a note for after the visit, or nothing. Thresholds come from held-out practice visits | The app |
| 5. Say it plainly | **Nemotron 3.5 Lightning** writes the card's one line. Code checks that any words it quotes were really said | Nebius Token Factory |
| 6. Re-read after the visit | **Nemotron 3 Ultra** reads the whole conversation once. Each item it finds must quote a real turn | Nebius Token Factory |

Steps 2 and 3 cover each other's blind spots:
- **Code** sees exact words. It knows that *cada seis horas* is "every 6 hours" and that *cada ocho horas* isn't.
- **The checker** is trained on meaning. It's there for changes words alone can't see, like “not this week” becoming *“esta semana sí puede”* (both sentences still contain a “no”) or a dose moved to a different medicine. How often it actually catches them is in the results below.
- **The rule** keeps either one from crying wolf without letting the dangerous changes through. A changed or missing allergy, “no”, medicine, left/right or dose limit is always a card. A changed number, unit or duration that the checker is confident still means the same (often a speech-recognition slip or a re-wording) becomes a quiet note for after the visit instead of an interruption.

## Results

_Filled in from `eval/RESULTS.md` and `eval/e2e.json` after the runs._

### End to end, from audio
We wrote six scripted visits with **20 planted errors**, following the JMIR 2026 design:
- 11 errors are in the patient's speech and 9 in the clinician's.
- They cover omissions, substitutions and additions.
- 34 other turns are faithful, some reworded the way interpreters normally do.

The visits were voiced with Kokoro, heard by Parakeet on our Nebius endpoint, and checked.

### Benchmark
From [`eval/RESULTS.md`](eval/RESULTS.md). Every method saw the same turns: a random sample of 400 per set (fixed seed), all 54 scripted turns. Thresholds were chosen on 632 held-out practice turns under one rule for every method (best F1 with no more than 5% false alarms there), never on these test sets.

| Method | Held-out synthetic visits: errors caught / false alarms | ACES (out-of-distribution): caught / false alarms | Time per turn | Cost per 1,000 turns |
|---|---|---|---|---|
| Code checks only | 50% / 9.6% | 49% / 5.1% | 2 ms | $0 |
| Nemotron 3.5 Lightning, asked per turn | 51% / 1.5% | 62% / 6.8% | 0.9 s | $0.03 |
| Nemotron 3 Ultra, asked per turn | 79% / 0.7% | 92% / 13.1% | 0.9 s | $0.36 |
| NVIDIA rerank-1b before fine-tuning | 20% / 7.0% | 13% / 1.1% | 0.5 s | GPU time |
| **Our fine-tuned checker alone** | **82% / 3.3%** | 51% / 0.6% | 0.5 s | GPU time |
| **Faithful: code + checker + decision rule** | **83% / 9.2%** | 58% / 2.8% | 0.5 s | GPU time |

What this shows:
- **Fine-tuning made the small model useful.** Before, the reranker caught 20% of held-out errors; after, 82%, level with Nemotron 3 Ultra asked turn by turn, at half the latency and on our own GPU, with no per-call model cost. The every-turn decision never needs a large model; Lightning only words the cards that appear.
- **The decision rule trades false alarms for safety.** It adds the code checks' always-a-card changes (an allergy, a "no", a medicine, a side, a dose limit), which lifts recall a little and false alarms from 3.3% to 9.2% on the synthetic visits. We chose that on purpose; the rule's thresholds came from the practice turns (re-confirm at a checker error score of 0.05 or more when code sees a change; a checker-only card at 0.95).
- **Out of domain, the checker is cautious.** On ACES, general-domain sentences with traps like antonyms and named entities, it flags few (51% caught) but is almost never wrong (0.6% false alarms). Ultra catches more (92%) but false-alarms on 13%. Our training visits are medical and interpreter-shaped; ACES isn't.
- **On our six scripted visits** (text, no speech recognition): Faithful 20 of 20 caught, 2 of 34 false alarms; Ultra 19 of 20, 0 of 34; code alone 17 of 20, 0 of 34. The code rules were written while looking at these visits, so treat this row as a demo check, not a benchmark.
- The rerank model before fine-tuning couldn't meet the 5% false-alarm limit on the practice turns at any threshold.

### Fine-tuning on Nebius
NVIDIA `llama-nemotron-rerank-1b-v2`, fine-tuned on 3,818 synthetic training pairs for 2 epochs on one L40S in a Nebius Serverless Job (161 seconds). On 1,449 held-out test pairs written by two other model families: **AUC 0.60 before fine-tuning, 0.95 after**; at the threshold chosen on dev, F1 0.29 before and 0.84 after (precision 0.88, recall 0.81). The job's report is in [`eval/training.json`](eval/training.json), copied from the endpoint.

## Honest limits

- Built and tested on synthetic and scripted visits only. No real patient audio was used.
- Nebius Serverless isn't HIPAA-eligible by default. A clinical deployment would use Nebius Compute or Managed Kubernetes under a BAA, and Token Factory with zero data retention.
- Covers Spanish and English only. Speakers of Indigenous languages such as Mam or K’iche’, and of Haitian Creole, aren't covered.
- A speech-recognition slip can look like an interpreting error. Cards therefore say “re-confirm”, never “wrong”, and show the words heard.
- It is not an interpreter and doesn't replace a qualified one.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 7860
```

| What you set | What works |
|---|---|
| Nothing | The replay and the code checks |
| `NEBIUS_API_KEY` | Lightning card wording and the Ultra review |
| `ENDPOINT_URL` and `ENDPOINT_TOKEN` | Our Nebius endpoint: speech recognition and the fine-tuned checker |

Pipeline steps run in GitHub Actions (**Run a pipeline step**):

| Step | What it does |
|---|---|
| `synth` | Writes the training and test visits |
| `tts` | Voices the scripted visits |
| `demo` | Runs speech recognition and the checks over the scripted visits |
| `eval` | Runs the benchmark |

The Nebius parts are in `train/`:

| File | What it is |
|---|---|
| `train_checker.py` | The fine-tuning Job |
| `server.py` | The endpoint |
| `nebius.sh` | The CLI commands to run both |

## License

Apache-2.0. Test voices: Kokoro-82M (Apache-2.0). ACES is CC BY-NC-SA 4.0 and isn't redistributed here.
