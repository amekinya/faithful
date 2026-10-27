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
- **The rule** stops either one from crying wolf. If code sees a big change but the checker is confident the meaning held (often a speech-recognition slip or a legitimate re-wording), the turn becomes a quiet note for later instead of an interruption.

## Results

_Filled in from `eval/RESULTS.md` and `eval/e2e.json` after the runs._

### End to end, from audio
We wrote six scripted visits with **20 planted errors**, following the JMIR 2026 design:
- 11 errors are in the patient's speech and 9 in the clinician's.
- They cover omissions, substitutions and additions.
- 34 other turns are faithful, some reworded the way interpreters normally do.

The visits were voiced with Kokoro, heard by Parakeet on our Nebius endpoint, and checked.

### Benchmark
| Test set | What it is |
|---|---|
| Held-out synthetic visits | Written by a different model family, with a different seed, than the training visits. Each turn is labeled by how it was written (a planted error or a faithful rewording), and test turns are never filtered by our own code |
| ACES English↔Spanish | A human-designed translation challenge set: numbers, units, dates, antonyms, omissions, additions, names. It is out-of-distribution, never seen in training, and downloaded at test time (CC BY-NC-SA 4.0) |
| Our 6 scripted visits | The text, without speech recognition. The code rules were written while looking at these, so the code-only score here is optimistic |

### Fine-tuning on Nebius
We report the base model against the fine-tuned model on held-out visits (AUC, and F1 at the threshold chosen on dev). The training job writes these to `metrics.json` next to the weights; a copy is published in `eval/training.json`.

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
