"""Voice the scripted visits so the demo and the end-to-end test have real audio (Kokoro-82M, open, runs on CPU).

Text-to-speech is only used to MAKE test audio. Faithful never speaks to the patient.
Writes app/static/audio/<visit>-<turn>-src.mp3 and -ren.mp3 (mono 48 kbps) and data/encounters/audio.json.
Usage: python -m scripts.tts   (needs: pip install kokoro soundfile; apt-get install espeak-ng ffmpeg)
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "app" / "static" / "audio"
CLINICIAN_VOICE = {"V1": "af_heart", "V2": "am_michael", "V3": "af_bella", "V4": "af_nicole", "V5": "am_adam", "V6": "af_sarah"}
PATIENT_VOICE = {"V1": "ef_dora", "V2": "ef_dora", "V3": "em_alex", "V4": "em_alex", "V5": "em_alex", "V6": "ef_dora"}
INTERPRETER_VOICE = {"en": "am_echo", "es": "em_santa"}


def main() -> None:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipes = {"en": KPipeline(lang_code="a"), "es": KPipeline(lang_code="e")}
    d = json.loads((ROOT / "data" / "encounters" / "visits.json").read_text())
    AUDIO.mkdir(parents=True, exist_ok=True)
    index = {}

    def say(text: str, lang: str, voice: str, out: Path) -> float:
        chunks = [a for _, _, a in pipes[lang](text, voice=voice, speed=1.0)]
        audio = np.concatenate([np.asarray(c) for c in chunks]) if chunks else np.zeros(2400)
        with tempfile.NamedTemporaryFile(suffix=".wav") as w:
            sf.write(w.name, audio, 24000)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", w.name, "-ac", "1", "-b:a", "48k", str(out)], check=True)
        return round(len(audio) / 24000, 2)

    for v in d["visits"]:
        for i, t in enumerate(v["turns"], 1):
            sl, rl = ("en", "es") if t["speaker"] == "clinician" else ("es", "en")
            src_voice = CLINICIAN_VOICE[v["id"]] if t["speaker"] == "clinician" else PATIENT_VOICE[v["id"]]
            key = f"{v['id']}-{i:02d}"
            s_sec = say(t["source"], sl, src_voice, AUDIO / f"{key}-src.mp3")
            r_sec = say(t["rendition"], rl, INTERPRETER_VOICE[rl], AUDIO / f"{key}-ren.mp3")
            index[key] = {"src": f"audio/{key}-src.mp3", "ren": f"audio/{key}-ren.mp3", "src_seconds": s_sec, "ren_seconds": r_sec,
                          "src_voice": src_voice, "ren_voice": INTERPRETER_VOICE[rl]}
            print(key, s_sec, r_sec)
    (ROOT / "data" / "encounters" / "audio.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} turns voiced")


if __name__ == "__main__":
    main()
