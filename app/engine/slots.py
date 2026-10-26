"""Critical content in a sentence, in English or Spanish: numbers, units, how often, how long, negation, allergy,
left/right, drug names and warning signs. Plain code, no AI, so every flag can point at the exact words.

extract(text, lang) -> list[Slot];  diff(source, rendition) -> list[Finding]
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass

SEVERITY = {"number": "high", "unit": "high", "negation": "high", "allergy": "high", "side": "high", "drug": "high",
            "warning": "high", "limit": "high", "frequency": "medium", "duration": "high", "order": "medium", "timing": "low"}
KIND_LABEL = {"number": "number", "unit": "unit", "negation": "a 'no' / 'not'", "allergy": "allergy", "side": "left/right",
              "drug": "medicine", "warning": "warning sign", "limit": "dose limit", "order": "order of the numbers", "frequency": "how often", "duration": "how long", "timing": "when to take it"}


@dataclass(frozen=True)
class Slot:
    kind: str
    value: str   # canonical, language-neutral ("6", "mg", "acetaminophen", "q6h")
    span: str    # the words as said
    ctx: str = ""  # for numbers: what they count ("hour", "mg", "tablet"), used to pair a changed number with its replacement


@dataclass(frozen=True)
class Finding:
    type: str        # omission | substitution | addition
    kind: str
    severity: str
    source: str      # words in the source ("" for additions)
    rendition: str   # words in the rendition ("" for omissions)
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


def fold(s: str) -> str:
    """lowercase, strip accents (so 'días' == 'dias'), one output character per input character (so spans map back)."""
    out = []
    for ch in unicodedata.normalize("NFC", s):
        f = "".join(c for c in unicodedata.normalize("NFD", ch.lower()) if unicodedata.category(c) != "Mn").replace("\u2019", "'")
        out.append(f[:1] or ch)
    return "".join(out)


def restore(text: str, span: str) -> str:
    """The original words (accents, capitals) for a folded span found in text."""
    t = unicodedata.normalize("NFC", text)
    i = fold(t).find(span)
    return t[i:i + len(span)] if i >= 0 and span else span


# ---------------------------------------------------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------------------------------------------------
EN_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
            "nineteen": 19}
EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ES_UNITS = {"cero": 0, "uno": 1, "una": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
            "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
            "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiuna": 21, "veintiun": 21, "veintidos": 22,
            "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29}
ES_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
ES_HUNDREDS = {"cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300, "trescientas": 300,
               "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500, "quinientas": 500, "seiscientos": 600, "seiscientas": 600,
               "setecientos": 700, "setecientas": 700, "ochocientos": 800, "ochocientas": 800, "novecientos": 900, "novecientas": 900}
# Words that make 'a' / 'un' / 'una' a real quantity ("a teaspoon", "una pastilla", "una vez") rather than an article.
DOSE_NOUNS = r"(tablet|tablets|pill|pills|capsule|capsules|teaspoon|teaspoons|tablespoon|tablespoons|drop|drops|puff|puffs|dose|doses|patch|" \
             r"cup|cups|glass|time|pastilla|tableta|pildora|comprimido|capsula|cucharadita|cucharada|gota|inhalacion|disparo|dosis|parche|" \
             r"taza|vaso|vez|sola)"
TIME_NOUNS = r"(hour|day|week|month|minute|hora|dia|semana|mes|minuto)"
EN_PRONOUN_BEFORE_ONE = {"the", "this", "that", "which", "new", "other", "another", "each", "any", "every", "no", "old", "right", "left", "last",
                         "first", "little", "big", "same", "only", "good", "bad", "red", "blue", "white"}


def _num_tokens(t: str, lang: str = "en") -> list[tuple[float, str, int, int]]:
    """All numbers in folded text -> (value, words, start, end). Digits and fractions in any language; number words
    in the sentence's language only (Spanish 'once' is 11, English 'once' is 1)."""
    out: list[tuple[float, str, int, int]] = []
    taken = [False] * (len(t) + 1)
    # digits: 500, 2.5, 2,5 (Spanish decimal), 1,000, 1/2
    for m in re.finditer(r"(?<![\w.])(\d+)\s*/\s*(\d+)(?![\w])|(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d{1,2})?)(?![\w])", t):
        if m.group(1):
            v = int(m.group(1)) / max(int(m.group(2)), 1)
        else:
            raw = m.group(3)
            v = float(raw.replace(",", "")) if re.fullmatch(r"\d{1,3}(?:,\d{3})+", raw) else float(raw.replace(",", "."))
        out.append((v, m.group(0), m.start(), m.end()))
        for i in range(m.start(), m.end()):
            taken[i] = True
    words = list(re.finditer(r"[a-z]+(?:-[a-z]+)?", t))
    W = [w.group(0) for w in words]
    nxt = lambda k: W[k] if k < len(W) else ""  # noqa: E731
    prv = lambda k: W[k] if k >= 0 else ""  # noqa: E731
    i = 0
    while i < len(words):
        w = W[i]
        if taken[words[i].start()]:
            i += 1
            continue
        val, j = None, i
        if lang == "en":
            parts = w.split("-")
            if all(p in EN_UNITS or p in EN_TENS for p in parts) and not (w == "one" and (prv(i - 1) in EN_PRONOUN_BEFORE_ONE or nxt(i + 1) == "of")):
                val = sum(EN_UNITS.get(p, 0) + EN_TENS.get(p, 0) for p in parts)
                j = i + 1
                while nxt(j) in ("hundred", "thousand"):
                    val *= 100 if W[j] == "hundred" else 1000
                    j += 1
                    if nxt(j) and all(p in EN_UNITS or p in EN_TENS for p in nxt(j).split("-")):
                        val += sum(EN_UNITS.get(p, 0) + EN_TENS.get(p, 0) for p in W[j].split("-"))
                        j += 1
                if nxt(j) == "and" and nxt(j + 1) == "a" and nxt(j + 2) == "half":
                    val += 0.5
                    j += 3
            elif w == "half" and prv(i - 1) not in ("and",):
                val, j = 0.5, i + 1
            elif w in ("once", "twice"):
                val, j = (1 if w == "once" else 2), i + 1
            elif w in ("a", "an") and prv(i - 1) not in ("half", "quarter", "at", "times", "twice", "once") and \
                    (re.fullmatch(DOSE_NOUNS, nxt(i + 1)) or (re.fullmatch(TIME_NOUNS, nxt(i + 1)) and prv(i - 1) in ("for", "in", "about", "within", "over", "after"))):
                val, j = 1, i + 1
            elif w == "quarter" and prv(i - 1) in ("a", "one"):
                val, j = 0.25, i + 1
        else:
            if w in ES_HUNDREDS or w in ES_TENS or (w in ES_UNITS and w not in ("un", "una", "uno")) or w == "mil":
                val = ES_HUNDREDS.get(w, 0) + ES_TENS.get(w, 0) + ES_UNITS.get(w, 0) + (1000 if w == "mil" else 0)
                j = i + 1
                if w in ES_HUNDREDS and (nxt(j) in ES_TENS or nxt(j) in ES_UNITS):
                    val += ES_TENS.get(W[j], 0) + ES_UNITS.get(W[j], 0)
                    j += 1
                if val in ES_TENS.values() or (val >= 100 and val % 100 in ES_TENS.values()):
                    if nxt(j) == "y" and nxt(j + 1) in ES_UNITS:
                        val += ES_UNITS[W[j + 1]]
                        j += 2
                if nxt(j) == "mil":
                    val *= 1000
                    j += 1
                if nxt(j) == "y" and nxt(j + 1) in ("media", "medio"):
                    val += 0.5
                    j += 2
            elif w in ("un", "una", "uno") and (re.fullmatch(DOSE_NOUNS, nxt(i + 1)) or re.fullmatch(TIME_NOUNS, nxt(i + 1)) or nxt(i + 1) == "cuarto"
                                                or (nxt(i + 1) == "y" and nxt(i + 2) in ("media", "medio"))):
                if nxt(i + 1) == "cuarto":
                    val, j = 0.25, i + 2
                elif nxt(i + 1) == "y":
                    val, j = 1.5, i + 3
                else:
                    val, j = 1, i + 1
            elif w in ("media", "medio") and prv(i - 1) not in ("y",) and (re.fullmatch(DOSE_NOUNS, nxt(i + 1)) or re.fullmatch(TIME_NOUNS, nxt(i + 1))):
                val, j = 0.5, i + 1
        if val is not None and nxt(j) in ("point", "punto") and nxt(j + 1) and (nxt(j + 1) in EN_UNITS or nxt(j + 1) in ES_UNITS):
            # spoken decimals: "two point five", "dos punto cinco"
            frac = EN_UNITS.get(W[j + 1], ES_UNITS.get(W[j + 1], 0))
            val = val + frac / (10 ** len(str(frac)))
            j += 2
        if val is not None:
            s0, e0 = words[i].start(), words[j - 1].end()
            out.append((float(val), t[s0:e0], s0, e0))
            i = j
        else:
            i += 1
    out.sort(key=lambda x: x[2])
    return out


def _fmt(v: float) -> str:
    return str(int(v)) if v == int(v) else f"{v:g}"


# ---------------------------------------------------------------------------------------------------------------------
# Lexicons (folded: no accents)
# ---------------------------------------------------------------------------------------------------------------------
UNITS = [
    ("mg", r"mg|milligrams?|miligramos?"), ("mcg", r"mcg|micrograms?|microgramos?"), ("g", r"grams?|gramos?"),
    ("ml", r"ml|milliliters?|millilitres?|mililitros?|cc"), ("tablet", r"tablets?|pills?|pastillas?|tabletas?|pildoras?|comprimidos?"),
    ("capsule", r"capsules?|capsulas?"), ("teaspoon", r"teaspoons?|tsp|cucharaditas?"), ("tablespoon", r"tablespoons?|tbsp|cucharadas?"),
    ("drop", r"drops?|gotas?"), ("puff", r"puffs?|inhalations?|inhalaciones|inhalacion|disparos?"), ("unit", r"units?|unidades|unidad"),
    ("patch", r"patch(?:es)?|parches?"), ("dose", r"doses?|dosis"),
]
TIME_UNITS = [("hour", r"hours?|hrs?|horas?"), ("day", r"days?|dias?"), ("week", r"weeks?|semanas?"), ("month", r"months?|meses|mes"),
              ("minute", r"minutes?|mins?|minutos?")]
DRUGS = {
    "acetaminophen": r"acetaminophen|acetaminofen|tylenol|tilenol|paracetamol", "ibuprofen": r"ibuprofen|ibuprofeno|advil|motrin",
    "amoxicillin": r"amoxicillin|amoxicilina|amoxil", "penicillin": r"penicillin|penicilina", "aspirin": r"aspirin|aspirina",
    "insulin": r"insulin|insulina", "metformin": r"metformin|metformina", "lisinopril": r"lisinopril", "amlodipine": r"amlodipine|amlodipino",
    "atorvastatin": r"atorvastatin|atorvastatina|lipitor", "warfarin": r"warfarin|warfarina|coumadin", "prednisone": r"prednisone|prednisona",
    "albuterol": r"albuterol|salbutamol|ventolin", "ondansetron": r"ondansetron|zofran", "cephalexin": r"cephalexin|cefalexina|keflex",
    "azithromycin": r"azithromycin|azitromicina|zithromax", "omeprazole": r"omeprazole|omeprazol|prilosec",
    "sulfa": r"sulfa|sulfas|bactrim|sulfamethoxazole|sulfametoxazol", "codeine": r"codeine|codeina", "morphine": r"morphine|morfina",
    "oxycodone": r"oxycodone|oxicodona", "hydrocodone": r"hydrocodone|hidrocodona|vicodin|norco", "naproxen": r"naproxen|naproxeno|aleve",
    "diphenhydramine": r"diphenhydramine|difenhidramina|benadryl", "cetirizine": r"cetirizine|cetirizina|zyrtec",
    "loratadine": r"loratadine|loratadina|claritin", "furosemide": r"furosemide|furosemida|lasix", "levothyroxine": r"levothyroxine|levotiroxina|synthroid",
    "clopidogrel": r"clopidogrel|plavix", "heparin": r"heparin|heparina", "nitroglycerin": r"nitroglycerin|nitroglicerina",
    "ceftriaxone": r"ceftriaxone|ceftriaxona", "antibiotic": r"antibiotics?|antibioticos?", "latex": r"latex", "peanut": r"peanuts?|cacahuates?|mani|cacahuetes?",
    "shellfish": r"shellfish|mariscos?", "iodine": r"iodine|yodo|contrast dye|medio de contraste", "egg": r"eggs?|huevos?",
    "blood thinner": r"blood thinners?|anticoagulantes?", "inhaler": r"inhaler|inhalador",
}
WARNINGS = {
    "emergency": r"emergency(?: room)?|\ber\b|urgencias|sala de emergencias?|emergencias?", "911": r"911|nine one one|nueve uno uno|novecientos once",
    "call": r"\bcall (?:us|your doctor|the doctor|the clinic|back)|\bllame|\bllamenos|\bllamar al", "return": r"come back|return|regrese|vuelva|regresar|volver",
    "breathing": r"(?:trouble|difficulty|hard time|problems?) breathing|can'?t breathe|short(?:ness)? of breath|dificultad para respirar|no puede respirar|le falta el aire|falta de aire",
    "chest pain": r"chest pain|pain in (?:the|his|her|your) chest|dolor (?:de|en el) pecho", "fever": r"fever|temperature|fiebre|calentura",
    "bleeding": r"bleed(?:ing|s)?|blood in|sangr(?:ado|a|e|ando)", "rash": r"rash|hives|[sz]arpullido|ronchas|erupcion",
    "swelling": r"swell(?:ing|ed|s)?|hinchaz[oó]n|hinchad[oa]|inflamad[oa]", "vomiting": r"vomit(?:ing|s)?|throw(?:ing)? up|vomit(?:o|a|ando|os)|devolv",
    "dizzy": r"dizz(?:y|iness)|faint(?:ing|ed)?|pass(?:es|ed)? out|mare(?:o|ad[oa]|os)|desmay", "seizure": r"seizures?|convulsi(?:on|ones)|ataques?",
    "unresponsive": r"unconscious|won'?t wake|unresponsive|inconsciente|no despierta|no reacciona", "pregnant": r"pregnan(?:t|cy)|embarazad[ao]|embarazo",
    "pain": r"\bpain\b|hurts?|\bdolor\b|\bduele\b",
}
BODY_EN = r"(?:side|arm|leg|ear|eye|hand|foot|knee|hip|shoulder|kidney|lung|breast|ankle|wrist|elbow|finger|toe|thumb|thigh|flank|lower|upper|chest|wrist)"
BODY_ES = r"(?:lado|costado|brazo|pierna|oido|oreja|ojo|mano|pie|rodilla|cadera|hombro|rinon|pulmon|seno|pecho|tobillo|muneca|codo|dedo|muslo)"
NEG_EN = r"\b(?:not|no|never|don'?t|doesn'?t|didn'?t|isn'?t|aren'?t|wasn'?t|can'?t|cannot|won'?t|shouldn'?t|mustn'?t|without|none|nothing|avoid|stop|quit)\b|n't\b"
NEG_ES = r"\b(?:no|nunca|jamas|ni|sin|nada|ningun|ninguna|ninguno|evite|evitar|deje|dejar|suspenda|suspender|pare de|tampoco)\b"
FREQ = [
    (r"every\s+(?P<n>\S+(?:\s+(?:and a half|hundred))?)\s+hours?|cada\s+(?P<m>\S+(?:\s+y\s+media)?)\s+horas?", "every_n_hours"),
    (r"(?P<n>\S+)\s+times?\s+(?:a|per|each)\s+day|(?P<m>\S+)\s+(?:vez|veces)\s+(?:al|por|cada|a el)\s+dia|(?P<o>once|twice)\s+(?:a|per|each)\s+day|(?P<o2>once|twice)\s+daily", "times_a_day"),
    (r"\bdaily\b|every day|each day|(?:a|per) day|todos los dias|cada dia|diario|diariamente|al dia|por dia", "daily"),
    (r"as needed|if needed|when needed|if (?:he|she|you) needs? it|si (?:es|lo) necesari[oa]|cuando lo necesite|segun sea necesario|si (?:le|lo) necesita", "as_needed"),
]
LIMIT = r"(?:no|not) more than|never more than|maximum|max(?:imum)? of|up to|at most|no mas de|nunca mas de|maximo|como maximo|hasta un maximo de|hasta"
TIMING = [
    (r"with food|with meals|after (?:eating|meals)|con (?:la )?comida|con (?:los )?alimentos|despues de comer", "with_food"),
    (r"empty stomach|before (?:eating|meals|breakfast)|en ayunas|estomago vacio|antes de comer|antes del desayuno", "empty_stomach"),
    (r"at bedtime|before bed|at night|before (?:you )?sleep|antes de dormir|antes de acostarse|en la noche|por la noche|de noche", "night"),
    (r"in the morning|every morning|en la manana|por la manana|cada manana|en las mananas", "morning"),
]
ALLERGY = r"allerg(?:y|ies|ic)|alerg(?:ia|ias|ico|ica|icos|icas)"


def _side_slots(t: str, lang: str) -> list[Slot]:
    out = []
    if lang == "en":
        pats = [rf"\b(left|right)\s+(?:\w+\s+)?{BODY_EN}\b", r"\bon (?:the|his|her|your|my) (left|right)\b", r"\bto the (left|right)\b"]
    else:
        pats = [rf"\b{BODY_ES}\s+(?:\w+\s+)?(izquierd[oa]|derech[oa])\b", r"\b(?:a|en) la (izquierda|derecha)\b", r"\b(izquierd[oa]|derech[oa])\b(?=.*\b(?:lado|costado)\b)"]
    seen = set()
    for p in pats:
        for m in re.finditer(p, t):
            if m.start() in seen:
                continue
            seen.add(m.start())
            w = m.group(1)
            out.append(Slot("side", "left" if w.startswith(("left", "izq")) else "right", m.group(0)))
    return out


def guess_lang(text: str) -> str:
    t = f" {fold(text)} "
    es = sum(t.count(f" {w} ") for w in ("el", "la", "de", "que", "y", "es", "por", "para", "con", "su", "le", "lo", "una", "los", "las", "esta", "tiene", "dice"))
    en = sum(t.count(f" {w} ") for w in ("the", "and", "is", "to", "of", "you", "he", "she", "it", "with", "for", "your", "has", "this", "take", "says"))
    return "es" if es > en else "en"


def extract(text: str, lang: str | None = None) -> list[Slot]:
    lang = lang or guess_lang(text)
    t = fold(text)
    slots: list[Slot] = []
    unit_rx = "|".join(f"(?P<u_{c}>{rx})" for c, rx in UNITS + TIME_UNITS)
    for v, words, _, end in _num_tokens(t, lang):
        m = re.match(rf"[\s-]*(?:{unit_rx})(?![a-z])", t[end:end + 40])
        ctx = next((k[2:] for k, val in (m.groupdict().items() if m else []) if val), "")
        slots.append(Slot("number", _fmt(v), words, ctx))
        if ctx in dict(TIME_UNITS):
            slots.append(Slot("duration", ctx, (words + t[end:end + m.end()]).strip()))
    for canon, rx in UNITS:
        for m in re.finditer(rf"(?<![a-z])(?:{rx})(?![a-z])", t):
            slots.append(Slot("unit", canon, m.group(0)))
    for canon, rx in DRUGS.items():
        for m in re.finditer(rf"(?<![a-z])(?:{rx})(?![a-z])", t):
            slots.append(Slot("drug", canon, m.group(0)))
    for canon, rx in WARNINGS.items():
        for m in re.finditer(rf"(?<![a-z])(?:{rx})", t):
            slots.append(Slot("warning", canon, m.group(0)))
            break  # presence is enough
    for rx, canon in FREQ:
        for m in re.finditer(rf"(?<![a-z])(?:{rx})", t):
            slots.append(Slot("frequency", canon, m.group(0)))
    for rx, canon in TIMING:
        for m in re.finditer(rf"(?<![a-z])(?:{rx})", t):
            slots.append(Slot("timing", canon, m.group(0)))
    for m in re.finditer(rf"(?<![a-z])(?:{LIMIT})(?![a-z])", t):
        slots.append(Slot("limit", "limit", m.group(0)))
    for m in re.finditer(rf"(?<![a-z])(?:{ALLERGY})", t):
        slots.append(Slot("allergy", "allergy", m.group(0)))
    neg = NEG_EN if lang == "en" else NEG_ES
    for m in re.finditer(neg, t):
        slots.append(Slot("negation", "neg", m.group(0)))
    slots += _side_slots(t, lang)
    # 'daily' inside 'times a day' / 'al dia' inside 'veces al dia' is the same instruction: keep the specific one.
    if any(s.value == "times_a_day" for s in slots):
        slots = [s for s in slots if s.value != "daily"]
    return slots


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def heard_as(canon: str, text: str) -> bool:
    """Is a near-spelling of this drug in the text (speech recognition slips: 'telenol', 'acitromicina')?
    Only words that are not themselves a known drug count, so a real sound-alike drug swap is still flagged."""
    known = re.compile(rf"(?<![a-z])(?:{'|'.join(DRUGS.values())})(?![a-z])")
    forms = [f for f in DRUGS[canon].split("|") if f.isalpha() and len(f) >= 6]
    for w in re.findall(r"[a-z]{5,}", fold(text)):
        if known.fullmatch(w):
            continue
        if any(_lev(w, f) <= (1 if len(f) < 8 else 2) and w[0] in (f[0], {"z": "s", "s": "z", "c": "z"}.get(f[0], f[0])) for f in forms):
            return True
    return False


PRESENCE_KINDS = ("negation", "allergy", "frequency", "timing", "limit")  # compared as present/absent, not counted


def diff(source: str, rendition: str, src_lang: str | None = None, ren_lang: str | None = None) -> list[Finding]:
    """What critical content changed between a sentence and its interpretation."""
    a = [Slot(x.kind, x.value, restore(source, x.span), x.ctx) for x in extract(source, src_lang)]
    b = [Slot(x.kind, x.value, restore(rendition, x.span), x.ctx) for x in extract(rendition, ren_lang)]
    out: list[Finding] = []
    for kind in SEVERITY:
        sa = [s for s in a if s.kind == kind]
        sb = [s for s in b if s.kind == kind]
        if kind in PRESENCE_KINDS:
            va, vb = {s.value for s in sa}, {s.value for s in sb}
            miss, extra = va - vb, vb - va
            if kind == "negation":
                miss, extra = (({"neg"} if sa and not sb else set()), ({"neg"} if sb and not sa else set()))
        else:
            ca, cb = Counter(s.value for s in sa), Counter(s.value for s in sb)
            miss, extra = set((ca - cb).elements()), set((cb - ca).elements())
            if kind == "warning":  # 'pain' alone is too common to flag as an omission
                miss.discard("pain"); extra.discard("pain")
            if kind == "drug":  # a misheard drug name ('acitromicina') is not a missing drug
                miss = {m for m in miss if not heard_as(m, rendition)}
                extra = {e for e in extra if not heard_as(e, source)}
        span = lambda xs, v: next((s.span for s in xs if s.value == v), v)  # noqa: E731
        sev = SEVERITY[kind]
        pair_kinds = ("number", "unit", "side", "drug", "duration")
        miss_l, extra_l = sorted(miss), sorted(extra)
        if kind == "number":  # pair a dropped number with an added one that counts the same thing ("every 6 hours" -> "every 8 hours")
            ctx_of = lambda xs, v: next((s.ctx for s in xs if s.value == v), "")  # noqa: E731
            for m in list(miss_l):
                e = next((e for e in extra_l if ctx_of(sa, m) and ctx_of(sb, e) == ctx_of(sa, m)), None)
                if e is not None:
                    miss_l.remove(m); extra_l.remove(e)
                    out.append(Finding("substitution", kind, sev, span(sa, m), span(sb, e),
                                       f"Changed {KIND_LABEL[kind]}: \u201c{span(sa, m)}\u201d became \u201c{span(sb, e)}\u201d"))
        if kind in pair_kinds:
            while miss_l and extra_l:
                m, e = miss_l.pop(0), extra_l.pop(0)
                out.append(Finding("substitution", kind, sev, span(sa, m), span(sb, e),
                                   f"Changed {KIND_LABEL[kind]}: “{span(sa, m)}” became “{span(sb, e)}”"))
        for m in miss_l:
            out.append(Finding("omission", kind, sev, span(sa, m), "", f"Not in the interpretation: {KIND_LABEL[kind]} “{span(sa, m)}”"))
        for e in extra_l:
            # added content is usually lower risk than dropped content, except an added 'no' (flips meaning)
            s2 = "high" if kind in ("negation", "drug", "number", "side") else "medium"
            out.append(Finding("addition", kind, s2, "", span(sb, e), f"Added in the interpretation: {KIND_LABEL[kind]} “{span(sb, e)}”"))
    # Same numbers, different order ("2.5 mg Mon/Fri, 5 mg other days" -> "5 mg Mon/Fri, 2.5 mg other days"): code can't tell
    # a harmless re-ordering from a swap, so it's a medium note for the checker to weigh.
    na = [s.value for s in a if s.kind == "number"]
    nb = [s.value for s in b if s.kind == "number"]
    if len(na) >= 2 and Counter(na) == Counter(nb) and na != nb:
        out.append(Finding("substitution", "order", SEVERITY["order"], " / ".join(na), " / ".join(nb),
                           "Same numbers in a different order: " + ", ".join(na) + " became " + ", ".join(nb)))
    return out


def worst(findings: list[Finding]) -> str:
    order = {"high": 3, "medium": 2, "low": 1}
    return max((f.severity for f in findings), key=lambda s: order[s], default="none")
