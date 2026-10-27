"""Critical content in a sentence, in English or Spanish: numbers, units, how often, how long, dose limits, negation,
allergy, left/right, drug names, warning signs and when to take a medicine. Plain code, no AI, so every flag can point
at the exact words.

extract(text, lang) -> list[Slot];  diff(source, rendition) -> list[Finding]

How it reads numbers: the sentence is split into tokens (punctuation kept, so "quinientos, dos veces" stays two
numbers), number phrases are parsed with a small grammar per language ("two hundred and fifty", "treinta y nueve",
"mil quinientos", "one and a half", "cucharadita y media", "0.125", Spanish "1.000"), and then phrases that use a
number are recognised on a copy of the sentence where each number is a placeholder: "every #0 hours" is a dosing
interval, "#0 times a day" a frequency, "#0 weeks" a duration (compared in hours, so 48 hours == 2 days).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass

SEVERITY = {"allergy": "high", "negation": "high", "drug": "high", "warning": "high", "limit": "high", "side": "high",
            "frequency": "high", "number": "high", "unit": "high", "duration": "high", "order": "medium", "timing": "low"}
# Clinical priority: the order findings are listed on a card.
PRIORITY = ["allergy", "negation", "drug", "warning", "limit", "side", "frequency", "number", "unit", "duration", "timing", "order"]
KIND_LABEL = {"number": "number", "unit": "unit", "negation": "a 'no' / 'not'", "allergy": "allergy", "side": "left/right",
              "drug": "medicine", "warning": "warning sign", "limit": "dose limit", "order": "order of the numbers",
              "frequency": "how often", "duration": "how long", "timing": "when to take it"}


@dataclass(frozen=True)
class Slot:
    kind: str
    value: str     # canonical, language-neutral ("6", "mg", "acetaminophen", "q6h", "48h")
    span: str      # the words as said (folded)
    ctx: str = ""  # for numbers: what they count ("mg", "tablet"); used to pair a changed number with its replacement
    pos: int = -1  # where the words start in the (NFC) text, so they map back exactly


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
        f = "".join(c for c in unicodedata.normalize("NFD", ch.lower()) if unicodedata.category(c) != "Mn")
        f = f.replace("’", "'").replace("‘", "'")
        out.append(f[:1] or ch)
    return "".join(out)


def restore(text: str, span: str, pos: int = -1) -> str:
    """The original words (accents, capitals) for a folded span: by position when known, else the first whole-word match."""
    t = unicodedata.normalize("NFC", text)
    if not span:
        return span
    f = fold(t)
    if 0 <= pos and f[pos:pos + len(span)] == span:
        return t[pos:pos + len(span)]
    m = re.search(rf"(?<![a-z0-9]){re.escape(span)}(?![a-z0-9])", f)
    i = m.start() if m else f.find(span)
    return t[i:i + len(span)] if i >= 0 else span


# ---------------------------------------------------------------------------------------------------------------------
# Tokens and number words
# ---------------------------------------------------------------------------------------------------------------------
TOKEN = re.compile(r"\d+(?:[.,:/]\d+)*|[½¼¾⅓⅔⅛]|[a-z]+(?:'[a-z]+)?|[^\sa-z\d]")
UNICODE_FRAC = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125}

EN_SMALL = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
            "eighteen": 18, "nineteen": 19}
EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
EN_FRAC = {"half": 0.5, "halves": 0.5, "quarter": 0.25, "quarters": 0.25, "third": 1 / 3, "thirds": 1 / 3, "eighth": 0.125, "eighths": 0.125}
ES_SMALL = {"cero": 0, "uno": 1, "una": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
            "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
            "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiuna": 21, "veintiun": 21, "veintidos": 22,
            "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
            "veintinueve": 29}
ES_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
ES_HUNDREDS = {"cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200, "trescientos": 300, "trescientas": 300,
               "cuatrocientos": 400, "cuatrocientas": 400, "quinientos": 500, "quinientas": 500, "seiscientos": 600, "seiscientas": 600,
               "setecientos": 700, "setecientas": 700, "ochocientos": 800, "ochocientas": 800, "novecientos": 900, "novecientas": 900}
ES_FRAC = {"medio": 0.5, "media": 0.5, "mitad": 0.5, "cuarto": 0.25, "cuartos": 0.25, "tercio": 1 / 3, "tercios": 1 / 3,
           "octavo": 0.125, "octavos": 0.125}
DIGIT_WORDS = {**{k: v for k, v in EN_SMALL.items() if v < 10}, **{k: v for k, v in ES_SMALL.items() if v < 10}, "oh": 0}

# What a number can count. Words map to a canonical unit. "dose" and "time" give context but aren't compared as units.
UNIT_WORDS = {
    "mg": "mg", "milligram": "mg", "milligrams": "mg", "miligramo": "mg", "miligramos": "mg",
    "mcg": "mcg", "microgram": "mcg", "micrograms": "mcg", "microgramo": "mcg", "microgramos": "mcg",
    "g": "g", "gram": "g", "grams": "g", "gramo": "g", "gramos": "g",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml", "mililitro": "ml", "mililitros": "ml", "cc": "ml",
    "l": "l", "liter": "l", "liters": "l", "litro": "l", "litros": "l",
    "tablet": "tablet", "tablets": "tablet", "pill": "tablet", "pills": "tablet", "pastilla": "tablet", "pastillas": "tablet",
    "tableta": "tablet", "tabletas": "tablet", "pildora": "tablet", "pildoras": "tablet", "comprimido": "tablet", "comprimidos": "tablet",
    "capsule": "capsule", "capsules": "capsule", "capsula": "capsule", "capsulas": "capsule",
    "teaspoon": "teaspoon", "teaspoons": "teaspoon", "tsp": "teaspoon", "cucharadita": "teaspoon", "cucharaditas": "teaspoon",
    "tablespoon": "tablespoon", "tablespoons": "tablespoon", "tbsp": "tablespoon", "cucharada": "tablespoon", "cucharadas": "tablespoon",
    "drop": "drop", "drops": "drop", "gota": "drop", "gotas": "drop",
    "puff": "puff", "puffs": "puff", "inhalation": "puff", "inhalations": "puff", "inhalacion": "puff", "inhalaciones": "puff",
    "disparo": "puff", "disparos": "puff",
    "unit": "unit", "units": "unit", "unidad": "unit", "unidades": "unit",
    "patch": "patch", "patches": "patch", "parche": "patch", "parches": "patch",
    "kg": "kg", "kgs": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg", "kilograms": "kg", "kilogramo": "kg", "kilogramos": "kg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb", "libra": "lb", "libras": "lb",
    "oz": "oz", "ounce": "oz", "ounces": "oz", "onza": "oz", "onzas": "oz",
    "degree": "degree", "degrees": "degree", "grado": "degree", "grados": "degree",
    "fahrenheit": "fahrenheit", "celsius": "celsius", "centigrados": "celsius", "centigrade": "celsius",
    "percent": "percent", "%": "percent", "por": "",  # 'por ciento' handled below
    "cup": "cup", "cups": "cup", "taza": "cup", "tazas": "cup", "glass": "glass", "glasses": "glass", "vaso": "glass", "vasos": "glass",
}
UNIT_WORDS.pop("por")
CTX_ONLY = {"dose": "dose", "doses": "dose", "dosis": "dose", "time": "times", "times": "times", "vez": "times", "veces": "times", "x": "times"}
TIME_WORDS = {"second": "s", "seconds": "s", "segundo": "s", "segundos": "s", "minute": "min", "minutes": "min", "min": "min", "mins": "min",
              "minuto": "min", "minutos": "min", "hour": "h", "hours": "h", "hr": "h", "hrs": "h", "hora": "h", "horas": "h",
              "day": "d", "days": "d", "dia": "d", "dias": "d", "week": "w", "weeks": "w", "semana": "w", "semanas": "w",
              "month": "mo", "months": "mo", "mes": "mo", "meses": "mo", "year": "y", "years": "y", "ano": "y", "anos": "y"}
HOURS = {"s": 1 / 3600, "min": 1 / 60, "h": 1, "d": 24, "w": 168, "mo": 720, "y": 8760}
NOUNS = set(UNIT_WORDS) | set(CTX_ONLY) | set(TIME_WORDS)


@dataclass
class Num:
    value: float
    i0: int    # first token
    i1: int    # one past the last token
    clock: bool = False


def _tok(t: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN.finditer(t)]


def _digits(s: str, lang: str) -> list[float] | None:
    """A digit token -> one or more values. 0.125, 1,000, 1.000 (Spanish thousands), 2,5 (Spanish decimal), 1/2, 10:30,
    140/90 (two numbers: a reading, not a fraction)."""
    if ":" in s:
        h, _, m = s.partition(":")
        return [int(h) + int(m[:2] or 0) / 60] if h.isdigit() and m[:2].isdigit() else None
    if "/" in s:
        a, _, b = s.partition("/")
        if a.isdigit() and b.isdigit():
            a_, b_ = int(a), int(b)
            return [a_ / b_] if 0 < b_ <= 16 and a_ < b_ else [float(a_), float(b_)]
        return None
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):
        return [float(s.replace(",", ""))]
    if lang == "es" and re.fullmatch(r"[1-9]\d{0,2}(?:\.\d{3})+", s):
        return [float(s.replace(".", ""))]       # Spanish 1.000 = one thousand
    if re.fullmatch(r"\d+,\d{1,2}", s):
        return [float(s.replace(",", "."))]       # 2,5 = 2.5
    try:
        return [float(s)]
    except ValueError:
        return None


def _cardinal(W: list[str], i: int, lang: str) -> tuple[float, int] | None:
    """A spelled-out whole number starting at token i -> (value, next index)."""
    total, cur, j, seen, last = 0.0, 0.0, i, False, ""
    small, tens = (EN_SMALL, EN_TENS) if lang == "en" else (ES_SMALL, ES_TENS)
    while j < len(W):
        w = W[j]
        if lang == "en":
            if w in ("a", "an") and not seen and j + 1 < len(W) and W[j + 1] in ("hundred", "thousand"):
                cur, seen, last = 1, True, "small"
            elif w in small and (not seen or last in ("tens", "hundred", "and", "thousand", "hyphen")):
                if last == "tens" and small[w] >= 10:
                    break
                cur += small[w]; seen, last = True, "small"
            elif w in tens and (not seen or last in ("hundred", "and", "thousand")):
                cur += tens[w]; seen, last = True, "tens"
            elif w == "hundred" and seen and last != "hundred":
                cur = (cur or 1) * 100; last = "hundred"
            elif w == "thousand" and seen:
                total += (cur or 1) * 1000; cur = 0; last = "thousand"
            elif w == "and" and last in ("hundred", "thousand") and j + 1 < len(W) and (W[j + 1] in small or W[j + 1] in tens):
                last = "and"
            elif w == "-" and last == "tens" and j + 1 < len(W) and W[j + 1] in small and small[W[j + 1]] < 10:
                last = "hyphen"
            elif w == "-" and last in ("small", "tens") and j + 1 < len(W) and W[j + 1] in ("hundred", "thousand"):
                pass
            else:
                break
        else:
            if w in ES_HUNDREDS and (not seen or last == "mil"):
                cur += ES_HUNDREDS[w]; seen, last = True, "hundred"
            elif w in tens and (not seen or last in ("hundred", "mil")):
                cur += tens[w]; seen, last = True, "tens"
            elif w in small and (not seen or last in ("hundred", "mil", "y")):
                if last in ("hundred", "mil") and j + 1 < len(W) and W[j + 1] in ("veces", "vez"):
                    break                                            # "trescientos dos veces" = 300, twice
                cur += small[w]; seen, last = True, "small"
            elif w == "y" and last == "tens" and j + 1 < len(W) and W[j + 1] in small and small[W[j + 1]] < 10:
                last = "y"
            elif w == "mil" and (not seen or last in ("small", "tens", "hundred")):
                total += (cur or 1) * 1000; cur = 0; seen, last = True, "mil"
            elif w in ("millon", "millones") and seen:
                total += (cur or 1) * 1_000_000; cur = 0; last = "mil"
            else:
                break
        j += 1
    if not seen:
        return None
    if last in ("and", "y", "hyphen"):
        j -= 1
    return total + cur, j


def _is_noun(w: str) -> bool:
    return w in NOUNS


def numbers(toks: list[tuple[str, int, int]], lang: str) -> list[Num]:
    W = [x[0] for x in toks]
    n = len(W)
    w_ = lambda k: W[k] if 0 <= k < n else ""  # noqa: E731
    out: list[Num] = []
    skip: set[int] = set()   # tokens already used by a number ("and a half" after the noun it counts)
    i = 0
    while i < n:
        if i in skip:
            i += 1
            continue
        w = W[i]
        got: Num | None = None
        extra: list[Num] = []
        if w[0].isdigit():
            vals = _digits(w, lang)
            if vals and len(vals) == 1 and w_(i + 1) in ("mil", "thousand"):
                vals = [vals[0] * 1000]
                got = Num(vals[0], i, i + 2)
            elif vals:
                got = Num(vals[0], i, i + 1, clock=":" in w)
                extra = [Num(v, i, i + 1) for v in vals[1:]]
                nxt = w_(i + 1)
                if len(vals) == 1 and ":" not in w and nxt and (nxt in UNICODE_FRAC or re.fullmatch(r"\d+/\d+", nxt)):  # 2 1/2, 2½
                    f = UNICODE_FRAC.get(nxt) or (_digits(nxt, lang) or [1])[0]
                    if f < 1:
                        got = Num(vals[0] + f, i, i + 2)
        elif w in UNICODE_FRAC:
            got = Num(UNICODE_FRAC[w], i, i + 1)
        elif lang == "en":
            got = _en_word_number(W, i)
        else:
            got = _es_word_number(W, i)
        if got is None:
            i += 1
            continue
        j = got.i1
        # spoken decimals: "two point five", "cero punto uno dos cinco"
        if w_(j) in ("point", "punto") and re.fullmatch(r"\d{1,3}", w_(j + 1)) and not got.clock:
            got = Num(got.value + float("0." + w_(j + 1)), got.i0, j + 2)
            j = got.i1
        elif w_(j) in ("point", "punto") and w_(j + 1) in DIGIT_WORDS:
            digits, j = "", j + 1
            while w_(j) in DIGIT_WORDS:
                digits += str(DIGIT_WORDS[w_(j)]); j += 1
            got = Num(got.value + float("0." + digits), got.i0, j)
            j = got.i1
        # "... and a half" / "y media" / "y cuarto" right after the number, or right after the noun it counts
        k = j + 1 if _is_noun(w_(j)) else j
        half = (w_(k), w_(k + 1), w_(k + 2)) == ("and", "a", "half")
        ymed = w_(k) == "y" and w_(k + 1) in ("medio", "media")
        ycuarto = k == j and w_(k) == "y" and w_(k + 1) == "cuarto"
        if half or ymed or ycuarto:
            width = 3 if half else 2
            got = Num(got.value + (0.25 if ycuarto else 0.5), got.i0, got.i1 if k != j else k + width, got.clock)
            if k != j:
                skip.update(range(k, k + width))
        out.append(got)
        out += extra
        i = max(got.i1, i + 1)
    # a noun with no number but "y media": "cucharadita y media" = 1.5 teaspoons
    for i in range(n - 2):
        if _is_noun(W[i]) and W[i + 1] == "y" and W[i + 2] in ("media", "medio") and i + 1 not in skip \
                and not any(nm.i0 <= i < nm.i1 or nm.i1 == i for nm in out):
            out.append(Num(1.5, i, i + 3))
    out.sort(key=lambda x: x.i0)
    # a year read in pairs ("nineteen seventy-eight", "twenty twenty-five") after a month or "in"/"since"/"born"
    if lang == "en":
        merged: list[Num] = []
        for nm in out:
            if merged:
                p = merged[-1]
                ctx_ok = any(x in MONTHS for x in W[max(0, p.i0 - 5):p.i0]) or w_(p.i0 - 1) in ("in", "since", "born", "year")
                if ctx_ok and nm.i0 == p.i1 and 10 <= p.value <= 20 and p.value == int(p.value) and 0 <= nm.value <= 99 \
                        and not W[p.i0][0].isdigit() and not W[nm.i0][0].isdigit() and p.i1 - p.i0 == 1:
                    merged[-1] = Num(p.value * 100 + nm.value, p.i0, nm.i1)
                    continue
            merged.append(nm)
        out = merged
    return out


MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
          "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "setiembre", "octubre", "noviembre", "diciembre"}
EN_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
          "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
          "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
DUR_PREP = {"for", "in", "within", "after", "over", "about", "than", "least", "past", "last", "nearly", "almost", "around",
            "under", "only", "just", "once", "every", "each", "maybe", "roughly", "approximately", "another", "half"}


def _en_word_number(W: list[str], i: int) -> Num | None:
    n = len(W)
    w = W[i]
    prv = W[i - 1] if i else ""
    nxt = W[i + 1] if i + 1 < n else ""
    nxt2 = W[i + 2] if i + 2 < n else ""
    if w in ("a", "an", "one"):
        if nxt in ("half", "quarter", "third", "eighth"):                      # a half, a quarter, one third
            j = i + 2
            if W[j:j + 2] in (["of", "a"], ["of", "an"], ["of", "one"]):        # a quarter of a tablet
                j += 2
            return Num(EN_FRAC[nxt], i, j)
        if nxt in ("hundred", "thousand"):
            r = _cardinal(W, i, "en")
            return Num(r[0], i, r[1]) if r else None
    if w in ("a", "an"):
        per = prv in ("times", "twice", "once", "x", "per") or prv in CTX_ONLY or prv in UNIT_WORDS or (prv and prv[0].isdigit()) \
            or prv in EN_SMALL or prv in EN_TENS or prv in ("hundred", "thousand", "half")
        if prv in ("half", "quarter", "third", "eighth", "of"):
            return None
        if (nxt in UNIT_WORDS or (nxt in CTX_ONLY and nxt != "x")) and not per:
            return Num(1, i, i + 1)
        if nxt in ("full", "whole", "entire", "single") and nxt2 in TIME_WORDS and (prv in DUR_PREP or prv in ("", ".", "?", "!", ",", "¿")):
            return Num(1, i, i + 1)
        # "a day" is a length of time only after for/in/within/about... or before "ago"; otherwise it's a rate ("two liters a day")
        clause_start = prv in ("", ".", "?", "!", ";", ":", "—", "-", ",", "and", "or")
        if prv == "in" and any(W[k] in UNIT_WORDS or W[k] in CTX_ONLY for k in range(max(0, i - 4), i - 1)):
            return None                                                      # "four puffs in a day" is a rate
        if nxt in TIME_WORDS and not per and (prv in DUR_PREP or clause_start or nxt2 in ("ago", "later", "earlier", "before", "after", "old", "away")
                                              or W[i + 2:i + 5] == ["and", "a", "half"]):
            return Num(1, i, i + 1)
        return None
    if w == "both" and "and" not in W[i + 1:i + 7] and prv not in ("you", "them", "us", "we", "they") \
            and ((nxt.endswith("s") and nxt.isalpha() and nxt not in ("is", "was", "has", "does", "says")) or nxt in ("of", "feet", "teeth")
                 or (not nxt.isalpha() and prv in ("to", "of", "for", "with", "on", "in"))):
        return Num(2, i, i + 1)                                               # "both allergies" = "las dos alergias"
    if w in EN_ORD and (prv in MONTHS or (prv == "the" and nxt == "of" and nxt2 in MONTHS)):
        return Num(EN_ORD[w], i, i + 1)                                        # "February third", "the third of February"
    if w in ("twenty", "thirty") and nxt == "-" and nxt2 in EN_ORD and EN_ORD[nxt2] < 10 and (prv in MONTHS or prv == "the"):
        return Num(EN_TENS[w] + EN_ORD[nxt2], i, i + 3)
    if w == "half":
        if prv in ("a", "an", "and", "one"):
            return None
        if nxt in ("a", "an"):
            return Num(0.5, i, i + 2)
        if nxt == "of" and nxt2 in ("a", "an", "the", "one", "his", "her", "your"):
            return Num(0.5, i, i + 2)
        if _is_noun(nxt):
            return Num(0.5, i, i + 1)
        return None
    if w == "once":
        return Num(1, i, i + 1) if nxt in ("a", "per", "daily", "every", "each", "weekly", "an") else None
    if w == "twice":
        return Num(2, i, i + 1)
    if w in EN_SMALL or w in EN_TENS:
        r = _cardinal(W, i, "en")
        if r is None:
            return None
        v, j = r
        nj = W[j] if j < n else ""
        if nj in EN_FRAC and v < 10 and nj not in ("half",):                   # three quarters
            return Num(v * EN_FRAC[nj], i, j + 1)
        if w == "one" and j == i + 1:
            # the pronoun "one" ("take one every morning", "the left one") is not a quantity
            if prv in ("the", "this", "that", "which", "new", "other", "another", "each", "any", "every", "no", "old", "right", "left",
                       "last", "first", "little", "big", "same", "only", "good", "bad") or nj == "of":
                return None
            if not (_is_noun(nj) or (nj == "and" and W[j + 1:j + 3] == ["a", "half"]) or nj in ("point",) or nj in EN_FRAC):
                return None
        return Num(v, i, j)
    return None


def _es_word_number(W: list[str], i: int) -> Num | None:
    n = len(W)
    w = W[i]
    prv = W[i - 1] if i else ""
    nxt = W[i + 1] if i + 1 < n else ""
    if w in ("ambos", "ambas"):
        return Num(2, i, i + 1)
    if w == "dos" and i >= 3 and W[i - 3] in ("cualquiera", "ninguna", "ninguno", "cual", "una", "uno") and W[i - 2] == "de":
        return None                                                          # "cualquiera de las dos" = either
    if w == "primero" and nxt == "de" and i + 2 < n and W[i + 2] in MONTHS:
        return Num(1, i, i + 1)
    if w in ("un", "una", "uno"):
        if nxt in ("cuarto", "tercio", "octavo", "medio"):                    # un cuarto (de tableta)
            j = i + 2
            if W[j:j + 2] in (["de", "una"], ["de", "un"]):
                j += 2
            return Num(ES_FRAC[nxt], i, j)
        if prv == "de" and i >= 2 and W[i - 2] in ES_FRAC:                     # la mitad de una pastilla
            return None
        if nxt == "mil":
            r = _cardinal(W, i, "es")
            return Num(r[0], i, r[1]) if r else None
        if nxt in TIME_WORDS and prv in ("en", "por", "al") and any(W[k] in UNIT_WORDS or W[k] in CTX_ONLY for k in range(max(0, i - 4), i - 1)):
            return None                                                      # "cuatro dosis en un día" is a rate
        if _is_noun(nxt) or (nxt == "y" and W[i + 2:i + 3] in (["medio"], ["media"])):
            return Num(1, i, i + 1)
        return None                                                          # an article: "una pregunta", "tome una"
    if w == "mitad":
        j = i + 1
        if W[j:j + 2] in (["de", "una"], ["de", "un"]):
            j += 2
        return Num(0.5, i, j)
    if w in ("medio", "media"):
        if prv in ("y",):
            return None
        return Num(0.5, i, i + 1) if _is_noun(nxt) else None
    if w in ES_SMALL or w in ES_TENS or w in ES_HUNDREDS or w == "mil":
        r = _cardinal(W, i, "es")
        if r is None:
            return None
        v, j = r
        nj = W[j] if j < n else ""
        if nj in ("cuartos", "tercios", "octavos") and v < 10:              # tres cuartos
            return Num(v * ES_FRAC[nj], i, j + 1)
        return Num(v, i, j)
    return None


def _num_tokens(t: str, lang: str = "en") -> list[tuple[float, str, int, int]]:
    """(value, words, start, end) for every number in folded text (kept for scripts that import it)."""
    toks = _tok(t)
    return [(nm.value, t[toks[nm.i0][1]:toks[nm.i1 - 1][2]], toks[nm.i0][1], toks[nm.i1 - 1][2]) for nm in numbers(toks, lang)]


def _fmt(v: float) -> str:
    return str(int(v)) if v == int(v) else f"{v:.4g}"


# ---------------------------------------------------------------------------------------------------------------------
# Lexicons (folded: no accents)
# ---------------------------------------------------------------------------------------------------------------------
DRUGS = {
    "acetaminophen": r"acetaminophen|acetaminofen|tylenol|tilenol|paracetamol", "ibuprofen": r"ibuprofen|ibuprofeno|advil|motrin",
    "amoxicillin": r"amoxicillin|amoxicilina|amoxil", "penicillin": r"penicillin|penicilina", "aspirin": r"aspirin|aspirina",
    "insulin": r"insulin|insulina", "inulin": r"inulin|inulina", "metformin": r"metformin|metformina", "metronidazole": r"metronidazole|metronidazol|flagyl",
    "lisinopril": r"lisinopril", "losartan": r"losartan", "amlodipine": r"amlodipine|amlodipino",
    "atorvastatin": r"atorvastatin|atorvastatina|lipitor", "simvastatin": r"simvastatin|simvastatina",
    "warfarin": r"warfarin|warfarina|coumadin", "apixaban": r"apixaban|eliquis", "rivaroxaban": r"rivaroxaban|xarelto",
    "prednisone": r"prednisone|prednisona", "prednisolone": r"prednisolone|prednisolona|orapred",
    "albuterol": r"albuterol|salbutamol|ventolin", "ondansetron": r"ondansetron|zofran", "cephalexin": r"cephalexin|cefalexina|keflex",
    "azithromycin": r"azithromycin|azitromicina|zithromax", "doxycycline": r"doxycycline|doxiciclina", "ciprofloxacin": r"ciprofloxacin|ciprofloxacino|cipro",
    "omeprazole": r"omeprazole|omeprazol|prilosec", "ranitidine": r"ranitidine|ranitidina|zantac", "famotidine": r"famotidine|famotidina|pepcid",
    "sulfa": r"sulfa|sulfas|bactrim|sulfamethoxazole|sulfametoxazol", "codeine": r"codeine|codeina", "morphine": r"morphine|morfina",
    "oxycodone": r"oxycodone|oxicodona", "hydrocodone": r"hydrocodone|hidrocodona|vicodin|norco", "tramadol": r"tramadol|ultram",
    "trazodone": r"trazodone|trazodona", "naproxen": r"naproxen|naproxeno|aleve", "celecoxib": r"celecoxib|celebrex",
    "citalopram": r"citalopram|celexa", "sertraline": r"sertraline|sertralina|zoloft",
    "hydroxyzine": r"hydroxyzine|hidroxicina|hidroxizina|atarax", "hydralazine": r"hydralazine|hidralazina",
    "hydrochlorothiazide": r"hydrochlorothiazide|hidroclorotiazida", "hydroxychloroquine": r"hydroxychloroquine|hidroxicloroquina|plaquenil",
    "digoxin": r"digoxin|digoxina|lanoxin", "gabapentin": r"gabapentin|gabapentina|neurontin",
    "diphenhydramine": r"diphenhydramine|difenhidramina|benadryl", "cetirizine": r"cetirizine|cetirizina|zyrtec",
    "loratadine": r"loratadine|loratadina|claritin", "furosemide": r"furosemide|furosemida|lasix", "levothyroxine": r"levothyroxine|levotiroxina|synthroid",
    "clopidogrel": r"clopidogrel|plavix", "heparin": r"heparin|heparina", "nitroglycerin": r"nitroglycerin|nitroglicerina",
    "ceftriaxone": r"ceftriaxone|ceftriaxona", "antibiotic": r"antibiotics?|antibioticos?", "latex": r"latex", "peanut": r"peanuts?|cacahuates?|mani|cacahuetes?",
    "shellfish": r"shellfish|mariscos?", "iodine": r"iodine|yodo|contrast dye|medio de contraste", "egg": r"eggs?|huevos?",
    "blood thinner": r"blood thinners?|anticoagulantes?|medicinas? para (?:adelgazar|diluir|aguar) la sangre|para la sangre rala",
    "inhaler": r"inhaler|inhalador",
}
WARNINGS = {
    "emergency": r"emergency(?: room)?|\ber\b|urgencias|sala de emergencias?|emergencias?",
    "911": r"911|nine one one|nueve uno uno|novecientos once",
    # an instruction to call ("call us if...", "llámenos"), not "we'll call you with the result" / "le llamamos"
    "call": r"\bcall(?:ing)? (?:us|me|911|the (?:doctor|clinic|office|nurse|pediatrician|ambulance)|your (?:doctor|pediatrician)|an ambulance|"
            r"right away|immediately|if|when|first|back|our)\b|\bcall\b(?=\s*(?:[.,;!?—-]|$))|let (?:us|me|the doctor) know|contact us|phone us|"
            r"\b(?:llame|llamen|llamenos|llameme|llamennos|llamar(?:nos|me|le)?|llamandonos|avis(?:e|enos|eme|ennos|ar|arnos|arme))\b",
    "return": r"come back|go back|return(?=\s*[?.!,]|\s+(?:to|if|here|right|in|for|on|immediately|tomorrow)\b)|go back to the|regres(?:e|ar|en|o|a)\b|vuelva(?! a [a-z]+(?:ar|er|ir)(?:se|lo|la|le)?\b)|volver(?! a [a-z]+(?:ar|er|ir)\b)",
    "breathing": r"(?:trouble|difficulty|hard time|problems?|struggling|struggles?) (?:to )?breath(?:e|ing)|hard to breathe|can'?t breathe|short(?:ness)? of breath|"
                 r"(?:dificultad|problemas?) (?:para|al) respirar|no puede respirar|le cuesta (?:trabajo )?respirar|le falta el aire|falta de aire|respira con dificultad|"
                 r"batall\w* (?:para|pa) respirar|(?:can'?t|couldn'?t|unable to) (?:get enough air|catch (?:his|her|my|your) breath)",
    "chest pain": r"chest pain|pain in (?:the|his|her|your) chest|dolor (?:de|en el) pecho", "fever": r"fever|temperature|fiebre|calentura|temperatura",
    "bleeding": r"bleed(?:ing|s)?|bled\b|bloody|nosebleeds?|blood in (?:the |his |her |your |my )?(?:urine|stool|stools|poop|vomit|pee)|black stools?|"
                r"sangrado|sangr(?:a|an|ar|o)\b|sangrando|sangre en (?:la |el |las |los |su )?(?:orina|heces|popo|excremento|vomito|pipi)|con sangre|heces negras|excremento negro",
    "rash": r"rash|hives|[sz]arpullido|salpullido|ronchas|erupcion", "swelling": r"swell(?:ing|ed|s)?|swollen|hinch(?:azon|ad[oa]s?|a|an|o|e|en|arse|ando)\b|inflamad[oa]",
    "vomiting": r"vomit(?:ing|s|ed)?|thr(?:ew|ows?|owing) (?:\w+ ){0,2}up|vomit(?:o|a|ando|os|e|ar|an)\b|devolv",
    "fainting": r"faint(?:ing|ed|s)?|pass(?:es|ed)? out|black(?:ed)? out|desmay|perd(?:io|ido|er|ió|e) el conocimiento|loses? consciousness|lost consciousness",
    "dizzy": r"dizz(?:y|iness)|light-?headed|mare(?:o|ad[oa]|os|e|a|an|arse)\b",
    "seizure": r"seizures?|convulsi(?:on|ones)|ataques?", "unresponsive": r"unconscious|(?:won'?t|will not|doesn'?t|does not|can'?t|cannot) (?:want to )?wake|unresponsive|inconsciente|no despierta|no reacciona|no (?:se )?quiere despertar|hard to wake|"
                     r"(?:dificil|cuesta(?: trabajo)?|trabajo) (?:de )?despertar|no (?:se )?despierta",
    "pregnant": r"pregnan(?:t|cy)|embarazad[ao]|embarazo",
    "pain": r"\bpain\b|hurts?|\bdolor\b|\bduele\b",
}
WARNING_MEDIUM: set[str] = set()   # warning signs whose change is only a note (none at present)
ALLERGY = r"allerg(?:y|ies|ic)|alerg(?:ia|ias|ico|ica|icos|icas)"
# Negation is compared by presence, sentence by sentence, not by counting words: languages negate differently ("no food,
# no water, nothing" vs "no le dé nada de comer ni de tomar"). Each negating word gets a class:
#   verbal   negates a verb or noun ("not", "don't", "never", "no" + word, "nunca")
#   nominal  a negative pronoun or connector ("nothing", "none", "nada", "ninguna", "ni")
#   answer   a bare "No," / "No." answering a question
#   weak     "without", "sin", "forget", "le falta": often a paraphrase of a plain "no", never flagged on its own
#   hardly   "hardly", "barely", "casi no", "apenas": flagged only when the other side has no negation at all
#   stop     "stop", "avoid", "deje de", "suspenda", "evite"
#   dontstop a negated stop ("don't stop", "no deje de", "no lo suspenda"), which means "keep going"
NEG_VERBAL_EN = {"not", "no", "never", "cannot"}
NEG_NOMINAL_EN = {"none", "nothing", "nobody", "neither", "nor", "nowhere"}
NEG_WEAK_EN = {"without", "forget", "forgets", "forgot", "forgotten", "forgetting"}
NEG_HARDLY_EN = {"hardly", "barely", "scarcely", "rarely", "seldom"}
STOP_EN = {"stop", "stops", "stopped", "stopping", "quit", "quits", "quitting", "avoid", "avoids", "avoided", "avoiding",
           "discontinue", "discontinued", "discontinuing", "skip", "skips", "skipped", "skipping"}
STOP_EN_OBJ = {"the", "his", "her", "your", "their", "my", "it", "them", "that", "this", "these", "those", "all", "any", "a", "an"}
NEG_VERBAL_ES = {"no", "nunca", "jamas"}
NEG_NOMINAL_ES = {"nada", "nadie", "ningun", "ninguna", "ninguno", "ningunas", "ningunos", "ni", "tampoco"}
NEG_WEAK_ES = {"sin", "apenas"}
STOP_ES = {"salte", "saltar", "saltarse", "saltes", "saltarme", "saltarle", "saltandose", "saltense"}
STOP_ES_RX = re.compile(r"(?:suspend|evit)(?:a|e|o|as|es|an|en|ar|er|ido|io|iendo|ando)(?:lo|la|los|las|le|les|se)?")
DEJAR = {"deje", "deja", "dejar", "dejen", "dejo", "dejes"}
PARAR = {"pare", "parar", "paren", "pararle", "pararlo"}
LACK_ES = {"falta", "faltaba", "falto", "faltan", "faltaban", "faltar"}
TIMING = [
    (r"with (?:food|meals|a meal)|after (?:eating|meals|a meal)|con (?:la |las )?comidas?|con (?:los )?alimentos|despues de comer|con cada comida", "with_food"),
    (r"(?:on an )?empty stomach|en ayunas|(?:con el )?estomago vacio", "empty_stomach"),
    (r"before (?:eating|meals|breakfast|lunch|dinner|supper)|antes de (?:comer|desayunar|almorzar|cenar|la cena|la comida|las comidas|el desayuno|el almuerzo)|antes del desayuno|antes del almuerzo", "before_meal"),
    (r"after (?:breakfast|lunch|dinner|supper)|despues de (?:desayunar|almorzar|cenar|la cena|el desayuno|el almuerzo)|despues del desayuno|despues del almuerzo", "after_meal"),
    (r"at bedtime|before bed|(?<!at #\d )(?<!at #\d\d )at night|every night|(?<!at #\d )(?<!at #\d\d )in the evening|before (?:you )?(?:go to )?sleep|antes de dormir|antes de acostarse|(?:en|por) la noche|cada noche|todas las noches|(?<!#\d )(?<!#\d\d )de (?:la )?noche", "night"),
    (r"(?<!at #\d )(?<!at #\d\d )(?<!tomorrow )(?<!this )in the morning|every morning|(?<!manana )(?<!hoy )(?:en|por) la manana|cada manana|todas las mananas|en las mananas|(?<!#\d )(?<!#\d\d )de la manana", "morning"),
]
TIMING_CONFLICT = {frozenset(("with_food", "empty_stomach")), frozenset(("before_meal", "after_meal")), frozenset(("morning", "night")),
                   frozenset(("empty_stomach", "after_meal"))}
PRN = r"as needed|if needed|when needed|as necessary|if necessary|when necessary|if (?:he|she|you|they) needs? (?:it|them)|when (?:he|she|you|they) needs? (?:it|them)|prn|" \
      r"si (?:lo |la |los |las |le |les )?necesita|cuando (?:lo |la |los |las |le |les )?necesite|cuando sea necesario|si es necesario|si (?:lo|la) necesita|" \
      r"segun sea necesario|si hace falta|cuando haga falta"
LIMIT_MAX = r"(?:no|not) (?:\w+ ){0,2}more (?:often )?than(?= #\d| every| each| a\b| an\b| one\b| the\b| \w+ (?:doses?|times?|pills?|tablets?|puffs?))|never more (?:often )?than|(?:do not|don't|never|not to) (?:\w+ ){0,2}more than|no (?:\w+ ){0,2}mas de|nunca mas de|maximum(?: of)?|max\b|(?<!got )(?<!get )(?<!gets )(?<!went )(?<!goes )(?<!rose )up to(?= #\d)|at most|como maximo|maximo(?: de)?|hasta un maximo de|(?<!llega )(?<!llego )(?<!sube )(?<!subio )(?<!subir )(?<!llegar )(?<!llegan )(?<!suben )hasta(?= #\d)|" \
            r"(?:do not|don't|no|never) (?:\w+ ){0,1}(?:exceed|pass|go over|go above)|(?:no|nunca|jamas) (?:\w+ ){0,2}(?:exceda|exceder|excedas|pase de|pasar de|pasarse de|sobrepase|sobrepasar)|" \
            r"no mas(?= (?:#\d|una?\b|uno\b))|(?:no|nunca|jamas) mas (?:seguido|frecuente|a menudo|seguidas?) (?:que|de)|" \
            r"sin (?:pasar(?:se)?|exceder|sobrepasar) (?:de )?"
LIMIT_MIN = r"at least|no less than|minimum(?: of)?|por lo menos|al menos|minimo(?: de)?|como minimo"
BODY_EN = r"(?:side|arm|leg|ear|eye|hand|foot|knee|hip|shoulder|kidney|lung|breast|ankle|wrist|elbow|finger|toe|thumb|thigh|flank|lower|upper|chest|one|quadrant|cheek|temple|heel|calf|shin)"


def _side_slots(t: str, lang: str) -> list[Slot]:
    """left/right said about a body part or a direction. English 'right' is also 'correct' and 'left' is also a verb, so
    English needs a cue (the/his/on the/to the/turn, or a body part after it); Spanish izquierdo/derecho almost always
    means a side, except 'derecho a' (a right to) and 'todo derecho' (straight ahead)."""
    out, seen = [], set()
    if lang == "en":
        pats = [rf"\b(left|right)\s+(?:\w+\s+)?{BODY_EN}\b", r"\b(?:on|to|in|at) (?:the|his|her|your|my|their) (left|right)\b",
                r"\b(?:the|his|her|your|my|their) (left|right)\b(?!\s*(?:now|away|here|there|thing|answer|time|person|place|way|to\b))",
                r"\bturn (left|right)\b", r"\b(left|right)\b(?= (?:or|and) (?:the )?(?:left|right)\b)",
                r"\b(?:left|right) (?:or|and) (?:the )?(left|right)\b"]
    else:
        pats = [r"(?<!todo )\b(izquierd[oa]s?|derech[oa]s?)\b(?! a (?:la|las|los|el|un|una|recibir|tener|ser)\b)"]
    for p in pats:
        for m in re.finditer(p, t):
            pos = m.start(1)
            if pos in seen:
                continue
            seen.add(pos)
            w = m.group(1)
            pair = re.match(r"(?:left|right) (?:or|and) ", m.group(0)) is not None   # "left or right": the second word
            out.append(Slot("side", "left" if w.startswith(("left", "izq")) else "right", w if pair else m.group(0),
                            pos=m.start(1) if pair else m.start()))
    return out


def _negations(toks: list[tuple[str, int, int]], lang: str, exclude: list[tuple[int, int]] = ()) -> list[Slot]:
    """Every negating word with its class (see NEG_* above) as the slot value."""
    W = [x[0] for x in toks]
    n = len(W)
    w_ = lambda k: W[k] if 0 <= k < n else ""  # noqa: E731
    punct = {",", ".", "!", "?", ";", ":", "-", "—", "–", "…", ")"}
    openers = {".", "!", "?", ";", ":", "—", "–", "-", "¿", "¡", '"', "“", "(", "…"}
    fillers = {"oh", "ah", "uh", "um", "well", "bueno", "pues", "ay", "eh", "este", "mmm", "hmm", "okay", "ok"}

    def answer_at(k: int) -> bool:
        """A bare 'No,' / 'No.' that opens a sentence (after an optional 'oh', 'bueno', ...) answers a question, and so
        does a reported one ("she says no,"). A 'no' after other words ("fiebre no", "creo que no", "fever, no") negates them."""
        if not (w_(k + 1) in punct or k + 1 == n):
            return False
        if w_(k - 1) in ("says", "said", "say", "dice", "dijo", "decir", "answers", "answered", "responde", "contesta"):
            return True
        j, conj = k - 1, False
        while j >= 0 and (W[j] in fillers or W[j] in (",", "and", "but", "y", "pero", "no")):
            conj = conj or W[j] in ("and", "but", "y", "pero")
            if conj and W[j] == ",":
                return True                                  # "..., y no, fiebre no ha tenido"
            j -= 1
        return j < 0 or W[j] in openers

    def in_question(k: int) -> bool:
        """Inside the asking part of a question ("¿no hay problema?", "isn't it?"): negation there is rhetorical, so it is
        weak. Not a statement with a tag ("Don't give him aspirin, okay?")."""
        j = k + 1
        while j < n and W[j] not in (".", "!", "?", ",", ";"):
            j += 1
        if j >= n or W[j] != "?":
            return False
        if lang == "es":
            i = k - 1
            while i >= 0 and W[i] not in (".", "!", "?", ",", ";", "¿"):
                i -= 1
            return i >= 0 and W[i] == "¿"
        return True

    found: list[list] = []   # [class, k]
    for k, w in enumerate(W):
        cls = ""
        if lang == "en":
            if w == "no" and w_(k + 1) == "?":
                continue                                     # tag question: "..., no?"
            if (w == "no" and w_(k + 1) == "problem") or (w in ("don't", "do") and W[k + 1:k + 2] == ["worry"]) \
                    or (w == "not" and w_(k - 1) == "do" and w_(k + 1) == "worry"):
                continue                                     # "no problem", "don't worry"
            if w == "not" and w_(k - 1) in ("course", "certainly", "definitely", "absolutely", "probably") and (w_(k + 1) in punct or k + 1 == n):
                found.append(["answer", k])                  # "of course not." answers like "No."
                continue
            if w in NEG_VERBAL_EN or w.endswith("n't"):
                cls = "answer" if w == "no" and answer_at(k) else "verbal"
            elif w in NEG_NOMINAL_EN:
                cls = "nominal"
            elif w in NEG_WEAK_EN:
                cls = "weak"
            elif w in NEG_HARDLY_EN:
                cls = "hardly"
            elif w in STOP_EN and (w.startswith(("avoid", "skip")) or w_(k + 1) in STOP_EN_OBJ or w_(k + 1).endswith("ing")
                                   or _is_medicine(w_(k + 1)) or w_(k - 1) in ("don't", "not", "never", "dont")):
                cls = "stop"                                 # an instruction ("stop the medicine"), not "it stopped quickly"
        else:
            if w == "no" and w_(k + 1) == "?":
                continue                                     # "¿verdad?, ¿no?"
            if w == "no" and (W[k + 1:k + 3] in (["hay", "problema"], ["pasa", "nada"], ["se", "preocupe"], ["te", "preocupes"], ["hay", "de"])
                              or W[k + 1:k + 4] == ["se", "le", "olvide"]):
                continue                                     # "no hay problema" = fine, "no hay de qué" = you're welcome
            if w == "no" and w_(k - 1) == "si" and (w_(k + 1) in punct or k + 1 == n):
                continue                                     # "si no," = otherwise
            if w == "nada" and w_(k - 1) in ("de", "por") and (w_(k + 1) in punct or k + 1 == n):
                continue                                     # "de nada" = you're welcome
            if w in NEG_VERBAL_ES:
                if w_(k - 1) == "casi":
                    cls = "hardly"                           # "casi no" = hardly
                else:
                    cls = "answer" if w == "no" and answer_at(k) else "verbal"
            elif w == "nada" and w_(k + 1) == "mas":
                cls = "weak"                                 # "nada más" = only / nothing else
            elif w in NEG_NOMINAL_ES:
                cls = "hardly" if w_(k - 1) == "casi" else "nominal"
            elif w == "apenas":
                cls = "hardly"
            elif w in NEG_WEAK_ES or w.startswith("olvid"):
                cls = "weak"
            elif w in STOP_ES or STOP_ES_RX.fullmatch(w):
                cls = "stop"
            elif w in LACK_ES:
                cls = "weak"                                 # "le falta el aire" = he can't get enough air
            elif w in DEJAR and (w_(k + 1) == "de"
                                 or (w_(k + 1) in ("el", "la", "los", "las", "su", "sus", "ese", "esa", "esta", "este")
                                     and any(_is_medicine(x) for x in W[k + 2:k + 4]))
                                 or (w_(k - 1) in ("lo", "la", "los", "las", "le", "les")
                                     and (w_(k + 1) in punct or k + 1 == n or w_(k + 1) in ("y", "o", "si", "hasta", "cuando", "ya", "nunca")))):
                cls = "stop"                                 # "no lo deje." / "déjelo de dar"; not "la dejo tapada"
            elif w.startswith("quit") and w_(k - 1) in ("lo", "la", "los", "las") and w_(k - 2) == "se":
                cls = "stop"                                 # "se la quito" = I stop giving it
            elif w in PARAR and w_(k + 1) in ("de", "el", "la", "los", "las", "su", "sus"):
                cls = "stop"
            if w in ("ninguna", "ningun") and w_(k - 1) == "de" and w_(k + 1) in ("manera", "forma", "modo"):
                cls = "verbal"                               # "de ninguna manera" = by no means
        if cls in ("verbal", "nominal", "answer", "hardly") and in_question(k):
            cls = "weak"
        if cls:
            found.append([cls, k])
    # a negated stop means "keep going": "don't stop", "do not stop", "no lo deje de tomar", "nunca suspenda";
    # a negated 'forget' means "remember": both words drop out
    for i, (cls, k) in enumerate(found):
        if cls == "weak" and (W[k].startswith(("olvid", "forget", "forgot"))):
            for j in range(i - 1, -1, -1):
                c2, k2 = found[j]
                if k - k2 > 3:
                    break
                if c2 == "verbal" and not any(x in punct for x in W[k2 + 1:k]):
                    found[i][0] = found[j][0] = ""
                    break
            continue
        if cls != "stop":
            continue
        for j in range(i - 1, -1, -1):
            c2, k2 = found[j]
            if k - k2 > 4:
                break
            if c2 == "verbal" and not any(x in punct for x in W[k2 + 1:k]):
                found[i][0] = "dontstop"
                found[j][0] = ""
                break
    out = []
    for cls, k in found:
        s_ = toks[k][1]
        if cls and not any(a <= s_ < b for a, b in exclude):
            out.append(Slot("negation", cls, W[k], pos=s_))
    return out


MEDICINE_WORDS = {"medicina", "medicinas", "medicamento", "medicamentos", "pastilla", "pastillas", "tableta", "tabletas", "tratamiento",
                  "jarabe", "inyecciones", "inhalador", "crema", "gotas", "terapia", "fisioterapia", "antibiotico", "dosis", "insulina",
                  "medicine", "medication", "medications", "medicines", "pills", "pill", "tablets", "tablet", "treatment", "therapy",
                  "antibiotic", "antibiotics", "inhaler", "drops", "doses", "dose", "insulin"}


def _is_medicine(w: str) -> bool:
    return w in MEDICINE_WORDS or bool(_KNOWN_DRUG.fullmatch(w))


def guess_lang(text: str) -> str:
    t = f" {fold(text)} "
    es = sum(t.count(f" {w} ") for w in ("el", "la", "de", "que", "y", "es", "por", "para", "con", "su", "le", "lo", "una", "los", "las", "esta", "tiene", "dice"))
    en = sum(t.count(f" {w} ") for w in ("the", "and", "is", "to", "of", "you", "he", "she", "it", "with", "for", "your", "has", "this", "take", "says"))
    return "es" if es > en else "en"


FREQ_UNIT = {"hour": "h", "hours": "h", "hr": "h", "hrs": "h", "hora": "h", "horas": "h", "day": "d", "days": "d", "dia": "d", "dias": "d",
             "week": "w", "weeks": "w", "semana": "w", "semanas": "w", "minute": "min", "minutes": "min", "minuto": "min", "minutos": "min",
             "month": "mo", "months": "mo", "mes": "mo", "meses": "mo"}
_FU = "|".join(sorted(FREQ_UNIT, key=len, reverse=True))
_TU = "|".join(sorted(TIME_WORDS, key=len, reverse=True))


def extract(text: str, lang: str | None = None) -> list[Slot]:
    lang = lang or guess_lang(text)
    t = fold(text)
    toks = _tok(t)
    nums = numbers(toks, lang)
    # A copy of the sentence where each number is a placeholder "#k", with the char offsets of every placeholder token.
    ns_parts, ns_pos = [], []   # text pieces and (start,end) in t for each piece
    starts = {nm.i0: k for k, nm in enumerate(nums)}
    inside = {i for nm in nums for i in range(nm.i0, nm.i1)}
    seen_first = set()
    for i, (w, s, e) in enumerate(toks):
        if i in starts and starts[i] not in seen_first:
            k = starts[i]
            seen_first.add(k)
            nm = nums[k]
            ns_parts.append(f"#{k}")
            ns_pos.append((toks[nm.i0][1], toks[nm.i1 - 1][2]))
        elif i in inside:
            continue
        else:
            ns_parts.append(w)
            ns_pos.append((s, e))
    ns, offs, c = "", [], 0
    for p, (s, e) in zip(ns_parts, ns_pos):
        if ns:
            ns += " "; c += 1
        offs.append((c, c + len(p), s, e))
        ns += p; c += len(p)

    def span_of(a: int, b: int) -> tuple[str, int]:
        hit = [(s, e) for (x, y, s, e) in offs if x < b and y > a]
        if not hit:
            return "", -1
        return t[hit[0][0]:hit[-1][1]], hit[0][0]

    used: set[int] = set()
    slots: list[Slot] = []

    def num_ids(m: re.Match) -> list[int]:
        return [int(x) for x in re.findall(r"#(\d+)", m.group(0))]

    def add(kind: str, value: str, m: re.Match, ctx: str = ""):
        sp, pos = span_of(m.start(), m.end())
        slots.append(Slot(kind, value, sp, ctx, pos))
        used.update(num_ids(m))

    # How often. Dosing intervals and "N times a day" carry their own numbers.
    for m in re.finditer(rf"(?<![a-z#])(?:every|cada|each)\s+(?:(#\d+)(?:\s*(?:-|to|or|a|o|u)\s*(#\d+))?\s+)?({_FU})\b", ns):
        lo = _fmt(nums[int(m.group(1)[1:])].value) if m.group(1) else "1"
        hi = ("-" + _fmt(nums[int(m.group(2)[1:])].value)) if m.group(2) else ""
        add("frequency", f"q{lo}{hi}{FREQ_UNIT[m.group(3)]}" if (m.group(1) or FREQ_UNIT[m.group(3)] != "d") else "1x/day", m)
    # "6 hours apart", "con 6 horas de diferencia": the same dosing interval
    for m in re.finditer(r"(#\d+)(?:\s*(?:-|to|or|a|o|u)\s*(#\d+))?\s+(hours?|horas)\s+(?:apart|between (?:doses|each dose|them)|de diferencia|entre (?:cada )?(?:dosis|toma|tomas|una y otra))\b", ns):
        if int(m.group(1)[1:]) in used:
            continue
        hi = ("-" + _fmt(nums[int(m.group(2)[1:])].value)) if m.group(2) else ""
        add("frequency", f"q{_fmt(nums[int(m.group(1)[1:])].value)}{hi}h", m)
    no_neg: list[tuple[int, int]] = []
    for m in re.finditer(r"(?<![a-z])(?:every other day|every second day|cada tercer dia|every third day|(?:un|1) dia si,? y (?:un dia|otro|1 dia) no|"
                         r"dia (?:de por medio|por medio)|cada otro dia)", t):
        slots.append(Slot("frequency", "q3d" if "tercer" in m.group(0) or "third" in m.group(0) else "q2d", m.group(0), pos=m.start()))
        no_neg.append((m.start(), m.end()))
        used.update(k for k, nm in enumerate(nums) if m.start() <= toks[nm.i0][1] < m.end())
    for m in re.finditer(r"(#\d+)\s+(?:times?\s+|x\s*|veces\s+|vez\s+|(?:nights|days|mornings|evenings|noches|dias|mananas|tardes)\s+)?"
                         r"(?:(?:a|per|each|al|por|cada|a la|en el|en la)\s+)(day|dia|week|semana|month|mes)\b(?!\s+(?:siguiente|anterior|before|after|next))", ns):
        k = int(m.group(1)[1:])
        if k in used:
            continue
        unit = {"day": "day", "dia": "day", "week": "week", "semana": "week", "month": "month", "mes": "month"}[m.group(2)]
        add("frequency", f"{_fmt(nums[k].value)}x/{unit}", m, ctx="said")
    for m in re.finditer(r"(#\d+)\s+(?:times?\s+|veces\s+)?(daily|diari[oa]s?|weekly|semanal(?:es)?)\b", ns):
        k = int(m.group(1)[1:])
        if k not in used:
            add("frequency", f"{_fmt(nums[k].value)}x/{'day' if m.group(2).startswith(('dai', 'dia')) else 'week'}", m, ctx="said")
    have_xday = any(s.kind == "frequency" and s.value.endswith("x/day") for s in slots)
    if not have_xday and not any(s.kind == "frequency" and s.value.startswith("q") and s.value.endswith(("h", "min")) for s in slots):
        for m in re.finditer(r"(?<![a-z])(?:daily|every day|each day|(?<!#\d )(?<!#\d\d )(?<!times )(?:a|per) day|todos los dias|cada dia|diari[oa]s?|diariamente|"
                             r"(?<!#\d )(?<!veces )(?:al|por) dia|una vez al dia)\b", ns):
            if re.match(r"\s+(?:siguiente|anterior|before|after|next)\b", ns[m.end():]):
                continue                                     # "al día siguiente" = the next day
            if not any(s.kind == "frequency" and s.value == "1x/day" for s in slots):
                add("frequency", "1x/day", m)
    for m in re.finditer(rf"(?<![a-z])(?:{PRN})\b", ns):
        add("frequency", "as_needed", m)

    # How long: a number with a time word, compared in hours (48 hours == 2 days). A clock time ("a las 10") isn't one.
    for m in re.finditer(rf"(#\d+)\s+({_TU})\b", ns):
        k = int(m.group(1)[1:])
        if k in used or nums[k].clock:
            continue
        hours = nums[k].value * HOURS[TIME_WORDS[m.group(2)]]
        add("duration", f"{hours:.6g}h", m, ctx=m.group(2))

    # Remaining numbers, with what they count.
    for k, nm in enumerate(nums):
        if k in used:
            continue
        nxt = toks[nm.i1][0] if nm.i1 < len(toks) else ""
        nxt2 = toks[nm.i1 + 1][0] if nm.i1 + 1 < len(toks) else ""
        ctx = UNIT_WORDS.get(nxt) or CTX_ONLY.get(nxt) or ("percent" if (nxt, nxt2) == ("por", "ciento") else "") or \
            ("clock" if nxt in ("am", "pm") or nm.clock else "")
        s0, e0 = toks[nm.i0][1], toks[nm.i1 - 1][2]
        slots.append(Slot("number", _fmt(nm.value), t[s0:e0], ctx, s0))

    # Units (compared on their own, so "por kilo" vs "por libra" is caught even without a number).
    for i, (w, s, e) in enumerate(toks):
        u = UNIT_WORDS.get(w)
        prv_ = toks[i - 1][0] if i else ""
        nxt_ = toks[i + 1][0] if i + 1 < len(toks) else ""
        if u == "drop" and lang == "en" and not (prv_[:1].isdigit() or prv_ in EN_SMALL or prv_ in ("eye", "ear", "nose", "nasal", "few", "some", "the", "these",
                                                                                                    "those", "of", "two", "a", "saline")
                                                  or nxt_ in ("of", "in", "into", "each", "per", "every")):
            continue                                         # "when your sugar drops" is a verb
        if u and not (w == "g" and i and toks[i - 1][0] in ("vitamin", "vitamina")) and not (w == "l" and not (i and toks[i - 1][0][0].isdigit())):
            slots.append(Slot("unit", u, w, pos=s))
    for canon, rx in DRUGS.items():
        for m in re.finditer(rf"(?<![a-z])(?:{rx})(?![a-z])", t):
            slots.append(Slot("drug", canon, m.group(0), pos=m.start()))
    for canon, rx in WARNINGS.items():
        m = re.search(rf"(?<![a-z])(?:{rx})", t)
        if m:
            slots.append(Slot("warning", canon, m.group(0), pos=m.start()))  # presence is enough
    for rx, canon in TIMING:
        for m in re.finditer(rf"(?<![a-z])(?:{rx})", ns):
            add("timing", canon, m)
    for m in re.finditer(rf"(?<![a-z])(?:{LIMIT_MAX})(?![a-z])", ns):
        add("limit", "max", m)
    for m in re.finditer(rf"(?<![a-z])(?:{LIMIT_MIN})(?![a-z])", ns):
        add("limit", "min", m)
    for m in re.finditer(rf"(?<![a-z])(?:{ALLERGY})", t):
        slots.append(Slot("allergy", "allergy", m.group(0), pos=m.start()))
    no_neg += [(x.pos, x.pos + len(x.span)) for x in slots if x.kind == "limit" and x.pos >= 0]  # "no more than" is a limit
    slots += _negations(toks, lang, no_neg)
    slots += _side_slots(t, lang)
    return slots


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_KNOWN_DRUG = re.compile(rf"(?<![a-z])(?:{'|'.join(DRUGS.values())})(?![a-z])")


def heard_as(canon: str, text: str) -> bool:
    """Is a one-letter misspelling of this drug in the text (speech recognition slips: 'telenol', 'acitromicina')?
    Only words that are not themselves a known drug count, so a real look-alike / sound-alike swap (prednisone vs
    prednisolone, insulin vs inulin, Celebrex vs Celexa) is still flagged."""
    forms = [f for f in DRUGS[canon].split("|") if f.isalpha() and len(f) >= 6]
    for w in re.findall(r"[a-z]{5,}", fold(text)):
        if _KNOWN_DRUG.fullmatch(w):
            continue
        if any(_lev(w, f) <= 1 and w[0] in (f[0], {"z": "s", "s": "z", "c": "z"}.get(f[0], f[0])) for f in forms):
            return True
    return False


COUNTED_KINDS = ("number", "unit", "side", "drug", "duration", "frequency", "negation", "warning", "limit")


def diff(source: str, rendition: str, src_lang: str | None = None, ren_lang: str | None = None) -> list[Finding]:
    """What critical content changed between a sentence and its interpretation (listed in clinical priority order)."""
    a = [Slot(x.kind, x.value, restore(source, x.span, x.pos), x.ctx, x.pos) for x in extract(source, src_lang)]
    b = [Slot(x.kind, x.value, restore(rendition, x.span, x.pos), x.ctx, x.pos) for x in extract(rendition, ren_lang)]
    out: list[Finding] = []
    q = lambda x: f"“{x}”"  # noqa: E731
    leftover: dict[str, tuple[list[Slot], list[Slot]]] = {}
    for kind in SEVERITY:
        if kind == "order":
            continue
        sa = [s for s in a if s.kind == kind]
        sb = [s for s in b if s.kind == kind]
        if kind == "negation":
            continue
        if kind in COUNTED_KINDS and kind not in ("warning",):
            ca, cb = Counter(s.value for s in sa), Counter(s.value for s in sb)
            miss, extra = list((ca - cb).elements()), list((cb - ca).elements())
        else:
            va, vb = {s.value for s in sa}, {s.value for s in sb}
            miss, extra = sorted(va - vb), sorted(vb - va)
        if kind == "warning":   # 'pain' alone is too common to flag on its own
            miss = [m for m in miss if m != "pain"]; extra = [e for e in extra if e != "pain"]
        if kind == "drug":      # a misheard drug name ('acitromicina') is not a missing drug
            miss = [m for m in miss if not heard_as(m, rendition)]
            extra = [e for e in extra if not heard_as(e, source)]
        if kind == "timing":    # only a contradiction is a substitution; an omitted 'with food' is a low note
            pass
        # spans: take slots in order of appearance, one per missing value
        pool_a, pool_b = list(sa), list(sb)

        def take(pool, v):
            for s in pool:
                if s.value == v:
                    pool.remove(s)
                    return s
            return Slot(kind, v, v)
        if kind == "number":
            ms, es = _match_numbers(sa, sb)
        else:
            ms = [take(pool_a, v) for v in miss]
            es = [take(pool_b, v) for v in extra]
        leftover[kind] = (ms, es)
    # "don't go under 6 hours between doses" said as a length of time on one side and as an interval on the other
    for side_d, side_f in ((0, 1), (1, 0)):
        ds, fs = leftover["duration"][side_d], leftover["frequency"][side_f]
        for d in list(ds):
            f = next((f for f in fs if re.fullmatch(r"q[\d.]+h", f.value) and abs(float(f.value[1:-1]) - float(d.value.rstrip("h"))) < 1e-6), None)
            if f:
                ds.remove(d); fs.remove(f)
    # A duration in one sentence and the same count as a bare number in the other ("dos años y medio" / "two and a half"):
    # the unit was left implicit, not changed.
    dm, de = leftover["duration"]
    nm, ne = leftover["number"]
    for d in list(dm):
        hit = next((x for x in ne if not x.ctx and abs(float(x.value) - _count(d)) < 1e-6), None)
        if hit:
            dm.remove(d); ne.remove(hit)
    for d in list(de):
        hit = next((x for x in nm if not x.ctx and abs(float(x.value) - _count(d)) < 1e-6), None)
        if hit:
            de.remove(d); nm.remove(hit)
    leftover["negation"] = ([], [])
    neg_findings = _negation_diff(source, rendition, [x for x in a if x.kind == "negation"], [x for x in b if x.kind == "negation"])
    for kind in PRIORITY:
        if kind == "order" or kind not in leftover:
            continue
        ms, es = leftover[kind]
        sev = SEVERITY[kind]
        if kind == "timing":
            for m in list(ms):
                e = next((e for e in es if frozenset((m.value, e.value)) in TIMING_CONFLICT), None)
                if e:
                    ms.remove(m); es.remove(e)
                    out.append(Finding("substitution", kind, "high", m.span, e.span, f"Changed {KIND_LABEL[kind]}: {q(m.span)} became {q(e.span)}"))
            for m in ms:
                out.append(Finding("omission", kind, "low", m.span, "", f"Not in the interpretation: {KIND_LABEL[kind]} {q(m.span)}"))
            for e in es:
                out.append(Finding("addition", kind, "medium", "", e.span, f"Added in the interpretation: {KIND_LABEL[kind]} {q(e.span)}"))
            continue
        if kind == "negation":
            out += neg_findings
            continue
        if kind == "number":    # pair a changed number with its replacement: same unit first, then in order
            for m in list(ms):
                e = next((e for e in es if m.ctx and e.ctx == m.ctx), None)
                if e:
                    ms.remove(m); es.remove(e)
                    out.append(Finding("substitution", kind, sev, m.span, e.span, f"Changed {KIND_LABEL[kind]}: {q(m.span)} became {q(e.span)}"))
        if kind == "number":    # a lone 'a' / 'una' / 'una vez' is rarely the number that changed
            weak_m = [m for m in ms if _weak_one(m)]; weak_e = [e for e in es if _weak_one(e)]
            ms = [m for m in ms if m not in weak_m]; es = [e for e in es if e not in weak_e]
        if kind in ("number", "unit", "side", "drug", "duration", "frequency", "limit", "warning"):
            while ms and es:
                m, e = ms.pop(0), es.pop(0)
                s_sub = "medium" if kind == "warning" and (m.value in WARNING_MEDIUM or e.value in WARNING_MEDIUM) else sev
                out.append(Finding("substitution", kind, s_sub, m.span, e.span, f"Changed {KIND_LABEL[kind]}: {q(m.span)} became {q(e.span)}"))
        if kind == "number":
            out += [Finding("omission", kind, "low", m.span, "", f"Not in the interpretation: {KIND_LABEL[kind]} {q(m.span)}") for m in weak_m]
            out += [Finding("addition", kind, "low", "", e.span, f"Added in the interpretation: {KIND_LABEL[kind]} {q(e.span)}") for e in weak_e]
        for m in ms:
            s_om = sev
            if kind == "frequency" and (m.value == "as_needed" or (m.value == "1x/day" and m.ctx != "said")):
                s_om = "medium"     # a dropped "daily" / "as needed" is a note; a dropped "once a day" is a card
            if kind == "warning" and m.value in WARNING_MEDIUM:
                s_om = "medium"
            if kind == "drug" and any(x.value == m.value for x in b if x.kind == "drug"):
                s_om = "medium"     # one of several mentions of a medicine that is still named
            if kind == "unit" and m.value in ("celsius", "fahrenheit"):
                s_om = "medium"     # "38 grados" for "38 degrees Celsius": the scale is implied
            out.append(Finding("omission", kind, s_om, m.span, "", f"Not in the interpretation: {KIND_LABEL[kind]} {q(m.span)}"))
        for e in es:
            # added content is usually lower risk than dropped content, except an added drug, number or side
            s_add = "high" if kind in ("drug", "number", "side") or (kind == "frequency" and e.value not in ("1x/day", "as_needed")) else "medium"
            if kind == "drug" and any(x.value == e.value for x in a if x.kind == "drug"):
                s_add = "medium"    # the same medicine named again ("allergic to it" -> "alérgico a la codeína")
            out.append(Finding("addition", kind, s_add, "", e.span, f"Added in the interpretation: {KIND_LABEL[kind]} {q(e.span)}"))
    # Same numbers in a different order ("2.5 mg Mon/Fri, 5 mg other days" -> "5 mg Mon/Fri, 2.5 mg other days"): code
    # can't tell a harmless re-ordering from a swap, so it's a medium note for the checker to weigh.
    na = [s.value for s in a if s.kind == "number"]
    nb = [s.value for s in b if s.kind == "number"]
    if len(na) >= 2 and Counter(na) == Counter(nb) and na != nb:
        out.append(Finding("substitution", "order", SEVERITY["order"], " / ".join(na), " / ".join(nb),
                           "Same numbers in a different order: " + ", ".join(na) + " became " + ", ".join(nb)))
    return out


ABBREV = {"dr", "dra", "sr", "sra", "srta", "mr", "mrs", "ms", "st", "vs", "etc", "approx", "aprox", "no", "num", "ud", "uds", "lic"}


def _sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) character ranges of the sentences in a text (split at . ? !; not at abbreviations, decimals or '...')."""
    t = fold(text)
    toks = _tok(t)
    cuts, k = [], 0
    while k < len(toks):
        w, s_, e = toks[k]
        if w in (".", "?", "!"):
            j = k
            while j + 1 < len(toks) and toks[j + 1][0] in (".", "?", "!"):
                j += 1
            prev = toks[k - 1][0] if k else ""
            ellipsis = w == "." and j > k and all(toks[x][0] == "." for x in range(k, j + 1))
            nxt = toks[k + 1][0] if k + 1 < len(toks) else ""
            abbrev = w == "." and j == k and ((len(prev) == 1 and prev.isalpha() and k + 1 < len(toks))
                                              or (prev in ABBREV and prev != "no" and nxt[:1].isalpha())
                                              or (prev == "no" and nxt[:1].isdigit()))
            if not ellipsis and not abbrev:
                cuts.append(toks[j][2])
            k = j + 1
            continue
        k += 1
    bounds, last = [], 0
    for c in cuts + [len(t)]:
        if re.search(r"[a-z0-9]", t[last:c]):
            bounds.append((last, c))
        last = c
    return bounds or [(0, len(t))]


def _neg_profile(slots_: list[Slot]) -> dict:
    return {c: [x for x in slots_ if x.value == c] for c in ("verbal", "nominal", "answer", "weak", "hardly", "stop", "dontstop")}


def _satisfied(A: dict, B: dict) -> tuple[bool, Slot | None]:
    """Is every negation in sentence A carried by sentence B? -> (ok, the slot in A that isn't)."""
    anyB = any(B[c] for c in B)
    strongB = bool(B["verbal"] or B["nominal"] or B["weak"] or B["hardly"] or B["stop"] or B["dontstop"])
    # more negated verbs than B has negations of any kind: one was dropped ("if you can't answer or she can't wake you"
    # -> "si puede contestar o ella no logra despertarlo"). Negative pronouns and 'sin' count on B's side, so
    # "no food, no water" vs "ni comida ni agua" is fine.
    if len(A["verbal"]) > len(B["verbal"]) + len(B["nominal"]) + len(B["weak"]) + len(B["hardly"]) + len(B["stop"]) + len(B["dontstop"]) \
            + (0 if A["answer"] else len(B["answer"])):
        return False, A["verbal"][-1]
    if A["verbal"]:
        # a negated verb needs a negation in B; a bare "No," in B only counts if A had no answer of its own
        # ("No, no tiene fiebre" -> "No, he has a fever" drops the second 'no')
        if not (strongB or (B["answer"] and not A["answer"])):
            return False, A["verbal"][0]
    for c in ("nominal", "stop", "hardly", "answer"):
        if A[c] and not anyB:
            return False, A[c][0]
    return True, None


def _negation_diff(source: str, rendition: str, na: list[Slot], nb: list[Slot]) -> list[Finding]:
    """Negation compared sentence by sentence when both texts have the same number of sentences, else over the whole turn."""
    sa, sb = _sentences(source), _sentences(rendition)
    if len(sa) == len(sb):
        groups = [([x for x in na if a0 <= x.pos < a1], [x for x in nb if b0 <= x.pos < b1]) for (a0, a1), (b0, b1) in zip(sa, sb)]
    else:
        groups = [(na, nb)]
    out: list[Finding] = []
    q = lambda x: f"“{x}”"  # noqa: E731
    lab = KIND_LABEL["negation"]
    for ga, gb in groups:
        A, B = _neg_profile(ga), _neg_profile(gb)
        # "don't stop" vs "stop" (or the reverse) flips the instruction
        if A["dontstop"] and B["stop"] and not B["dontstop"]:
            out.append(Finding("substitution", "negation", "high", A["dontstop"][0].span, B["stop"][0].span,
                               f"Changed {lab}: {q(A['dontstop'][0].span)} became {q(B['stop'][0].span)}"))
            continue
        if A["stop"] and B["dontstop"] and not A["dontstop"]:
            out.append(Finding("substitution", "negation", "high", A["stop"][0].span, B["dontstop"][0].span,
                               f"Changed {lab}: {q(A['stop'][0].span)} became {q(B['dontstop'][0].span)}"))
            continue
        ok, miss = _satisfied(A, B)
        if not ok:   # a dropped bare "No," answer is often carried by the rest of the sentence: a note, not a card
            out.append(Finding("omission", "negation", "medium" if miss.value == "answer" else "high", miss.span, "",
                               f"Not in the interpretation: {lab} {q(miss.span)}"))
        ok, extra = _satisfied(B, A)
        if not ok:
            out.append(Finding("addition", "negation", "medium" if extra.value == "answer" else "high", "", extra.span,
                               f"Added in the interpretation: {lab} {q(extra.span)}"))
    return out


def _weak_one(x: Slot) -> bool:
    """A 1 that is just an article or 'once' ('a pill', 'una pastilla', 'una vez'): its absence on the other side is a
    matter of grammar, not a dropped number."""
    return x.value == "1" and (fold(x.span).strip() in ("a", "an", "un", "una", "uno") or x.ctx == "times")


def _match_numbers(sa: list[Slot], sb: list[Slot]) -> tuple[list[Slot], list[Slot]]:
    """Numbers left over on each side after matching equal values: same value and same unit first, then same value."""
    a, b = list(sa), list(sb)
    for key in (lambda x: (x.value, x.ctx), lambda x: x.value):
        for x in list(a):
            y = next((y for y in b if key(y) == key(x)), None)
            if y is not None:
                a.remove(x); b.remove(y)
    # a body temperature converted between Fahrenheit and Celsius (100.4 = 38) is the same number
    for x in list(a):
        y = next((y for y in b if _same_temp(float(x.value), float(y.value))), None)
        if y is not None:
            a.remove(x); b.remove(y)
    return a, b


def _same_temp(u: float, v: float) -> bool:
    f, c = (u, v) if u > v else (v, u)
    return 95 <= f <= 110 and 35 <= c <= 43.5 and abs(f - (c * 9 / 5 + 32)) <= 0.5


def _count(d: Slot) -> float:
    """The number said in a duration slot (value is in hours; ctx is the time word)."""
    unit = TIME_WORDS.get(d.ctx, "h")
    return float(d.value.rstrip("h")) / HOURS[unit]


def worst(findings: list[Finding]) -> str:
    order = {"high": 3, "medium": 2, "low": 1}
    return max((f.severity for f in findings), key=lambda s: order[s], default="none")
