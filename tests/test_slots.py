from app.engine.slots import diff, extract


def kinds(s, r, a, b, sev="high"):
    return sorted({(f.type, f.kind) for f in diff(s, r, a, b) if f.severity == sev})


def test_faithful_pairs_are_clean():
    pairs = [
        ("Ya le di dos dosis de Tylenol hoy y es alérgico a la penicilina.", "She already gave him two doses of Tylenol today, and he is allergic to penicillin.", "es", "en"),
        ("Give him 5 milliliters every six hours, no more than four doses a day.", "Dele cinco mililitros cada seis horas, no más de cuatro dosis al día.", "en", "es"),
        ("Take one tablet twice a day with food for ten days.", "Tome una pastilla dos veces al día con comida por diez días.", "en", "es"),
        ("Le doy media cucharadita en la noche.", "I give him half a teaspoon at night.", "es", "en"),
        ("Tiene treinta y ocho y medio de fiebre desde hace tres días.", "He's had a fever of 38.5 for three days.", "es", "en"),
        ("Doscientos cincuenta miligramos.", "250 mg.", "es", "en"),
        ("Give him 5 milliliters of azithromycin once a day for 5 days.", "Dele 5 mililitros de azitromicina una vez al día por cinco días.", "en", "es"),
        ("And the new one, how many times?", "¿Y el nuevo, cuántas veces?", "en", "es"),
        ("Come back in one week.", "Regrese en una semana.", "en", "es"),
        ("Drink half a cup of juice.", "Tómese media taza de jugo.", "en", "es"),
    ]
    for s, r, a, b in pairs:
        assert kinds(s, r, a, b) == [], (s, r, diff(s, r, a, b))


def test_critical_changes_are_found():
    assert ("omission", "allergy") in kinds("Ya le di dos dosis de Tylenol hoy y es alérgico a la penicilina.", "She gave him Tylenol today.", "es", "en")
    assert ("substitution", "unit") in kinds("Give 5 milliliters.", "Dele 5 miligramos.", "en", "es")
    assert ("substitution", "frequency") in kinds("every six hours", "cada ocho horas", "en", "es")
    assert ("substitution", "side") in kinds("Me duele la rodilla derecha.", "My left knee hurts.", "es", "en")
    assert ("omission", "negation") in kinds("Do not take it with alcohol.", "Tómelo con alcohol.", "en", "es")
    assert ("substitution", "duration") in kinds("for 2 weeks", "por 2 días", "en", "es")
    assert ("omission", "limit") in kinds("every 6 hours, no more than 4 doses a day", "cada 6 horas", "en", "es")


def test_number_words_follow_language():
    assert [s.value for s in extract("once a day", "en") if s.kind == "frequency"] == ["1x/day"]
    assert [s.value for s in extract("Once you finish, come back.", "en") if s.kind == "number"] == []
    assert [s.value for s in extract("once pastillas", "es") if s.kind == "number"] == ["11"]


def test_reordered_numbers_noted():
    f = diff("Take 2.5 milligrams on Mondays and 5 milligrams on other days.", "Tome 5 miligramos los lunes y 2.5 miligramos los demás días.", "en", "es")
    assert any(x.kind == "order" for x in f)


def test_speech_recognition_slips_are_tolerated():
    assert kinds("Le puedo seguir dando el Tylenol también.", "Le puedo seguir dando el telenol también.", "es", "es") == []
    assert kinds("Take 2.5 milligrams.", "Take two point five milligrams.", "en", "en") == []
    assert kinds("llame al 911", "llame al novecientos once", "es", "es") == []


def test_real_drug_swap_still_flagged():
    assert ("substitution", "drug") in kinds("He is allergic to penicillin.", "Es alérgico a la amoxicilina.", "en", "es")


# --- review fixes: faithful re-wordings that must NOT raise a card (code only) --------------------------------------
FAITHFUL = [
    ("Take no more than four doses a day.", "No tome más de cuatro dosis al día.", "en", "es"),
    ("Don't take more than 4 a day.", "No tome más de 4 al día.", "en", "es"),
    ("Empezó hace una semana.", "It started a week ago.", "es", "en"),
    ("Él tiene un año.", "He is one year old.", "es", "en"),
    ("Tiene un año y medio.", "He's a year and a half old.", "es", "en"),
    ("Tiene dos años y medio.", "He's two and a half.", "es", "en"),
    ("Take one every morning.", "Tome una cada mañana.", "en", "es"),
    ("One more question.", "Una pregunta más.", "en", "es"),
    ("Okay, one second.", "Bueno, un segundo.", "en", "es"),
    ("Once you finish the antibiotic, come back.", "Cuando termine el antibiótico, regrese.", "en", "es"),
    ("Anoche vomitó dos veces.", "He threw up twice last night.", "es", "en"),
    ("If he has trouble breathing, call 911.", "Si le cuesta respirar, llame al 911.", "en", "es"),
    ("His temperature was 101.5.", "Su temperatura fue de ciento uno punto cinco.", "en", "es"),
    ("Return the forms at the front desk.", "Entregue los formularios en recepción.", "en", "es"),
    ("Do you take blood thinners?", "¿Toma medicinas para adelgazar la sangre?", "en", "es"),
    ("We need a blood test.", "Necesitamos un análisis de sangre.", "en", "es"),
    ("Le doy una cucharadita y media.", "I give him one and a half teaspoons.", "es", "en"),
    ("Le doy cucharadita y media.", "I give him a teaspoon and a half.", "es", "en"),
    ("Le di la mitad de una pastilla.", "I gave him half a pill.", "es", "en"),
    ("Give him a quarter of a tablet.", "Dele un cuarto de tableta.", "en", "es"),
    ("Give 0.5 mg.", "Dele medio miligramo.", "en", "es"),
    ("Take 1000 milligrams.", "Tome 1.000 miligramos.", "en", "es"),
    ("Take 1,500 milligrams.", "Tome mil quinientos miligramos.", "en", "es"),
    ("Tome dos mil quinientos miligramos.", "Take 2,500 milligrams.", "es", "en"),
    ("Tomo metformina, quinientos, dos veces al día.", "I take metformin, 500, twice a day.", "es", "en"),
    ("Give twenty two units.", "Póngase veintidós unidades.", "en", "es"),
    ("Tiene noventa y nueve de fiebre.", "He has a temperature of ninety-nine.", "es", "en"),
    ("Take two hundred and fifty milligrams.", "Tome doscientos cincuenta miligramos.", "en", "es"),
    ("Tiene 101 de fiebre.", "He has a fever of a hundred and one.", "es", "en"),
    ("Give 5mg.", "Dele 5 mg.", "en", "es"),
    ("Take it 3x a day.", "Tómelo 3 veces al día.", "en", "es"),
    ("Give ½ tablet.", "Dele media tableta.", "en", "es"),
    ("Give 2 1/2 teaspoons.", "Dele dos cucharaditas y media.", "en", "es"),
    ("Su presión es 140/90.", "Her blood pressure is 140 over 90.", "es", "en"),
    ("The last dose was at 10:30.", "La última dosis fue a las diez y media.", "en", "es"),
    ("Take it every 4-6 hours.", "Tómelo cada 4 a 6 horas.", "en", "es"),
    ("Keep it dry for 48 hours.", "Manténgala seca por dos días.", "en", "es"),
    ("Take it every other day.", "Tómelo un día sí y un día no.", "en", "es"),
    ("Use two puffs as needed.", "Use dos inhalaciones si las necesita.", "en", "es"),
    ("Me duele del lado derecho.", "It hurts on my right side.", "es", "en"),
    ("Turn right at the hall.", "Doble a la derecha en el pasillo.", "en", "es"),
    ("No, nada de eso.", "No, nothing like that.", "es", "en"),
    ("Keep taking it, don't stop.", "Siga tomándolo, no lo deje.", "en", "es"),
    ("Le di diez mililitros de jarabe a las seis de la mañana.", "I gave him ten milliliters of syrup at six in the morning.", "es", "en"),
    ("Ya le di dos dosis de Tylenol hoy.", "I gave him Tylenol twice today.", "es", "en"),
]


def test_faithful_rewordings_raise_no_card():
    from app.engine.check import check_turn
    bad = [(s, r) for s, r, a, b in FAITHFUL if check_turn(s, r, a, b, use_checker=False, explain=False)["decision"]["level"] == "card"]
    assert bad == []


# --- review fixes: dangerous changes that MUST raise a card (code only) ---------------------------------------------
ERRORS = [
    ("Take levothyroxine 0.125 milligrams.", "Tome levotiroxina 0.175 miligramos.", "en", "es"),
    ("Take digoxin 0.125 mg daily.", "Tome digoxina 0.0125 mg diario.", "en", "es"),
    ("Give 5mg.", "Dele 50mg.", "en", "es"),
    ("Toma 10.000 unidades.", "He takes 10 units.", "es", "en"),
    ("Toma 1.000 unidades.", "He takes 100 units.", "es", "en"),
    ("Pesa veintidós kilos.", "He weighs 22 pounds.", "es", "en"),
    ("Give 15 mg per kilo.", "Dele 15 mg por libra.", "en", "es"),
    ("No, no tiene fiebre.", "No, he has a fever.", "es", "en"),
    ("Don't stop taking it, even if he feels better.", "Deje de dárselo cuando se sienta mejor.", "en", "es"),
    ("Stop taking it if you feel better.", "No deje de tomarlo si se siente mejor.", "en", "es"),
    ("Give 2 puffs every 4 hours.", "Dele 4 inhalaciones cada 2 horas.", "en", "es"),
    ("Take it every day.", "Tómelo cada tercer día.", "en", "es"),
    ("Take it as needed.", "Tómelo todos los días.", "en", "es"),
    ("Take it twice daily.", "Tómelo diario.", "en", "es"),
    ("Take it with food.", "Tómelo en ayunas.", "en", "es"),
    ("Take it in the morning.", "Tómelo en la noche.", "en", "es"),
    ("Inject 12 units of insulin before dinner.", "Inyecte 12 unidades de insulina después de la cena.", "en", "es"),
    ("El izquierdo.", "The right one.", "es", "en"),
    ("Which ankle? The left?", "¿Cuál tobillo? ¿El derecho?", "en", "es"),
    ("Se desmayó ayer.", "He got dizzy yesterday.", "es", "en"),
    ("Take prednisone 20 mg.", "Tome prednisolona 20 mg.", "en", "es"),
    ("He takes insulin.", "Toma inulina.", "en", "es"),
    ("Take Celebrex.", "Tome Celexa.", "en", "es"),
    ("Take tramadol.", "Tome trazodona.", "en", "es"),
    ("Take hydroxyzine.", "Tome hidralazina.", "en", "es"),
    ("At least 3 times a day.", "Hasta 3 veces al día.", "en", "es"),
    ("Wait at least 4 hours between doses.", "Espere 4 horas entre dosis.", "en", "es"),
    ("He has had a fever for 3 days.", "Tiene fiebre desde hace 3 semanas.", "en", "es"),
    ("Le empezó la tos hace tres días.", "The cough started three weeks ago.", "es", "en"),
]


def test_dangerous_changes_raise_a_card():
    from app.engine.check import check_turn
    missed = [(s, r) for s, r, a, b in ERRORS if check_turn(s, r, a, b, use_checker=False, explain=False)["decision"]["level"] != "card"]
    assert missed == []


def test_numbers_parse():
    from app.engine.slots import _num_tokens, fold
    cases = {("Take 0.125 mg", "en"): [0.125], ("Toma 10.000 unidades", "es"): [10000], ("Give 2 1/2 teaspoons", "en"): [2.5],
             ("dos mil quinientos", "es"): [2500], ("a hundred and one", "en"): [101], ("ninety-nine", "en"): [99],
             ("ciento uno punto cinco", "es"): [101.5], ("Su presión es 140/90", "es"): [140, 90], ("quinientos, dos", "es"): [500, 2]}
    for (t, lang), want in cases.items():
        assert [round(v, 4) for v, *_ in _num_tokens(fold(t), lang)] == want, t


def test_card_points_at_the_dropped_words():
    # the dropped part is "no more than 4 doses in 24 hours"; "every 6 hours" was interpreted and must not be blamed
    from app.engine.slots import diff
    f = diff("Give ibuprofen, 100 milligrams every 6 hours, but no more than 4 doses in 24 hours.",
             "Dele ibuprofeno, 100 miligramos cada 6 horas.", "en", "es")
    spans = {x.source for x in f if x.type == "omission"}
    assert "24 hours" in spans and not any("6 hours" in s_ for s_ in spans)


def test_spans_point_at_whole_words():
    from app.engine.slots import extract, restore
    t = "Le di la medicina anoche, no hoy."
    assert [s.pos for s in extract(t, "es") if s.kind == "negation"] == [t.index(", no") + 2]   # not the "no" inside "anoche"
    assert restore("Tome otra pastilla, not another.", "not") == "not"
    assert restore("cafe\u0301 no", "no") == "no"   # decomposed accents still map back


def test_findings_in_clinical_priority_order():
    from app.engine.slots import PRIORITY, diff
    f = diff("Sí, ya le di dos dosis de Tylenol hoy, la última a las diez. Y es alérgico a la penicilina.", "Yes, I gave him Tylenol today.", "es", "en")
    kinds_ = [x.kind for x in f]
    assert kinds_[0] == "allergy" and kinds_ == sorted(kinds_, key=PRIORITY.index)
