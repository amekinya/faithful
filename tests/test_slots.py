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
    assert ("substitution", "number") in kinds("every six hours", "cada ocho horas", "en", "es")
    assert ("substitution", "side") in kinds("Me duele la rodilla derecha.", "My left knee hurts.", "es", "en")
    assert ("omission", "negation") in kinds("Do not take it with alcohol.", "Tómelo con alcohol.", "en", "es")
    assert ("substitution", "duration") in kinds("for 2 weeks", "por 2 días", "en", "es")
    assert ("omission", "limit") in kinds("every 6 hours, no more than 4 doses a day", "cada 6 horas", "en", "es")


def test_number_words_follow_language():
    assert [s.value for s in extract("once a day", "en") if s.kind == "number"] == ["1"]
    assert [s.value for s in extract("once pastillas", "es") if s.kind == "number"] == ["11"]


def test_reordered_numbers_noted():
    f = diff("Take 2.5 milligrams on Mondays and 5 milligrams on other days.", "Tome 5 miligramos los lunes y 2.5 miligramos los demás días.", "en", "es")
    assert any(x.kind == "order" for x in f)
