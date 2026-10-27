from fastapi.testclient import TestClient

from app.engine import gate
from app.main import app

HIGH = [{"severity": "high", "type": "omission", "kind": "allergy", "source": "alérgico", "rendition": "", "message": "x"}]
TH = {"confirm": 0.35, "alone": 0.85}


def test_gate_rules():
    assert gate.decide(HIGH, None, TH)["level"] == "card"            # code alone when the checker is off
    assert gate.decide([], None, TH)["level"] == "ok"
    assert gate.decide(HIGH, 0.2, TH)["level"] == "card"             # p_error 0.8 >= confirm
    assert gate.decide(HIGH, 0.9, TH)["level"] == "card"             # an allergy change can't be quieted by the checker
    NUM = [HIGH[0] | {"kind": "number"}]
    assert gate.decide(NUM, 0.9, TH)["level"] == "note"              # a number the checker thinks is a fine re-wording
    assert gate.decide([], 0.1, TH)["title"] == "Possible change"    # checker very sure, code found nothing
    assert gate.decide([], 0.5, TH)["level"] == "ok"


def test_api_check_and_visits():
    c = TestClient(app)
    assert c.get("/api/health").status_code == 200
    vs = c.get("/api/visits").json()
    assert len(vs) == 6
    v = c.get(f"/api/visit/{vs[0]['id']}").json()
    assert v["turns"] and "check" in v["turns"][0]
    r = c.post("/api/check", json={"source": "Do not give him ibuprofen.", "rendition": "Dele ibuprofeno.", "src_lang": "en"}).json()
    assert r["decision"]["level"] == "card"
    r = c.post("/api/check", json={"source": "Give him 5 mL every 6 hours.", "rendition": "Dele 5 mL cada 6 horas.", "src_lang": "en"}).json()
    assert r["decision"]["level"] == "ok"
    assert c.get("/").status_code == 200
