from tala.common.io import load_json, save_json


def test_json_round_trip(tmp_path):
    path = tmp_path / "nested" / "sample.json"
    payload = {"stage": "scaffold", "count": 3}
    save_json(payload, path)
    assert load_json(path) == payload
