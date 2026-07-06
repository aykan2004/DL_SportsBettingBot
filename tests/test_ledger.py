from quantbet.ledger import append_bet, load_ledger, save_ledger

SAMPLE_BET = {
    "h_name": "Arsenal",
    "h_id": 42,
    "a_name": "Chelsea",
    "a_id": 49,
    "l_id": 39,
    "date": "2026-07-05T15:00",
    "match": "Arsenal vs Chelsea",
    "bet": "Home",
    "prob": 0.55,
    "odds": 2.10,
    "ev": 0.155,
}


class TestLedger:
    def test_missing_file_is_empty_ledger(self, tmp_path):
        assert load_ledger(tmp_path / "nope.json") == []

    def test_append_and_reload_roundtrip(self, tmp_path):
        path = tmp_path / "ledger.json"
        record = append_bet(SAMPLE_BET, stake=12.5, path=path)
        loaded = load_ledger(path)
        assert len(loaded) == 1
        assert loaded[0] == record
        assert loaded[0]["status"] == "pending"
        assert loaded[0]["stake"] == 12.5
        assert loaded[0]["model_prob"] == 0.55

    def test_feature_snapshot_is_persisted(self, tmp_path):
        path = tmp_path / "ledger.json"
        snapshot = {"cont": [0.1] * 6, "seq": [[0, 0, 0, 0]] * 5}
        append_bet(SAMPLE_BET, stake=5.0, feature_snapshot=snapshot, path=path)
        assert load_ledger(path)[0]["features"] == snapshot

    def test_appending_preserves_existing_records(self, tmp_path):
        path = tmp_path / "ledger.json"
        append_bet(SAMPLE_BET, stake=1.0, path=path)
        append_bet(SAMPLE_BET, stake=2.0, path=path)
        assert [b["stake"] for b in load_ledger(path)] == [1.0, 2.0]

    def test_save_ledger_roundtrip(self, tmp_path):
        path = tmp_path / "ledger.json"
        save_ledger([{"a": 1}], path=path)
        assert load_ledger(path) == [{"a": 1}]
