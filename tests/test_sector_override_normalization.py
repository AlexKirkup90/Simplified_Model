import backend


def test_load_sector_overrides_normalizes_tickers():
    overrides = backend._load_sector_overrides()
    assert overrides.get("BRK-B") == "Financial Services"
    assert "BRK.B" not in overrides
    assert overrides.get("GOOG") == "Communication Services"
