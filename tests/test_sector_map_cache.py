import streamlit as st
import sys, pathlib, types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_get_sector_map_uses_cache(monkeypatch):
    calls = {"count": 0}

    class FakeTicker:
        def __init__(self, symbol):
            self.fast_info = {}

        def get_info(self):
            calls["count"] += 1
            return {"sector": "Tech"}

    # Replace yfinance.Ticker with our fake
    monkeypatch.setattr(backend.yf, "Ticker", FakeTicker)

    # Clear caches
    backend._SECTOR_CACHE.clear()
    backend.get_sector_map.clear()

    # First call fetches sector for AAA
    backend.get_sector_map(["AAA"])
    assert calls["count"] == 1

    # Clearing streamlit cache forces function to run but AAA is already cached
    backend.get_sector_map.clear()
    backend.get_sector_map(["AAA", "BBB"])

    # Should only fetch sector for BBB on second run
    assert calls["count"] == 2
