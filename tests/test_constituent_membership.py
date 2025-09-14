import types
import pandas as pd
import pytest

import strategy_core as sc


def test_momentum_respects_constituents():
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    prices = pd.DataFrame({
        "A": [100, 110, 121, 133.1],
        "B": [100, 200, 400, 800],
    }, index=idx)

    members = {
        pd.Timestamp("2020-03-31"): ["A"],
        pd.Timestamp("2020-04-30"): ["A", "B"],
    }

    def get_members(dt):
        return members.get(dt, ["A", "B"])

    rets, _ = sc.run_backtest_momentum(
        prices,
        lookback_m=1,
        top_n=1,
        cap=1.0,
        get_constituents=get_members,
    )

    assert rets.loc[pd.Timestamp("2020-03-31")] == pytest.approx(0.1)


def test_constituent_cache(monkeypatch):
    sc._NDX_CONSTITUENT_CACHE.clear()
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append(params.get("date"))
        text = "ticker\nAAPL\nMSFT\n"
        return types.SimpleNamespace(text=text, status_code=200, raise_for_status=lambda: None)

    monkeypatch.setattr(sc.requests, "get", fake_get)
    first = sc.get_nasdaq_100_plus_tickers(as_of="2024-01-01")
    second = sc.get_nasdaq_100_plus_tickers(as_of="2024-01-01")
    assert first == second == ["AAPL", "MSFT"]
    assert len(calls) == 1
