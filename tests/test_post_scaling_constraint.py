import pandas as pd
import streamlit as st
import sys, pathlib, types
from datetime import date
import pytest

# Provide empty secrets to avoid backend errors
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def _mock_env(
    monkeypatch,
    base_weights: dict[str, float] | None = None,
    exposure: float = 2.0,
    sector_map: dict[str, str] | None = None,
):
    base_weights = base_weights or {"AAA": 0.25, "BBB": 0.25}
    sector_map = sector_map or {"AAA": "Tech", "BBB": "Tech"}

    def fake_get_universe(choice):
        tickers = list(base_weights.keys())
        return tickers, sector_map, "Label"

    def fake_fetch_price_volume(tickers, start, end):
        idx = pd.date_range("2024-06-01", periods=10, freq="B")
        close = pd.DataFrame(100.0, index=idx, columns=tickers)
        vol = pd.DataFrame(1000.0, index=idx, columns=tickers)
        return close, vol

    def fake_fetch_fundamental_metrics(tickers):
        return pd.DataFrame(index=tickers)

    def fake_fundamental_quality_filter(df, min_profitability, max_leverage):
        return df.index.tolist()

    def fake_build_weights(close, params, sectors_map, use_enhanced_features=True):
        # already respecting caps before scaling
        return pd.Series(base_weights)

    monkeypatch.setattr(backend, "get_universe", fake_get_universe)
    monkeypatch.setattr(backend, "fetch_price_volume", fake_fetch_price_volume)
    monkeypatch.setattr(backend, "fetch_fundamental_metrics", fake_fetch_fundamental_metrics)
    monkeypatch.setattr(backend, "fundamental_quality_filter", fake_fundamental_quality_filter)
    monkeypatch.setattr(backend, "_build_isa_weights_fixed", fake_build_weights)
    monkeypatch.setattr(backend, "compute_regime_metrics", lambda hist: {})
    monkeypatch.setattr(backend, "get_regime_adjusted_exposure", lambda metrics: exposure)
    monkeypatch.setattr(backend, "is_rebalance_today", lambda today, idx: True)


def test_scaling_enforces_caps(monkeypatch):
    _mock_env(monkeypatch)

    calls = {"count": 0}
    orig_enforce = backend.enforce_caps_iteratively

    def tracking_enforce(*args, **kwargs):
        calls["count"] += 1
        return orig_enforce(*args, **kwargs)

    monkeypatch.setattr(backend, "enforce_caps_iteratively", tracking_enforce)

    preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
    disp, raw, decision = backend.generate_live_portfolio_isa_monthly(
        preset, None, as_of=date(2024, 6, 3)
    )

    assert calls["count"] == 1
    assert raw["Weight"].max() <= preset["mom_cap"] + 1e-9
    assert raw["Weight"].sum() <= preset["sector_cap"] + 1e-9


def test_borderline_scaling_triggers_reenforcement(monkeypatch):
    _mock_env(
        monkeypatch,
        base_weights={"AAA": 0.25, "BBB": 0.25},
        exposure=1.0005,
        sector_map={"AAA": "Tech", "BBB": "Health"},
    )

    calls = {"count": 0}
    orig_enforce = backend.enforce_caps_iteratively

    def tracking_enforce(*args, **kwargs):
        calls["count"] += 1
        return orig_enforce(*args, **kwargs)

    monkeypatch.setattr(backend, "enforce_caps_iteratively", tracking_enforce)

    preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
    disp, raw, decision = backend.generate_live_portfolio_isa_monthly(
        preset, None, as_of=date(2024, 6, 3)
    )

    assert calls["count"] == 1
    assert raw["Weight"].max() <= preset["mom_cap"] + 1e-9


def test_scaling_raises_on_unfixed(monkeypatch):
    _mock_env(monkeypatch)

    monkeypatch.setattr(
        backend,
        "enforce_caps_iteratively",
        lambda weights, *a, **k: weights,
    )

    preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
    with pytest.raises(ValueError):
        backend.generate_live_portfolio_isa_monthly(
            preset, None, as_of=date(2024, 6, 3)
        )
