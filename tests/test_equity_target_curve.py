import pathlib
import sys
import types

import streamlit as st

st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend  # noqa: E402


def test_equity_target_extends_above_one():
    metrics = {"regime_score": 95.0}
    target = backend._determine_equity_target("Risk-On", metrics, base_target=0.9)
    assert 1.05 <= target <= 1.10


def test_equity_target_clamps_in_risk_off():
    metrics = {"regime_score": 15.0}
    target = backend._determine_equity_target("Extreme Risk-Off", metrics, base_target=0.6)
    assert target <= 0.35


def test_equity_target_neutral_band():
    metrics = {"regime_score": 55.0}
    target = backend._determine_equity_target("Neutral", metrics, base_target=0.7)
    assert 0.60 <= target <= 0.90
