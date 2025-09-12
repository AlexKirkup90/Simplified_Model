import streamlit as st
import sys, pathlib, types
import pytest

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


@pytest.fixture
def high_vol_metrics():
    return {"qqq_vol_10d": 0.05}


@pytest.fixture
def low_vol_metrics():
    return {"qqq_vol_10d": 0.01}


def test_map_metrics_high_volatility(high_vol_metrics):
    settings = backend.map_metrics_to_settings(high_vol_metrics)
    assert settings["top_n"] == 5
    assert settings["name_cap"] == pytest.approx(0.20)
    assert settings["sector_cap"] == pytest.approx(0.25)


def test_map_metrics_low_volatility(low_vol_metrics):
    settings = backend.map_metrics_to_settings(low_vol_metrics)
    assert settings["top_n"] == 10
    assert settings["name_cap"] == pytest.approx(0.30)
    assert settings["sector_cap"] == pytest.approx(0.35)
