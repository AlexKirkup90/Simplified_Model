import streamlit as st
import sys, pathlib, types, json
import pandas as pd

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend

def test_update_parameter_mapping_sets_new_defaults():
    st.session_state.clear()
    data = {
        "metrics": [json.dumps({"qqq_vol_10d": v}) for v in [0.01, 0.015, 0.02, 0.03, 0.05, 0.06]],
        "portfolio_ret": [0.03, 0.02, 0.01, -0.01, -0.02, -0.03],
        "benchmark_ret": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)
    backend.update_parameter_mapping(df)
    mapping = st.session_state.get("param_map_defaults")
    assert mapping is not None
    assert mapping["low_vol"] != backend.PARAM_MAP_DEFAULTS["low_vol"]
    assert mapping["high_vol"] != backend.PARAM_MAP_DEFAULTS["high_vol"]

def test_update_parameter_mapping_adjusts_caps():
    st.session_state.clear()
    data = {
        "metrics": [json.dumps({"qqq_vol_10d": v}) for v in [0.01, 0.015, 0.02, 0.05, 0.06, 0.07]],
        "portfolio_ret": [0.5, 0.4, 0.3, -0.3, -0.4, -0.5],
        "benchmark_ret": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)
    backend.update_parameter_mapping(df)
    mapping = st.session_state.get("param_map_defaults")
    assert mapping is not None
    assert mapping["name_cap_low"] > backend.PARAM_MAP_DEFAULTS["name_cap_low"]
    assert mapping["name_cap_high"] < backend.PARAM_MAP_DEFAULTS["name_cap_high"]
    assert mapping["sector_cap_low"] > backend.PARAM_MAP_DEFAULTS["sector_cap_low"]
    assert mapping["sector_cap_high"] < backend.PARAM_MAP_DEFAULTS["sector_cap_high"]
    assert mapping["top_n_low"] > backend.PARAM_MAP_DEFAULTS["top_n_low"]
    assert mapping["top_n_high"] < backend.PARAM_MAP_DEFAULTS["top_n_high"]
