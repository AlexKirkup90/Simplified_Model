import pandas as pd
import streamlit as st
import types, sys, pathlib

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

# Ensure backend is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_group_cap_violations_detected():
    weights = pd.Series({
        "CRWD": 0.30,  # Software:Security
        "SNOW": 0.25,  # Software:Data/AI
    })
    sectors_map = {"CRWD": "Software", "SNOW": "Software"}
    group_caps = {
        "Software": 0.50,
        "Software:Security": 0.25,
        "Software:Data/AI": 0.20,
    }
    violations = backend.check_constraint_violations(
        weights,
        sectors_map,
        name_cap=1.0,
        sector_cap=1.0,
        group_caps=group_caps,
    )
    assert any(v.startswith("Software:Security") for v in violations)
    assert any(v.startswith("Software:Data/AI") for v in violations)
    assert any(v.startswith("Software:") and v.count(':') == 1 for v in violations)
