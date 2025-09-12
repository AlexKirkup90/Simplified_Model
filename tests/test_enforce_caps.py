import pandas as pd
import pytest
import streamlit as st
import types, sys, pathlib

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

# Ensure backend is importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend

def test_enforce_caps_hierarchical():
    weights = pd.Series({
        "CRWD": 0.30,  # Software:Security
        "ZS": 0.20,    # Software:Security
        "SNOW": 0.25,  # Software:Data/AI
        "DDOG": 0.15,  # Software:Data/AI
        "APP": 0.10,   # Software:AdTech
    })
    sector_labels = {
        "CRWD": "Software:Security",
        "ZS": "Software:Security",
        "SNOW": "Software:Data/AI",
        "DDOG": "Software:Data/AI",
        "APP": "Software:AdTech",
    }
    group_caps = {
        "Software": 0.50,
        "Software:Security": 0.25,
        "Software:Data/AI": 0.20,
        "Software:AdTech": 0.15,
    }

    out = backend.enforce_caps_iteratively(
        weights,
        sector_labels,
        name_cap=0.25,
        sector_cap=1.0,  # no generic sector limit
        group_caps=group_caps,
    )

    # Each security is within the name cap
    assert (out <= 0.25 + 1e-9).all()

    # Each sub-bucket respects its cap
    ser_sector = pd.Series(sector_labels)
    sec_sum = out[ser_sector[ser_sector == "Software:Security"].index].sum()
    assert sec_sum <= 0.25 + 1e-9

    data_sum = out[ser_sector[ser_sector == "Software:Data/AI"].index].sum()
    assert data_sum <= 0.20 + 1e-9

    adtech_sum = out[ser_sector[ser_sector == "Software:AdTech"].index].sum()
    assert adtech_sum <= 0.15 + 1e-9

    # Parent bucket respects its cap
    assert out.sum() <= 0.50 + 1e-9

    # Trimming occurred (iterative process)
    assert any(out[k] < weights[k] for k in weights.index)

