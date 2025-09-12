import pandas as pd
import numpy as np
import backend


def test_older_signals_get_lower_weights(monkeypatch):
    dates = pd.date_range('2023-01-01', '2023-02-28', freq='D')
    prices = pd.DataFrame({
        'AAA': np.linspace(100, 120, len(dates)),
        'BBB': np.linspace(100, 120, len(dates)),
    }, index=dates)
    sectors = {'AAA': 'Tech', 'BBB': 'Tech'}

    def fake_composite_score(hist):
        curr = hist.index[-1]
        if curr.month == 1:
            return pd.DataFrame([{'AAA': 1.0, 'BBB': 0.5}], index=[curr])
        else:
            return pd.DataFrame([{'AAA': 1.0, 'BBB': 1.0}], index=[curr])

    def fake_blended_momentum_z(monthly):
        curr = monthly.index[-1]
        if curr.month == 1:
            return pd.Series({'AAA': 1.0, 'BBB': -1.0})
        else:
            return pd.Series({'AAA': 1.0, 'BBB': 1.0})

    def fake_momentum_stable_names(hist, top_n, days):
        curr = hist.index[-1]
        if curr.month == 1:
            return ['AAA']
        else:
            return ['AAA', 'BBB']

    monkeypatch.setattr(backend, 'composite_score', fake_composite_score)
    monkeypatch.setattr(backend, 'blended_momentum_z', fake_blended_momentum_z)
    monkeypatch.setattr(backend, 'momentum_stable_names', fake_momentum_stable_names)
    monkeypatch.setattr(backend, 'get_volatility_adjusted_caps', lambda raw, hist, base_cap: {})
    monkeypatch.setattr(backend, 'cap_weights', lambda raw, cap, vol_adjusted_caps=None: raw)
    monkeypatch.setattr(backend, 'get_enhanced_sector_map', lambda tickers, base_map: {t: 'Other' for t in tickers})
    monkeypatch.setattr(backend, 'build_group_caps', lambda sectors_map: {})
    monkeypatch.setattr(
        backend,
        'enforce_caps_iteratively',
        lambda raw, enhanced_sectors, name_cap, sector_cap, group_caps, debug: raw,
    )
    monkeypatch.setattr(backend, 'compute_regime_metrics', lambda hist: {})
    monkeypatch.setattr(backend, 'get_regime_adjusted_exposure', lambda metrics: 1.0)

    calls = []
    original_decay = backend.apply_signal_decay

    def spy_apply_signal_decay(momentum_scores, signal_age_days, half_life=45):
        result = original_decay(momentum_scores, signal_age_days, half_life)
        calls.append((momentum_scores, signal_age_days, result))
        return result

    monkeypatch.setattr(backend, 'apply_signal_decay', spy_apply_signal_decay)

    backend.run_momentum_composite_param(
        prices,
        sectors,
        top_n=2,
        name_cap=1.0,
        sector_cap=1.0,
        stickiness_days=0,
        use_enhanced_features=True,
    )

    assert len(calls) == 2
    _, ages, result = calls[1]
    assert ages['AAA'] > ages['BBB']
    assert result['AAA'] < result['BBB']
