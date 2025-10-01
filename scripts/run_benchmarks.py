#!/usr/bin/env python3
"""Benchmark harness for Hybrid150 strategy variants."""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import backend


@dataclass
class ScenarioResult:
    name: str
    equity_gross: pd.Series
    equity_net: pd.Series
    benchmark: pd.Series
    turnover: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series
    payload: Dict[str, object]
    regime_timeline: pd.DataFrame


def _annualize_return(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    compounded = (1 + returns).prod()
    periods = returns.shape[0]
    years = periods / 12.0
    if years <= 0:
        return float("nan")
    return float(compounded ** (1 / years) - 1)


def _annualized_vol(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float(returns.std(ddof=0) * math.sqrt(12))


def _sortino_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    downside = returns[returns < 0]
    if downside.empty:
        return float("inf")
    downside_vol = downside.std(ddof=0) * math.sqrt(12)
    if downside_vol == 0:
        return float("inf")
    annual_ret = returns.mean() * 12
    return float(annual_ret / downside_vol)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min())


def _calmar(cagr: float, max_dd: float) -> float:
    if math.isnan(cagr) or math.isnan(max_dd) or max_dd == 0:
        return float("nan")
    return float(cagr / abs(max_dd))


def _hit_rate(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def _avg_trades_per_year(turnover: pd.Series) -> float:
    if turnover.empty:
        return float("nan")
    avg_monthly = float(turnover.mean())
    return float((avg_monthly * 12) / backend.AVG_TRADE_SIZE_DEFAULT)


def _ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _canonical_payload(
    gross: pd.Series,
    net: pd.Series,
    bench: pd.Series,
    turnover: pd.Series,
    show_net: bool = True,
) -> Dict[str, object]:
    return backend.standardize_backtest_payload(
        strategy_cum_gross=gross,
        strategy_cum_net=net,
        qqq_cum=bench,
        hybrid_tno=turnover,
        show_net=show_net,
    )


def _regime_timeline(daily_prices: pd.DataFrame, months: Iterable[pd.Timestamp]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for ts in months:
        window = daily_prices.loc[:ts].tail(252)
        if window.empty or len(window) < 60:
            continue
        metrics = backend.compute_regime_metrics(window)
        label, score, comps = backend.compute_regime_label(metrics)
        records.append(
            {
                "date": ts,
                "regime": label,
                "regime_score": score,
            }
        )
    if not records:
        return pd.DataFrame(columns=["regime", "regime_score"])
    df = pd.DataFrame(records).drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").set_index("date")
    return df


def _bucket_label(label: str) -> str:
    norm = (label or "").lower()
    if "extreme" in norm:
        return "Extreme Risk-Off"
    if "risk-off" in norm:
        return "Risk-Off"
    if "risk-on" in norm:
        return "Risk-On"
    return "Neutral"


def _regime_metrics(returns: pd.Series, timeline: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if returns.empty or timeline.empty:
        return {}
    joined = timeline.reindex(returns.index, method="ffill").dropna()
    out: Dict[str, Dict[str, float]] = {}
    for bucket in {"Risk-On", "Neutral", "Risk-Off", "Extreme Risk-Off"}:
        mask = joined["regime"].apply(_bucket_label) == bucket
        if not mask.any():
            continue
        bucket_rets = returns.loc[mask.index[mask]]
        metrics = {
            "cagr": _annualize_return(bucket_rets),
            "vol": _annualized_vol(bucket_rets),
        }
        vol = metrics["vol"]
        ann_ret = bucket_rets.mean() * 12 if not bucket_rets.empty else float("nan")
        metrics["sharpe"] = float(ann_ret / vol) if vol and not math.isnan(vol) and vol != 0 else float("nan")
        metrics["max_drawdown"] = _max_drawdown((1 + bucket_rets).cumprod())
        metrics["hit_rate"] = _hit_rate(bucket_rets)
        out[bucket] = metrics
    return out


def _run_backtest_scenario(
    name: str,
    start: str,
    end: str,
    universe: str,
    roundtrip_bps: float,
    min_dollar_volume: float,
    use_enhanced_features: bool = True,
) -> ScenarioResult:
    strat_cum_gross, strat_cum_net, qqq_cum, turnover, _, _ = backend.run_backtest_isa_dynamic(
        roundtrip_bps=roundtrip_bps,
        min_dollar_volume=min_dollar_volume,
        show_net=True,
        start_date=start,
        end_date=end,
        universe_choice=universe,
        use_enhanced_features=use_enhanced_features,
        auto_optimize=False,
    )
    if strat_cum_net is None or qqq_cum is None:
        raise RuntimeError(f"Scenario {name} failed to produce equity curves")

    gross = pd.Series(strat_cum_gross).astype(float) if strat_cum_gross is not None else pd.Series(dtype=float)
    net = pd.Series(strat_cum_net).astype(float)
    bench = pd.Series(qqq_cum).astype(float)
    tno = pd.Series(turnover).astype(float) if turnover is not None else pd.Series(dtype=float)
    returns = net.pct_change().dropna()
    bench_returns = bench.pct_change().dropna()

    close, _, _, _ = backend._prepare_universe_for_backtest(universe, start, end)
    timeline = _regime_timeline(close, net.index)

    payload = _canonical_payload(gross, net, bench, tno, show_net=True)

    return ScenarioResult(
        name=name,
        equity_gross=gross,
        equity_net=net,
        benchmark=bench,
        turnover=tno,
        returns=returns,
        benchmark_returns=bench_returns,
        payload=payload,
        regime_timeline=timeline,
    )


def _run_leadership_scenario(start: str, end: str) -> ScenarioResult:
    tickers = list(dict.fromkeys(
        backend.FALLBACK_BROAD_ETFS + backend.FALLBACK_SECTOR_ETFS + [backend.FALLBACK_CORE_TICKER]
    ))
    daily = backend.fetch_market_data(tickers, start, end)
    daily = backend._ensure_unique_sorted_index(daily)
    if daily.empty:
        raise RuntimeError("Leadership ETF data unavailable")

    monthly = daily.resample("M").last().dropna(how="all")
    monthly_returns = monthly.pct_change().dropna(how="all")

    weights_by_month: Dict[pd.Timestamp, pd.Series] = {}
    equity = [1.0]
    dates: List[pd.Timestamp] = []
    returns: List[float] = []

    for ts in monthly_returns.index:
        window = daily.loc[:ts].tail(252)
        metrics = backend.compute_regime_metrics(window) if not window.empty else {}
        weights, _, target, _, _ = backend.compose_graceful_fallback(
            stock_weights=None,
            regime_metrics=metrics,
            regime_label=metrics.get("regime"),
            base_target=0.8,
            min_names=backend.MIN_ELIGIBLE_FALLBACK,
            eligible_pool=0,
            leadership_slice=backend.LEADERSHIP_SLICE_DEFAULT,
            core_slice=backend.CORE_SPY_SLICE_DEFAULT,
        )
        aligned = monthly_returns.loc[ts].reindex(weights.index).fillna(0.0)
        ret = float((weights * aligned).sum())
        weights_by_month[ts] = weights
        equity.append(equity[-1] * (1 + ret))
        dates.append(ts)
        returns.append(ret)

    equity_series = pd.Series(equity[1:], index=pd.Index(dates, name="date"))
    net = equity_series
    gross = net.copy()
    bench = monthly.loc[net.index, "QQQ"].pct_change().add(1).cumprod() if "QQQ" in monthly.columns else net
    turnover = pd.Series(0.0, index=net.index)
    rets = pd.Series(returns, index=net.index)
    bench_rets = bench.pct_change().dropna()

    timeline = _regime_timeline(daily, net.index)
    payload = _canonical_payload(gross, net, bench, turnover, show_net=True)

    return ScenarioResult(
        name="Leadership_ETF_Blend",
        equity_gross=gross,
        equity_net=net,
        benchmark=bench,
        turnover=turnover,
        returns=rets,
        benchmark_returns=bench_rets,
        payload=payload,
        regime_timeline=timeline,
    )


def _scenario_metrics(result: ScenarioResult) -> Dict[str, object]:
    cagr = _annualize_return(result.returns)
    vol = _annualized_vol(result.returns)
    sharpe = float((result.returns.mean() * 12) / vol) if vol and not math.isnan(vol) and vol != 0 else float("nan")
    sortino = _sortino_ratio(result.returns)
    max_dd = _max_drawdown(result.equity_net)
    calmar = _calmar(cagr, max_dd)
    hit_rate = _hit_rate(result.returns)
    avg_trades = _avg_trades_per_year(result.turnover)
    regimes = _regime_metrics(result.returns, result.regime_timeline)

    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "MaxDD": max_dd,
        "MAR": calmar,
        "HitRate": hit_rate,
        "AvgTradesPerYear": avg_trades,
        "CashUtilization": None,
        "HedgeUtilization": None,
        "Regimes": regimes,
    }


def run_benchmarks(start: str, end: str, export: str, universe_choice: str) -> None:
    roundtrip_bps = backend.ROUNDTRIP_BPS_DEFAULT
    min_dollar_volume = 0.0

    if universe_choice.lower() == "locked":
        base_universe = "Hybrid Top150"
    else:
        base_universe = universe_choice

    scenarios: List[ScenarioResult] = []
    scenarios.append(
        _run_backtest_scenario(
            "Hybrid150",
            start,
            end,
            base_universe,
            roundtrip_bps,
            min_dollar_volume,
        )
    )
    scenarios.append(
        _run_backtest_scenario(
            "SP500_Fallback",
            start,
            end,
            "S&P500 (All)",
            roundtrip_bps,
            min_dollar_volume,
        )
    )
    scenarios.append(_run_leadership_scenario(start, end))

    metrics: Dict[str, Dict[str, object]] = {}
    equity_rows: List[Dict[str, object]] = []
    turnover_rows: List[Dict[str, object]] = []
    regime_rows: List[Dict[str, object]] = []
    payloads: Dict[str, Dict[str, object]] = {}

    for result in scenarios:
        metrics[result.name] = _scenario_metrics(result)
        payloads[result.name] = result.payload
        for ts, value in result.equity_net.items():
            equity_rows.append(
                {
                    "date": ts.isoformat(),
                    "scenario": result.name,
                    "equity_net": float(value),
                    "equity_gross": float(result.equity_gross.get(ts, value)),
                    "benchmark": float(result.benchmark.get(ts, np.nan)),
                }
            )
        for ts, value in result.turnover.items():
            turnover_rows.append(
                {
                    "date": ts.isoformat(),
                    "scenario": result.name,
                    "turnover": float(value),
                }
            )
        if not result.regime_timeline.empty:
            for ts, row in result.regime_timeline.iterrows():
                regime_rows.append(
                    {
                        "date": ts.isoformat(),
                        "scenario": result.name,
                        "regime": row.get("regime"),
                        "regime_score": float(row.get("regime_score", np.nan)),
                    }
                )

    _ensure_directory(export)

    metrics_path = os.path.join(export, "metrics.json")
    equity_path = os.path.join(export, "equity_curves.csv")
    turnover_path = os.path.join(export, "turnover.csv")
    regime_path = os.path.join(export, "regime_timeline.csv")
    payload_path = os.path.join(export, "payload.json")

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump({"scenarios": metrics}, fh, indent=2, default=float)

    pd.DataFrame(equity_rows).to_csv(equity_path, index=False)
    pd.DataFrame(turnover_rows).to_csv(turnover_path, index=False)
    pd.DataFrame(regime_rows).to_csv(regime_path, index=False)

    with open(payload_path, "w", encoding="utf-8") as fh:
        json.dump(payloads, fh, indent=2, default=float)

    print(f"Metrics written to {metrics_path}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    today = date.today().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Run Hybrid150 benchmark scenarios")
    parser.add_argument("--start", default="2017-07-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=today, help="Backtest end date (YYYY-MM-DD or 'today')")
    parser.add_argument("--export", default="out", help="Directory for exported artefacts")
    parser.add_argument("--universe", default="locked", help="Universe selection (locked/Hybrid Top150/etc)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    end = args.end
    if end.lower() == "today":
        end = date.today().strftime("%Y-%m-%d")
    run_benchmarks(args.start, end, args.export, args.universe)


if __name__ == "__main__":
    main()
