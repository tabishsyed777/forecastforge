#!/usr/bin/env python3
"""Forecast a regular numeric time series with quantile and conformal intervals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


def feature_row(history: list[float], timestamp: pd.Timestamp, index: int, season_lag: int) -> list[float]:
    if len(history) < season_lag:
        raise ValueError("not enough history for the seasonal lag")
    recent = history[-7:] if len(history) >= 7 else history
    return [history[-1], history[-season_lag], float(np.mean(recent)), float(index), float(np.sin(2 * np.pi * timestamp.dayofweek / 7)), float(np.cos(2 * np.pi * timestamp.dayofweek / 7))]


def make_training(series: pd.Series, season_lag: int) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    x, y, dates = [], [], []
    values = series.to_numpy(dtype=float)
    for index in range(season_lag, len(values)):
        x.append(feature_row(list(values[:index]), series.index[index], index, season_lag)); y.append(values[index]); dates.append(series.index[index])
    return np.asarray(x), np.asarray(y), dates


def fit_models(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, GradientBoostingRegressor]:
    models = {}
    for name, quantile in (("lower", 0.1), ("median", 0.5), ("upper", 0.9)):
        models[name] = GradientBoostingRegressor(loss="quantile", alpha=quantile, n_estimators=180, learning_rate=0.035, max_depth=2, random_state=seed).fit(x, y)
    return models


def recursive_forecast(models: dict[str, GradientBoostingRegressor], history: list[float], start: pd.Timestamp, horizon: int, season_lag: int, start_index: int, conformal: float) -> list[dict[str, Any]]:
    output = []
    rolling = list(history)
    for step in range(horizon):
        timestamp = start + pd.to_timedelta(int(step), unit="D")
        features = np.asarray([feature_row(rolling, timestamp, start_index + step, season_lag)])
        median = float(models["median"].predict(features)[0]); low = float(models["lower"].predict(features)[0]) - conformal; high = float(models["upper"].predict(features)[0]) + conformal
        output.append({"timestamp": timestamp.isoformat(), "lower": round(max(0.0, low), 4), "forecast": round(median, 4), "upper": round(max(median, high), 4)})
        rolling.append(median)
    return output


def analyze(frame: pd.DataFrame, timestamp: str, value: str, horizon: int, season_lag: int, seed: int) -> dict[str, Any]:
    if timestamp not in frame.columns or value not in frame.columns:
        raise ValueError(f"CSV must contain '{timestamp}' and '{value}' columns")
    data = frame[[timestamp, value]].copy(); data[timestamp] = pd.to_datetime(data[timestamp], utc=True, errors="coerce"); data[value] = pd.to_numeric(data[value], errors="coerce"); data = data.dropna().sort_values(timestamp)
    if len(data) < season_lag + horizon + 10:
        raise ValueError(f"at least {season_lag + horizon + 10} valid rows are required")
    data = data.drop_duplicates(timestamp).set_index(timestamp); series = data[value].astype(float); deltas = series.index.to_series().diff().dropna().dt.total_seconds() / 86400
    warnings = []
    if len(deltas) and (deltas.median() != 1 or deltas.max() > 2): warnings.append("timestamps are not strictly daily; calendar features assume a regular cadence")
    x_all, y_all, _ = make_training(series, season_lag); split = len(series) - horizon; train_series = series.iloc[:split]; x_train, y_train, _ = make_training(train_series, season_lag); models = fit_models(x_train, y_train, seed)
    calibration_size = max(7, min(21, len(y_train) // 5)); calibration_x, calibration_y = x_train[-calibration_size:], y_train[-calibration_size:]; residuals = np.abs(calibration_y - models["median"].predict(calibration_x)); conformal = float(np.quantile(residuals, 0.9, method="higher")) if len(residuals) else 0.0
    holdout = recursive_forecast(models, list(train_series.to_numpy()), series.index[split], horizon, season_lag, split, conformal); actual = series.iloc[split:].to_numpy(); predictions = np.asarray([item["forecast"] for item in holdout]); lowers = np.asarray([item["lower"] for item in holdout]); uppers = np.asarray([item["upper"] for item in holdout]); coverage = float(np.mean((actual >= lowers) & (actual <= uppers)))
    full_x, full_y, _ = make_training(series, season_lag); full_models = fit_models(full_x, full_y, seed); future_start = series.index[-1] + pd.to_timedelta(1, unit="D"); forecast = recursive_forecast(full_models, list(series.to_numpy()), future_start, horizon, season_lag, len(series), conformal)
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "method": "gradient_boosting_quantiles_with_conformal_residual", "design": {"timestamp": timestamp, "value": value, "horizon": horizon, "season_lag": season_lag, "conformal_quantile": round(conformal, 4)}, "sample": {"rows": int(len(series)), "start": series.index[0].isoformat(), "end": series.index[-1].isoformat(), "value_min": round(float(series.min()), 4), "value_max": round(float(series.max()), 4)}, "backtest": {"holdout_rows": horizon, "mae": round(float(mean_absolute_error(actual, predictions)), 4), "mape_percent": round(float(np.mean(np.abs((actual - predictions) / np.maximum(np.abs(actual), 1e-8))) * 100), 4), "interval_coverage": round(coverage, 4)}, "forecast": forecast, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("input", type=Path); parser.add_argument("--timestamp", default="timestamp"); parser.add_argument("--value", default="value"); parser.add_argument("--horizon", type=int, default=7); parser.add_argument("--season-lag", type=int, default=7); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output", type=Path); args = parser.parse_args()
    if args.horizon < 1 or args.season_lag < 2: parser.error("horizon must be positive and season lag must be at least 2")
    if not args.input.is_file(): parser.error(f"input file does not exist: {args.input}")
    try: report = analyze(pd.read_csv(args.input), args.timestamp, args.value, args.horizon, args.season_lag, args.seed)
    except ValueError as error: parser.error(str(error))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output: args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered, encoding="utf-8"); print(f"Report written to {args.output}")
    else: print(rendered, end="")


if __name__ == "__main__": main()
