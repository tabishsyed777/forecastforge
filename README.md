# ForecastForge

ForecastForge is a local probabilistic forecasting lab for regularly spaced numeric time series. It builds lag and calendar features, runs a walk-forward holdout, trains quantile regressors, and widens the forecast band with a conformal residual calibration step.

## Docker

```sh
docker build -t forecastforge .
mkdir -p reports
docker run --rm -v "$PWD/samples:/data:ro" -v "$PWD/reports:/reports" forecastforge /data/demand.csv --timestamp timestamp --value value --horizon 7 --output /reports/report.json
```

## Input and method

The CSV needs a timestamp column and a numeric value column. Timestamps are sorted and expected to be regularly spaced. The model uses lag 1, lag 7, a rolling mean, trend position, and weekly sine/cosine terms. A median Gradient Boosting model produces the central forecast; quantile models and a conformal residual quantile produce the interval.

The final report includes a held-out walk-forward MAE, MAPE, interval coverage, and the next horizon. These metrics are only meaningful for a series with enough history and a stable sampling cadence. Forecast intervals are empirical uncertainty bands, not guarantees.

The included data is synthetic and intended to exercise the pipeline, not establish production accuracy.

## Browser preview

Open `index.html` for a static preview. It reads a local CSV and shows cadence, trend, range, and a short data-quality checklist. The full model runs through Python or Docker.

## License

MIT. See [LICENSE](LICENSE).
