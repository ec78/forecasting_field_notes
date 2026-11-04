# Data

Two series are used across the demos, both stored as `date,y` CSVs under `data/raw/`.

## `electricity_ciso.csv`

- **Source:** U.S. Energy Information Administration (EIA), hourly electricity demand.
- **Balancing authority:** CISO (California ISO).
- **Frequency:** hourly.
- **Coverage:** 2019-01-01 through 2026-08-18 (see `electricity_ciso_metadata.json`).
- **License:** U.S. government work product; public domain.
- **Role in this repo:** the seasonal, high-signal series used throughout Post 01's three demos and again in Posts 03 and 04. Daily (24h) and weekly (168h) seasonality make it a good testbed for seasonal-naive baselines, Fourier features, and residual-memory correction.

## `spy.csv`

- **Source:** Yahoo Finance, SPY (S&P 500 ETF) adjusted close.
- **Frequency:** daily (trading days only).
- **Coverage:** 1993-01-29 through 2026-08-17 (see `spy_metadata.json`).
- **Role in this repo:** a low-signal-to-noise example. Used in Post 01 Demo 1 to illustrate how a pretrained model and a hybrid combination behave when the series is close to a random walk; used again in Post 04 as the control series where an online corrector should find nothing to learn (Post 02 is a synthetic system, Post 03 is CISO-only).

Both files were reused as-is from the companion repo for
[*Transformers vs. ARIMA*](https://medium.com/@eclow/transformers-vs-arima-do-more-complex-models-produce-better-forecasts-28ca372052d4)
so that results here are directly comparable to that earlier post.

## Regenerating the data

`src/dataimport.py` can refresh both series:

```bash
# electricity demand requires a free EIA API key
echo "EIA_API_KEY=your_key_here" > .env
python -m src.dataimport --dataset electricity --balancing-authority CISO
python -m src.dataimport --dataset spy
```

This is optional — the checked-in CSVs are sufficient to run every notebook.
