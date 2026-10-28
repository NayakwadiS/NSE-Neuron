# Tests

Automated test suite for NSE-Neuron.

## Run

```bash
pip install -r requirements-dev.txt
pytest
```

Useful selections:

```bash
pytest -m "not slow"          # skip anything that imports TensorFlow
pytest test/test_preprocessor.py -v
pytest --collect-only -q      # list every test
```

## What is covered

| File | Area |
|------|------|
| `test_preprocessor.py`     | Column selection/renaming, numeric + volume parsing, date sorting, duplicate-date removal, `config.HISTORIC_DATA` publication and isolation |
| `test_data_fetcher.py`     | Raw CSV cache: hit/miss behaviour, per-day cache key, directory creation |
| `test_regime_detector.py`  | BULL/BEAR/SIDEWAYS classification, insufficient-data fallback, confidence boost/penalty and clamping |
| `test_pattern_detector.py` | TA-Lib candlestick pattern columns, pattern score, de-duplication |
| `test_model_registry.py`   | Weight cache: signature hashing, `fresh`/`warm`/`miss` tiering, expiry, save/load round-trip, scaler pairing, delete/purge |
| `test_nse_service.py`      | Chart OHLCV shaping, forecast-day construction, regime/insight mapping, `fetch_data` race-safety, symbol search |
| `test_api.py`              | Every FastAPI route: validation, error mapping, job lifecycle, cache management endpoints |

## Design notes

- **No network, no training.** The NSE API, TensorFlow and the job queue are
  stubbed, so the full suite runs in seconds.
- Tests marked `slow` only import heavy dependencies (TensorFlow / TA-Lib);
  they are skipped automatically when those packages are unavailable.
- `config.HISTORIC_DATA` is a module-level global — an autouse fixture in
  `conftest.py` restores it after every test so state never leaks.

