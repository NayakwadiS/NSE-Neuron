"""
Smoke / contract tests for the FastAPI backend.

Every route is exercised with the heavy service layer stubbed out, so these
verify wiring, validation and error mapping - not model quality.
"""
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

fastapi_testclient = pytest.importorskip("fastapi.testclient")
main = pytest.importorskip(
    "src.backend.main",
    reason="backend dependencies (tensorflow / nselib / talib) unavailable",
)

from src.backend.routers import analysis as analysis_router          # noqa: E402
from src.backend.routers import data as data_router                  # noqa: E402
from src.backend.routers import forecast as forecast_router          # noqa: E402
from src.backend.services.job_manager import JobManager, JobStatus   # noqa: E402


@pytest.fixture
def client():
    return fastapi_testclient.TestClient(main.app)


# ── root / health ─────────────────────────────────────────────────────────────

def test_root_is_alive(client):
    assert client.get("/").status_code == 200


def test_health_returns_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_openapi_schema_builds(client):
    assert client.get("/openapi.json").status_code == 200


# ── /api/forecast ─────────────────────────────────────────────────────────────

@pytest.fixture
def captured_submit(monkeypatch):
    """Replace the job queue so no model ever trains during the tests."""
    calls = []

    def fake_submit(func, *args, description="", **kwargs):
        calls.append({"func": func, "args": args, "description": description})
        return "job-123"

    monkeypatch.setattr(forecast_router.job_manager, "submit", fake_submit)
    return calls


@pytest.mark.parametrize("algo", ["lstm", "bilstm", "gru", "cnn_lstm", "all"])
def test_forecast_accepts_every_valid_algorithm(client, captured_submit, algo):
    res = client.post("/api/forecast", json={"symbol": "SUZLON", "algorithm": algo})
    assert res.status_code == 200
    assert res.json() == {"job_id": "job-123", "status": "pending"}


def test_forecast_rejects_unknown_algorithm(client, captured_submit):
    res = client.post("/api/forecast", json={"symbol": "SUZLON", "algorithm": "transformer"})
    assert res.status_code == 400
    assert not captured_submit


def test_forecast_requires_a_symbol(client):
    assert client.post("/api/forecast", json={"algorithm": "lstm"}).status_code == 422


def test_forecast_normalises_symbol_and_algorithm(client, captured_submit):
    client.post("/api/forecast", json={"symbol": " SUZLON ", "algorithm": " LSTM "})
    assert captured_submit[0]["args"][0] == "SUZLON"
    assert captured_submit[0]["args"][1] == "lstm"


def test_forecast_defaults_force_retrain_to_false(client, captured_submit):
    client.post("/api/forecast", json={"symbol": "SUZLON", "algorithm": "lstm"})
    assert captured_submit[0]["args"][2] is False


def test_forecast_forwards_force_retrain(client, captured_submit):
    client.post("/api/forecast",
                json={"symbol": "SUZLON", "algorithm": "lstm", "force_retrain": True})
    assert captured_submit[0]["args"][2] is True
    assert "forced retrain" in captured_submit[0]["description"]


def test_forecast_all_routes_to_the_multi_model_runner(client, captured_submit):
    client.post("/api/forecast", json={"symbol": "SUZLON", "algorithm": "all"})
    assert captured_submit[0]["func"].__name__ == "run_all_forecast"


def test_forecast_single_routes_to_the_single_model_runner(client, captured_submit):
    client.post("/api/forecast", json={"symbol": "SUZLON", "algorithm": "gru"})
    assert captured_submit[0]["func"].__name__ == "run_single_forecast"


# ── /api/jobs ─────────────────────────────────────────────────────────────────

def test_unknown_job_returns_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_job_lifecycle_reports_a_result():
    manager = JobManager(max_workers=1)
    job_id = manager.submit(lambda job: {"ok": True}, description="unit")
    for _ in range(200):
        job = manager.get(job_id)
        if job.status in (JobStatus.DONE, JobStatus.ERROR):
            break
        import time; time.sleep(0.01)
    payload = manager.to_dict(job)
    assert payload["status"] == JobStatus.DONE
    assert payload["result"] == {"ok": True}
    assert payload["error"] is None


def test_job_lifecycle_captures_errors():
    manager = JobManager(max_workers=1)

    def boom(job):
        raise RuntimeError("kaboom")

    job_id = manager.submit(boom)
    for _ in range(200):
        job = manager.get(job_id)
        if job.status in (JobStatus.DONE, JobStatus.ERROR):
            break
        import time; time.sleep(0.01)
    assert job.status == JobStatus.ERROR
    assert "kaboom" in job.error


def test_job_progress_is_visible_to_the_worker_function():
    manager = JobManager(max_workers=1)

    def work(job):
        job.progress = "halfway"
        return job.progress

    job_id = manager.submit(work)
    for _ in range(200):
        job = manager.get(job_id)
        if job.status in (JobStatus.DONE, JobStatus.ERROR):
            break
        import time; time.sleep(0.01)
    assert job.result == "halfway"


def test_job_ids_are_unique():
    manager = JobManager(max_workers=1)
    ids = {manager.submit(lambda job: None) for _ in range(20)}
    assert len(ids) == 20


# ── /api/regime ───────────────────────────────────────────────────────────────

def test_regime_returns_payload(client, monkeypatch):
    monkeypatch.setattr(analysis_router, "run_regime_analysis",
                        lambda sym: {"symbol": sym, "patterns": [], "historical": []})
    res = client.post("/api/regime/SUZLON")
    assert res.status_code == 200
    assert res.json()["symbol"] == "SUZLON"


def test_regime_maps_value_error_to_404(client, monkeypatch):
    def boom(sym):
        raise ValueError("Symbol 'ZZZ' not found in NSE equity list.")

    monkeypatch.setattr(analysis_router, "run_regime_analysis", boom)
    res = client.post("/api/regime/ZZZ")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"]


def test_regime_maps_unexpected_error_to_500(client, monkeypatch):
    def boom(sym):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(analysis_router, "run_regime_analysis", boom)
    assert client.post("/api/regime/SUZLON").status_code == 500


# ── /api/symbols ──────────────────────────────────────────────────────────────

def test_symbols_search(client, monkeypatch):
    monkeypatch.setattr(data_router, "search_symbols",
                        lambda q: [{"symbol": "SUZLON", "name": "Suzlon Energy Limited"}])
    res = client.get("/api/symbols", params={"q": "suz"})
    assert res.status_code == 200
    assert res.json()["symbols"][0]["symbol"] == "SUZLON"


def test_symbols_rejects_empty_query(client):
    assert client.get("/api/symbols", params={"q": ""}).status_code == 422


# ── /api/historical ───────────────────────────────────────────────────────────

@pytest.fixture
def stub_fetch_data(monkeypatch, price_df_factory):
    hist = price_df_factory([100.0, 101.0, 102.0])
    monkeypatch.setattr(data_router, "fetch_data",
                        lambda sym: (hist, {"scheme_name": "X"}, hist))
    return hist


def test_historical_returns_ohlcv(client, stub_fetch_data):
    res = client.get("/api/historical/SUZLON")
    assert res.status_code == 200
    body = res.json()
    assert body["symbol"] == "SUZLON"
    assert len(body["data"]) == 3
    assert {"date", "open", "high", "low", "close", "volume"} <= set(body["data"][0])


def test_historical_respects_days_parameter(client, stub_fetch_data):
    assert len(client.get("/api/historical/SUZLON", params={"days": 2}).json()["data"]) == 2


def test_historical_keeps_rows_with_missing_volume(client, monkeypatch, price_df_factory):
    hist = price_df_factory([100.0, 101.0, 102.0])
    hist["volume"] = hist["volume"].astype(float)
    hist.loc[1, "volume"] = None
    monkeypatch.setattr(data_router, "fetch_data", lambda sym: (hist, {}, hist))
    body = client.get("/api/historical/SUZLON").json()
    assert len(body["data"]) == 3
    assert body["data"][1]["volume"] is None


def test_historical_404_for_unknown_symbol(client, monkeypatch):
    def boom(sym):
        raise ValueError("Symbol 'ZZZ' not found in NSE equity list.")

    monkeypatch.setattr(data_router, "fetch_data", boom)
    assert client.get("/api/historical/ZZZ").status_code == 404


# ── /api/models/cached ────────────────────────────────────────────────────────

ENTRY = {
    "symbol": "SUZLON", "model": "LSTM", "trained_at": "2024-06-01T10:00:00",
    "n_rows": 1000, "last_date": "2024-06-01", "metrics": {"rmse": 0.01},
    "size_kb": 128.0,
}


def test_list_cached_models(client, monkeypatch):
    from utils import model_registry
    monkeypatch.setattr(model_registry, "list_cached", lambda: [ENTRY])
    body = client.get("/api/models/cached").json()
    assert body["count"] == 1
    assert body["total_size_kb"] == pytest.approx(128.0)
    assert "enabled" in body and "max_stale_days" in body


def test_list_cached_models_for_one_symbol(client, monkeypatch):
    from utils import model_registry
    monkeypatch.setattr(model_registry, "list_cached", lambda: [ENTRY])
    body = client.get("/api/models/cached/SUZLON").json()
    assert body["symbol"] == "SUZLON"
    assert body["count"] == 1


def test_list_cached_models_for_unknown_symbol_is_empty(client, monkeypatch):
    from utils import model_registry
    monkeypatch.setattr(model_registry, "list_cached", lambda: [ENTRY])
    assert client.get("/api/models/cached/SBIN").json()["count"] == 0


def test_delete_cached_symbol(client, monkeypatch):
    from utils import model_registry
    monkeypatch.setattr(model_registry, "delete", lambda sym, model=None: 2)
    body = client.delete("/api/models/cached/SUZLON").json()
    assert body == {"deleted": 2, "symbol": "SUZLON"}


def test_delete_cached_symbol_404_when_nothing_removed(client, monkeypatch):
    from utils import model_registry
    monkeypatch.setattr(model_registry, "delete", lambda sym, model=None: 0)
    assert client.delete("/api/models/cached/SUZLON").status_code == 404


def test_delete_single_cached_model(client, monkeypatch):
    from utils import model_registry
    monkeypatch.setattr(model_registry, "delete", lambda sym, model=None: 1)
    body = client.delete("/api/models/cached/SUZLON/LSTM").json()
    assert body["model"] == "LSTM"


def test_purge_expired_models(client, monkeypatch):
    from utils import model_registry
    monkeypatch.setattr(model_registry, "purge_expired", lambda: 3)
    assert client.post("/api/models/cached/purge").json()["purged"] == 3

