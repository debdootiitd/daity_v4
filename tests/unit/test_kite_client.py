"""Unit tests for the Kite client wrapper.

Mirrors `test_groww_client.py`. The actual `KiteConnect` is mocked so these
tests don't burn rate limit and don't need a fresh access token.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timezone
from unittest.mock import MagicMock

import polars as pl
import pytest

from daity.data.kite import (
    INTERVAL_MAP,
    Candle,
    KiteClient,
    _as_datetime,
    _historical_with_retry,
    _iter_time_chunks,
    _to_candles,
    _to_ist_naive,
)
from daity.utils.env import KiteConfig

# ----- chunking -----


def test_chunk_single_window_when_below_max() -> None:
    s = datetime(2026, 4, 1)
    e = datetime(2026, 4, 5)
    assert _iter_time_chunks(s, e, max_days=100) == [(s, e)]


def test_chunk_splits_at_exact_max() -> None:
    # 5min cap = 100 days; 200-day range → 2 chunks
    s = datetime(2026, 1, 1)
    e = datetime(2026, 7, 20)  # 200 days
    chunks = _iter_time_chunks(s, e, max_days=100)
    assert len(chunks) == 2


def test_chunk_empty_when_start_after_end() -> None:
    assert _iter_time_chunks(datetime(2026, 5, 1), datetime(2026, 4, 1), max_days=100) == []


def test_chunk_rejects_nonpositive_max() -> None:
    with pytest.raises(ValueError):
        _iter_time_chunks(datetime(2026, 1, 1), datetime(2026, 1, 5), max_days=0)


# ----- date coercion -----


def test_as_datetime_passes_through_datetime() -> None:
    dt = datetime(2026, 4, 1, 9, 15)
    assert _as_datetime(dt) is dt


def test_as_datetime_promotes_date_to_midnight() -> None:
    assert _as_datetime(date(2026, 4, 1)) == datetime(2026, 4, 1, 0, 0)


def test_as_datetime_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        _as_datetime("2026-04-01")  # type: ignore[arg-type]


def test_to_ist_naive_converts_utc_to_ist_naive() -> None:
    """A UTC tz-aware datetime should become its IST equivalent with tzinfo stripped."""
    utc = datetime(2026, 4, 1, 9, 15, tzinfo=UTC)  # 09:15 UTC = 14:45 IST
    naive_ist = _to_ist_naive(utc)
    assert naive_ist == datetime(2026, 4, 1, 14, 45)
    assert naive_ist.tzinfo is None


def test_to_ist_naive_passes_naive_through() -> None:
    """A naive datetime is assumed to already be IST and is returned unchanged."""
    naive = datetime(2026, 4, 1, 9, 15)
    assert _to_ist_naive(naive) is naive


def test_candles_calls_sdk_with_ist_naive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the SDK call must receive IST-naive datetimes (Kite's
    server interprets the strftime'd input as IST), not UTC-aware ones."""
    captured: list = []

    def fake_historical(token, start, end, kite_iv, oi=False):
        captured.append((start, end))
        return []

    fake_api = MagicMock()
    fake_api.historical_data.side_effect = fake_historical
    monkeypatch.setattr("daity.data.kite.KiteConnect", MagicMock(return_value=fake_api))
    cfg = KiteConfig(api_key="ak", api_secret=None, access_token=None)
    client = KiteClient(cfg)
    client._api = fake_api

    # Pass UTC tz-aware; the SDK should receive IST-naive equivalents.
    client.candles(
        738561,
        interval="day",
        start=datetime(2026, 4, 1, 9, 15, tzinfo=UTC),
        end=datetime(2026, 4, 5, 9, 15, tzinfo=UTC),
    )
    sent_start, sent_end = captured[0]
    assert sent_start == datetime(2026, 4, 1, 14, 45)
    assert sent_start.tzinfo is None
    assert sent_end == datetime(2026, 4, 5, 14, 45)


# ----- candle parsing -----


def test_to_candles_normalizes_ist_to_utc_epoch() -> None:
    """Kite returns IST-aware datetimes (+05:30); we convert to UTC epoch
    so downstream BQ writes are tz-agnostic."""
    from datetime import timedelta

    ist = timezone(timedelta(hours=5, minutes=30))
    rows = [
        {"date": datetime(2026, 4, 1, 9, 15, tzinfo=ist),
         "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.5, "volume": 1234},
    ]
    out = _to_candles(rows)
    assert len(out) == 1
    # 2026-04-01 09:15 IST = 2026-04-01 03:45 UTC
    expected_utc = datetime(2026, 4, 1, 3, 45, tzinfo=UTC)
    assert out[0].ts_epoch == int(expected_utc.timestamp())


def test_to_candles_handles_naive_datetime_as_ist() -> None:
    """A naive datetime (no tzinfo) is treated as IST — Kite's native zone.

    Defaulting to UTC would silently shift every bar by 5h30m if the SDK
    ever returns a tz-naive value. Mirror's the convention `_to_ist_naive`
    establishes on the outbound side.
    """
    from datetime import timedelta as _td
    ist = timezone(_td(hours=5, minutes=30))
    rows = [{"date": datetime(2026, 4, 1, 0, 0),
             "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10}]
    out = _to_candles(rows)
    assert len(out) == 1
    # 2026-04-01 00:00 IST = 2026-03-31 18:30 UTC
    assert out[0].ts_epoch == int(datetime(2026, 4, 1, tzinfo=ist).timestamp())
    assert out[0].ts_epoch != int(datetime(2026, 4, 1, tzinfo=UTC).timestamp())


def test_to_candles_handles_oi_field() -> None:
    rows = [{"date": datetime(2026, 4, 1, tzinfo=UTC),
             "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10, "oi": 999}]
    out = _to_candles(rows)
    assert out[0].open_interest == 999


def test_to_candles_skips_malformed() -> None:
    rows = [
        {"date": datetime(2026, 4, 1, tzinfo=UTC),
         "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10},
        {"foo": "bar"},                             # no date → skipped
        {"date": "2026-04-02",
         "open": 2.0, "high": 3.0, "low": 1.5, "close": 2.5, "volume": 20},  # str date → skipped
        {"date": datetime(2026, 4, 3, tzinfo=UTC),
         "open": "x", "high": 3.0, "low": 1.5, "close": 2.5, "volume": 20},  # bad open → skipped
    ]
    out = _to_candles(rows)
    assert len(out) == 1
    assert out[0].close == 1.5


def test_to_candles_empty() -> None:
    assert _to_candles([]) == []
    assert _to_candles(None) == []


# ----- config / api construction -----


def test_build_api_sets_access_token_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeKite:
        def __init__(self, *, api_key: str, timeout: int = 7) -> None:
            captured["api_key"] = api_key
            captured["timeout"] = timeout
        def set_access_token(self, t: str) -> None:
            captured["token"] = t

    monkeypatch.setattr("daity.data.kite.KiteConnect", FakeKite)
    cfg = KiteConfig(api_key="ak", api_secret="sk", access_token="tok123")
    KiteClient(cfg)
    assert captured["api_key"] == "ak"
    assert captured["token"] == "tok123"
    # The wrapper bumps timeout above the SDK default (7s) for the historical
    # endpoint, which routinely needs > 7s under load.
    assert captured["timeout"] > 7


def test_build_api_skips_set_access_token_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeKite:
        def __init__(self, *, api_key: str, timeout: int = 7) -> None:
            captured["api_key"] = api_key
        def set_access_token(self, t: str) -> None:
            captured["token"] = t

    monkeypatch.setattr("daity.data.kite.KiteConnect", FakeKite)
    cfg = KiteConfig(api_key="ak", api_secret=None, access_token=None)
    KiteClient(cfg)
    assert captured["api_key"] == "ak"
    assert "token" not in captured


def test_build_api_rejects_empty_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daity.data.kite.KiteConnect", MagicMock())
    cfg = KiteConfig(api_key="", api_secret=None, access_token=None)
    with pytest.raises(ValueError, match="KITE_API_KEY"):
        KiteClient(cfg)


# ----- identifier lookup -----


def _instruments_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "exchange": ["NSE", "NSE", "NSE", "NFO", "NSE"],
            "tradingsymbol": ["RELIANCE", "TCS", "INFY", "RELIANCE25APRFUT", "ZOMATO"],
            "instrument_token": [738561, 2953217, 408065, 99999999, 5097985],
            "name": ["RELIANCE", "Tata Consultancy", "Infosys", "RELIANCE FUT", "Zomato"],
            "instrument_type": ["EQ", "EQ", "EQ", "FUT", "EQ"],
            "segment": ["NSE", "NSE", "NSE", "NFO-FUT", "NSE"],
        }
    )


def _client_with_instruments(monkeypatch: pytest.MonkeyPatch, frame: pl.DataFrame) -> KiteClient:
    monkeypatch.setattr("daity.data.kite.KiteConnect", MagicMock())
    cfg = KiteConfig(api_key="ak", api_secret=None, access_token=None)
    client = KiteClient(cfg)
    client._instruments = frame
    return client


def test_lookup_finds_canonical_match(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_instruments(monkeypatch, _instruments_frame())
    row = client.lookup("RELIANCE")
    assert row is not None
    assert row["instrument_token"] == 738561


def test_lookup_returns_none_for_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_instruments(monkeypatch, _instruments_frame())
    assert client.lookup("NOTREAL") is None


def test_bulk_lookup_filters_to_eq_only_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_instruments(monkeypatch, _instruments_frame())
    out = client.bulk_lookup(["RELIANCE", "TCS", "INFY", "ZOMATO"])
    assert out.height == 4
    by_sym = {r["tradingsymbol"]: r for r in out.to_dicts()}
    # Cash equity wins over the FUT row that shares the RELIANCE root.
    assert by_sym["RELIANCE"]["instrument_token"] == 738561
    assert by_sym["TCS"]["instrument_token"] == 2953217


def test_bulk_lookup_records_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_instruments(monkeypatch, _instruments_frame())
    out = client.bulk_lookup(["RELIANCE", "GHOSTSYM"])
    by_sym = {r["tradingsymbol"]: r for r in out.to_dicts()}
    assert by_sym["GHOSTSYM"]["instrument_token"] is None


# ----- candles end-to-end (mocked) -----


def test_candles_passes_through_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunked calls are made with the right interval string."""
    calls: list[tuple] = []

    def fake_historical(token, start, end, kite_iv, oi=False):
        calls.append((token, start, end, kite_iv, oi))
        epoch = int(start.timestamp())
        return [{"date": start, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10}]

    fake_api = MagicMock()
    fake_api.historical_data.side_effect = fake_historical
    monkeypatch.setattr("daity.data.kite.KiteConnect", MagicMock(return_value=fake_api))
    cfg = KiteConfig(api_key="ak", api_secret=None, access_token=None)
    client = KiteClient(cfg)
    client._api = fake_api

    # 5min: max=100 days. 250-day range → 3 chunks.
    out = client.candles(
        738561,
        interval="5m",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 9, 8),
    )
    assert fake_api.historical_data.call_count == 3
    assert all(c[3] == "5minute" for c in calls)
    # All 3 chunks emitted distinct ts_epochs → 3 candles.
    assert len(out) == 3


def test_candles_rejects_unknown_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daity.data.kite.KiteConnect", MagicMock())
    cfg = KiteConfig(api_key="ak", api_secret=None, access_token=None)
    client = KiteClient(cfg)
    with pytest.raises(ValueError, match="Unknown interval"):
        client.candles(738561, interval="2m", start=datetime(2026, 1, 1), end=datetime(2026, 1, 5))


def test_interval_map_covers_design_resolutions() -> None:
    for label in ("5m", "15m", "60m", "day"):
        assert label in INTERVAL_MAP


def test_candle_dataclass_immutable() -> None:
    c = Candle(ts_epoch=1, open=1.0, high=2.0, low=0.5, close=1.5, volume=10, open_interest=None)
    with pytest.raises(Exception):
        c.open = 2.0  # type: ignore[misc]


# ----- retry helper -----


def test_historical_with_retry_succeeds_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    api = MagicMock()
    api.historical_data.return_value = [{"date": datetime(2026, 4, 1, tzinfo=UTC),
                                         "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                                         "volume": 10}]
    out = _historical_with_retry(api, 738561, datetime(2026, 4, 1), datetime(2026, 4, 5),
                                 "day", attempts=3, base_delay=0)
    assert api.historical_data.call_count == 1
    assert len(out) == 1


def test_historical_with_retry_recovers_from_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient ReadTimeout on the first call → retry → succeed on the second."""
    import requests.exceptions
    api = MagicMock()
    api.historical_data.side_effect = [
        requests.exceptions.ReadTimeout("simulated"),
        [{"date": datetime(2026, 4, 1, tzinfo=UTC), "open": 1.0,
          "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10}],
    ]
    out = _historical_with_retry(api, 738561, datetime(2026, 4, 1), datetime(2026, 4, 5),
                                 "day", attempts=3, base_delay=0)
    assert api.historical_data.call_count == 2
    assert len(out) == 1


def test_historical_with_retry_does_not_retry_kite_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A KiteException (server-side reject) is fatal — retrying won't help."""
    from kiteconnect.exceptions import TokenException
    api = MagicMock()
    api.historical_data.side_effect = TokenException("Invalid token")
    with pytest.raises(TokenException):
        _historical_with_retry(api, 738561, datetime(2026, 4, 1), datetime(2026, 4, 5),
                               "day", attempts=3, base_delay=0)
    # Only ONE call — no retry on KiteException.
    assert api.historical_data.call_count == 1


def test_historical_with_retry_exhausts_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """All attempts time out → final ReadTimeout propagates out."""
    import requests.exceptions
    api = MagicMock()
    api.historical_data.side_effect = requests.exceptions.ReadTimeout("simulated")
    with pytest.raises(requests.exceptions.ReadTimeout):
        _historical_with_retry(api, 738561, datetime(2026, 4, 1), datetime(2026, 4, 5),
                               "day", attempts=3, base_delay=0)
    assert api.historical_data.call_count == 3
