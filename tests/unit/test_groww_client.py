"""Unit tests for the Groww client wrapper.

The actual `GrowwAPI` is mocked — these tests verify the chunking, response
parsing, identifier-lookup, and config-branching logic without burning any
real API quota.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import polars as pl
import pytest

from daity.data.groww import (
    INTERVAL_MAP,
    Candle,
    GrowwClient,
    _extract_candles,
    _format_ts,
    _iter_time_chunks,
)
from daity.utils.env import GrowwConfig

# ----- chunking -----


def test_chunk_single_window_when_below_max() -> None:
    s = datetime(2026, 4, 1, 9, 15)
    e = datetime(2026, 4, 5, 15, 30)
    chunks = _iter_time_chunks(s, e, max_days=30)
    assert chunks == [(s, e)]


def test_chunk_splits_at_exact_max() -> None:
    s = datetime(2026, 1, 1)
    e = datetime(2026, 4, 1)  # 90 days
    chunks = _iter_time_chunks(s, e, max_days=30)
    assert len(chunks) == 3
    assert chunks[0] == (s, datetime(2026, 1, 31))
    assert chunks[1] == (datetime(2026, 1, 31), datetime(2026, 3, 2))
    assert chunks[2] == (datetime(2026, 3, 2), datetime(2026, 4, 1))


def test_chunk_handles_partial_last() -> None:
    s = datetime(2026, 1, 1)
    e = datetime(2026, 2, 10)  # 40 days
    chunks = _iter_time_chunks(s, e, max_days=30)
    assert len(chunks) == 2
    assert chunks[1][1] == e


def test_chunk_empty_when_start_after_end() -> None:
    s = datetime(2026, 4, 5)
    e = datetime(2026, 4, 1)
    assert _iter_time_chunks(s, e, max_days=30) == []


def test_chunk_emits_one_when_start_equals_end() -> None:
    s = e = datetime(2026, 4, 5)
    assert _iter_time_chunks(s, e, max_days=30) == [(s, e)]


def test_chunk_rejects_nonpositive_max() -> None:
    with pytest.raises(ValueError):
        _iter_time_chunks(datetime(2026, 1, 1), datetime(2026, 1, 5), max_days=0)


# ----- timestamp formatting -----


def test_format_ts_matches_groww_spec() -> None:
    assert _format_ts(datetime(2026, 4, 1, 9, 15, 0)) == "2026-04-01 09:15:00"
    assert _format_ts(datetime(2026, 4, 1, 15, 30, 5)) == "2026-04-01 15:30:05"


# ----- response parsing -----


def test_extract_candles_top_level_shape() -> None:
    res = {"candles": [[1700000000, 100.0, 105.0, 99.0, 104.5, 1234, None]]}
    out = _extract_candles(res)
    assert len(out) == 1
    c = out[0]
    assert c.ts_epoch == 1700000000
    assert c.open == 100.0 and c.close == 104.5 and c.volume == 1234
    assert c.open_interest is None


def test_extract_candles_payload_wrapped_shape() -> None:
    res = {"payload": {"candles": [[1700000000, 1.0, 2.0, 0.5, 1.5, 10, 42]]}}
    out = _extract_candles(res)
    assert len(out) == 1
    assert out[0].open_interest == 42


def test_extract_candles_skips_malformed_rows() -> None:
    res = {"candles": [
        [1, 1.0, 2.0, 0.5, 1.5, 10],          # OK
        "not-a-row",                          # invalid
        [2, "x", 2.0, 0.5, 1.5, 10],          # invalid float
        [3, 1.0, 2.0, 0.5, 1.5, 10, None],    # OK
    ]}
    out = _extract_candles(res)
    assert [c.ts_epoch for c in out] == [1, 3]


def test_extract_candles_empty_or_unknown_shape() -> None:
    assert _extract_candles({}) == []
    assert _extract_candles({"foo": "bar"}) == []
    assert _extract_candles(None) == []
    assert _extract_candles("not a dict") == []


# ----- config / auth branching -----


def test_build_api_uses_jwt_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeAPI:
        def __init__(self, token: str) -> None:
            captured["token"] = token

    monkeypatch.setattr("daity.data.groww.GrowwAPI", FakeAPI)
    cfg = GrowwConfig(api_key="eyJfaketoken", api_secret="ignored", totp_secret=None)
    client = GrowwClient(cfg)
    assert isinstance(client._api, FakeAPI)
    assert captured["token"] == "eyJfaketoken"


def test_build_api_exchanges_key_when_not_jwt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}

    class FakeAPI:
        @staticmethod
        def get_access_token(*, api_key: str, secret: str) -> str:
            calls["mint"] = (api_key, secret)
            return "minted-token"

        def __init__(self, token: str) -> None:
            calls["init"] = token

    monkeypatch.setattr("daity.data.groww.GrowwAPI", FakeAPI)
    cfg = GrowwConfig(api_key="short-key", api_secret="the-secret", totp_secret=None)
    GrowwClient(cfg)
    assert calls["mint"] == ("short-key", "the-secret")
    assert calls["init"] == "minted-token"


def test_build_api_raises_when_no_secret_for_non_jwt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daity.data.groww.GrowwAPI", MagicMock())
    cfg = GrowwConfig(api_key="short-key", api_secret=None, totp_secret=None)
    with pytest.raises(ValueError, match="cannot mint an access token"):
        GrowwClient(cfg)


# ----- identifier lookup -----


def _instruments_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "exchange": ["NSE", "NSE", "NSE", "BSE", "NSE", "NSE"],
            "trading_symbol": ["RELIANCE", "TCS", "ZOMATO", "RELIANCE", "RELIANCE", "RELIANCE"],
            "groww_symbol": ["NSE-RELIANCE", "NSE-TCS", "NSE-ZOMATO", "BSE-RELIANCE",
                             "NSE-RELIANCE-FUT", "NSE-RELIANCE-OPT"],
            "isin": ["INE002A01018", "INE467B01029", "INE758T01015", "INE002A01018", None, None],
            "exchange_token": ["2885", "11536", "11078", "500325", "53000", "54000"],
            "segment": ["CASH", "CASH", "CASH", "CASH", "FNO", "FNO"],
            "name": ["Reliance Industries", "Tata Consultancy", "Zomato", "Reliance Industries",
                     "Reliance Fut", "Reliance Opt"],
        }
    )


def _client_with_instruments(monkeypatch: pytest.MonkeyPatch, frame: pl.DataFrame) -> GrowwClient:
    monkeypatch.setattr("daity.data.groww.GrowwAPI", MagicMock())
    cfg = GrowwConfig(api_key="eyJfake", api_secret=None, totp_secret=None)
    client = GrowwClient(cfg)
    client._instruments = frame
    return client


def test_lookup_finds_canonical_match(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_instruments(monkeypatch, _instruments_frame())
    row = client.lookup("RELIANCE", exchange="NSE")
    assert row is not None
    assert row["groww_symbol"] == "NSE-RELIANCE"
    assert row["isin"] == "INE002A01018"


def test_lookup_returns_none_for_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_instruments(monkeypatch, _instruments_frame())
    assert client.lookup("NONESUCH") is None


def test_lookup_distinguishes_exchanges(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_instruments(monkeypatch, _instruments_frame())
    row = client.lookup("RELIANCE", exchange="BSE")
    assert row is not None
    assert row["groww_symbol"] == "BSE-RELIANCE"


def test_bulk_lookup_dedupes_to_cash_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_with_instruments(monkeypatch, _instruments_frame())
    out = client.bulk_lookup(["RELIANCE", "TCS", "ZOMATO"])
    assert out.height == 3
    by_sym = {r["trading_symbol"]: r for r in out.to_dicts()}
    # Cash-segment rows kept; FNO duplicates filtered out.
    assert by_sym["RELIANCE"]["groww_symbol"] == "NSE-RELIANCE"
    assert by_sym["RELIANCE"]["isin"] == "INE002A01018"


def test_bulk_lookup_records_missing(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    client = _client_with_instruments(monkeypatch, _instruments_frame())
    out = client.bulk_lookup(["RELIANCE", "GHOSTSYM"])
    assert out.height == 2
    by_sym = {r["trading_symbol"]: r for r in out.to_dicts()}
    assert by_sym["GHOSTSYM"]["groww_symbol"] is None
    assert by_sym["GHOSTSYM"]["isin"] is None


# ----- candles end-to-end (mocked) -----


def test_candles_chunks_calls_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunked calls are made; duplicate ts_epochs (chunk seam) are deduped."""
    calls: list[dict] = []

    def fake_get_historical_candles(**kw):
        calls.append(kw)
        # Each chunk returns one candle keyed off the start-time string.
        ts = int(datetime.fromisoformat(kw["start_time"].replace(" ", "T")).timestamp())
        # Emit two candles where the second one is the boundary (overlaps next chunk).
        return {"candles": [
            [ts, 1.0, 2.0, 0.5, 1.5, 10, None],
            [ts + 86400 * 30, 2.0, 3.0, 1.5, 2.5, 20, None],  # boundary
        ]}

    fake_api = MagicMock()
    fake_api.get_historical_candles.side_effect = fake_get_historical_candles
    monkeypatch.setattr("daity.data.groww.GrowwAPI", MagicMock(return_value=fake_api))
    cfg = GrowwConfig(api_key="eyJfake", api_secret=None, totp_secret=None)
    client = GrowwClient(cfg)
    client._api = fake_api  # bypass the actual SDK constructor branch

    start = datetime(2026, 1, 1)
    end = datetime(2026, 4, 1)  # 90 days; 5m max=30 → 3 chunks
    out = client.candles("NSE-RELIANCE", interval="5m", start=start, end=end)

    # Three calls were made.
    assert fake_api.get_historical_candles.call_count == 3
    # Even though each chunk returned 2 rows (6 total), the boundary duplicates
    # collapse: 3 distinct chunk-start-ts + 2 distinct boundary-ts = 5 candles.
    # Verify dedupe: no two candles share a ts_epoch.
    ts_epochs = [c.ts_epoch for c in out]
    assert len(ts_epochs) == len(set(ts_epochs))
    # And the result is sorted ascending.
    assert ts_epochs == sorted(ts_epochs)


def test_candles_rejects_unknown_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daity.data.groww.GrowwAPI", MagicMock())
    cfg = GrowwConfig(api_key="eyJfake", api_secret=None, totp_secret=None)
    client = GrowwClient(cfg)
    with pytest.raises(ValueError, match="Unknown interval"):
        client.candles("NSE-X", interval="2m", start=datetime(2026, 1, 1),
                       end=datetime(2026, 1, 2))


def test_interval_map_covers_design_resolutions() -> None:
    """The four resolutions DESIGN cares about must all map to a Groww interval."""
    for label in ("5m", "15m", "60m", "day"):
        assert label in INTERVAL_MAP


def test_candle_dataclass_immutable() -> None:
    c = Candle(ts_epoch=1, open=1.0, high=2.0, low=0.5, close=1.5, volume=10, open_interest=None)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        c.open = 2.0  # type: ignore[misc]
