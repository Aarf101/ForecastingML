"""Tests de l'ingestion (producteur/consumer) — aucun Docker requis."""

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from retail_forecast.data.simulate import simulate_sales
from retail_forecast.ingest.producer import (
    decode,
    encode,
    record_from_row,
    records_from_frame,
)
from retail_forecast.ingest.storage import chunks, raw_key, to_gzip_jsonl
from retail_forecast.ingest.to_s3 import LocalSink, batch_key, run


def test_record_from_row_types_and_date():
    rec = record_from_row(
        {
            "date": pd.Timestamp("2024-03-05"),
            "store_id": "S01",
            "item_id": "I002",
            "category": "grocery",
            "unit_price": 4.25,
            "promo": 1,
            "is_holiday": 0,
            "lead_time_days": 3,
            "stock_on_hand": 12,
            "stockout": 0,
            "units": 7,
        },
        event_time="2026-01-01T00:00:00+00:00",
    )
    assert rec["date"] == "2024-03-05"
    assert rec["units"] == 7 and isinstance(rec["units"], int)
    assert rec["unit_price"] == pytest.approx(4.25)
    assert rec["store_id"] == "S01"
    assert rec["event_time"] == "2026-01-01T00:00:00+00:00"


def test_record_from_row_handles_nan():
    rec = record_from_row({"date": "2024-01-01", "units": float("nan"), "store_id": None})
    assert rec["units"] == 0
    assert rec["store_id"] is None


def test_encode_decode_roundtrip():
    rec = {"date": "2024-01-01", "units": 3, "store_id": "S01", "promo": 0}
    assert decode(encode(rec)) == rec


def test_records_from_frame_respects_limit():
    df = simulate_sales(n_stores=1, n_items=1, days=10, seed=1)
    assert len(list(records_from_frame(df))) == 10
    assert len(list(records_from_frame(df, limit=3))) == 3


def test_raw_key_is_partitioned_by_ingest_date():
    key = raw_key("2024-05-06", "batch-1")
    assert key == "raw/sales/ingest_date=2024-05-06/batch-1.json.gz"


def test_batch_key_uses_utc_date():
    now = datetime(2024, 5, 6, 23, 59, 59, tzinfo=timezone.utc)
    ingest_date, key = batch_key(now, seq=2)
    assert ingest_date == "2024-05-06"
    assert key.endswith("-0002.json.gz")
    assert "ingest_date=2024-05-06" in key


def test_to_gzip_jsonl_is_readable_jsonl():
    body = to_gzip_jsonl([{"a": 1}, {"b": 2}])
    lines = gzip.decompress(body).decode("utf-8").strip().split("\n")
    assert [json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]


def test_chunks_keeps_remainder():
    assert list(chunks(range(7), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    with pytest.raises(ValueError):
        list(chunks(range(3), 0))


class _Msg:
    def __init__(self, value):
        self.value = value


class _FakeConsumer:
    """Itère des messages et compte les commits (pas de broker)."""

    def __init__(self, records):
        self._records = records
        self.commits = 0

    def __iter__(self):
        return iter(_Msg(r) for r in self._records)

    def commit(self):  # pragma: no cover - appelé par run()
        self.commits += 1


def test_run_writes_expected_batches_and_commits(tmp_path):
    records = [{"i": i, "units": i} for i in range(7)]
    consumer = _FakeConsumer(records)
    sink = LocalSink(tmp_path)
    stats = run(
        consumer,
        sink,
        batch_size=3,
        flush_seconds=3600,
        now_fn=lambda: datetime(2024, 5, 6, 12, 0, 0, tzinfo=timezone.utc),
        log=lambda *_: None,
    )

    assert stats.messages == 7
    assert stats.batches == 3            # 3 + 3 + 1
    assert consumer.commits == 3
    assert stats.bytes > 0

    written = sorted(Path(tmp_path).rglob("*.json.gz"))
    assert len(written) == 3
    recovered = []
    for path in written:
        recovered += [
            json.loads(line)
            for line in gzip.decompress(path.read_bytes()).decode().strip().split("\n")
        ]
    assert recovered == records          # aucun message perdu, aucun dupliqué


def test_json_value_serializer_supports_both_protocols():
    import json as _json

    from retail_forecast.ingest.producer import JsonValueSerializer

    ser = JsonValueSerializer()
    record = {"store_id": "S01", "units": 4}
    assert _json.loads(ser.serialize("sales.events", [], record)) == record
    assert _json.loads(ser(record)) == record
    assert ser.close() is None


def test_json_value_deserializer_supports_both_protocols():
    import json as _json

    from retail_forecast.ingest.to_s3 import JsonValueDeserializer

    deser = JsonValueDeserializer()
    payload = _json.dumps({"store_id": "S01", "units": 4}).encode()
    expected = {"store_id": "S01", "units": 4}
    assert deser.deserialize("sales.events", [], payload) == expected  # kafka-python 3
    assert deser(payload) == expected  # kafka-python 2.x
    assert deser.close() is None


def test_run_respects_max_messages(tmp_path):
    consumer = _FakeConsumer([{"i": i} for i in range(100)])
    stats = run(
        consumer,
        LocalSink(tmp_path),
        batch_size=10,
        flush_seconds=3600,
        max_messages=25,
        log=lambda *_: None,
    )
    assert stats.messages == 25
    assert sum(
        len(gzip.decompress(p.read_bytes()).decode().strip().split("\n"))
        for p in Path(tmp_path).rglob("*.json.gz")
    ) == 25
