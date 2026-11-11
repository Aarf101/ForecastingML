"""Tests du pont Kafka -> Firehose (DirectPut), sans broker ni compte AWS."""

from __future__ import annotations

import pytest

from retail_forecast.ingest import kafka_to_firehose as kf


# --- doublures ---------------------------------------------------------------


class FakeFirehose:
    """Compte les appels ; peut refuser les ``fail_first`` premiers lots."""

    def __init__(self, fail_first: int = 0) -> None:
        self.calls: list[list] = []
        self.fail_first = fail_first

    def put_record_batch(self, DeliveryStreamName, Records):  # noqa: N803
        self.calls.append(list(Records))
        if self.fail_first > 0:
            self.fail_first -= 1
            return {
                "FailedPutCount": len(Records),
                "RequestResponses": [
                    {"ErrorCode": "ServiceUnavailable", "ErrorMessage": "throttlé"}
                    for _ in Records
                ],
            }
        return {
            "FailedPutCount": 0,
            "RequestResponses": [{"RecordId": f"r{i}"} for i in range(len(Records))],
        }


class Msg:
    def __init__(self, value):
        self.value = value


class FakeConsumer:
    """Itère des messages et compte les commits."""

    def __init__(self, values: list) -> None:
        self._values = values
        self.commits = 0
        self.closed = False

    def __iter__(self):
        for value in self._values:
            yield Msg(value)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Les reprises dorment : on neutralise l'attente pour garder le test à ~0 s."""
    monkeypatch.setattr(kf.time, "sleep", lambda *_a, **_k: None)


# --- sérialisation -----------------------------------------------------------


def test_records_are_jsonl_with_trailing_newline():
    recs = kf.to_records([{"a": 1}, {"b": 2}])
    assert [r["Data"] for r in recs] == [b'{"a":1}\n', b'{"b":2}\n']


def test_bytes_value_gets_newline_added():
    """Firehose concatène sans délimiteur : le pont doit le fournir."""
    recs = kf.to_records([b'{"a":1}'])
    assert recs[0]["Data"] == b'{"a":1}\n'


def test_newline_not_duplicated():
    recs = kf.to_records([b'{"a":1}\n'])
    assert recs[0]["Data"] == b'{"a":1}\n'


def test_oversized_record_is_rejected():
    big = {"x": "a" * (kf.MAX_RECORD_BYTES + 10)}
    with pytest.raises(ValueError, match="limite Firehose"):
        kf.to_records([big])


# --- envoi d'un lot ----------------------------------------------------------


def test_put_batch_rejects_more_than_500_records():
    with pytest.raises(ValueError, match="limite Firehose"):
        kf.put_batch(FakeFirehose(), "s", [{"i": i} for i in range(kf.MAX_RECORDS + 1)])


def test_put_batch_reports_accepted_and_failed():
    fh = FakeFirehose(fail_first=1)
    ok, failed, err = kf.put_batch(fh, "s", [{"i": 1}, {"i": 2}])
    assert ok == 0
    assert len(failed) == 2
    assert err == "throttlé"


def test_put_batch_uses_the_stream_name():
    class Recorder(FakeFirehose):
        def put_record_batch(self, DeliveryStreamName, Records):  # noqa: N803
            self.name = DeliveryStreamName
            return super().put_record_batch(DeliveryStreamName, Records)

    fh = Recorder()
    kf.put_batch(fh, "retail-sales-direct", [{"i": 1}])
    assert fh.name == "retail-sales-direct"


# --- boucle de pont ----------------------------------------------------------


def test_run_batches_by_size_and_commits_once_per_batch():
    consumer = FakeConsumer([{"i": i} for i in range(1200)])
    fh = FakeFirehose()

    stats = kf.run(consumer, fh, "s", batch_size=500, flush_seconds=999)

    assert [len(c) for c in fh.calls] == [500, 500, 200]
    assert stats.messages == 1200
    assert stats.sent == 1200
    assert consumer.commits == 3


def test_run_respects_max_messages():
    consumer = FakeConsumer([{"i": i} for i in range(100)])
    fh = FakeFirehose()

    stats = kf.run(consumer, fh, "s", batch_size=10, flush_seconds=999, max_messages=25)

    assert stats.messages == 25
    assert sum(len(c) for c in fh.calls) == 25


def test_run_retries_failed_records_then_succeeds():
    consumer = FakeConsumer([{"i": i} for i in range(10)])
    fh = FakeFirehose(fail_first=1)

    stats = kf.run(consumer, fh, "s", batch_size=10, flush_seconds=999)

    assert stats.retries == 1
    assert stats.sent == 10
    assert consumer.commits == 1


def test_run_gives_up_after_max_retries():
    consumer = FakeConsumer([{"i": 1}])
    fh = FakeFirehose(fail_first=99)

    with pytest.raises(RuntimeError, match="refusé"):
        kf.run(consumer, fh, "s", batch_size=10, flush_seconds=999, max_retries=2)

    assert consumer.commits == 0


def test_no_commit_when_firehose_raises():
    class Boom:
        def put_record_batch(self, **_kw):
            raise RuntimeError("réseau")

    consumer = FakeConsumer([{"i": 1}])
    with pytest.raises(RuntimeError, match="réseau"):
        kf.run(consumer, Boom(), "s", batch_size=10, flush_seconds=999, max_messages=1)

    assert consumer.commits == 0
