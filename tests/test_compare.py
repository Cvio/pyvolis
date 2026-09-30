"""Port of the tests in Rust `ring.rs` and `compare.rs`."""

import numpy as np

from pyvolis import compare
from pyvolis.asr import AsrError, AsrResult, Memory
from pyvolis.audio import SAMPLE_RATE
from pyvolis.ring import UtteranceRing


def test_the_ring_keeps_the_newest_and_drops_the_oldest():
    ring = UtteranceRing(3)
    for i in range(5):
        ring.push(i * 1000, np.zeros(10, np.float32))
    assert len(ring) == 3
    assert [u.index for u in ring.recent()] == [5, 4, 3], "newest first, oldest evicted"


def test_indexes_count_every_utterance_not_just_the_retained_ones():
    ring = UtteranceRing(1)
    ring.push(0, np.zeros(1, np.float32))
    ring.push(1, np.zeros(1, np.float32))
    assert ring.push(2, np.zeros(1, np.float32)).index == 3


def test_duration_comes_from_the_sample_count():
    ring = UtteranceRing(1)
    assert ring.push(0, np.zeros(SAMPLE_RATE * 3 // 2, np.float32)).duration_ms() == 1500


class Fake:
    def __init__(self, name, reply="", fail=False):
        self.name, self.reply, self.fail = name, reply, fail

    def transcribe(self, audio, language):
        if self.fail:
            raise AsrError("no model loaded")
        return AsrResult(self.reply)

    def memory(self):
        return Memory()

    def close(self):
        pass


class FakeEngines(compare.Engines):
    def __init__(self, fakes):
        self._fakes = fakes

    def each(self):
        for fake in self._fakes:
            yield fake.name, fake, False


def utterance():
    return UtteranceRing(1).push(0, np.zeros(SAMPLE_RATE, np.float32))


def test_every_engine_gets_the_same_audio():
    c = compare.run_all(FakeEngines([Fake("parakeet", "hola"), Fake("whisper", "Hola.")]), utterance(), "es")
    assert c.segment_ms == 1000
    assert [r.text for r in c.runs] == ["hola", "Hola."]
    assert all(r.ok for r in c.runs)


def test_the_table_marks_the_configured_engine_and_keeps_the_duration():
    c = compare.run_all(FakeEngines([Fake("parakeet", "hola"), Fake("whisper", "")]), utterance(), "es")
    table = compare.format_table(c, "whisper")
    assert "1000 ms of audio" in table
    assert "* whisper" in table and "* parakeet" not in table
    assert "(no text)" in table


def test_one_failing_engine_does_not_lose_the_others():
    c = compare.run_all(FakeEngines([Fake("broken", fail=True), Fake("working", "hola")]), utterance(), "es")
    assert not c.runs[0].ok and "no model loaded" in c.runs[0].text
    assert c.runs[1].ok


def test_the_table_is_rusts_line_for_line():
    """parity/asr.py parses Rust's table; pyvolis's must read the same way."""
    c = compare.run_all(FakeEngines([Fake("parakeet", "hola")]), utterance(), "es")
    assert compare.format_table(c, "parakeet").splitlines() == [
        "utterance 1 at 0 ms - 1000 ms of audio",
        "  * parakeet       0 ms (0.00x realtime)",
        "      hola",
    ]
