"""Holding speech for revision ([context] hold_speech): with revision and the
voice both on, a short sentence waits for the next one before it is spoken,
so that it is spoken as revised. The translation thread with a scripted
translator and a voice that only records; no models."""

import queue
import time
from types import SimpleNamespace

from pyvolis import events as ev, sentences
from pyvolis.config import PyvolisConfig
from pyvolis.pipeline import Options, Route, Stats, TranslationThread

from test_revision import Scripted

DRIVE = {
    "Yo manejo.": "I manage.",
    "Mi carro está afuera.": "My car is outside.",
    "Yo manejo. Mi carro está afuera.": "I'll drive. My car is outside.",
    "Vámonos.": "Let's go.",
    "Mi carro está afuera. Vámonos.": "My car is outside. Let's go.",
    "Yo manejo. Mi carro está afuera. Vámonos.": "I'll drive. My car is outside. Let's go.",
}
LONG = "El sistema de reservas no ha respondido a ninguna de nuestras llamadas desde las nueve de la mañana."


class Voice:
    def __init__(self) -> None:
        self.said: list[tuple[str, str, float]] = []

    def submit(self, sentence_id, text, language, cut_at, folder="", generation=None) -> None:
        self.said.append((sentence_id, text, time.monotonic()))

    def texts(self) -> list[str]:
        return [text for _, text, _ in self.said]


def thread(answers, hold_speech=True, hold_s=5.0, mode="revision", lossless=True):
    py = PyvolisConfig()
    py.context.mode, py.context.hold_speech, py.context.hold_speech_s = mode, hold_speech, hold_s
    seen: queue.Queue = queue.Queue()
    pipeline = SimpleNamespace(pyconfig=py, options=Options(), stats=Stats(), emit=seen.put, glossary=[],
                               generation=0, translator_name="scripted", source=SimpleNamespace(lossless=lossless))
    translator = Scripted(answers)
    translator.close = lambda: None
    voice = Voice()
    return TranslationThread(pipeline, translator, "en", voice), voice, pipeline, seen


def say(t: TranslationThread, n: int, text: str, generation: int = 0) -> None:
    t.submit(sentences.Sentence(f"{n}.1", n, text, float(n), float(n) + 0.5, False), "es", 0.0,
             Route(generation=generation))


def settle(t: TranslationThread) -> None:
    while not t.queue.empty():
        time.sleep(0.01)
    time.sleep(0.1)


def test_a_short_sentence_is_spoken_as_revised_once_the_next_is_heard():
    t, voice, pipeline, seen = thread(DRIVE)
    say(t, 1, "Yo manejo.")
    settle(t)
    assert voice.said == [], "held: the next sentence may change it"
    say(t, 2, "Mi carro está afuera.")
    settle(t)
    assert voice.texts() == ["I'll drive."], "spoken as revised; the newest is held in its turn"
    t.finish(wait=True)
    assert voice.texts() == ["I'll drive.", "My car is outside."]
    assert pipeline.stats.revisions == 1
    revised = [e for e in list(seen.queue) if isinstance(e, ev.Revised)]
    assert [(e.id, e.new) for e in revised] == [("1.1", "I'll drive.")]


def test_what_has_been_spoken_is_not_revised_again():
    t, voice, pipeline, _ = thread(DRIVE)
    for n, text in enumerate(("Yo manejo.", "Mi carro está afuera.", "Vámonos."), 1):
        say(t, n, text)
        settle(t)
    t.finish(wait=True)
    assert voice.texts() == ["I'll drive.", "My car is outside.", "Let's go."]
    assert pipeline.stats.revisions == 1, "sentence 1 was spoken before the third pass"


def test_a_held_sentence_is_spoken_when_nothing_follows_in_time():
    t, voice, _, _ = thread(DRIVE, hold_s=0.3)
    began = time.monotonic()
    say(t, 1, "Yo manejo.")
    while not voice.said and time.monotonic() - began < 3:
        time.sleep(0.01)
    assert voice.texts() == ["I manage."]
    assert 0.25 <= voice.said[0][2] - began < 1.5
    t.finish(wait=True)
    assert voice.texts() == ["I manage."], "and only once"


def test_the_end_of_a_turn_speaks_the_held_sentence_at_once():
    t, voice, _, _ = thread(DRIVE, hold_s=30.0)
    say(t, 1, "Yo manejo.")
    t.end_of_turn()
    settle(t)
    assert voice.texts() == ["I manage."]
    t.finish(wait=True)


def test_a_long_sentence_never_waits_and_comes_after_the_one_held_before_it():
    t, voice, _, _ = thread({**DRIVE, LONG: "The booking system has not answered.",
                             f"Yo manejo. {LONG}": "I'll drive. The booking system has not answered."}, hold_s=30.0)
    say(t, 1, LONG)
    settle(t)
    assert voice.texts() == ["The booking system has not answered."], "never revised, so never held"
    say(t, 2, "Yo manejo.")
    say(t, 3, LONG)
    settle(t)
    assert voice.texts()[1:] == ["I'll drive.", "The booking system has not answered."], "revised, and in order"
    t.finish(wait=True)


def test_stopping_at_once_drops_the_held_sentence():
    t, voice, _, _ = thread(DRIVE, hold_s=30.0)
    say(t, 1, "Yo manejo.")
    settle(t)
    t.finish(wait=False)
    assert voice.said == []


def test_off_by_default_speech_is_never_delayed_and_nothing_spoken_is_revised():
    assert PyvolisConfig().context.hold_speech is False
    t, voice, pipeline, _ = thread(DRIVE, hold_speech=False)
    say(t, 1, "Yo manejo.")
    settle(t)
    assert voice.texts() == ["I manage."]
    say(t, 2, "Mi carro está afuera.")
    t.finish(wait=True)
    assert voice.texts() == ["I manage.", "My car is outside."]
    assert pipeline.stats.revisions == 0


def test_without_revision_the_setting_does_nothing():
    t, voice, _, _ = thread(DRIVE, mode="carry")
    say(t, 1, "Yo manejo.")
    settle(t)
    assert voice.texts() == ["I manage."]
    t.finish(wait=True)
