"""LocalAgreement, with a scripted recognizer: no models needed."""

import numpy as np

from pyvolis.asr import AsrResult, Word
from pyvolis.asr.streaming import LocalAgreement, agreed_prefix

SECOND = 16_000


class Scripted:
    """Answers each call with the next scripted hypothesis, and remembers how
    much audio it was given."""

    def __init__(self, *hypotheses, timed: bool = False) -> None:
        self.hypotheses = list(hypotheses)
        self.timed = timed
        self.lengths: list[int] = []

    def transcribe(self, audio, language, timestamps=True):
        self.lengths.append(len(audio))
        text = self.hypotheses.pop(0)
        words = None
        if self.timed:  # one word per second of the audio it was given
            words = [Word(t, float(i), float(i + 1)) for i, t in enumerate(text.split())]
        return AsrResult(text, words, None, language, 0.1)


def audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SECOND), np.float32)


def test_agreement_ignores_case_and_punctuation():
    assert agreed_prefix(["Hola,", "buenos", "días"], ["hola", "buenos", "dias."]) == 2
    assert agreed_prefix([], ["hola"]) == 0


def test_only_what_two_passes_agree_on_is_committed():
    la = LocalAgreement(Scripted("Se me", "Se me murió el", "Se me murió el perro ayer"), "es")
    first = la.update(audio(1))
    assert first.text == "" and first.provisional == "Se me", "one pass alone commits nothing"
    second = la.update(audio(2))
    assert second.text == "Se me" and second.provisional == "murió el"
    third = la.update(audio(3))
    assert third.text == "murió el" and third.provisional == "perro ayer"


def test_a_word_the_model_changes_its_mind_about_is_not_committed():
    la = LocalAgreement(Scripted("Si me", "Se me murió", "Se me murió el"), "es")
    assert la.update(audio(1)).text == ""
    second = la.update(audio(2))
    assert second.text == "" and second.provisional == "Se me murió", "the first word changed: nothing agreed"
    assert la.update(audio(3)).text == "Se me murió"


def test_committed_text_never_changes_even_if_a_later_pass_disagrees():
    la = LocalAgreement(Scripted("Hola buenos", "Hola buenos días", "Ola buenos días a todos"), "es")
    la.update(audio(1))
    assert la.update(audio(2)).text == "Hola buenos"
    third = la.update(audio(3))
    assert third.text == "" and third.provisional == "días a todos", "what was committed stays as it was"


def test_finishing_commits_everything_that_is_left():
    la = LocalAgreement(Scripted("Hola buenos", "Hola buenos días", "Hola buenos días a todos."), "es")
    la.update(audio(1))
    assert la.update(audio(2)).text == "Hola buenos"
    final = la.finish(audio(3))
    assert final.text == "días a todos." and final.provisional == ""
    assert la.passes == 3


def test_an_utterance_too_short_for_any_pass_is_committed_whole_at_the_end():
    la = LocalAgreement(Scripted("Sí."), "es")
    assert la.finish(audio(0.8)).text == "Sí."


def test_the_buffer_is_trimmed_at_a_committed_sentence_end_once_it_is_long():
    """With word timings, the audio before a committed sentence end is dropped
    from later passes, so a pass never grows towards Whisper's 30 seconds."""
    sentence = "uno dos tres cuatro cinco seis siete ocho nueve diez once."  # 11 words, 11 s
    recognizer = Scripted(sentence, sentence + " doce", "doce trece", timed=True)
    la = LocalAgreement(recognizer, "es")
    la.update(audio(11))
    second = la.update(audio(12))
    assert second.text == sentence
    assert la.offset == 11 * SECOND, "cut at the end of the sentence's last word"
    third = la.update(audio(14))
    assert recognizer.lengths == [11 * SECOND, 12 * SECOND, 3 * SECOND], "the third pass heard only what follows"
    assert third.text == "doce" and third.provisional == "trece"
    assert third.committed[0].start == 11.0, "times stay on the utterance's timeline after a trim"


def test_without_word_timings_nothing_is_trimmed():
    sentence = "uno dos tres cuatro cinco seis siete ocho nueve diez once."
    la = LocalAgreement(Scripted(sentence, sentence + " doce", sentence + " doce trece"), "es")
    la.update(audio(11))
    la.update(audio(12))
    assert la.offset == 0
    assert la.update(audio(14)).text == "doce"


# ---------------------------------------------------------------- through the pipeline


def test_the_pipeline_streams_commits_sentences_early_and_ends_consistent(monkeypatch):
    """A fixture utterance through the real detector, with a recognizer that
    'hears' two words per second of audio: provisional text while the speech
    goes on, sentences committed before the utterance ends, and a final
    transcript that is exactly those sentences."""
    import json
    import queue

    import pytest

    from pyvolis import asr, paths
    from pyvolis import events as ev
    from pyvolis.config import Config, PyvolisConfig
    from pyvolis.filesource import read_16k_mono
    from pyvolis.pipeline import ArraySource, Options, Pipeline, run_to_end

    root = paths.app_root()
    folder = root / "tests" / "fixtures" / "fleurs" / "es_419"
    if not (folder / "refs.jsonl").is_file() or not paths.vad_model_file(root).is_file():
        pytest.skip("needs the VAD model and tests/fetch-fixtures.ps1")
    clip = read_16k_mono(folder / json.loads((folder / "refs.jsonl").read_text(encoding="utf-8").splitlines()[0])["file"])
    script = "uno dos tres cuatro cinco. seis siete ocho nueve diez. once doce trece catorce quince. " \
             "dieciséis diecisiete dieciocho diecinueve veinte. veintiuno veintidós veintitrés".split()

    class TwoWordsASecond:
        name = "scripted"

        def transcribe(self, audio, language, timestamps=True):
            heard = min(len(script), int(len(audio) / SECOND * 2))
            return AsrResult(" ".join(script[:heard]), None, None, language, 0.01)

        def prepare(self, language):
            pass

        def memory(self):
            return asr.Memory()

        def close(self):
            pass

    monkeypatch.setattr(asr, "load", lambda engine: TwoWordsASecond())
    config = Config.parse('[asr]\nengine = "parakeet-tdt-0.6b-v3-int8"\n[languages]\nsource = "es"\n')
    events: queue.Queue = queue.Queue()
    silence = np.zeros(SECOND, np.float32)
    source = ArraySource(np.concatenate([silence, clip, silence]))
    out = run_to_end(Pipeline(root, config, Options(translate=False, streaming=True), events, source,
                              PyvolisConfig()), events)
    assert not out.errors, out.errors
    kinds = [type(e).__name__ for e in out.events]
    partials, finals, sentences = out.of(ev.Partial), out.of(ev.Final), out.of(ev.SentenceMsg)
    assert len(partials) >= 5, "a pass for about every second of speech"
    assert any(p.text for p in partials), "provisional text was shown"
    assert len(finals) == 1
    first_sentence, final_at = kinds.index("SentenceMsg"), kinds.index("Final")
    assert first_sentence < final_at, "a sentence was committed while the utterance was still being spoken"
    assert " ".join(s.text for s in sentences) == finals[0].text, "the transcript is exactly its sentences"
    assert sentences[0].text == "uno dos tres cuatro cinco."
    assert [s.id for s in sentences] == [f"1.{n}" for n in range(1, len(sentences) + 1)]
    stats = out.of(ev.Summary)[-1].stats
    assert stats["asr_passes"] == len(partials) + 1, "every pass is counted in the cost"


def test_segment_mode_makes_one_pass_and_shows_nothing_provisional(monkeypatch):
    import queue

    from pyvolis import asr, paths
    from pyvolis import events as ev
    from pyvolis.config import Config, PyvolisConfig
    from pyvolis.pipeline import ArraySource, Options, Pipeline, run_to_end

    calls = []

    class Counts:
        name = "counts"

        def transcribe(self, audio, language, timestamps=True):
            calls.append(len(audio))
            return AsrResult("hola.", None, None, language, 0.01)

        def prepare(self, language):
            pass

        def memory(self):
            return asr.Memory()

        def close(self):
            pass

    import sys as _sys

    _sys.path.insert(0, str(paths.app_root() / "scripts"))
    import pytest

    try:
        from vad_cuts import conversation

        audio_in = conversation()[0][: 16 * SECOND]
    except SystemExit:
        pytest.skip("needs tests/fetch-fixtures.ps1")
    monkeypatch.setattr(asr, "load", lambda engine: Counts())
    config = Config.parse('[asr]\nengine = "parakeet-tdt-0.6b-v3-int8"\n[languages]\nsource = "es"\n')
    events: queue.Queue = queue.Queue()
    out = run_to_end(Pipeline(paths.app_root(), config, Options(translate=False, streaming=False), events,
                              ArraySource(audio_in), PyvolisConfig()), events)
    # One pass per utterance. (A lone "hola." for ten seconds of speech is
    # dropped by the too-few-words guard: still one pass.)
    assert out.of(ev.Partial) == []
    assert len(calls) == len(out.of(ev.Final)) + len(out.of(ev.Dropped)) >= 1
