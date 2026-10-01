"""The hallucination guards (new in pyvolis)."""

from pathlib import Path

import numpy as np
import pytest

from pyvolis import paths
from pyvolis.asr import guards as g
from pyvolis.filesource import read_16k_mono

ROOT = paths.app_root()
PHRASES = g.load_phrases(paths.hallucinations_file(ROOT))
MODEL = paths.vad_model_file(ROOT)
FIXTURES = ROOT / "tests" / "fixtures" / "fleurs" / "es_419"


def guards(**kw) -> g.Guards:
    return g.Guards(PHRASES, **{"vad_probability": False, **kw})


def test_a_repeated_phrase_is_kept_once():
    assert g.collapse_repeats("gracias gracias gracias gracias")[0] == "gracias"
    text, why = g.collapse_repeats("Bueno, vamos a ver. Vamos a ver. Vamos a ver. Ya está.")
    assert text == "Bueno, vamos a ver. Ya está." and "3 times" in why


def test_twice_is_not_a_hallucination():
    assert g.collapse_repeats("no no, eso no") == ("no no, eso no", None)
    assert g.collapse_repeats("Sí, sí.")[1] is None


def test_the_shipped_phrase_list_parses_and_covers_the_languages():
    assert {"es", "en", "ar"} <= PHRASES.keys()


@pytest.mark.parametrize(
    "text,language",
    [
        ("Gracias por ver el video.", "es-MX"),
        ("Subtítulos realizados por la comunidad de Amara.org", "es"),
        ("Thank you for watching!", "en"),
        ("ترجمة نانسي قنقر", "ar"),
    ],
)
def test_stock_phrases_are_dropped_with_the_reason(text, language):
    verdict = guards().check(text, language)
    assert verdict.dropped and verdict.text == "" and verdict.reasons


def test_a_trailing_stock_phrase_is_cut_and_the_rest_kept_exactly():
    verdict = guards().check("¿Dónde está la estación? Repito.", "es")
    assert verdict.text == "¿Dónde está la estación?" and not verdict.dropped


def test_real_speech_and_its_punctuation_pass_untouched():
    text = "¿Se me murió el perro, no? Oigan, ¿quién maneja?"
    verdict = guards().check(text, "es")
    assert verdict.text == text and not verdict.changed


def test_a_phrase_list_is_per_language():
    assert not guards().check("Thank you.", "es").dropped


def test_far_too_few_words_for_the_speech_is_dropped():
    """6 s of speech-shaped noise came out of Whisper as a lone "y"."""
    verdict = guards().check("y", "es", speech_seconds=6.2)
    assert verdict.dropped and "too few" in verdict.reasons[0]
    assert not guards().check("Sí.", "es", speech_seconds=1.2).dropped, "a short answer is short speech"
    assert not guards().check("Bueno, vamos.", "es", speech_seconds=5.0).dropped, "two words in 5 s is slow, not noise"
    assert not guards(sparse=False).check("y", "es", speech_seconds=6.2).dropped
    assert guards().check("y", "es", np.zeros(16000 * 7, np.float32)).dropped, "the audio's length stands in"


def test_each_guard_can_be_switched_off():
    off = guards(repeats=False, stock_phrases=False)
    assert off.check("Gracias por ver el video.", "es").text == "Gracias por ver el video."
    assert off.check("sí sí sí sí", "es").text == "sí sí sí sí"


def test_a_bad_phrase_file_names_itself(tmp_path):
    bad = tmp_path / "hallucinations.toml"
    bad.write_text("[es]\nwhole = [\"x\"]\nfull = []\n", encoding="utf-8")
    with pytest.raises(g.GuardError, match=str(bad.absolute()).replace("\\", "\\\\")):
        g.load_phrases(bad)


@pytest.mark.skipif(not MODEL.is_file() or not (FIXTURES / "refs.jsonl").is_file(),
                    reason="needs the VAD model and tests/fetch-fixtures.ps1")
def test_the_speech_scorer_tells_speech_from_noise():
    scorer = g.SpeechScorer(MODEL, 0.8)
    clip = next(FIXTURES.glob("*.wav"))
    assert scorer.has_clear_speech(read_16k_mono(clip))
    rng = np.random.default_rng(1)
    assert not scorer.has_clear_speech((rng.standard_normal(48000) * 0.2).astype(np.float32))
    assert not scorer.has_clear_speech(np.zeros(48000, np.float32))
    verdict = g.Guards(PHRASES, scorer=scorer).check("Hola.", "es", np.zeros(16000, np.float32))
    assert verdict.dropped and "speech probability" in verdict.reasons[0]
