"""Port of the tests in Rust `shared.rs`: which side pressed, and so where
the turn's words go."""

from pathlib import Path

import pytest

from pyvolis import shared
from pyvolis.config import Shared
from pyvolis.models import Engine, ModelFile
from pyvolis.shared import LEFT, RIGHT, Direction, SharedError, direction


def engine(dir_name: str, backend: str, languages: list[str], present: bool = True, **more) -> Engine:
    return Engine(dir_name, Path(dir_name), f"{dir_name} (test)", "segment", backend, languages,
                  files=[ModelFile("model", "model.onnx", Path("model.onnx"), present)], **more)


def whisper() -> Engine:
    return engine("whisper", "whisper", ["en", "es"])


def voice(dir_name: str, language: str, **more) -> Engine:
    return engine(dir_name, "vits", [language], **more)


def voices() -> list[Engine]:
    return [voice("piper-en", "en"), voice("piper-es-es", "es"), voice("piper-es-mx", "es")]


def settings(**changes) -> Shared:
    """English on the left, Spanish on the right, the defaults."""
    s = Shared()
    for name, value in changes.items():
        setattr(s, name, value)
    return s


def test_the_left_key_translates_left_into_right():
    r = direction(LEFT, settings(), [whisper()], "whisper", voices())
    assert r.direction == Direction(LEFT, "whisper", "en", "es", "piper-es-es")
    assert r.warnings == []


def test_the_right_key_translates_right_into_left():
    r = direction(RIGHT, settings(), [whisper()], "whisper", voices())
    assert (r.direction.source, r.direction.target) == ("es", "en")
    assert r.direction.voice == "piper-en", "a voice in the LEFT person's language"


def test_a_voice_is_chosen_by_folder_so_two_spanish_voices_can_be_told_apart():
    r = direction(LEFT, settings(left_voice="piper-es-mx"), [whisper()], "whisper", voices())
    assert r.direction.voice == "piper-es-mx"


def test_a_missing_voice_falls_back_and_says_so():
    r = direction(LEFT, settings(left_voice="deleted-voice"), [whisper()], "whisper", voices())
    assert r.direction.voice == "piper-es-es", "the first Spanish voice"
    assert "deleted-voice" in " ".join(r.warnings)
    # A voice in the wrong language is no better than a missing one.
    r = direction(LEFT, settings(left_voice="piper-en"), [whisper()], "whisper", voices())
    assert r.direction.voice == "piper-es-es" and r.warnings


def test_a_recognizer_that_guesses_the_language_is_allowed_with_a_note():
    parakeet = engine("parakeet", "nemo_transducer", ["en", "es"])
    r = direction(LEFT, settings(), [parakeet], "parakeet", voices())
    assert r.direction.asr == "parakeet"
    assert any("decides the language itself" in w for w in r.warnings)


def test_each_side_gets_its_own_recognizer():
    # English on the left with Parakeet, Spanish on the right with a
    # Spanish-only Whisper: the example from dialect-per-side.md.
    both = [engine("parakeet", "nemo_transducer", ["en"]), engine("whisper-es", "whisper", ["es"])]
    s = settings(left_asr="parakeet", right_asr="whisper-es")
    assert direction(LEFT, s, both, "", voices()).direction.asr == "parakeet"
    assert direction(RIGHT, s, both, "", voices()).direction.asr == "whisper-es"
    # Left unset: the best match for English, never the Spanish-only model.
    s.left_asr = ""
    assert direction(LEFT, s, both, "whisper-es", voices()).direction.asr == "parakeet"


def test_a_configured_recognizer_is_never_swapped_for_another():
    with pytest.raises(SharedError, match="deleted-model"):
        direction(LEFT, settings(left_asr="deleted-model"), [whisper()], "whisper", voices())


def test_a_recognizer_missing_one_language_refuses_only_that_side():
    english_only = [engine("whisper-en", "whisper", ["en"])]
    direction(LEFT, settings(), english_only, "whisper-en", voices())
    with pytest.raises(SharedError, match='"es"'):
        direction(RIGHT, settings(), english_only, "whisper-en", voices())


def test_no_voice_for_the_target_language_is_refused():
    with pytest.raises(SharedError, match='"es"'):
        direction(LEFT, settings(), [whisper()], "whisper", [voice("piper-en", "en")])


def test_broken_models_and_bad_settings_are_refused_with_a_reason():
    broken = engine("whisper", "whisper", ["en", "es"], present=False)
    with pytest.raises(SharedError):
        direction(LEFT, settings(), [broken], "whisper", voices())
    with pytest.raises(SharedError):
        direction(LEFT, settings(), [], "", voices())
    with pytest.raises(SharedError, match="different language"):
        direction(LEFT, settings(right_language="en"), [whisper()], "whisper", voices())
    # A broken voice is never chosen, even when it is the configured one.
    vs = voices()
    vs[2] = engine("piper-es-mx", "vits", ["es"], present=False)
    r = direction(LEFT, settings(left_voice="piper-es-mx"), [whisper()], "whisper", vs)
    assert r.direction.voice == "piper-es-es" and r.warnings


def test_a_side_set_to_a_variety_hears_its_language_and_an_unknown_tag_is_refused():
    r = direction(RIGHT, settings(right_language="es-MX"), [whisper()], "whisper", voices())
    assert r.direction.source == "es-MX"
    with pytest.raises(SharedError, match="xx-YY"):
        direction(RIGHT, settings(right_language="xx-YY"), [whisper()], "whisper", voices())


def test_the_picker_lists_only_usable_voices_for_the_language():
    assert [v.engine.dir_name for v in shared.voices_for("es", voices())] == ["piper-es-es", "piper-es-mx"]


# ---------------------------------------------------------------- pyvolis


def test_a_side_set_to_a_variety_is_spoken_by_the_voice_tuned_for_it_first():
    """M7.7's check: Spanish (Mexico) offers the Mexico-tuned voice first."""
    vs = [voice("piper-en", "en"), voice("piper-es-es", "es", varieties=["es-ES"]),
          voice("piper-es-mx", "es", varieties=["es-MX"])]
    r = direction(LEFT, settings(right_language="es-MX"), [whisper()], "whisper", vs)
    assert r.direction.voice == "piper-es-mx" and r.direction.target == "es-MX"
    labels = [f"{v.engine.dir_name}: {v.fit.label('es-MX')}" for v in shared.voices_for("es-MX", vs)]
    assert labels[0].startswith("piper-es-mx: tuned for") and "Mexico" in labels[0]


def test_a_downloaded_model_that_does_not_say_its_languages_may_be_chosen_with_a_note():
    unknown = engine("hf-model", "transformers", [], languages_known=False)
    r = direction(RIGHT, settings(right_asr="hf-model"), [whisper(), unknown], "whisper", voices())
    assert r.direction.asr == "hf-model"
    assert any("doesn't say which languages" in w for w in r.warnings)
    # Unset, a model that lists the language is preferred to one that doesn't say.
    assert direction(RIGHT, settings(), [unknown, whisper()], "", voices()).direction.asr == "whisper"


def test_only_parakeet_decides_the_language_itself():
    assert not shared.takes_language(engine("p", "nemo_transducer", ["en"]))
    assert shared.takes_language(engine("w", "whisper", ["en"]))
    assert shared.takes_language(engine("h", "transformers", ["en"]))


def test_the_main_recognizer_is_preferred_among_equally_good_ones():
    both = [engine("whisper-a", "whisper", ["en", "es"]), engine("whisper-b", "whisper", ["en", "es"])]
    assert direction(LEFT, settings(), both, "whisper-b", voices()).direction.asr == "whisper-b"
    assert direction(LEFT, settings(), both, "", voices()).direction.asr == "whisper-a"
