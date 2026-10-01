"""Port of the tests in Rust `playback.rs` and `tts.rs`. The ones that need a
sound card use the VB-Audio cable when it is installed (silent), and are
skipped otherwise. `scripts/gate_check.py` is the end-to-end check that
pyvolis doesn't hear itself."""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pyvolis import audio, models, paths, playback, tts
from pyvolis.models import Engine
from pyvolis.playback import TAIL_SECONDS, Gate

ROOT = paths.app_root()
CABLE = "CABLE Input (VB-Audio Virtual Cable)"


def test_the_gate_is_closed_while_speaking_and_through_the_tail():
    gate = Gate(True)
    assert not gate.is_closed(), "idle"
    gate.speaking_started()
    assert gate.is_closed(), "speaking"
    gate.speaking_ended()
    assert gate.is_closed(), "still closed during the tail"
    time.sleep(TAIL_SECONDS + 0.06)
    assert not gate.is_closed(), "open again after the tail"


def test_a_disabled_gate_never_closes():
    gate = Gate(False)  # half_duplex = false, for headphones
    gate.speaking_started()
    assert not gate.is_closed()
    gate.speaking_ended()
    assert not gate.is_closed()


def test_taking_a_turn_opens_the_gate_at_once():
    gate = Gate(True)
    gate.speaking_started()
    assert gate.is_closed()
    gate.open_now()
    assert not gate.is_closed(), "the turn's first words would be lost"


def test_a_second_utterance_keeps_the_gate_closed():
    gate = Gate(True)
    gate.speaking_started()
    gate.speaking_ended()
    gate.speaking_started()
    time.sleep(TAIL_SECONDS + 0.06)
    assert gate.is_closed(), "speaking again: the first tail expiring must not open it"


def cable_or_skip():
    try:
        names = [d.name for d in audio.list_output_devices()]
    except audio.AudioError:
        pytest.skip("no audio host API")
    if CABLE not in names:
        pytest.skip("needs the VB-Audio Virtual Cable")


def test_a_reply_held_through_a_turn_closes_the_gate_only_when_it_plays():
    cable_or_skip()
    gate = Gate(True)
    player = playback.Player(CABLE, gate)
    try:
        control = player.control()
        control.begin_turn()
        player.play(np.full(1600, 0.1, np.float32), 16_000)  # a reply arrives mid-turn
        assert not gate.is_closed(), "a held reply must not gate the turn"
        assert player.queued() > 0, "it waits rather than being dropped"
        control.end_turn()
        assert gate.is_closed(), "the held reply now plays, and gates the microphone"
    finally:
        player.close()


def test_taking_a_turn_cuts_off_the_reply_that_was_playing():
    cable_or_skip()
    player = playback.Player(CABLE, Gate(True))
    try:
        player.play(np.full(48_000, 0.5, np.float32), 16_000)
        player.control().begin_turn()
        assert player.queued() == 0
    finally:
        player.close()


def test_speech_plays_out_and_reports_its_end():
    cable_or_skip()
    ended = threading.Event()
    gate = Gate(True)
    player = playback.Player(CABLE, gate, on_ended=ended.set)
    try:
        player.play(np.full(8_000, 0.2, np.float32), 16_000)  # half a second
        assert gate.is_closed(), "closed before the first sample reaches the speakers"
        assert ended.wait(3.0), "the end of speech was never reported"
        time.sleep(TAIL_SECONDS + 0.1)
        assert not gate.is_closed()
    finally:
        player.close()


def test_streams_opened_from_different_threads_coexist():
    """PortAudio (WASAPI) refuses a stream opened off the thread that
    initialised it; every open goes through audio.on_audio_thread."""
    cable_or_skip()
    first = playback.Player(CABLE, Gate(False))
    made, failed = [], []

    def other() -> None:
        try:
            made.append(playback.Player(CABLE, Gate(False)))
        except playback.PlaybackError as e:
            failed.append(e)

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    for player in [first] + made:
        player.close()
    assert not failed, failed


def test_an_unknown_output_device_is_an_error_naming_it():
    cable_or_skip()
    with pytest.raises(playback.PlaybackError, match="Definitely Not Speakers"):
        playback.select_output_device("Definitely Not Speakers")


# ---------------------------------------------------------------- voices


def voice(dir_name: str, languages: list[str], tuned: list[str]) -> Engine:
    return Engine(dir_name, Path(dir_name), dir_name, "segment", "vits", languages, tuned)


def test_the_voice_is_the_best_fit_for_the_variety():
    voices = [voice("es-spain", ["es"], ["es-ES"]), voice("es-mexico", ["es"], ["es-MX"]), voice("en", ["en"], [])]
    assert tts.choose(voices, "es-MX").dir_name == "es-mexico"
    assert tts.choose(voices, "es-ES").dir_name == "es-spain"
    assert tts.choose(voices, "en-US").dir_name == "en"


def test_an_uninstalled_language_is_refused_not_substituted():
    voices = [voice("en", ["en"], [])]
    with pytest.raises(tts.TtsError, match="no installed voice") as e:
        tts.choose(voices, "ja")
    assert "en (en)" in str(e.value), "it says which voices are installed"


def test_speaks_the_target_language():
    """Rust's speaks_the_target_language, with the real voice."""
    engines = [e for e in models.discover(paths.tts_dir(ROOT), models.Role.TTS) if isinstance(e, Engine)]
    try:
        engine = tts.for_language(engines, "en")
    except tts.TtsError:
        pytest.skip("no English voice installed")
    speech = tts.Voice(engine).speak("Where is the station?")
    assert len(speech.samples) > 0
    assert speech.duration_ms() > 400, f"only {speech.duration_ms()} ms for a four-word question"
    assert float(np.abs(speech.samples).max()) > 0.05, "the audio is silent"
