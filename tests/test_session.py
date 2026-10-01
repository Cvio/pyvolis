"""Port of the tests of Rust `gui::Session` that apply before turns, pairing
and sharing, plus pyvolis's rows per sentence. No Qt is imported here."""

import sys

from pyvolis import events as ev
from pyvolis.gui import session as ses
from pyvolis.gui.session import ComparisonLine, DroppedLine, Nothing, Row, Session


def heard(s: Session, utterance: int, text: str, lang: str = "es", sentence: str | None = None) -> str:
    """An utterance recognised as one sentence: Final, then its sentence."""
    sid = sentence or f"{utterance}.1"
    s.apply(ev.Final(utterance, text, lang, 1500, 120, 1.0, 2.5))
    s.apply(ev.SentenceMsg(sid, utterance, text, lang, 1.0, 2.5))
    return sid


def test_the_session_imports_no_qt():
    assert not any(name.startswith("PySide6") for name in sys.modules if "session" in name)
    import pyvolis.gui.session as module

    assert "PySide6" not in open(module.__file__, encoding="utf-8").read().replace('"PySide6"', "")


def test_right_to_left_text_is_told_apart():
    assert ses.has_rtl("ساعدني في إدخال مشترياتي.")
    assert ses.has_rtl("שלום")
    assert ses.has_rtl("Uber إلى المطار")
    assert ses.has_rtl("می‌روم")
    assert not ses.has_rtl("¿Dónde está la estación?")
    assert not ses.has_rtl("Help me get my groceries in.")


def test_an_utterance_collects_its_translation_and_timings():
    s = Session()
    s.apply(ev.Listening())
    sid = heard(s, 1, "¿Dónde está la estación?")
    s.apply(ev.Translated(sid, "Where is the station?", "en", 340, "cpu", "qwen"))
    s.apply(ev.SpeakingStarted(sid, 650))
    row = s.latest_timed()
    assert row.target == "Where is the station?"
    assert (row.translate_ms, row.first_audio_ms, row.asr_ms, row.speech_ms) == (340, 650, 120, 1500)
    assert s.speaking and row.spoken
    s.apply(ev.SpeakingEnded())
    assert not s.speaking


def test_a_refused_translation_is_shown_on_its_own_row():
    s = Session()
    heard(s, 1, "Hola.")
    second = heard(s, 2, "Me scables.")
    s.apply(ev.NotTranslated(second, "the model recited its own instructions", "recited"))
    first_row, second_row = s.lines
    assert first_row.problem is None, "the wrong row was marked"
    assert "recited" in second_row.problem
    assert s.latest_timed() is None, "nothing was translated"


def test_loading_listening_and_stopping_move_the_state():
    s = Session()
    s.apply(ev.Loading("the translation model"))
    assert (s.state, s.loading) == (ses.STARTING, "the translation model")
    s.apply(ev.Listening())
    assert s.state == ses.LISTENING
    s.apply(ev.Level(-20.0))
    s.apply(ev.SpeakingStarted("9.1", 1))
    s.apply(ev.Stopped())
    assert s.state == ses.STOPPED
    assert not s.speaking, "stopping ends speech"
    assert s.level_db is None and not s.has_level, "no meter once stopped"


def test_an_error_is_kept_until_the_next_start_listens():
    s = Session()
    s.apply(ev.Error('no installed voice speaks "ja"'))
    s.apply(ev.Stopped())
    assert s.last_error, "a failed start must stay on screen"
    s.apply(ev.Listening())
    assert s.last_error is None


def test_a_row_keeps_the_languages_it_was_spoken_in():
    s = Session()
    s.begin("en", "es", False, True)
    heard(s, 1, "Hello.", "en")
    s.begin("es", "en", False, True)
    heard(s, 2, "Hola.", "es")
    first, second = s.lines
    assert (first.source_lang, first.target_lang) == ("en", "es")
    assert (second.source_lang, second.target_lang) == ("es", "en")


def test_speech_with_no_words_is_shown_not_hidden():
    s = Session()
    s.apply(ev.NothingRecognized(4, 2182))
    assert isinstance(s.lines[0], Nothing) and (s.lines[0].index, s.lines[0].speech_ms) == (4, 2182)


def test_comparisons_become_their_own_lines():
    s = Session()
    heard(s, 1, "Hola.")
    s.apply(ev.ComparisonMsg(object()))
    assert [type(line) for line in s.lines] == [Row, ComparisonLine]


def test_the_pane_keeps_a_bounded_history():
    s = Session()
    for i in range(ses.MAX_LINES + 50):
        heard(s, i, f"línea {i}")
    assert len(s.lines) == ses.MAX_LINES
    assert s.lines[-1].utterance == ses.MAX_LINES + 49


def test_comparing_never_speaks():
    s = Session()
    s.begin("es", "en", comparing=True, speaks=True)
    assert not s.speaks
    assert s.state == ses.STARTING


# ---------------------------------------------------------------- pyvolis


def test_an_utterance_of_two_sentences_is_two_rows_with_their_own_translations():
    s = Session()
    s.apply(ev.Final(3, "Yo manejo. Mi carro está aquí.", "es", 3000, 200, 10.0, 13.0))
    s.apply(ev.SentenceMsg("3.1", 3, "Yo manejo.", "es", 10.0, 11.0))
    s.apply(ev.SentenceMsg("3.2", 3, "Mi carro está aquí.", "es", 11.0, 13.0))
    s.apply(ev.Translated("3.2", "My car is here.", "en", 500, "cpu", "qwen"))
    s.apply(ev.Translated("3.1", "I'll drive.", "en", 400, "cpu", "qwen"))
    assert [(r.id, r.target) for r in s.rows()] == [("3.1", "I'll drive."), ("3.2", "My car is here.")]
    assert all(r.asr_ms == 200 for r in s.rows()), "each sentence carries its utterance's timings"


def test_a_dropped_hallucination_is_a_line_with_its_reasons():
    s = Session()
    s.apply(ev.Dropped(2, "Gracias por ver el video.", ["a stock phrase"], 4.0))
    assert isinstance(s.lines[0], DroppedLine) and s.lines[0].reasons == ["a stock phrase"]
    assert s.counts()["dropped"] == 1


def test_a_revision_replaces_the_translation_and_keeps_the_old_one():
    s = Session()
    sid = heard(s, 1, "Yo manejo.")
    s.apply(ev.Translated(sid, "I handle it.", "en", 300, "cpu", "qwen"))
    s.apply(ev.Revised(sid, "I handle it.", "I'll drive."))
    row = s.row(sid)
    assert (row.target, row.revised, row.history) == ("I'll drive.", True, ["I handle it."])
    assert s.counts()["revisions"] == 1


def test_loaded_models_report_their_memory():
    s = Session()
    s.begin("es", "en", False, False)
    s.apply(ev.ModelLoaded("recognizer", "whisper", "cuda", 1_600_000_000, 0))
    s.apply(ev.ModelLoaded("translator", "qwen", "cpu", 0, 1_100_000_000))
    assert s.memory() == (1_600_000_000, 1_100_000_000)
    s.begin("es", "en", False, False)
    assert s.memory() == (0, 0), "a new run starts with nothing loaded"


def test_file_progress_stats_and_stalls_are_kept():
    s = Session()
    s.apply(ev.Progress(12.0, 120.0))
    s.apply(ev.Stall(180))
    s.apply(ev.Stall(160))
    s.apply(ev.Summary({"asr_rtf": 0.2}))
    assert s.progress == (12.0, 120.0) and s.worst_stall_ms == 180 and s.stats == {"asr_rtf": 0.2}
    assert ses.clock(75) == "1:15" and ses.clock(3675) == "1:01:15"
