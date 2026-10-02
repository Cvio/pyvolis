"""Shared-machine mode in the window's state: port of Rust's shared tests in
`gui.rs`. No Qt here."""

from pyvolis import events as ev
from pyvolis.gui import session as ses
from pyvolis.gui.session import Session


def shared_session() -> Session:
    s = Session()
    s.begin("es", "en", False, True, False)
    s.apply(ev.Listening())
    s.apply(ev.Mode("shared"))
    return s


def heard(s: Session, utterance: int, text: str, lang: str) -> str:
    sid = f"{utterance}.1"
    s.apply(ev.Final(utterance, text, lang, 800, 400, 0.0, 1.0))
    s.apply(ev.SentenceMsg(sid, utterance, text, lang, 0.0, 1.0))
    return sid


def test_shared_one_person_at_a_time():
    s = shared_session()
    assert s.shared_press("left")[0] == "begin" and s.shared_press("right")[0] == "begin"

    s.apply(ev.TurnStarted("left"))
    assert s.active_side == "left"
    assert s.shared_press("left")[0] == "end"
    action, why = s.shared_press("right")
    assert action == "ignore" and "left" in why, "the right key does nothing during the left turn"

    s.apply(ev.TurnEnded())
    assert s.shared_press("left") == ("ignore", "working - wait")

    sid = heard(s, 1, "Hello.", "en")
    s.apply(ev.Translated(sid, "Hola.", "es", 200, "cpu", "qwen"))
    s.apply(ev.SpeakingStarted(sid, 900))
    for side in ("left", "right"):
        assert s.shared_press(side) == ("ignore", "speaking - wait"), "neither key works while the machine speaks"

    s.apply(ev.SpeakingEnded())
    assert s.turn == ses.IDLE and s.active_side == ""
    assert s.shared_press("right")[0] == "begin"


def test_shared_rows_belong_to_their_side_and_carry_their_languages():
    s = shared_session()
    s.apply(ev.TurnStarted("right"))
    s.apply(ev.TurnEnded())
    sid = heard(s, 3, "Hola.", "es")
    s.apply(ev.Translated(sid, "Hello.", "en", 200, "cpu", "qwen"))
    row = s.row(sid)
    assert row.side == "right"
    assert (row.source_lang, row.target_lang) == ("es", "en")


def test_escape_cancels_only_when_there_is_something_to_cancel():
    s = shared_session()
    assert not s.shared_can_cancel(), "Escape must still close a dropdown"
    s.apply(ev.TurnStarted("left"))
    assert s.shared_can_cancel()
    s.apply(ev.TurnCancelled())
    assert s.turn == ses.IDLE and s.active_side == "" and not s.shared_can_cancel()
    assert s.shared_press("right")[0] == "begin"


def test_shared_keys_do_nothing_outside_shared_mode():
    s = Session()
    s.apply(ev.Listening())
    s.apply(ev.Mode("turn"))
    assert s.shared_press("left")[0] == "ignore"
    assert not s.shared_can_cancel()


# ---------------------------------------------------------------- pyvolis


def test_a_turn_of_several_sentences_keeps_its_side_until_the_last_is_spoken():
    s = shared_session()
    s.apply(ev.TurnStarted("left"))
    s.apply(ev.TurnEnded())
    s.apply(ev.Final(1, "Hello. How are you?", "en", 2000, 300, 0.0, 2.0))
    s.apply(ev.SentenceMsg("1.1", 1, "Hello.", "en", 0.0, 1.0))
    s.apply(ev.SentenceMsg("1.2", 1, "How are you?", "en", 1.0, 2.0))
    s.apply(ev.Translated("1.1", "Hola.", "es", 200, "cpu", "qwen"))
    s.apply(ev.SpeakingStarted("1.1", 900))
    s.apply(ev.SpeakingEnded())
    assert s.active_side == "left" and s.shared_press("right")[0] == "ignore", "the second sentence is still to come"
    s.apply(ev.Translated("1.2", "¿Cómo estás?", "es", 200, "cpu", "qwen"))
    s.apply(ev.SpeakingStarted("1.2", 900))
    s.apply(ev.SpeakingEnded())
    assert s.active_side == "" and s.shared_press("right")[0] == "begin"
    assert [r.side for r in s.rows()] == ["left", "left"]


def test_a_sides_recognizer_problem_is_kept_until_it_is_ready_again():
    s = shared_session()
    s.apply(ev.SharedSide("right", 'the recognizer "deleted-model" is not installed'))
    assert "deleted-model" in s.side_problems["right"] and "left" not in s.side_problems
    s.apply(ev.SharedSide("right"))
    assert s.side_problems == {}


def test_the_indicator_says_whose_turn_it_is():
    s = shared_session()
    assert ses.indicator(s, "Space", False)[0].startswith("READY")
    s.apply(ev.TurnStarted("right"))
    assert ses.indicator(s, "Space", False)[0] == "RIGHT IS TALKING"
    s.apply(ev.TurnEnded())
    assert ses.indicator(s, "Space", False)[0] == "WORKING..."
