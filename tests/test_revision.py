"""Revision mode (P8): which sentences are translated again, how the joint
translation is shared back out, and what is never touched. No models here;
tests/test_translate.py runs it through the real translator."""

from pyvolis import translate as tr
from pyvolis.translate import context as ctx
from pyvolis.translate.revision import Done, Reviser, align


class Scripted:
    """A translator that answers from a table and remembers what it was asked."""

    device = "cpu"

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.requests: list[tr.TranslationRequest] = []

    def translate(self, request: tr.TranslationRequest) -> tr.TranslationResult:
        self.requests.append(request)
        if request.text not in self.answers:
            raise tr.Refused("echo", "not in the script")
        return tr.TranslationResult(self.answers[request.text], "", 0.1, "cpu")


def done(sid: str, source: str, translation: str, end: float, spoken: bool = False, lang: str = "es") -> Done:
    return Done(sid, source, lang, translation, end, spoken)


def test_a_later_sentence_changes_an_earlier_translation():
    reviser = Reviser(sentences=3)
    reviser.add(done("1.1", "Yo manejo.", "I manage.", 1.0))
    reviser.add(done("2.1", "Mi carro está afuera.", "My car is outside.", 3.0))
    translator = Scripted({"Yo manejo. Mi carro está afuera.": "I'll drive. My car is outside."})
    changes = reviser.revise(translator, [], "en", [])
    assert [(c.id, c.old, c.new) for c in changes] == [("1.1", "I manage.", "I'll drive.")]
    assert reviser.recent[0].translation == "I'll drive.", "the next pass compares with what is on screen now"
    assert reviser.passes == 1


def test_the_window_is_the_last_k_sentences_and_what_is_before_them_is_context():
    reviser = Reviser(sentences=2)
    history = ctx.History(sentences=4)
    for i, (source, translation) in enumerate([("Uno.", "One."), ("Dos.", "Two."), ("Tres.", "Three.")]):
        reviser.add(done(f"{i}.1", source, translation, float(i)))
        history.add(source, translation)
    translator = Scripted({"Dos. Tres.": "Two. Three."})
    assert reviser.revise(translator, history.context(len), "en", ["Bellas Artes"]) == []
    [request] = translator.requests
    assert request.text == "Dos. Tres."
    assert [(t.source, t.translation) for t in request.context] == [("Uno.", "One.")]
    assert request.glossary == ["Bellas Artes"] and (request.source, request.target) == ("es", "en")


def test_a_spoken_sentence_is_never_revised_nor_anything_before_it():
    reviser = Reviser(sentences=3)
    reviser.add(done("1.1", "Yo manejo.", "I manage.", 1.0))
    reviser.add(done("2.1", "Mi carro está afuera.", "My car is outside.", 3.0, spoken=True))
    reviser.add(done("3.1", "Vámonos.", "Let's go.", 4.0))
    assert reviser.window() == [], "the sentence before the newest was spoken"
    translator = Scripted({})
    assert reviser.revise(translator, [], "en", []) == [] and translator.requests == []


def test_when_every_sentence_is_spoken_the_translator_is_never_asked_again():
    reviser = Reviser(sentences=3)
    for i in range(3):
        reviser.add(done(f"{i}.1", f"Frase {i}.", f"Sentence {i}.", float(i), spoken=True))
    assert reviser.window() == []


def test_a_sentence_older_than_the_limit_is_never_revised():
    reviser = Reviser(sentences=3, max_age=30.0)
    reviser.add(done("1.1", "Yo manejo.", "I manage.", 1.0))
    reviser.add(done("2.1", "Mi carro está afuera.", "My car is outside.", 20.0))
    reviser.add(done("3.1", "Vámonos.", "Let's go.", 40.0))
    assert [d.id for d in reviser.window()] == ["2.1", "3.1"], "1.1 ended 39 s before the newest"
    reviser.add(done("4.1", "Ya.", "Now.", 80.0))
    assert reviser.window() == []


def test_a_change_of_language_ends_the_window():
    reviser = Reviser(sentences=3)
    reviser.add(done("1.1", "I'll drive.", "Yo manejo.", 1.0, lang="en"))
    reviser.add(done("2.1", "Mi carro está afuera.", "My car is outside.", 3.0))
    assert reviser.window() == []


def test_a_translation_that_does_not_split_the_same_way_changes_nothing():
    window = [done("1.1", "Yo manejo.", "I manage.", 1.0), done("2.1", "Mi carro está afuera.", "My car is outside.", 3.0)]
    assert align(window, "I'll drive, my car is outside.") == []
    assert align(window, "I'll drive. My car. It is outside.") == []
    assert [d.translation for d in window] == ["I manage.", "My car is outside."]


def test_punctuation_and_case_alone_are_not_a_revision():
    window = [done("1.1", "Hola.", "Hello.", 1.0), done("2.1", "¿Qué tal?", "How are you?", 2.0)]
    assert align(window, "hello! How are you?") == []


def test_the_newest_sentence_is_never_revised():
    window = [done("1.1", "Yo manejo.", "I manage.", 1.0), done("2.1", "Mi carro está afuera.", "My car is outside.", 3.0)]
    changes = align(window, "I'll drive. My car is out front.")
    assert [(c.id, c.new) for c in changes] == [("1.1", "I'll drive.")]
    assert window[1].translation == "My car is outside."


def test_a_long_sentence_is_not_revised_and_alone_does_not_ask_the_translator():
    long = "Se recomienda a los viajeros que se informen sobre cualquier riesgo del clima."
    reviser = Reviser(sentences=3, max_words=8)
    reviser.add(done("1.1", long, "Travellers are advised to find out about any weather risk.", 1.0))
    reviser.add(done("2.1", "Mi carro está afuera.", "My car is outside.", 3.0))
    translator = Scripted({})
    assert reviser.window() == [] and reviser.revise(translator, [], "en", []) == []
    assert translator.requests == []
    reviser.add(done("3.1", "Vámonos.", "Let's go.", 4.0))
    assert [d.id for d in reviser.window()] == ["1.1", "2.1", "3.1"], "2.1 is short: the stretch is translated again"
    answers = {f"{long} Mi carro está afuera. Vámonos.": "Travellers should check the weather. My car is out front. Let's go."}
    changes = reviser.revise(Scripted(answers), [], "en", [])
    assert [c.id for c in changes] == ["2.1"], "the long one keeps its translation"


def test_a_sentence_is_revised_at_most_once():
    reviser = Reviser(sentences=3)
    reviser.add(done("1.1", "No lo encuentro.", "It does not find it.", 1.0))
    reviser.add(done("2.1", "Mi hermano no contesta.", "My brother doesn't answer.", 2.0))
    first = reviser.revise(Scripted({"No lo encuentro. Mi hermano no contesta.": "I can't find him. My brother doesn't answer."}),
                           [], "en", [])
    assert [c.new for c in first] == ["I can't find him."]
    reviser.add(done("3.1", "Estoy preocupado.", "I'm worried.", 3.0))
    again = Scripted({"No lo encuentro. Mi hermano no contesta. Estoy preocupado.":
                      "I don't find him. My brother does not answer. I'm worried."})
    second = reviser.revise(again, [], "en", [])
    assert [c.id for c in second] == ["2.1"], "1.1 was revised once and keeps that wording"
    assert reviser.recent[0].translation == "I can't find him."


def test_a_refused_joint_translation_leaves_everything_as_it_is():
    reviser = Reviser(sentences=3)
    reviser.add(done("1.1", "Yo manejo.", "I manage.", 1.0))
    reviser.add(done("2.1", "Mi carro está afuera.", "My car is outside.", 3.0))
    assert reviser.revise(Scripted({}), [], "en", []) == []
    assert reviser.recent[0].translation == "I manage."


def test_revising_fewer_than_two_sentences_is_off():
    reviser = Reviser(sentences=1)
    reviser.add(done("1.1", "Yo manejo.", "I manage.", 1.0))
    reviser.add(done("2.1", "Mi carro está afuera.", "My car is outside.", 3.0))
    assert reviser.window() == []


def test_history_takes_the_revised_wording_even_when_it_holds_fewer_turns():
    history = ctx.History(sentences=4)
    history.add("Mi carro está afuera.", "My car is outside.")
    history.replace_last(["I'll drive.", "My car is out front."])
    assert [t.translation for t in history.context(len)] == ["My car is out front."]
