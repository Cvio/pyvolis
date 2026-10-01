"""Carry-forward context, the glossary and fragment holding (P7): the logic,
without models. tests/test_translate.py runs them through the real translator."""

from pyvolis.sentences import Sentence
from pyvolis.translate import context as ctx


def words(text: str) -> int:
    return len(text.split())


def test_the_most_recent_sentences_are_carried_forward():
    history = ctx.History(sentences=2, token_budget=400)
    for i in range(4):
        history.add(f"fuente {i}", f"source {i}")
    assert [(t.source, t.translation) for t in history.context(words)] == [
        ("fuente 2", "source 2"), ("fuente 3", "source 3")]


def test_the_oldest_half_leaves_at_once_so_prompts_keep_their_beginning():
    """With 4 sentences of context, the window grows to 4 and then drops 2
    together, rather than sliding by one each time: consecutive prompts then
    share their start, which the translator doesn't evaluate again."""
    history = ctx.History(sentences=4, token_budget=400)
    seen = []
    for i in range(9):
        history.add(f"s{i}", f"t{i}")
        seen.append([t.source for t in history.context(words)])
    assert seen[3] == ["s0", "s1", "s2", "s3"]
    assert seen[4] == ["s2", "s3", "s4"], "two dropped together"
    assert seen[5] == ["s2", "s3", "s4", "s5"], "then it grows again from the same start"
    assert all(len(c) <= 4 for c in seen)


def test_the_token_budget_drops_the_oldest_first():
    history = ctx.History(sentences=4, token_budget=10)
    history.add("uno dos tres cuatro", "one two three four")  # 8 tokens
    history.add("cinco seis", "five six")  # 4
    history.add("siete", "seven")  # 2
    kept = history.context(words)
    assert [t.source for t in kept] == ["cinco seis", "siete"], "the newest that fit, in order"


def test_no_context_when_it_is_off():
    history = ctx.History(sentences=0)
    history.add("hola", "hello")
    assert history.context(words) == []


def test_a_revision_replaces_the_wording_context_carries():
    history = ctx.History(sentences=3)
    history.add("Yo manejo.", "I handle it.")
    history.add("Mi carro está aquí.", "My car is here.")
    history.replace_last(["I'll drive.", "My car is here."])
    assert [t.translation for t in history.context(words)] == ["I'll drive.", "My car is here."]


def test_the_glossary_is_one_sentence_and_absent_when_empty():
    assert ctx.glossary_line(["Susie Wolff", " Bellas Artes "]) == \
        "Keep these names and terms exactly: Susie Wolff, Bellas Artes."
    assert ctx.glossary_line([]) == "" and ctx.glossary_line(["  "]) == ""
    assert ctx.parse_glossary("Susie Wolff, Bellas Artes;  Metro\nZócalo,") == [
        "Susie Wolff", "Bellas Artes", "Metro", "Zócalo"]


# ---------------------------------------------------------------- fragments


def sentence(sid: str, text: str, start: float = 0.0, end: float = 1.0) -> Sentence:
    return Sentence(sid, int(sid.split(".")[0]), text, start, end, False)


def test_what_counts_as_a_fragment():
    holder = ctx.FragmentHolder()
    assert holder.is_fragment("Yo manejo")
    assert not holder.is_fragment("Yo manejo."), "final punctuation: a sentence, however short"
    assert not holder.is_fragment("¿Quién maneja?")
    assert not holder.is_fragment("vamos a la tienda ahora"), "four words or more"
    assert holder.is_fragment("sí claro")


def test_a_fragment_is_held_and_joined_to_the_next_sentence():
    holder = ctx.FragmentHolder()
    ready, held = holder.offer(sentence("1.1", "Yo manejo", 0.0, 1.0), "es", now=10.0)
    assert ready == [] and held.text == "Yo manejo"
    ready, held = holder.offer(sentence("2.1", "Mi carro está aquí.", 1.4, 3.0), "es", now=10.9)
    assert held is None
    [(joined, source, _)] = ready
    assert joined.text == "Yo manejo Mi carro está aquí."
    assert (joined.id, joined.start, joined.end, source) == ("1.1", 0.0, 3.0, "es")


def test_a_fragment_nothing_follows_is_released_after_the_hold():
    holder = ctx.FragmentHolder(hold_seconds=1.5)
    holder.offer(sentence("1.1", "Yo manejo"), "es", now=10.0)
    assert holder.due(11.0) == [], "still within the hold"
    [(released, _, _)] = holder.due(11.6)
    assert released.text == "Yo manejo"
    assert holder.due(20.0) == []


def test_a_sentence_arriving_too_late_is_not_joined():
    holder = ctx.FragmentHolder(hold_seconds=1.5)
    holder.offer(sentence("1.1", "Yo manejo"), "es", now=10.0)
    ready, held = holder.offer(sentence("2.1", "Mi carro está aquí."), "es", now=13.0)
    assert [s.text for s, _, _ in ready] == ["Yo manejo", "Mi carro está aquí."] and held is None


def test_fragment_after_fragment_joins_and_is_held_again():
    holder = ctx.FragmentHolder()
    holder.offer(sentence("1.1", "Pues"), "es", now=1.0)
    ready, held = holder.offer(sentence("2.1", "yo manejo"), "es", now=1.5)
    assert ready == [] and held.text == "Pues yo manejo", "three words: still a fragment"
    ready, _ = holder.offer(sentence("3.1", "mi carro."), "es", now=2.0)
    assert [s.text for s, _, _ in ready] == ["Pues yo manejo mi carro."]


def test_holding_off_passes_everything_straight_through():
    holder = ctx.FragmentHolder(enabled=False)
    ready, held = holder.offer(sentence("1.1", "Yo manejo"), "es", now=1.0)
    assert [s.text for s, _, _ in ready] == ["Yo manejo"] and held is None


def test_the_end_of_a_run_releases_what_is_held():
    holder = ctx.FragmentHolder()
    holder.offer(sentence("1.1", "Yo manejo"), "es", now=1.0)
    assert [s.text for s, _, _ in holder.flush()] == ["Yo manejo"]
    assert holder.flush() == []
