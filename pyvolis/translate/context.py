"""Translation with context: what came earlier, the glossary, and short
fragments held to be joined to what follows.

New in pyvolis. In carry-forward mode each sentence is translated knowing the
recent conversation: the last few committed source sentences and their
translations are placed in the prompt as earlier turns of the chat (user turn
= source, assistant turn = translation), the most natural form for a chat
model, which then continues the pattern.

Context is untrusted text: it came from a recognizer. It never changes the
instructions, the guards still run on every output, and the "recites the
prompt" guard recognises recited context too (guards.leaks_the_prompt).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from . import Turn

SENTENCE_END = (".", "?", "!", "…", "؟")
GLOSSARY_LINE = "Keep these names and terms exactly: {terms}."


def glossary_line(terms: list[str]) -> str:
    """The sentence added to the system text for a session glossary, or ""."""
    terms = [t.strip() for t in terms if t.strip()]
    return GLOSSARY_LINE.format(terms=", ".join(terms)) if terms else ""


def parse_glossary(text: str) -> list[str]:
    """Names and terms as typed: separated by commas, semicolons or new lines."""
    for separator in (";", "\n"):
        text = text.replace(separator, ",")
    return [t.strip() for t in text.split(",") if t.strip()]


class History:
    """The committed sentences of a session and their translations."""

    def __init__(self, sentences: int = 4, token_budget: int = 400) -> None:
        self.sentences = sentences
        self.token_budget = token_budget
        self._turns: deque[Turn] = deque()

    def add(self, source: str, translation: str) -> None:
        """Remember a sentence. When there are more than `sentences`, the
        oldest half goes at once rather than one each time: the prompt then
        starts the same way for several sentences running, and the translator
        keeps what it has already evaluated instead of re-reading the whole
        context for every sentence."""
        if self.sentences <= 0:
            return
        self._turns.append(Turn(source, translation))
        if len(self._turns) > self.sentences:
            for _ in range(max(1, self.sentences // 2)):
                self._turns.popleft()

    def replace_last(self, translations: list[str]) -> None:
        """P8: the last sentences were re-translated; keep the new wording.
        Aligned from the end: the history may hold fewer turns than that."""
        for turn, new in zip(reversed(self._turns), reversed(translations)):
            turn.translation = new

    def context(self, count_tokens) -> list[Turn]:
        """The most recent turns that fit: at most `sentences`, and never
        more than `token_budget` tokens, newest kept first. `count_tokens`
        measures a text with the translator's own tokenizer."""
        if self.sentences <= 0:
            return []
        chosen: list[Turn] = []
        used = 0
        for turn in reversed(self._turns):
            cost = count_tokens(turn.source) + count_tokens(turn.translation)
            if used + cost > self.token_budget:
                break
            chosen.append(turn)
            used += cost
        if len(chosen) < len(self._turns):
            # Over the budget: forget what didn't fit, so the next prompt
            # starts where this one does.
            for _ in range(len(self._turns) - len(chosen)):
                self._turns.popleft()
        return list(reversed(chosen))

    def clear(self) -> None:
        self._turns.clear()


@dataclass
class Held:
    sentence: object  # sentences.Sentence
    source: str  # language
    since: float  # on the holder's clock
    cut_at: float


class FragmentHolder:
    """Holds a short sentence with no final punctuation, to join it to the
    next one before translating. "Yo manejo" translated alone came out as "I
    handle it"; with the next words, "Mi carro...", the meaning is "I'll
    drive". Held for at most `hold_seconds`, on whatever clock the run uses
    (the wall for the microphone, the file's time for a fast file run)."""

    def __init__(self, min_words: int = 4, hold_seconds: float = 1.5, enabled: bool = True) -> None:
        self.min_words = min_words
        self.hold_seconds = hold_seconds
        self.enabled = enabled
        self.held: Held | None = None

    def is_fragment(self, text: str) -> bool:
        text = text.strip().rstrip("\"'»”’)]")
        return len(text.split()) < self.min_words and not text.endswith(SENTENCE_END)

    def offer(self, sentence, source: str, now: float, cut_at: float = 0.0) -> tuple[list, object | None]:
        """A committed sentence arrives. Returns (ready, newly_held): the
        sentences to translate now, as (sentence, source, cut_at) tuples, and
        the sentence now being held, if any."""
        ready = []
        if self.held is not None:
            held, self.held = self.held, None
            if now - held.since <= self.hold_seconds and held.source == source:
                sentence = join(held.sentence, sentence)
                cut_at = held.cut_at or cut_at
            else:
                ready.append((held.sentence, held.source, held.cut_at))
        if self.enabled and self.is_fragment(sentence.text):
            self.held = Held(sentence, source, now, cut_at)
            return ready, sentence
        ready.append((sentence, source, cut_at))
        return ready, None

    def due(self, now: float) -> list:
        """Whatever has been held long enough."""
        if self.held is not None and now - self.held.since > self.hold_seconds:
            return self.flush()
        return []

    def flush(self) -> list:
        """Release what is held (the end of a turn or of the run)."""
        if self.held is None:
            return []
        held, self.held = self.held, None
        return [(held.sentence, held.source, held.cut_at)]


def join(first, second):
    """Two sentences as one: the first's id and start, the second's end."""
    import dataclasses

    return dataclasses.replace(first, text=f"{first.text.rstrip()} {second.text.lstrip()}", end=second.end,
                               approximate=first.approximate or second.approximate)
