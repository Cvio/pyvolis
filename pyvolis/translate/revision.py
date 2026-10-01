"""Revision mode: re-translating the last few sentences together once a new
one arrives, and replacing an earlier translation that changes.

New in pyvolis. "Yo manejo." alone is "I manage."; followed by "Mi carro está
afuera." it is "I'll drive.", and only the later sentence can say so. After
each newly translated sentence the last K source sentences are translated
again as one text, with the conversation before them as context, and the
result is split back into sentences. Where a sentence's translation differs
from the one on screen, it is replaced.

What is never revised:

  - a sentence that has been handed to the voice: what was said can't be unsaid;
  - a sentence that ended more than `max_age` seconds before the newest one:
    the reader has moved on;
  - a sentence of more than `max_words` words: a long sentence carries its
    own meaning, and translating it again only rewords it. On read speech
    (long, unrelated sentences) revising everything gave 20 revisions in 14
    sentences and a worse translation (chrF 61.2 against 63.6);
  - the newest sentence: it was translated a moment ago with everything
    before it, and nothing new has been said;
  - a sentence that has been revised once already: a second look mostly
    flips between two wordings ("doesn't" / "does not" and back);
  - anything, when the new translation doesn't split into the same number of
    sentences as the source: there is then no telling which part belongs to
    which sentence, and a guess would put words under the wrong one.

Differences of case, spacing or punctuation alone are not revisions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .. import sentences
from . import Refused, TranslateError, TranslationRequest, Turn, guards


@dataclass
class Done:
    """A sentence that has been translated and shown."""

    id: str
    source: str  # the text
    lang: str
    translation: str
    end: float  # seconds on the source's timeline
    spoken: bool  # handed to the voice
    revised: bool = False  # its translation has been replaced once


@dataclass
class Revision:
    id: str
    old: str
    new: str


class Reviser:
    def __init__(self, sentences: int = 3, max_age: float = 30.0, max_words: int = 8) -> None:
        self.sentences = max(sentences, 0)
        self.max_age = max_age
        self.max_words = max_words
        self.recent: deque[Done] = deque(maxlen=max(self.sentences, 1))
        self.passes = 0  # how many times the translator was asked again

    def add(self, done: Done) -> None:
        self.recent.append(done)

    def clear(self) -> None:
        self.recent.clear()

    def window(self) -> list[Done]:
        """The sentences to translate again: the newest and those just before
        it that may be reached, oldest first. Empty when none of the earlier
        ones is short enough to be revised: the translator isn't asked."""
        if self.sentences < 2 or not self.recent:
            return []
        newest = self.recent[-1]
        chosen = [newest]
        for done in list(self.recent)[-2::-1]:
            if done.spoken or done.lang != newest.lang or newest.end - done.end > self.max_age:
                break  # nothing before it is reached either: the text must be one stretch
            chosen.append(done)
        if not any(self.revisable(done) for done in chosen[1:]):
            return []
        return list(reversed(chosen))

    def revisable(self, done: Done) -> bool:
        return not done.spoken and not done.revised and len(done.source.split()) <= self.max_words

    def revise(self, translator, context: list[Turn], target: str, glossary: list[str]) -> list[Revision]:
        """Translate the window again and return what changed. `context` is
        the conversation up to and including the newest sentence, as
        carry-forward would send it; the window's own turns are taken off."""
        window = self.window()
        if not window:
            return []
        earlier = context[: -len(window)] if len(context) > len(window) else []
        request = TranslationRequest(" ".join(d.source for d in window), window[-1].lang, target, earlier, glossary)
        self.passes += 1
        try:
            result = translator.translate(request)
        except (Refused, TranslateError):
            return []  # the translations on screen stand
        return align(window, result.text, self.revisable)


def align(window: list[Done], translation: str, revisable=lambda done: not done.spoken) -> list[Revision]:
    """The joint translation split back over the window's sentences; the
    changes among the earlier ones, with each Done updated. Nothing if the
    split doesn't match."""
    parts = sentences.split_text(translation)
    if len(parts) != len(window):
        return []
    changes = []
    for done, new in list(zip(window, parts))[:-1]:
        if not revisable(done) or guards.normalise(new) == guards.normalise(done.translation):
            continue
        changes.append(Revision(done.id, done.translation, new))
        done.translation, done.revised = new, True
    return changes
