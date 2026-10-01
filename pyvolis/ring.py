"""The last few utterances, kept so they can be re-run. Port of Rust `ring.rs`."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE

# Utterances retained by default: about ten minutes of conversation.
DEFAULT_CAPACITY = 20


@dataclass(frozen=True)
class Utterance:
    index: int  # 1-based, in capture order this session
    start_ms: int  # since capture started
    pcm: np.ndarray

    def duration_ms(self) -> int:
        return len(self.pcm) * 1000 // SAMPLE_RATE


class UtteranceRing:
    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        self.capacity = max(1, capacity)
        self._items: deque[Utterance] = deque()
        self._captured = 0

    def push(self, start_ms: int, pcm: np.ndarray) -> Utterance:
        """Store an utterance, evicting the oldest if full. Returns it as stored."""
        self._captured += 1
        utterance = Utterance(self._captured, start_ms, pcm)
        if len(self._items) == self.capacity:
            self._items.popleft()
        self._items.append(utterance)
        return utterance

    def next_index(self) -> int:
        """The index the next utterance will have (streaming names an
        utterance before it has ended)."""
        return self._captured + 1

    def recent(self) -> list[Utterance]:
        """Newest first."""
        return list(reversed(self._items))

    def get(self, index: int) -> Utterance | None:
        return next((u for u in self._items if u.index == index), None)

    def latest(self) -> Utterance | None:
        return self._items[-1] if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)
