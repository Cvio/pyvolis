"""The floor token (SPEC §9): in turn-based paired mode exactly one peer holds
the floor, and the microphone is live only on that peer.

Port of Rust `floor.rs`. This is the rule by itself, with no sockets and no
clock of its own, so every case is tested directly: what each event does to
the floor, and what must be sent or done as a result. The peer thread feeds
it events and carries out the actions.

The rules, all from §9:

* Pressing the turn key sends `FloorRequest`. The microphone opens only
  when `FloorGrant` arrives.
* Two requests at once: the peer whose name sorts first wins, and the other
  is told the other party has the floor. No negotiation.
* No grant within two seconds fails the turn visibly. The microphone never
  opens on a timeout.
* The floor is released after the turn's utterance has been sent, and
  force-released if the connection drops.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import wire

GRANT_TIMEOUT = 2.0  # seconds to wait for FloorGrant

# Who holds the floor, as the window shows it.
FREE, ASKING, ME, THEM = "free", "asking", "me", "them"


@dataclass(frozen=True)
class Holder:
    who: str  # FREE | ASKING | ME | THEM
    name: str = ""  # the other PC's, when it is THEM


# What the peer thread must do because the floor changed.


@dataclass(frozen=True)
class Send:
    message: wire.Wire


@dataclass(frozen=True)
class BeginTurn:
    """The floor is ours: open the microphone."""


@dataclass(frozen=True)
class EndTurn:
    """The floor has gone: close the microphone and finish the turn."""


@dataclass(frozen=True)
class Refused:
    """A turn was asked for and will not happen. Shown on screen."""

    why: str


@dataclass(frozen=True)
class Changed:
    holder: Holder


class Floor:
    """The floor, as this PC sees it."""

    def __init__(self, them: str, wins_ties: bool) -> None:
        self.them = them
        self.wins_ties = wins_ties  # this PC wins when both ask at once
        self._state = FREE
        self._seq = 0  # of the request or the hold
        self._since = 0.0
        self._cancelled = False  # the key was released before the answer came
        self._next_seq = 1

    def holder(self) -> Holder:
        return Holder(THEM, self.them) if self._state == THEM else Holder(self._state)

    def request(self, now: float) -> list:
        """The turn key was pressed."""
        if self._state == FREE:
            self._seq, self._next_seq = self._next_seq, self._next_seq + 1
            self._state, self._since, self._cancelled = ASKING, now, False
            return [Send(wire.FloorRequest(self._seq)), Changed(Holder(ASKING))]
        if self._state == THEM:
            return [Refused(f"{self.them} has the floor. Wait for them to finish.")]
        return []  # already asking, or already ours

    def cancel(self) -> list:
        """The turn key was released, or pressed again, before the answer came."""
        if self._state == ASKING:
            self._cancelled = True
        return []

    def release(self) -> list:
        """The turn's utterance has been sent, or there was none to send."""
        if self._state != ME:
            return []
        self._state = FREE
        return [Send(wire.FloorRelease(self._seq)), Changed(Holder(FREE))]

    def tick(self, now: float) -> list:
        """Check the grant timeout."""
        if self._state != ASKING or now - self._since < GRANT_TIMEOUT:
            return []
        self._state = FREE
        actions: list = [Changed(Holder(FREE))]
        if not self._cancelled:
            actions.append(Refused(f"{self.them} did not answer within {GRANT_TIMEOUT:g} s, so the microphone "
                                   "stayed closed. Try again."))
        return actions

    def receive(self, message: wire.Wire) -> list:
        """A message about the floor arrived from the other side."""
        if isinstance(message, wire.FloorRequest):
            return self._their_request(message.seq)
        if isinstance(message, wire.FloorGrant):
            return self._their_grant(message.seq)
        if isinstance(message, wire.FloorRelease):
            if self._state == THEM and self._seq == message.seq:
                self._state = FREE
                return [Changed(Holder(FREE))]
        return []

    def disconnected(self) -> list:
        """The connection has gone: the floor is force-released."""
        before, cancelled = self._state, self._cancelled
        self._state = FREE
        if before == FREE:
            return []
        if before == ME:
            return [EndTurn(), Changed(Holder(FREE))]
        if before == ASKING:
            actions: list = [Changed(Holder(FREE))]
            if not cancelled:
                actions.append(Refused("the connection dropped before the floor was granted"))
            return actions
        return [Changed(Holder(FREE))]

    def _grant(self, seq: int) -> Send:
        self._state, self._seq = THEM, seq
        return Send(wire.FloorGrant(seq))

    def _their_request(self, seq: int) -> list:
        if self._state in (FREE, THEM):
            return [self._grant(seq), Changed(self.holder())]
        if self._state == ASKING:
            # Both asked at once. The winner ignores the other's request and
            # waits for its own grant, which the loser is about to send.
            if self.wins_ties and not self._cancelled:
                return []
            cancelled = self._cancelled
            actions: list = [self._grant(seq), Changed(self.holder())]
            if not cancelled:
                actions.append(Refused(f"{self.them} asked at the same moment and has the floor."))
            return actions
        return []  # we are talking. No answer: they time out and are told.

    def _their_grant(self, seq: int) -> list:
        if self._state == ASKING and self._seq == seq and not self._cancelled:
            self._state = ME
            return [BeginTurn(), Changed(Holder(ME))]
        if self._state == ME and self._seq == seq:
            return []  # the same grant twice: the turn is already ours
        # A grant nobody is waiting for: cancelled, timed out, or stale. The
        # other side now believes we hold the floor, so hand it back at once
        # rather than leave both ends disagreeing.
        was_asking = self._state == ASKING and self._seq == seq
        if was_asking:
            self._state = FREE
        actions = [Send(wire.FloorRelease(seq))]
        if was_asking:
            actions.append(Changed(Holder(FREE)))
        return actions


def wins_ties(me: str, them: str, my_addr: str, their_addr: str) -> bool:
    """Whether this PC wins a simultaneous request: its name sorts first. Two
    PCs with the same name fall back to their addresses, which the two ends
    see the same way round, so they still agree. Compared as Rust compares
    strings, by their UTF-8 bytes."""
    return (me.encode(), my_addr.encode()) < (them.encode(), their_addr.encode())
