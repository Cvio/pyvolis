"""Port of the tests in Rust `peer.rs`: two ends on this PC, over loopback.
What SPEC §9 asks for, without a microphone or a model: the handshake, an
utterance each way, the floor, and a dropped connection force-releasing the
floor on both ends."""

import queue
import socket
import time

import pytest

from pyvolis import events as ev
from pyvolis import peer
from pyvolis.config import Config


class End:
    def __init__(self, name: str, speaks: str, sends: str, port: int) -> None:
        config = Config()
        config.peer.enabled = True
        config.peer.display_name = name
        config.peer.listen_addr = f"127.0.0.1:{port}"
        config.peer.discovery = False
        config.languages.source, config.languages.target = speaks, sends
        self.events: queue.Queue = queue.Queue()
        self.pipeline: queue.Queue = queue.Queue()  # what the peer asked of the pipeline
        self.spoken: queue.Queue = queue.Queue()
        wiring = peer.Wiring(
            begin_turn=lambda: self.pipeline.put("begin_turn"), end_turn=lambda: self.pipeline.put("end_turn"),
            speak=lambda lang, text: self.spoken.put((lang, text)) or "ok", emit=self.events.put)
        self.peer = peer.start(config, "turn", wiring)

    def wait_for(self, what: str, want):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            if want(event):
                return event
        pytest.fail(f"never saw {what}")

    def stop(self) -> None:
        self.peer.stop()


@pytest.fixture
def ends():
    made = []

    def make(*args) -> End:
        made.append(End(*args))
        return made[-1]

    yield make
    for end in made:
        end.stop()


def state(kind: str):
    return lambda e: isinstance(e, ev.PeerMsg) and e.kind == kind


def floor_is(holder: str, name: str = ""):
    return lambda e: isinstance(e, ev.FloorChanged) and e.holder == holder and (not name or e.name == name)


def test_an_address_is_an_ip_with_an_optional_port_and_never_a_name():
    assert peer.parse_address("192.168.50.2") == peer.Address("192.168.50.2", 47800)
    assert peer.parse_address(" 10.0.0.7:47802 ") == peer.Address("10.0.0.7", 47802)
    assert peer.parse_address("[fe80::1]:47802") == peer.Address("fe80::1", 47802)
    assert str(peer.Address("::1", 47800)) == "[::1]:47800"
    with pytest.raises(peer.AddressError, match="never looked up"):
        peer.parse_address("laptop-b.local")
    with pytest.raises(peer.AddressError):
        peer.parse_address("")
    with pytest.raises(peer.AddressError):
        peer.parse_address("10.0.0.7:99999")


def test_a_timeout_names_the_firewall_and_the_network_profile():
    address = peer.Address("192.168.50.2", 47800)
    text = peer.diagnose(TimeoutError(), address, False, 47800)
    assert "firewall" in text and "Private" in text, text
    assert "This PC has not received a connection" in text, text
    assert "Nothing is listening" in peer.diagnose(ConnectionRefusedError(), address, True, 47800)


def test_the_name_is_the_configured_one_or_the_computers():
    assert peer.display_name(" front desk ") == "front desk"
    assert peer.display_name("") == socket.gethostname()[:64]
    assert peer.display_name("a\nb" + "x" * 100) == ("ab" + "x" * 100)[:64]


def test_two_ends_pair_talk_take_turns_and_notice_a_drop(ends):
    a = ends("laptop-a", "es", "en", 47911)
    b = ends("laptop-b", "en", "es", 47912)
    a.wait_for("a listening", state("waiting"))
    b.wait_for("b listening", state("waiting"))

    a.peer.connect(peer.parse_address("127.0.0.1:47912"))
    connected = a.wait_for("a connected", state("connected"))
    assert (connected.name, connected.sends) == ("laptop-b", "es")
    b.wait_for("b connected", state("connected"))

    # A takes a turn: the microphone opens only on the grant.
    a.peer.want_turn()
    a.wait_for("a holds the floor", floor_is("me"))
    assert a.pipeline.get(timeout=5) == "begin_turn"
    b.wait_for("b sees a has the floor", floor_is("them", "laptop-a"))

    # B presses its key while A talks: refused, microphone stays shut.
    b.peer.want_turn()
    b.wait_for("b refused", lambda e: isinstance(e, ev.FloorRefused))
    with pytest.raises(queue.Empty):
        b.pipeline.get(timeout=0.3)  # b opened its microphone while a held the floor

    # A's sentence arrives at B as text in B's language, to be spoken.
    a.peer.deliver(peer.Outgoing("1.1", "en", "Where is the station?", "es", "¿Dónde está la estación?"))
    sent = a.wait_for("a sent", lambda e: isinstance(e, ev.Sent))
    assert (sent.id, sent.to) == ("1.1", "laptop-b")
    remote = b.wait_for("b received", lambda e: isinstance(e, ev.Remote))
    assert (remote.sender, remote.text, remote.source_text) == \
        ("laptop-a", "Where is the station?", "¿Dónde está la estación?")
    assert b.spoken.get(timeout=5) == ("en", "Where is the station?"), "the voice follows the utterance's own language"

    # A hands the floor back; B may now talk.
    a.peer.release_floor()
    b.wait_for("b sees the floor free", floor_is("free"))
    b.peer.want_turn()
    b.wait_for("b holds the floor", floor_is("me"))
    a.wait_for("a sees b has the floor", floor_is("them"))

    # B goes away mid-turn. A is told, and the floor is free again.
    b.stop()
    dropped = a.wait_for("a disconnected", state("disconnected"))
    assert "laptop-b" in dropped.reason
    a.wait_for("a's floor force-released", floor_is("free"))


def test_a_link_that_dies_while_this_end_talks_closes_the_microphone(ends):
    a = ends("laptop-a", "es", "en", 47951)
    b = ends("laptop-b", "en", "es", 47952)
    a.peer.connect(peer.parse_address("127.0.0.1:47952"))
    a.wait_for("a connected", state("connected"))
    a.peer.want_turn()
    a.wait_for("a holds the floor", floor_is("me"))
    assert a.pipeline.get(timeout=5) == "begin_turn"
    b.stop()
    a.wait_for("a disconnected", state("disconnected"))
    assert a.pipeline.get(timeout=5) == "end_turn", "the turn ends when the floor is force-released"
    a.peer.deliver(peer.Outgoing("1.1", "en", "hello", "es", "hola"))
    not_sent = a.wait_for("not sent", lambda e: isinstance(e, ev.NotSent))
    assert "not connected" in not_sent.reason


def test_a_second_connection_is_refused_with_a_reason(ends):
    a = ends("laptop-a", "es", "en", 47921)
    b = ends("laptop-b", "en", "es", 47922)
    c = ends("laptop-c", "en", "es", 47923)
    a.peer.connect(peer.parse_address("127.0.0.1:47922"))
    a.wait_for("a connected", state("connected"))
    b.wait_for("b connected", state("connected"))
    c.peer.connect(peer.parse_address("127.0.0.1:47922"))
    refused = c.wait_for("c refused", state("disconnected"))
    assert "already paired with laptop-a" in refused.reason, refused.reason
    b.wait_for("b says it refused", lambda e: isinstance(e, ev.Error) and "laptop-c" in e.message)


def test_dialling_each_other_at_once_ends_with_one_pairing(ends):
    a = ends("laptop-a", "es", "en", 47931)
    b = ends("laptop-b", "en", "es", 47932)
    a.peer.connect(peer.parse_address("127.0.0.1:47932"))
    b.peer.connect(peer.parse_address("127.0.0.1:47931"))
    a.wait_for("a connected", state("connected"))
    b.wait_for("b connected", state("connected"))
    # Whichever connection survived, it carries utterances both ways.
    time.sleep(0.5)
    for sender, receiver in ((a, b), (b, a)):
        sender.peer.deliver(peer.Outgoing("1.1", "en", "hello", "es", "hola"))
        receiver.wait_for("the utterance", lambda e: isinstance(e, ev.Remote))


def test_a_hand_typed_line_is_understood_and_nonsense_is_reported(ends):
    # What netcat would do: connect, say hello, send a line.
    a = ends("laptop-a", "es", "en", 47941)
    a.wait_for("listening", state("waiting"))
    with socket.create_connection(("127.0.0.1", 47941)) as nc:
        nc.sendall(b'{"t":"Hello","name":"netcat","speaks":"en","sends":"es-MX","proto":2}\n'
                   b"this is not json\n"
                   b'{"t":"Utterance","seq":1,"lang":"es-MX","text":"Hola","source_lang":"en","source_text":"Hi"}\n')
        a.wait_for("nonsense reported", lambda e: isinstance(e, ev.Error))
        a.wait_for("the utterance", lambda e: isinstance(e, ev.Remote) and e.text == "Hola" and e.lang == "es-MX")
        nc.sendall(b'{"t":"Hello","name":"x","speaks":"en","sends":"es","proto":9}\n')
        a.wait_for("the version named", lambda e: isinstance(e, ev.Error) and "version 9" in e.message)


def test_a_port_already_in_use_makes_pairing_unavailable_not_a_crash(ends):
    a = ends("laptop-a", "es", "en", 47961)
    a.wait_for("a listening", state("waiting"))
    b = ends("laptop-b", "en", "es", 47961)
    unavailable = b.wait_for("b unavailable", state("unavailable"))
    assert "47961" in unavailable.reason and "another Volis" in unavailable.reason
