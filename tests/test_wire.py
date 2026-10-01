"""Port of the tests in Rust `wire.rs`, `floor.rs` and `discovery.rs`."""

import json

import pytest

from pyvolis import discovery, floor, wire
from pyvolis.floor import ASKING, FREE, ME, THEM, BeginTurn, EndTurn, Floor, Holder, Refused, Send

# ---------------------------------------------------------------- wire.rs

SPEC_LINES = [
    '{"t":"Hello","name":"laptop-a","speaks":"es","sends":"en","proto":2}',
    '{"t":"Utterance","seq":17,"lang":"en","text":"Where is the station?","source_lang":"es",'
    '"source_text":"¿Dónde está la estación?"}',
    '{"t":"FloorRequest","seq":18}',
    '{"t":"FloorGrant","seq":18}',
    '{"t":"FloorRelease","seq":18}',
    '{"t":"Ping"}',
    '{"t":"Pong"}',
    '{"t":"Bye"}',
]


def test_the_spec_examples_parse():
    for line in SPEC_LINES:
        wire.decode(line.encode())
    assert wire.decode(SPEC_LINES[1].encode()) == wire.Utterance(
        17, "en", "Where is the station?", "es", "¿Dónde está la estación?")


def test_what_is_sent_is_one_line_that_decodes_to_itself():
    message = wire.Utterance(3, "es", 'Hola, "amigo".', "en", 'Hello, "friend".')
    line = wire.encode(message)
    assert line.endswith("\n") and line.count("\n") == 1, "one message is one line"
    assert wire.decode(line.encode()) == message


def test_what_is_sent_is_byte_for_byte_what_rust_sends():
    """serde_json's output for the same messages: field order, no spaces,
    non-ASCII as it is."""
    assert wire.encode(wire.Hello("laptop-a", "es", "en")) == SPEC_LINES[0] + "\n"
    assert wire.encode(wire.Utterance(17, "en", "Where is the station?", "es", "¿Dónde está la estación?")) \
        == SPEC_LINES[1] + "\n"
    assert wire.encode(wire.FloorRequest(18)) == SPEC_LINES[2] + "\n"
    assert wire.encode(wire.Ping()) == '{"t":"Ping"}\n'


def test_an_unknown_protocol_version_is_refused_by_name():
    with pytest.raises(wire.UnknownProto, match="version 1") as error:
        wire.decode(b'{"t":"Hello","name":"x","speaks":"es","sends":"en","proto":1}')
    assert error.value.proto == 1


def test_hostile_input_is_refused_or_made_harmless():
    with pytest.raises(wire.NotUtf8):
        wire.decode(b"\xff\xfe{}")
    with pytest.raises(wire.TooLong):
        wire.decode(b" " * (wire.MAX_LINE + 1))
    with pytest.raises(wire.Malformed):
        wire.decode(b"not json")
    with pytest.raises(wire.Malformed):
        wire.decode(b'{"t":"Shutdown"}')
    # A language code is where a path or a prompt would try to hide.
    with pytest.raises(wire.Malformed):
        wire.decode(b'{"t":"Utterance","seq":1,"lang":"../../models","text":"x","source_lang":"es","source_text":"y"}')
    # Too long is refused, not truncated into a different sentence.
    long = json.dumps({"t": "Utterance", "seq": 1, "lang": "en", "text": "a" * (wire.MAX_TEXT + 1),
                       "source_lang": "es", "source_text": "y"})
    with pytest.raises(wire.Malformed):
        wire.decode(long.encode())
    # Control characters, including escaped newlines, become spaces.
    sneaky = '{"t":"Hello","name":"a\\nb\\u0007c","speaks":"es","sends":"en","proto":2}'
    assert wire.decode(sneaky.encode()).name == "a b c"


def test_what_serde_refuses_python_refuses_too():
    for line in (
        '{"t":"Utterance","seq":-1,"lang":"en","text":"x","source_lang":"es","source_text":"y"}',
        '{"t":"Utterance","seq":1.5,"lang":"en","text":"x","source_lang":"es","source_text":"y"}',
        '{"t":"Utterance","seq":true,"lang":"en","text":"x","source_lang":"es","source_text":"y"}',
        '{"t":"Utterance","seq":1,"lang":"en","text":7,"source_lang":"es","source_text":"y"}',
        '{"t":"Utterance","seq":1,"lang":"en","text":"x","source_lang":"es"}',
        '{"t":"FloorRequest"}',
        '{"t":"Hello","name":"x","speaks":"es","sends":"en","proto":"2"}',
        '{"t":"Hello","name":"x","speaks":"","sends":"en","proto":2}',
        '{"t":"Hello","name":"x","speaks":"español","sends":"en","proto":2}',
        '[1,2]', '"Ping"', '{"t":7}', '{"t":"Ping","x":NaN}',
    ):
        with pytest.raises(wire.WireError):
            wire.decode(line.encode())
    with pytest.raises(wire.NotUtf8):  # half a surrogate pair
        wire.decode(b'{"t":"Hello","name":"\\ud800","speaks":"es","sends":"en","proto":2}')
    # Fields nobody asked for are ignored, as serde ignores them.
    assert wire.decode(b'{"t":"Ping","extra":1}') == wire.Ping()


def test_a_bye_may_say_why():
    assert wire.decode(b'{"t":"Bye","reason":"already paired with laptop-a"}') == wire.Bye("already paired with laptop-a")
    assert wire.encode(wire.Bye()) == '{"t":"Bye"}\n'


# ---------------------------------------------------------------- floor.rs


def sent(actions) -> list:
    return [a.message for a in actions if isinstance(a, Send)]


def test_the_microphone_opens_only_when_the_grant_arrives():
    f = Floor("laptop-b", True)
    asked = f.request(0.0)
    assert sent(asked) == [wire.FloorRequest(1)]
    assert BeginTurn() not in asked, "opened before the grant"
    assert f.holder() == Holder(ASKING)
    granted = f.receive(wire.FloorGrant(1))
    assert BeginTurn() in granted and f.holder() == Holder(ME)
    assert sent(f.release()) == [wire.FloorRelease(1)]
    assert f.holder() == Holder(FREE)


def test_no_grant_within_two_seconds_fails_visibly_and_never_opens():
    f = Floor("laptop-b", True)
    f.request(0.0)
    assert f.tick(1.9) == [], "too early"
    timed_out = f.tick(2.0)
    assert any(isinstance(a, Refused) for a in timed_out) and BeginTurn() not in timed_out
    assert f.holder() == Holder(FREE)
    # The grant turns up late. The other side thinks we have the floor; it is
    # handed back, and the microphone still does not open.
    late = f.receive(wire.FloorGrant(1))
    assert BeginTurn() not in late and sent(late) == [wire.FloorRelease(1)]


def test_a_request_is_granted_and_the_floor_is_theirs_until_released():
    f = Floor("laptop-a", False)
    assert sent(f.receive(wire.FloorRequest(7))) == [wire.FloorGrant(7)]
    assert f.holder() == Holder(THEM, "laptop-a")
    refused = f.request(0.0)  # pressing the key now is refused locally, with their name
    assert sent(refused) == [] and "laptop-a" in refused[0].why
    f.receive(wire.FloorRelease(7))
    assert f.holder() == Holder(FREE)


def test_simultaneous_requests_go_to_the_name_that_sorts_first():
    a = Floor("laptop-b", floor.wins_ties("laptop-a", "laptop-b", "", ""))
    b = Floor("laptop-a", floor.wins_ties("laptop-b", "laptop-a", "", ""))
    a_asks, b_asks = sent(a.request(0.0)), sent(b.request(0.0))
    a_hears, b_hears = a.receive(b_asks[0]), b.receive(a_asks[0])
    assert a_hears == [], "the winner ignores the other's request"
    assert sent(b_hears) == [wire.FloorGrant(1)]
    assert any(isinstance(x, Refused) for x in b_hears), "the loser is told the other party has the floor"
    assert b.holder() == Holder(THEM, "laptop-a")
    assert BeginTurn() in a.receive(sent(b_hears)[0]) and a.holder() == Holder(ME)


def test_a_turn_cancelled_before_the_grant_hands_the_floor_straight_back():
    f = Floor("laptop-b", True)
    f.request(0.0)
    f.cancel()
    granted = f.receive(wire.FloorGrant(1))
    assert BeginTurn() not in granted and sent(granted) == [wire.FloorRelease(1)]
    assert f.holder() == Holder(FREE)


def test_a_dropped_connection_force_releases_the_floor():
    f = Floor("laptop-b", True)
    f.request(0.0)
    f.receive(wire.FloorGrant(1))
    assert EndTurn() in f.disconnected(), "the microphone stays open"
    assert f.holder() == Holder(FREE)
    theirs = Floor("laptop-a", False)
    theirs.receive(wire.FloorRequest(1))
    theirs.disconnected()
    assert theirs.holder() == Holder(FREE)


def test_a_request_while_talking_gets_no_answer():
    f = Floor("laptop-b", False)
    f.request(0.0)
    f.receive(wire.FloorGrant(1))
    assert f.receive(wire.FloorRequest(4)) == [] and f.holder() == Holder(ME)


def test_ties_between_equal_names_are_still_decided_the_same_way_on_both_ends():
    a = floor.wins_ties("pc", "pc", "10.0.0.1:50000", "10.0.0.2:47800")
    b = floor.wins_ties("pc", "pc", "10.0.0.2:47800", "10.0.0.1:50000")
    assert a != b


# ---------------------------------------------------------------- discovery.rs


def packet(name: str, port: int, proto: int, ident: int) -> bytes:
    return json.dumps({"t": "Announce", "name": name, "port": port, "proto": proto, "id": ident}).encode()


def test_another_pc_is_found_and_this_one_is_not():
    assert discovery.parse(packet("laptop-b", 47800, wire.PROTO, 2), 1) == ("laptop-b", 47800)
    assert discovery.parse(packet("me", 47800, wire.PROTO, 1), 1) is None, "our own"
    assert discovery.parse(packet("new", 47800, wire.PROTO + 1, 2), 1) is None
    assert discovery.parse(b"hello", 1) is None
    assert discovery.parse(packet("x", 0, wire.PROTO, 2), 1) is None


def test_a_name_cannot_carry_control_characters():
    name, _ = discovery.parse(packet("evil\n\x07pc", 47800, wire.PROTO, 2), 1)
    assert not any(wire.is_control(c) for c in name)


def test_broadcast_goes_to_every_local_network_and_never_further(monkeypatch):
    monkeypatch.setattr(discovery, "interfaces", lambda: [
        ("Wi-Fi", "192.168.1.156", "255.255.255.0"), ("Ethernet", "169.254.49.46", "255.255.0.0"),
        ("VPN", "100.64.100.6", "255.255.255.255")])
    assert discovery.broadcast_targets() == ["169.254.255.255", "192.168.1.255", "255.255.255.255"]
