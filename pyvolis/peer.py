"""Paired mode (SPEC §9): two Volis instances on a local network as the two
ends of one conversation. Port of Rust `peer.rs`; the two pair with each other.

Each machine runs its own complete pipeline and sends the other only the
translated **text**. The receiver shows it and speaks it with its own voice.
No audio, no models and no inference cross the wire.

Everything here runs on its own threads, so a stalled or dead peer can never
block local capture, recognition or the window:

* the **peer thread** owns the connection, the handshake and the floor token,
  and is the only thing that writes to the socket;
* an **acceptor** waits for the other side to dial in;
* a **dialler** makes one outgoing attempt when the user presses Connect;
* a **reader** per connection turns lines into messages.

Only IP addresses are accepted. A name would be resolved by the system's
resolver, which can mean a public DNS server, and the first hard constraint
forbids that. Everything works on a cable or a switch with no router, no DHCP
and no DNS.
"""

from __future__ import annotations

import errno
import ipaddress
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass

from . import discovery, floor, varieties, wire
from .events import Discovered, Error, FloorChanged, FloorRefused, NotSent, PeerMsg, Remote, Sent

log = logging.getLogger(__name__)

DEFAULT_PORT = 47800  # used when an address is typed without one
CONNECT_TIMEOUT = 4.0  # how long a dial may take before it counts as unanswered
PING_EVERY = 2.0  # how often a live connection is pinged
# Silence after which the other side is taken to be gone. A pulled cable sends
# nothing at all, not even a reset; this is how it is noticed.
DEAD_AFTER = 6.0
HELLO_TIMEOUT = 5.0  # how long a connection may go without saying Hello
WRITE_TIMEOUT = 2.0  # a write that takes longer than this means the link is gone
TICK = 0.1  # how often the peer thread wakes with nothing to do, for the timers


class AddressError(ValueError):
    pass


@dataclass(frozen=True)
class Address:
    """An IP address and a port. Ordered as Rust orders a SocketAddr: IPv4
    before IPv6, then by address, then by port."""

    host: str
    port: int

    def __str__(self) -> str:
        return f"[{self.host}]:{self.port}" if ":" in self.host else f"{self.host}:{self.port}"

    def key(self) -> tuple:
        ip = ipaddress.ip_address(self.host.split("%")[0])
        return (ip.version, int(ip), self.port)


def parse_address(text: str) -> Address:
    """What the user typed as the other PC's address. An IP address, with the
    port optional. Names are refused rather than looked up."""
    text = text.strip()
    if not text:
        raise AddressError("type the other PC's address, e.g. 192.168.50.2")
    host, port = text, DEFAULT_PORT
    try:
        if text.startswith("[") and "]:" in text:  # [v6]:port
            host, _, tail = text[1:].partition("]:")
            port = int(tail)
        elif text.count(":") == 1:  # v4:port
            host, _, tail = text.partition(":")
            port = int(tail)
        ipaddress.ip_address(host)
        if not 0 <= port <= 65535:
            raise ValueError(port)
    except ValueError:
        raise AddressError(
            f'"{text}" is not an IP address. Type the numbers shown on the other PC, e.g. 192.168.50.2 '
            "(names are never looked up, because that could need the internet)") from None
    return Address(host, port)


def display_name(configured: str) -> str:
    """`[peer].display_name`, or the computer's name when it is empty."""
    name = configured.strip() or socket.gethostname()
    name = "".join(c for c in name if not wire.is_control(c))[: wire.MAX_NAME]
    return name or "Volis"


@dataclass(frozen=True)
class Me:
    """Who this PC is, as Hello says it."""

    name: str
    speaks: str
    sends: str


@dataclass
class Outgoing:
    """One translated sentence on its way to the other side."""

    id: str  # the local sentence it belongs to
    lang: str
    text: str
    source_lang: str
    source_text: str


@dataclass
class Wiring:
    """Where the peer thread sends things."""

    begin_turn: callable  # straight to the pipeline: a granted floor opens the microphone
    end_turn: callable
    speak: callable  # (lang, text) -> "ok" | "off" | "full": a received utterance to say aloud
    emit: callable  # an event for the window


class _Link:
    """One TCP connection, before or after Hello."""

    def __init__(self, ident: int, sock: socket.socket, outbound: bool) -> None:
        self.id = ident
        self.sock = sock
        self.local = Address(*sock.getsockname()[:2])
        self.remote = Address(*sock.getpeername()[:2])
        self.outbound = outbound  # whether this PC dialled it
        self.hello: Me | None = None
        self.opened = self.last_ping = time.monotonic()

    def dialler(self) -> Address:
        """The end that dialled. Two connections between the same pair are
        told apart by it, and both ends see it the same way round."""
        return self.local if self.outbound else self.remote

    def name(self) -> str:
        return self.hello.name if self.hello else self.remote.host

    def write(self, message: wire.Wire) -> None:
        self.sock.sendall(wire.encode(message).encode("utf-8"))

    def shut(self) -> None:
        for action in (lambda: self.sock.shutdown(socket.SHUT_RDWR), self.sock.close):
            try:
                action()
            except OSError:
                pass


class Peer:
    """Paired mode, running. Every public method only queues a request for
    the peer thread and returns at once."""

    def __init__(self, me: Me, mode: str, listen_addr: str, discovery_on: bool, wiring: Wiring) -> None:
        self.me = me
        self.mode = mode
        self.wiring = wiring
        self.port = 0
        self._listen_addr = listen_addr.strip()
        self._discovery_on = discovery_on
        self._inbox: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._links: list[_Link] = []
        self._active: int | None = None
        self._floor: floor.Floor | None = None
        self._next_id = 1
        self._next_seq = 1
        self._ever_accepted = False  # part of the firewall diagnostic
        self._dialling: Address | None = None
        self._listening = False
        self._helpers: list[threading.Thread] = []
        self._thread = threading.Thread(target=self._run, name="pyvolis-peer", daemon=True)

    # ---- for the rest of pyvolis

    def start(self) -> Peer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def connect(self, address: Address) -> None:
        self._inbox.put(("connect", address))

    def disconnect(self) -> None:
        self._inbox.put(("disconnect",))

    def want_turn(self) -> None:
        """The turn key asked for a turn. In turn mode, while connected, this
        asks for the floor; otherwise the turn starts at once."""
        self._inbox.put(("want_turn",))

    def end_turn(self) -> None:
        """The turn key ended the turn, or cancelled a request still in flight."""
        self._inbox.put(("end_turn",))

    def set_mode(self, mode: str) -> None:
        self._inbox.put(("set_mode", mode))

    def deliver(self, out: Outgoing) -> None:
        self._inbox.put(("deliver", out))

    def release_floor(self) -> None:
        """The turn's sentences have been sent, or there were none."""
        self._inbox.put(("release_floor",))

    # ---- the peer thread

    def _emit(self, event) -> None:
        self.wiring.emit(event)

    def _state(self, kind: str, **fields) -> None:
        self._emit(PeerMsg(kind, **fields))

    def _run(self) -> None:
        self._listen()
        while not self._stop.is_set():
            try:
                self._handle(self._inbox.get(timeout=TICK))
            except queue.Empty:
                pass
            except Exception:  # a bug here must not take pairing down silently
                log.exception("paired mode: an input could not be handled")
            self._timers()
        self._shut_down()
        for helper in self._helpers:
            helper.join(2.0)
        log.info("paired mode stopped")

    def _listen(self) -> None:
        """Bind the listener and start the helper threads."""
        try:
            try:
                address = parse_address(self._listen_addr)
                if self._listen_addr.count(":") == 0:
                    raise AddressError()
            except AddressError:
                raise OSError(f'[peer].listen_addr = "{self._listen_addr}" is not an address and port') from None
            family = socket.AF_INET6 if ":" in address.host else socket.AF_INET
            listener = socket.socket(family, socket.SOCK_STREAM)
            try:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: never share the port
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                listener.bind((address.host, address.port))
                listener.listen(8)
            except OSError as e:
                listener.close()
                raise OSError(f"cannot listen on {address}. Is another Volis already running on this PC? Two on "
                              f"one PC need different [peer].listen_addr ports: {e}") from None
        except OSError as e:
            log.warning("%s", e)
            self._state("unavailable", reason=str(e))
            return
        self.port = listener.getsockname()[1]
        self._listening = True
        log.info('paired mode: listening on port %d as "%s"', self.port, self.me.name)
        self._state("waiting", port=self.port)
        acceptor = threading.Thread(target=self._accept, args=(listener,), name="pyvolis-accept", daemon=True)
        acceptor.start()
        self._helpers.append(acceptor)
        # Discovery is convenience only: if it cannot start, typing the
        # address still works, so a failure is logged and nothing more.
        if self._discovery_on:
            try:
                self._helpers.append(discovery.start(self.me.name, self.port, self._stop,
                                                     lambda found: self._inbox.put(("found", found))))
            except OSError as e:
                log.warning("discovery is off: cannot open UDP port %d for discovery: %s", discovery.PORT, e)

    def _accept(self, listener: socket.socket) -> None:
        """Wait for the other side to dial in, noticing the stop."""
        listener.settimeout(0.1)
        try:
            while not self._stop.is_set():
                try:
                    sock, _remote = listener.accept()
                except TimeoutError:
                    continue
                except OSError as e:
                    log.warning("paired mode: accept failed: %s", e)
                    time.sleep(0.2)
                    continue
                self._inbox.put(("accepted", sock))
        finally:
            listener.close()

    def _handle(self, item: tuple) -> None:
        kind = item[0]
        if kind == "accepted":
            self._ever_accepted = True
            log.info("paired mode: %s connected in", Address(*item[1].getpeername()[:2]))
            self._add_link(item[1], outbound=False)
        elif kind == "dialled":
            _, address, result = item
            if self._dialling != address:
                if not isinstance(result, OSError):
                    result.close()
                return
            self._dialling = None
            if isinstance(result, OSError):
                log.warning("paired mode: cannot connect to %s: %s", address, result)
                self._state("disconnected", port=self.port,
                            reason=diagnose(result, address, self._ever_accepted, self.port))
            else:
                log.info("paired mode: connected out to %s", address)
                self._add_link(result, outbound=True)
        elif kind == "line":
            _, ident, message = item
            if isinstance(message, wire.WireError):
                sender = self._link_name(ident)
                log.warning("paired mode: from %s: %s", sender, message)
                self._emit(Error(f"From {sender}: {message}"))
                # A version mismatch or a runaway line cannot be recovered
                # from on this connection. Anything else is skipped, which is
                # what makes a hand-typed netcat session usable.
                if isinstance(message, (wire.UnknownProto, wire.TooLong)):
                    self._close(ident, str(message), f"{sender} could not be understood: {message}")
            else:
                self._receive(ident, message)
        elif kind == "closed":
            _, ident, reason = item
            self._drop_link(ident, f"{self._link_name(ident)} {reason}")
        elif kind == "found":
            self._emit(Discovered([(f.name, f.host, f.port) for f in item[1]]))
        else:
            self._command(item)

    def _command(self, item: tuple) -> None:
        kind = item[0]
        if kind == "connect":
            address = item[1]
            link = self._active_link()
            if link is not None:
                self._emit(Error(f"Already paired with {link.name()}. Disconnect first."))
                return
            if not self._listening:
                self._emit(Error("Paired mode is unavailable on this PC; see the peer panel"))
                return
            self._dialling = address
            self._state("connecting", addr=str(address))

            def dial() -> None:
                try:
                    result = socket.create_connection((address.host, address.port), CONNECT_TIMEOUT)
                except OSError as e:
                    result = e
                self._inbox.put(("dialled", address, result))

            threading.Thread(target=dial, name="pyvolis-dial", daemon=True).start()
        elif kind == "disconnect":
            self._dialling = None
            if self._active is not None:
                self._close(self._active, None, "you disconnected")
            else:
                self._state("waiting", port=self.port)
        elif kind == "want_turn":
            if self._floor is not None and self.mode == "turn":
                self._act(self._floor.request(time.monotonic()))
            else:  # not connected, or continuous: nobody to ask
                self.wiring.begin_turn()
        elif kind == "end_turn":
            if self._floor is not None:
                self._act(self._floor.cancel())
            self.wiring.end_turn()
        elif kind == "set_mode":
            self.mode = item[1]
        elif kind == "deliver":
            self._deliver(item[1])
        elif kind == "release_floor":
            if self._floor is not None:
                self._act(self._floor.release())

    def _deliver(self, out: Outgoing) -> None:
        seq, self._next_seq = self._next_seq, self._next_seq + 1
        if self._active is None:
            self._emit(NotSent(out.id, "not connected to another PC"))
            return
        to = self._write_to(self._active, wire.Utterance(seq, out.lang, out.text, out.source_lang, out.source_text))
        if to is None:
            self._emit(NotSent(out.id, "the connection dropped"))
        else:
            log.debug("paired mode: sentence %s sent to %s", out.id, to)
            self._emit(Sent(out.id, to))

    def _receive(self, ident: int, message: wire.Wire) -> None:
        if isinstance(message, wire.Hello):
            self._hello(ident, Me(message.name, message.speaks, message.sends))
            return
        if isinstance(message, wire.Bye):
            name = self._link_name(ident)
            why = f"{name} ended the connection" + (f": {message.reason}" if message.reason else "")
            self._drop_link(ident, why)
            return
        if self._active != ident:
            return  # nothing but Hello and Bye counts before the handshake
        if isinstance(message, wire.Utterance):
            sender = self._link_name(ident)
            log.info("paired mode: from %s\n  [%s] %s\n  [%s] %s", sender, message.source_lang, message.source_text,
                     message.lang, message.text)
            self._emit(Remote(sender, message.lang, message.text, message.source_lang, message.source_text))
            # "off": no speaker, speech is off. Captions are enough.
            if self.wiring.speak(message.lang, message.text) == "full":
                self._emit(Error(f"too much arrived from {sender} at once; an utterance was shown but not spoken"))
        elif isinstance(message, (wire.FloorRequest, wire.FloorGrant, wire.FloorRelease)):
            if self._floor is not None:
                self._act(self._floor.receive(message))
        elif isinstance(message, wire.Ping):
            self._write_to(ident, wire.Pong())

    def _hello(self, ident: int, hello: Me) -> None:
        """A connection has said who it is: make it the peer, or refuse it."""
        new = self._link(ident)
        if new is None or new.hello is not None:
            return
        new.hello = hello
        if self._active is None:
            self._activate(ident)
            return
        current = self._active_link()
        # Both ends dialled at once: two connections to the same PC. Both
        # ends keep the one with the lower dialling address, so they settle
        # on the same one.
        if current is not None and current.remote.host == new.remote.host and current.hello \
                and current.hello.name == hello.name:
            log.debug("paired mode: two connections to %s; keeping one", hello.name)
            if new.dialler().key() < current.dialler().key():
                old, self._active, self._floor = self._active, None, None
                self._discard(old, "duplicate connection")
                self._activate(ident)
            else:
                self._discard(ident, "duplicate connection")
            return
        # Exactly one peer at a time. The newcomer is told why.
        paired_with = current.name() if current else ""
        reason = f"{self.me.name} is already paired with {paired_with}"
        log.warning("paired mode: refused %s: %s", hello.name, reason)
        self._emit(Error(f"Refused a connection from {hello.name}: already paired with {paired_with}"))
        self._discard(ident, reason)

    def _activate(self, ident: int) -> None:
        """Make a connection that has said Hello the peer."""
        link = self._link(ident)
        if link is None or link.hello is None:
            return
        hello = link.hello
        self._active = ident
        self._dialling = None
        self._floor = floor.Floor(hello.name, floor.wins_ties(self.me.name, hello.name, str(link.local), str(link.remote)))
        log.info('paired mode: paired with "%s" at %s, who speaks %s and sends %s', hello.name, link.remote,
                 hello.speaks, hello.sends)
        self._state("connected", name=hello.name, addr=str(link.remote), speaks=hello.speaks, sends=hello.sends)
        self._emit(FloorChanged(floor.FREE))
        # What they send is what this PC will speak. Say so if it is not the
        # language this PC's person speaks; it is not an error, and the
        # receiving voice is chosen by each utterance's own language anyway.
        # Compared by language: es-MX sent to someone set to plain es is fine.
        if varieties.language_of(hello.sends).lower() != varieties.language_of(self.me.speaks).lower():
            self._emit(Error(f"{hello.name} translates into {hello.sends}, but this PC is set up for someone who "
                             f"speaks {self.me.speaks}. Check the languages on both PCs."))

    def _act(self, actions: list) -> None:
        """Carry out what the floor decided."""
        for action in actions:
            if isinstance(action, floor.Send):
                if self._active is not None:
                    self._write_to(self._active, action.message)
            elif isinstance(action, floor.BeginTurn):
                self.wiring.begin_turn()
            elif isinstance(action, floor.EndTurn):
                self.wiring.end_turn()
            elif isinstance(action, floor.Refused):
                log.info("paired mode: turn refused: %s", action.why)
                self._emit(FloorRefused(action.why))
            elif isinstance(action, floor.Changed):
                self._emit(FloorChanged(action.holder.who, action.holder.name))

    def _timers(self) -> None:
        now = time.monotonic()
        if self._floor is not None:
            self._act(self._floor.tick(now))
        for link in list(self._links):
            if link.hello is None and now - link.opened > HELLO_TIMEOUT:
                self._close(link.id, "no Hello", f"{link.name()} connected but never said who it was; is it Volis?")
            elif link.id == self._active and now - link.last_ping >= PING_EVERY:
                link.last_ping = now
                self._write_to(link.id, wire.Ping())

    def _add_link(self, sock: socket.socket, outbound: bool) -> None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            reader = sock.dup()  # its own timeout: a read may wait far longer than a write
            # Anything at all, a Pong included, arrives at least every
            # PING_EVERY on a live link. A read that waits longer than
            # DEAD_AFTER is a dead link.
            reader.settimeout(DEAD_AFTER)
            sock.settimeout(WRITE_TIMEOUT)
            link = _Link(self._next_id, sock, outbound)
        except OSError as e:
            log.warning("paired mode: cannot use a new connection: %s", e)
            sock.close()
            return
        self._next_id += 1
        try:
            link.write(wire.Hello(self.me.name, self.me.speaks, self.me.sends))
        except OSError as e:
            log.warning("paired mode: cannot greet %s: %s", link.remote, e)
            reader.close()
            link.shut()
            return
        self._links.append(link)
        threading.Thread(target=self._read, args=(link.id, reader), name="pyvolis-peer-read", daemon=True).start()

    def _read(self, ident: int, sock: socket.socket) -> None:
        """Turn one connection's bytes into messages. Ends when the connection does."""
        buffer = b""
        reason = ""
        try:
            while not reason:
                newline = buffer.find(b"\n")
                if newline < 0:
                    if len(buffer) > wire.MAX_LINE:  # an over-long line is seen as such, not split in two
                        self._inbox.put(("line", ident, wire.TooLong()))
                        reason = "sent a message that was too long"
                        break
                    try:
                        data = sock.recv(4096)
                    except TimeoutError:
                        reason = f"went silent: nothing heard for {DEAD_AFTER:g} s. Was a cable pulled?"
                        break
                    except OSError as e:
                        reason = f"dropped the connection ({e})"
                        break
                    if not data:
                        reason = "closed the connection"
                        break
                    buffer += data
                    continue
                line, buffer = buffer[: newline + 1], buffer[newline + 1:]
                if not line.strip():
                    continue
                try:
                    message = wire.decode(line)
                except wire.WireError as e:
                    message = e
                self._inbox.put(("line", ident, message))
        finally:
            sock.close()
        self._inbox.put(("closed", ident, reason))

    def _write_to(self, ident: int, message: wire.Wire) -> str | None:
        """Write to one connection. A failed write closes it. Returns the
        other side's name when the write went out."""
        link = self._link(ident)
        if link is None:
            return None
        try:
            link.write(message)
            return link.name()
        except OSError as e:
            self._drop_link(ident, f"{link.name()} stopped answering ({e})")
            return None

    def _close(self, ident: int, reason: str | None, why: str) -> None:
        """Say goodbye on a connection and close it."""
        link = self._link(ident)
        if link is not None:
            try:
                link.write(wire.Bye(reason))
            except OSError:
                pass
        self._drop_link(ident, why)

    def _discard(self, ident: int, reason: str) -> None:
        """Close a connection that is not the peer, quietly."""
        link = self._link(ident)
        if link is None:
            return
        self._links.remove(link)
        try:
            link.write(wire.Bye(reason))
        except OSError:
            pass
        link.shut()

    def _drop_link(self, ident: int, why: str) -> None:
        """A connection is gone. If it was the peer, the floor is
        force-released and the window is told."""
        link = self._link(ident)
        if link is None:
            return
        self._links.remove(link)
        link.shut()
        if self._active == ident:
            self._active = None
            held, self._floor = self._floor, None
            # Both ends dialled at once, and the other end closed the spare
            # connection first: carry on over the one that is left.
            spare = next((other for other in self._links if other.remote.host == link.remote.host
                          and other.hello is not None and link.hello is not None
                          and other.hello.name == link.hello.name), None)
            if spare is not None:
                log.debug("paired mode: carrying on over the other connection")
                self._activate(spare.id)
                return
            log.warning("paired mode: disconnected: %s", why)
            # The disconnected state first, so the window never shows a free
            # floor on a link that is already gone.
            self._state("disconnected", port=self.port, reason=why)
            if held is not None:
                self._act(held.disconnected())
        elif link.outbound and link.hello is None:
            # Our own dial, refused before it was ever a pairing.
            self._state("disconnected", port=self.port, reason=why)

    def _link(self, ident: int) -> _Link | None:
        return next((link for link in self._links if link.id == ident), None)

    def _active_link(self) -> _Link | None:
        return self._link(self._active) if self._active is not None else None

    def _link_name(self, ident: int) -> str:
        link = self._link(ident)
        return link.name() if link else "the other PC"

    def _shut_down(self) -> None:
        for link in self._links:
            try:
                link.write(wire.Bye("Volis was stopped"))
            except OSError:
                pass
            link.shut()
        self._links.clear()


def start(config, mode: str, wiring: Wiring) -> Peer:
    """Start listening, and the thread that runs paired mode. Returns at
    once; a port that cannot be opened is reported as "unavailable", not as
    an error, so the rest of pyvolis still runs."""
    me = Me(display_name(config.peer.display_name), config.languages.source, config.languages.target)
    return Peer(me, mode, config.peer.listen_addr, config.peer.discovery, wiring).start()


REFUSED = {errno.ECONNREFUSED, 10061}
UNREACHABLE = {errno.EHOSTUNREACH, errno.ENETUNREACH, 10065, 10051}
NOT_AVAILABLE = {errno.EADDRNOTAVAIL, 10049}


def diagnose(error: OSError, address: Address, ever_accepted: bool, port: int) -> str:
    """Explain a failed dial in terms of what to do about it (SPEC §10).

    A refusal means the packets arrived and nothing was listening. A timeout
    means they were silently dropped, which on a cable or a router-less switch
    is nearly always the Windows firewall treating an unidentified network as
    Public."""
    code = getattr(error, "winerror", None) or error.errno
    if isinstance(error, ConnectionRefusedError) or code in REFUSED:
        return (f'Nothing is listening at {address}. On that PC, is Volis running with "Pair with another PC" '
                f"ticked and Start pressed? Is {address.port} the port it shows?")
    if isinstance(error, TimeoutError) or code in (errno.ETIMEDOUT, 10060):
        text = (f"No answer from {address} within {CONNECT_TIMEOUT:g} s. The connection is being silently dropped, "
                "which is almost always a firewall. On Windows, a cable or a switch with no router makes a network "
                "Windows cannot identify, and it usually calls it Public, which blocks incoming connections. On "
                "the other PC, open Settings › Network & internet, choose this network's adapter, and set its "
                "network profile to Private, or allow Volis through Windows Defender Firewall. On Linux, allow "
                f"TCP port {address.port} (ufw allow {address.port}/tcp).")
        if not ever_accepted:
            text += (f" This PC has not received a connection either since it started listening on port {port}, "
                     "so check this PC's firewall and network profile too.")
        return text
    if code in UNREACHABLE:
        return (f"There is no route to {address}. Are both PCs on the same cable, switch or Wi-Fi network, and is "
                "the address typed exactly as the other PC shows it?")
    if code in NOT_AVAILABLE:
        return f'{address} cannot be dialled from this PC. Type the address the other PC shows under "This PC".'
    return f"Cannot connect to {address}: {error}"


def local_addresses() -> list[tuple[str, str]]:
    """This PC's network addresses, for the other person to type in, as
    (interface, address). Loopback is left out; nobody else can reach it.
    IPv4 only: it is what people type."""
    return sorted((name, address) for name, address, _mask in discovery.interfaces())
