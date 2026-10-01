"""Finding other Volis PCs on the local link (SPEC §9), so the peer panel can
offer a pick-list instead of making someone type an address.

Port of Rust `discovery.rs`; the announcement is the same, so pyvolis and
Rust volis find each other. Convenience only: managed switches and VLANs drop
broadcast, so typing the address must always work and nothing depends on
this. A UDP broadcast on port 47801: every couple of seconds each Volis
announces its name and port to every local network it is on, and listens for
the others. Broadcast never leaves the local link, needs no router, no DHCP
and no DNS, and works on a bare cable.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass

from .wire import MAX_NAME, PROTO, is_control

log = logging.getLogger(__name__)

PORT = 47801
ANNOUNCE_EVERY = 2.0  # seconds
FORGET_AFTER = 7.0  # a PC not heard from for this long is taken off the list
MAX_PACKET = 512  # real announcements are under 150 bytes


@dataclass(frozen=True)
class Found:
    """Another Volis, heard on the local network."""

    name: str
    host: str  # the address it was heard from
    port: int  # the port it listens on


def start(name: str, port: int, stop: threading.Event, report) -> threading.Thread:
    """Start announcing and listening. `report` hears the list of other PCs
    whenever it changes. Raises OSError if the port can't be opened."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", PORT))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(0.25)
    except OSError:
        sock.close()
        raise
    own_id = run_id()
    announce = json.dumps({"t": "Announce", "name": name, "port": port, "proto": PROTO, "id": own_id}).encode()
    log.info('discovery: announcing "%s" on UDP port %d', name, PORT)

    def run() -> None:
        seen: dict[tuple[str, int], tuple[str, float]] = {}
        last_announce = None
        reported: list[Found] = []
        try:
            while not stop.is_set():
                if last_announce is None or time.monotonic() - last_announce >= ANNOUNCE_EVERY:
                    last_announce = time.monotonic()
                    for target in broadcast_targets():
                        # A network that refuses broadcast is simply not
                        # searched; typing the address still works there.
                        try:
                            sock.sendto(announce, (target, PORT))
                        except OSError as e:
                            log.debug("discovery: cannot announce to %s: %s", target, e)
                try:
                    packet, sender = sock.recvfrom(MAX_PACKET)
                    parsed = parse(packet, own_id)
                    if parsed is not None:
                        seen[(sender[0], parsed[1])] = (parsed[0], time.monotonic())
                except (TimeoutError, OSError):
                    pass
                now = time.monotonic()
                seen = {k: v for k, v in seen.items() if now - v[1] < FORGET_AFTER}
                found = [Found(name, host, port) for (host, port), (name, _) in sorted(seen.items())]
                if found != reported:
                    reported = found
                    report(list(found))
        finally:
            sock.close()

    thread = threading.Thread(target=run, name="pyvolis-discovery", daemon=True)
    thread.start()
    return thread


def parse(packet: bytes, own_id: int) -> tuple[str, int] | None:
    """Read an announcement. None for this PC's own, another version's, or
    anything that is not an announcement at all."""
    try:
        data = json.loads(packet.decode("utf-8"))
        t, name, port, proto, ident = data["t"], data["name"], data["port"], data["proto"], data["id"]
    except (ValueError, KeyError, TypeError):
        return None
    if not (isinstance(name, str) and all(type(x) is int for x in (port, proto, ident))):
        return None
    if t != "Announce" or proto != PROTO or ident == own_id or not 0 < port <= 65535:
        return None
    cleaned = "".join(" " if is_control(c) else c for c in name)[:MAX_NAME]
    return cleaned.strip(), port


def interfaces() -> list[tuple[str, str, str | None]]:
    """(interface name, IPv4 address, netmask) for every network card but
    loopback. Read from the operating system; nothing is looked up."""
    import psutil

    found = []
    for name, addresses in psutil.net_if_addrs().items():
        for a in addresses:
            if a.family == socket.AF_INET and not a.address.startswith("127."):
                found.append((name, a.address, a.netmask))
    return found


def broadcast_targets() -> list[str]:
    """Every local IPv4 network's broadcast address, and the all-networks one.

    Windows sends 255.255.255.255 out of one interface only, so each network's
    own broadcast address is used as well; that is what reaches a second
    network card with a cable to the other laptop."""
    targets = {"255.255.255.255"}
    for _name, address, netmask in interfaces():
        try:
            network = ipaddress.ip_network(f"{address}/{netmask}", strict=False)
        except (ValueError, TypeError):
            continue
        if network.prefixlen < 31:
            targets.add(str(network.broadcast_address))
    return sorted(targets)


def run_id() -> int:
    """A number that differs between runs."""
    return (time.time_ns() ^ (os.getpid() << 32)) & (2**64 - 1)
