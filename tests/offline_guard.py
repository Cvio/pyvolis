"""Run Python code with every connection off the local network refused and
recorded.

    python tests/offline_guard.py <script.py> [args...]

The guard patches socket connection and name lookup before the script runs.
An attempt to reach anything but loopback, private (RFC 1918), link-local or
unique-local addresses raises, and is also written to stderr as a line
starting `OFFLINE-GUARD:` with the address and the Python stack that made it,
so a library that catches the error and carries on is still caught. Local
addresses stay allowed: paired mode uses them, under the Rust rule "would it
still work with the WAN cable unplugged and no DNS server on the segment?".
Name lookups are allowed only for "localhost" and literal IP addresses; any
other name would need a DNS server.
"""

from __future__ import annotations

import ipaddress
import runpy
import socket
import sys
import traceback

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo
_real_create_connection = socket.create_connection


def is_local(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False  # a name: resolving it needs DNS
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified


class OfflineViolation(OSError):
    pass


def _refuse(what: str, host: str) -> None:
    stack = "".join(traceback.format_stack(limit=25)[:-2])
    sys.stderr.write(f"OFFLINE-GUARD: {what} {host!r} refused\n{stack}OFFLINE-GUARD-END\n")
    sys.stderr.flush()
    raise OfflineViolation(f"offline guard: {what} {host!r} is not on the local network")


def _host_of(address) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return ""  # AF_UNIX paths and the like are local


def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not is_local(_host_of(address)):
        _refuse("connect to", _host_of(address))
    return _real_connect(self, address)


def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not is_local(_host_of(address)):
        _refuse("connect to", _host_of(address))
    return _real_connect_ex(self, address)


def getaddrinfo(host, *args, **kwargs):
    if host is not None and not is_local(host if isinstance(host, str) else host.decode()):
        _refuse("name lookup of", str(host))
    return _real_getaddrinfo(host, *args, **kwargs)


def install() -> None:
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo


if __name__ == "__main__":
    install()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
