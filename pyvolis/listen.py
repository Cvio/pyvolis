"""`--listen`: the pipeline with no window. Port of Rust `listen.rs`.

The pipeline logs everything worth reading itself; this only starts it, stops
it when time is up, and makes sure an error reaches the exit code.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import time
from pathlib import Path

from .config import Config
from .pipeline import Error, Final, Listening, Options, Pipeline, Stopped

log = logging.getLogger(__name__)


def run(root: Path, config: Config, seconds: float | None, options: Options) -> int:
    # A terminal has no turn key, so --listen always listens continuously, and
    # doesn't pair (both need the window).
    if config.mode.kind != "continuous":
        log.info("--listen listens continuously; turn-taking is in the window")
        config = dataclasses.replace(config, mode=dataclasses.replace(config.mode, kind="continuous"))
    if config.peer.enabled:
        log.info("--listen does not pair; paired mode is in the window")
        config = dataclasses.replace(config, peer=dataclasses.replace(config.peer, enabled=False))

    events: queue.Queue = queue.Queue()
    pipeline = Pipeline(root, config, options, events).start()
    deadline = None
    errors: list[str] = []
    heard = 0
    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                pipeline.stop()
                deadline = None
            try:
                msg = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(msg, Listening):
                # The clock starts once the device is open, not while models load.
                if seconds is not None:
                    log.info("listening for %s s", seconds)
                    deadline = time.monotonic() + seconds
                else:
                    log.info("Ctrl-C to stop")
            elif isinstance(msg, Final):
                heard += 1
            elif isinstance(msg, Error):
                errors.append(msg.message)
            elif isinstance(msg, Stopped):
                break
    except KeyboardInterrupt:
        pipeline.stop()
        pipeline.join()
    if heard == 0 and not errors:
        log.info(
            "no speech recognised. If that is wrong, check --devices, and consider lowering "
            "[vad].threshold from %s.", config.vad.threshold,
        )
    # A pipeline that failed to start is a failed run.
    if errors and heard == 0:
        log.error("%s", errors[0])
        return 1
    return 0
