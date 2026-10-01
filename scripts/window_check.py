"""Drive the window through a file run with no screen, and report how
responsive it stayed (the P5 measurement).

    .venv\\Scripts\\python.exe scripts\\window_check.py <audio file> --from ar --to en --asr <folder> [--shot out.png] [--export dir]

Opens the file, starts a fast run, and while recognition and translation
work, a 20 ms timer on the window's own thread records how late it fires:
the longest gap is how long the window was ever frozen. Nothing is saved to
volis.toml.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# The offscreen platform has no fonts of its own; without these a screenshot is all boxes.
os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "Fonts"))

from pyvolis import paths  # noqa: E402

paths.apply_offline_environment(paths.app_root())
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pyvolis.config import Config, PyvolisConfig  # noqa: E402
from pyvolis.gui import session as ses  # noqa: E402
from pyvolis.gui.window import MainWindow, _select  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--from", dest="source", required=True)
    parser.add_argument("--to", dest="target", required=True)
    parser.add_argument("--asr", required=True)
    parser.add_argument("--shot", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    root = paths.app_root()
    app = QApplication([])
    config, _ = Config.load(paths.config_file(root))
    window = MainWindow(root, config, PyvolisConfig.load(paths.pyvolis_config_file(root))[0])
    window.save = lambda: None  # a check must not touch the user's settings
    window.show()
    window._filling = True
    _select(window.source_lang, args.source)
    _select(window.target_lang, args.target)
    window._fill_recognizers()
    _select(window.recognizer, args.asr)
    window._filling = False
    window.config.languages.source, window.config.languages.target = args.source, args.target
    window.config.asr.engine = args.asr
    window.open_file(args.file)
    window.fast.setChecked(True)

    gaps, last = [], [time.perf_counter()]

    def beat() -> None:
        now = time.perf_counter()
        gaps.append(now - last[0])
        last[0] = now

    timer = QTimer()
    timer.timeout.connect(beat)
    timer.start(20)
    window.start_file()
    seen_loading = False
    while window.running():
        app.processEvents()
        seen_loading = seen_loading or window.session.state == ses.STARTING
        time.sleep(0.002)
    for _ in range(100):  # the scores arrive from their own thread
        app.processEvents()
        time.sleep(0.02)
    timer.stop()

    rows = window.session.rows()
    worst = max(gaps, default=0) * 1000
    late = sorted(g * 1000 for g in gaps)
    print(f"rows: {len(rows)} sentences, {sum(1 for r in rows if r.target)} translated")
    print(f"the window showed \"Loading...\" while models loaded: {'yes' if seen_loading else 'no'}")
    print(f"window thread: {len(gaps)} timer beats at 20 ms; median {late[len(late) // 2]:.0f} ms, "
          f"99th percentile {late[int(len(late) * 0.99)]:.0f} ms, longest freeze {worst:.0f} ms")
    print(window.status.text())
    if window.error.text():
        print("error shown:", window.error.text())
    for row in rows[:3]:
        print(f"  {ses.clock(row.start)}  {row.source}\n        {row.target}")
    if rows:
        window.row_clicked(0, 0)  # plays the first row's stretch of the file
        print("clicked row 1: playing", "yes" if window.player and window.player.queued() > 0 else "no")
        window.player.clear() if window.player else None
    if args.shot:
        window.table.scrollToTop()
        app.processEvents()
        window.grab().save(str(args.shot))
        print("screenshot:", args.shot)
    if args.export:
        window.export_to(args.export)
        print("exported:", sorted(p.name for p in args.export.iterdir()))
    window.close()
    return 0 if rows and worst < 500 else 1


if __name__ == "__main__":
    sys.exit(main())
