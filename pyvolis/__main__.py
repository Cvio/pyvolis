"""Entry point: `pyvolis`, `python -m pyvolis`, or the built pyvolis.exe.

The offline and cache environment variables are set here, before anything
but the standard library is imported: Hugging Face and PyTorch read them when
they are first imported, and a later setting would be ignored.
"""

import sys


def main() -> None:
    from pyvolis import paths

    root = paths.app_root()
    paths.apply_offline_environment(root)
    # Arabic and Persian must reach the console and the log intact; a Windows
    # console defaults to a code page that can't encode them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    from pyvolis import cli

    sys.exit(cli.run(sys.argv[1:], root))


if __name__ == "__main__":
    main()
