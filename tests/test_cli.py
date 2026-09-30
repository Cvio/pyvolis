"""Port of the tests in Rust `cli.rs` (for the commands that exist so far)."""

import pytest

from pyvolis import cli


def test_no_arguments_opens_the_window():
    assert cli.parse([]) == cli.Command("gui")  # a double-click passes nothing
    assert cli.parse(["--report"]) == cli.Command("report")


def test_a_bad_argument_is_rejected_with_the_help_text():
    with pytest.raises(cli.UsageError) as e:
        cli.parse(["--transcribe"])
    assert "--transcribe" in str(e.value) and "USAGE" in str(e.value)
    with pytest.raises(cli.UsageError):
        cli.parse(["--report", "--wav"])


def test_listen_takes_its_options_in_any_order():
    assert cli.parse(["--listen", "--wav", "--seconds", "20"]) == cli.Command("listen", 20, True, False)
    assert cli.parse(["--listen", "--seconds", "5"]) == cli.Command("listen", 5, False, False)


def test_compare_is_a_listen_option():
    assert cli.parse(["--listen", "--compare"]) == cli.Command("listen", None, False, True)


def test_bad_listen_options_are_rejected():
    for args in (["--seconds"], ["--listen", "--seconds", "soon"], ["--listen", "--seconds"], ["--devices", "--wav"]):
        with pytest.raises(cli.UsageError):
            cli.parse(args)
