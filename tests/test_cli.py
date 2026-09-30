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
