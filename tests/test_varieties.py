"""Port of the tests in Rust `varieties.rs`."""

import pytest

from pyvolis import varieties as v


def test_every_row_is_well_formed():
    for row in v.TABLE:
        assert v.lookup(v.language_of(row.tag)) is not None, f"{row.tag} has no language row"
        assert row.display and row.prompt, row.tag
        assert sum(1 for w in v.TABLE if w.tag.lower() == row.tag.lower()) == 1, row.tag


def test_tags_split_into_language_and_variety():
    assert v.language_of("es-MX") == "es"
    assert v.language_of("es") == "es"
    assert v.has_variety("ar-IQ")
    assert not v.has_variety("ar")


def test_lookups_give_the_display_and_prompt_names():
    row = v.require("es-MX")
    assert (row.display, row.prompt) == ("Spanish (Mexico)", "Mexican Spanish")
    assert v.require("ar-iq").prompt == "Iraqi Arabic"
    assert v.require("ar").prompt == "Arabic"


def test_an_unknown_tag_is_an_error_not_a_fallback():
    with pytest.raises(ValueError) as why:
        v.require("xx-YY")
    assert "xx-YY" in str(why.value) and "varieties" in str(why.value)
    assert v.display_name("xx-YY") == "xx-YY (unknown)"


def test_a_language_lists_only_its_own_varieties():
    assert [row.tag for row in v.varieties_of("ar")] == ["ar-IQ", "ar-JO"]
    assert v.varieties_of("de") == []


def test_the_table_matches_rust():
    """The same tags, in the same order, as src/varieties.rs, which both apps'
    shared volis.toml depends on. Update both together."""
    assert [row.tag for row in v.TABLE] == [
        "en", "en-US", "es", "es-MX", "es-ES", "ar", "ar-IQ", "ar-JO",
        "de", "fr", "it", "pt", "ru",
    ]
