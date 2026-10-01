"""Languages and their varieties (dialects), by BCP 47 tag.

Port of Rust `varieties.rs`, with the same table. A tag is a language,
optionally with a region: `es` is Spanish with no particular dialect, `es-MX`
is Mexican Spanish. Adding a dialect means adding one row to `TABLE`. A tag
that isn't in the table is an error shown to the user, not a silent fallback.

The table must stay identical to Rust's: both apps read the same `volis.toml`,
and a tag one knows and the other doesn't would break the shared file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variety:
    tag: str  # BCP 47, e.g. "es-MX"
    display: str  # shown in the window, e.g. "Spanish (Mexico)"
    prompt: str  # written into the translation prompt, e.g. "Mexican Spanish"


TABLE: tuple[Variety, ...] = (
    Variety("en", "English", "English"),
    Variety("en-US", "English (US)", "American English"),
    Variety("es", "Spanish", "Spanish"),
    Variety("es-MX", "Spanish (Mexico)", "Mexican Spanish"),
    Variety("es-ES", "Spanish (Spain)", "Peninsular Spanish"),
    Variety("ar", "Arabic", "Arabic"),
    Variety("ar-IQ", "Arabic (Iraq)", "Iraqi Arabic"),
    Variety("ar-JO", "Arabic (Jordan)", "Jordanian Arabic"),
    Variety("fa", "Persian", "Persian"),
    Variety("fa-IR", "Persian (Iran)", "Iranian Persian"),
    # Languages the translation prompt already knew by name, kept so they
    # don't stop working. They have no varieties yet.
    Variety("de", "German", "German"),
    Variety("fr", "French", "French"),
    Variety("it", "Italian", "Italian"),
    Variety("pt", "Portuguese", "Portuguese"),
    Variety("ru", "Russian", "Russian"),
)


def lookup(tag: str) -> Variety | None:
    """The table's entry for `tag`, matched without regard to case."""
    tag = tag.strip().lower()
    return next((v for v in TABLE if v.tag.lower() == tag), None)


def require(tag: str) -> Variety:
    """The same, raising an error a person can act on."""
    found = lookup(tag)
    if found is None:
        known = ", ".join(v.tag for v in TABLE)
        raise ValueError(
            f'"{tag.strip()}" is not a language or variety Volis knows. Known: {known}. '
            "To add one, add a row to pyvolis/varieties.py and to Rust volis's "
            "src/varieties.rs."
        )
    return found


def language_of(tag: str) -> str:
    """The language part of a tag: `es` for both `es` and `es-MX`. This is what
    a recognizer is told, and what `engine.toml`'s `languages` lists."""
    return tag.strip().split("-")[0]


def has_variety(tag: str) -> bool:
    """Whether a tag names a region as well as a language."""
    return "-" in tag.strip()


def varieties_of(language: str) -> list[Variety]:
    """The varieties the table lists for a language, not the language itself."""
    return [
        v
        for v in TABLE
        if has_variety(v.tag) and language_of(v.tag).lower() == language.lower()
    ]


def display_name(tag: str) -> str:
    """How to show a tag in the window. An unknown tag is shown, marked so."""
    found = lookup(tag)
    return found.display if found else f"{tag.strip()} (unknown)"


# ISO 639-3 codes for the languages above, for models that name languages that
# way (MMS's adapters: "spa", "eng", "ara", "fas").
ISO639_3 = {
    "en": "eng", "es": "spa", "ar": "ara", "fa": "fas", "de": "deu",
    "fr": "fra", "it": "ita", "pt": "por", "ru": "rus",
}


def iso639_3(tag: str) -> str | None:
    """The ISO 639-3 code for a tag's language, or None if unknown here."""
    return ISO639_3.get(language_of(tag).lower())


def from_iso639_3(code: str) -> str:
    """Back to the two-letter code where known; otherwise the code itself."""
    for two, three in ISO639_3.items():
        if three == code:
            return two
    return code
