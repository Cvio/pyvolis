# Copied verbatim from model-bench (bench/textclean.py), so pyvolis's quick
# scores clean text exactly as model-bench does. Change it there first.

"""One text-cleaning function per language, used by every score (CER, chrF,
the dialect score). Both texts in a comparison are cleaned the same way.

- Arabic: remove diacritics (harakat, shadda, sukun, superscript alef) and
  tatweel; unify alef forms (أ إ آ ٱ -> ا); ة -> ه; ى -> ي; remove
  punctuation; collapse spaces.
- Persian: all of the Arabic rules, then Arabic ي -> Persian ی and Arabic
  ك -> Persian ک; the zero-width non-joiner (the "half-space" inside words
  like می‌روم) becomes a space; Persian and Arabic-Indic digits -> 0-9.
- English: lowercase, remove punctuation, collapse spaces.

Tags may carry a region (ar-IQ, fa-IR): the language part decides.
"""

import re
import unicodedata

DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
ARABIC_LETTERS = str.maketrans({"ة": "ه", "ى": "ي"})
PERSIAN_LETTERS = str.maketrans({"ي": "ی", "ك": "ک"})
ZWNJ = "‌"
DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def strip_punctuation(text: str) -> str:
    # Unicode punctuation (P*) and symbols (S*) of every script: ، ؛ ؟ « » too.
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)


def collapse(text: str) -> str:
    return " ".join(text.split())


def clean_arabic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = text.translate(ALEF).translate(ARABIC_LETTERS)
    return collapse(strip_punctuation(text))


def clean_persian(text: str) -> str:
    # The half-space and other invisible joiners first: after punctuation
    # removal they'd otherwise glue words together.
    text = unicodedata.normalize("NFC", text).replace(ZWNJ, " ").replace("‍", "")
    text = clean_arabic(text)
    return collapse(text.translate(PERSIAN_LETTERS).translate(DIGITS))


def clean_english(text: str) -> str:
    return collapse(strip_punctuation(unicodedata.normalize("NFC", text).lower()))


def clean(text: str, language: str) -> str:
    lang = (language or "").split("-")[0].lower()
    if lang == "fa":
        return clean_persian(text or "")
    if lang == "ar":
        return clean_arabic(text or "")
    return clean_english(text or "")
