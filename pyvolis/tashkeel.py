"""Vowel marks for Arabic before it is spoken ([tts] diacritize).

New in pyvolis. Arabic is written without its short vowels, and a Piper voice
pronounces what it is given: Piper itself restores the marks first, with a
small model (libtashkeel), and its Arabic voices were trained on text marked
that way. sherpa-onnx runs the voice without that step, so unmarked words
come out wrong. This is that step: a port of piper-phonemize's `tashkeel.cpp`
(MIT), running the same model file, `libtashkeel_model.ort`, with onnxruntime.

Marks already in the text are removed and predicted again, as Piper does. The
model takes 315 characters at a time; longer text is marked piece by piece,
cut at spaces.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PAD_ID, UNK_ID = 0, 1
MAX_INPUT_CHARS = 315
# Piper's list leaves out U+064B (fathatan), though the model writes it: text that
# already had one would come out with two. It is removed here with the rest.
HARAKAT = set("ًٌٍَُِّْ")
INVALID_OUTPUT_IDS = {UNK_ID, 8}

INPUT_VOCAB = {
    "\u0009": 8, " ": 28, " ": 84, "«": 74, "­": 40, "°": 5, "´": 110, "»": 30,
    "έ": 69, "ί": 112, "α": 47, "γ": 80, "ε": 7, "θ": 51, "ι": 36, "κ": 35,
    "μ": 54, "ν": 63, "ο": 114, "π": 116, "ρ": 26, "σ": 27, "τ": 78, "υ": 20,
    "χ": 14, "ψ": 12, "ω": 89, "ό": 77, "ώ": 103, "ו": 64, "؛": 17, "؟": 101,
    "ء": 120, "آ": 15, "أ": 73, "ؤ": 50, "إ": 119, "ئ": 56, "ا": 68, "ب": 118,
    "ة": 107, "ت": 22, "ث": 71, "ج": 59, "ح": 86, "خ": 19, "د": 104, "ذ": 97,
    "ر": 65, "ز": 92, "س": 82, "ش": 18, "ص": 75, "ض": 111, "ط": 93, "ظ": 11,
    "ع": 95, "غ": 24, "ـ": 9, "ف": 46, "ق": 38, "ك": 72, "ل": 29, "م": 48,
    "ن": 81, "ه": 49, "و": 6, "ى": 39, "ي": 70, "٪": 91, "ٰ": 45, "ٱ": 67,
    "ی": 105, "ے": 37, "۵": 109, "۷": 106, "۸": 10, "​": 52, "‍": 31, "‎": 117,
    "‏": 60, "–": 42, "‘": 34, "’": 41, "“": 55, "”": 85, "•": 62, "…": 23,
    "‫": 94, "‬": 108, "‰": 115, "ﮐ": 53, "﴾": 44, "﴿": 25, "ﺁ": 16, "ﺂ": 96,
    "ﺃ": 87, "ﺄ": 61, "ﺇ": 57, "ﺈ": 58, "ﺋ": 100, "ﺌ": 90, "ﺑ": 32, "ﺒ": 113,
    "ﺔ": 76, "ﻓ": 33, "ﻛ": 13, "ﻟ": 99, "ﻠ": 66, "ﻣ": 43, "ﻧ": 102, "ﻴ": 88,
    "ﻵ": 83, "ﻷ": 98, "ﻹ": 21, "ﻻ": 79,
}

OUTPUT_VOCAB = {
    4: "ـ", 5: "َ", 6: "ُّ", 7: "َّ", 8: "ـ", 9: "ِّ", 10: "ّ",
    11: "ّْ", 12: "ٍّ", 13: "ِّ", 14: "ٍّ", 15: "ٌّ",
    16: "َّ", 17: "ُ", 18: "ٌّ", 19: "ًّ", 20: "ْ", 21: "ٍ",
    22: "ِ", 23: "ُّ", 24: "ًّ", 25: "ٌ", 26: "ً", 27: "ّّ",
}


class TashkeelError(Exception):
    pass


class Tashkeel:
    def __init__(self, model: Path) -> None:
        if not model.is_file():
            raise TashkeelError(f"the vowel-marking model is not at {model.absolute()} "
                                "(.\\fetch-models.ps1 -Only tashkeel fetches it)")
        import onnxruntime

        began = time.perf_counter()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 2  # as the voice: the recognizer may be working
        options.log_severity_level = 3
        try:
            self._session = onnxruntime.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise TashkeelError(f"cannot load the vowel-marking model {model.absolute()}: {e}") from e
        self._input = self._session.get_inputs()[0].name
        log.info("loaded the vowel-marking model %s in %d ms", model, (time.perf_counter() - began) * 1000)

    def run(self, text: str) -> str:
        """`text` with its vowel marks predicted."""
        return "".join(self._piece(piece) for piece in pieces(strip(text)))

    def _piece(self, text: str) -> str:
        ids = np.full((1, MAX_INPUT_CHARS), PAD_ID, dtype=np.float32)
        ids[0, : len(text)] = [INPUT_VOCAB.get(c, UNK_ID) for c in text]
        scores = self._session.run(None, {self._input: ids})[0][0]  # characters x marks
        return mark(text, scores.argmax(axis=1))


def strip(text: str) -> str:
    return "".join(c for c in text if c not in HARAKAT)


def pieces(text: str, limit: int = MAX_INPUT_CHARS) -> list[str]:
    """`text` in pieces of at most `limit` characters, cut after a space where
    there is one. Joined, they are the text again."""
    out = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit) + 1 or limit
        out.append(text[:cut])
        text = text[cut:]
    return out + [text] if text else out


def mark(text: str, ids) -> str:
    """Each character followed by the mark predicted for it, if any."""
    out = []
    for char, mark_id in zip(text, ids):
        out.append(char)
        mark_id = int(mark_id)
        if mark_id not in INVALID_OUTPUT_IDS and mark_id in OUTPUT_VOCAB:
            out.append(OUTPUT_VOCAB[mark_id])
    return "".join(out) + text[len(ids):]
