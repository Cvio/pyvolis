"""Download the test clips into tests/fixtures/ (not committed). Run through
tests/fetch-fixtures.ps1. A development tool that uses the internet, like
fetch-model.ps1; pyvolis never runs it.

For each language, the first 10 clips of FLEURS's test split, with their
reference transcripts. FLEURS publishes each split's audio as one large
tar.gz; it is streamed and the download stops once 10 clips are in, so a few
MB come down instead of hundreds.

    tests/fixtures/fleurs/<lang>/<id>.wav    16 kHz mono, as published
    tests/fixtures/fleurs/<lang>/refs.jsonl  {"id", "file", "transcript", "raw_transcript", "gender"}

Then each clip gets its sentence in other languages from FLORES+, which
FLEURS was read from: "references" in refs.jsonl, keyed by language tag
(en, es, ar, ar-IQ, fa). FLEURS's ids aren't FLORES+ row numbers, so the
sentence is found by its English text (FLEURS's English list shares the ids).
FLORES+ is gated: accept its terms on Hugging Face, then .\fetch-model.ps1 -Login.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "fixtures" / "fleurs"
BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
LANGUAGES = {"es_419": "es", "fa_ir": "fa", "ar_eg": "ar", "en_us": "en"}
COUNT = 10


def fetch(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "pyvolis-fetch-fixtures"})
    return urllib.request.urlopen(request, timeout=60)


def rows(lang: str) -> dict[str, dict]:
    """test.tsv: id, file name, raw transcription, normalised transcription,
    phonemes, sample count, gender. Keyed by file name."""
    with fetch(f"{BASE}/{lang}/test.tsv") as response:
        text = response.read().decode("utf-8")
    out = {}
    for fields in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(fields) >= 7:
            out[fields[1]] = {"id": fields[0], "raw_transcript": fields[2], "transcript": fields[3],
                              "gender": fields[6]}
    return out


FLORES = "openlanguagedata/flores_plus"
FLORES_LANGS = {"en": "eng_Latn", "es": "spa_Latn", "ar": "arb_Arab", "ar-IQ": "acm_Arab", "fa": "pes_Arab"}


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def add_references() -> None:
    """Every clip's sentence in each FLORES_LANGS language, matched through
    its English text."""
    import os

    os.environ["HF_HOME"] = str(REPO / ".uv" / "hf")
    from huggingface_hub import hf_hub_download

    english = {}  # FLEURS id -> English sentence
    with fetch(f"{BASE}/en_us/test.tsv") as response:
        for fields in csv.reader(io.StringIO(response.read().decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(fields) >= 3:
                english[fields[0]] = fields[2]
    rows = {}  # (split, row) -> {tag: text}
    by_english = {}
    for split in ("dev", "devtest"):
        for tag, code in FLORES_LANGS.items():
            path = hf_hub_download(FLORES, f"{split}/{code}.jsonl", repo_type="dataset")
            for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
                rows.setdefault((split, n), {})[tag] = json.loads(line)["text"]
        for (sp, n), texts in rows.items():
            if sp == split:
                by_english[_key(texts["en"])] = texts
    for lang in LANGUAGES:
        refs_path = OUT / lang / "refs.jsonl"
        refs = [json.loads(x) for x in refs_path.read_text(encoding="utf-8").splitlines()]
        matched = 0
        for ref in refs:
            texts = by_english.get(_key(english.get(ref["id"], "")))
            if texts:
                ref["references"] = dict(texts)
                matched += 1
        with refs_path.open("w", encoding="utf-8") as f:
            for ref in refs:
                f.write(json.dumps(ref, ensure_ascii=False) + "\n")
        print(f"{lang}: {matched} of {len(refs)} clips matched to FLORES+")


def main() -> int:
    for lang in LANGUAGES:
        target = OUT / lang
        refs_path = target / "refs.jsonl"
        if refs_path.is_file() and sum(1 for _ in refs_path.open(encoding="utf-8")) >= COUNT:
            print(f"{lang}: already present")
            continue
        target.mkdir(parents=True, exist_ok=True)
        table = rows(lang)
        refs = []
        with fetch(f"{BASE}/{lang}/audio/test.tar.gz") as response:
            with tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    name = Path(member.name).name
                    if not member.isfile() or name not in table:
                        continue
                    data = archive.extractfile(member).read()
                    (target / name).write_bytes(data)
                    refs.append({"file": name, **table[name]})
                    if len(refs) == COUNT:
                        break  # stop reading: the rest of the archive never downloads
        with refs_path.open("w", encoding="utf-8") as f:
            for ref in refs:
                f.write(json.dumps(ref, ensure_ascii=False) + "\n")
        print(f"{lang}: {len(refs)} clips in {target}")
    add_references()
    return 0


if __name__ == "__main__":
    sys.exit(main())
