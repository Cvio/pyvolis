"""File mode (P4): sentences, scoring, the export, and a fixture run end to end."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pyvolis import export, paths, scoring, sentences
from pyvolis.asr import Word
from pyvolis.events import Configuration, Final, NotTranslated, SentenceMsg, Summary, Translated

ROOT = paths.app_root()


# ---------------------------------------------------------------- sentences


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yo manejo. Mi carro está aquí.", ["Yo manejo.", "Mi carro está aquí."]),
        ("¿Dónde está? No sé... Pregunta al Sr. Pérez en EE. UU. mañana.",
         ["¿Dónde está?", "No sé...", "Pregunta al Sr. Pérez en EE. UU. mañana."]),
        ("Cuesta 3.5 dólares!", ["Cuesta 3.5 dólares!"]),
        ("هل أنت بخير؟ نعم.", ["هل أنت بخير؟", "نعم."]),
        ("sin puntuación ninguna", ["sin puntuación ninguna"]),
        ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
    ],
)
def test_sentences_end_at_final_punctuation(text, expected):
    assert sentences.split_text(text) == expected


def test_sentence_ids_and_times_from_words():
    words = [Word("Yo", 0.0, 0.3), Word("manejo.", 0.3, 0.8), Word("Mi", 1.0, 1.2), Word("carro.", 1.2, 1.6)]
    first, second = sentences.split("Yo manejo. Mi carro.", 7, 10.0, 12.0, words)
    assert (first.id, first.start, first.end, first.approximate) == ("7.1", 10.0, 10.8, False)
    assert (second.id, second.start, second.end) == ("7.2", 11.0, 11.6)


def test_without_words_times_are_shared_by_length_and_marked():
    first, second = sentences.split("Uno dos. Tres cuatro.", 1, 0.0, 2.0)
    assert first.approximate and second.approximate
    assert first.start == 0.0 and second.end == 2.0 and first.end == second.start


# ---------------------------------------------------------------- scoring


def test_scores_use_model_benchs_cleaning():
    assert scoring.cer("¿Dónde está?", "donde esta", "es") > 0  # accents are kept, as model-bench does
    assert scoring.cer("¿Dónde está?", "dónde está", "es") == 0.0  # case and punctuation are not
    assert scoring.cer("إلى المدرسة", "الى المدرسة", "ar") == 0.0  # alef forms unified
    assert scoring.cer("می‌روم", "می روم", "fa") == 0.0  # the half-space is a space
    assert scoring.chrf("The cat sat on the mat.", "the cat sat on the mat", "en") == 100.0


def test_a_reference_is_found_beside_the_file(tmp_path):
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")
    assert scoring.find_reference(audio) is None
    (tmp_path / "talk.m4a.ref.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHola.\n\n2\n00:00:03,000 --> 00:00:04,000\nAdiós.\n",
                                               encoding="utf-8")
    assert scoring.find_reference(audio).transcript == "Hola. Adiós."
    (tmp_path / "talk.m4a.ref.json").write_text('{"transcript": "Hola.", "translation": "Hello."}', encoding="utf-8")
    assert scoring.find_reference(audio).translation == "Hello."
    (tmp_path / "talk.m4a.ref.json").write_text("not json", encoding="utf-8")
    with pytest.raises(scoring.ReferenceError, match="talk.m4a.ref.json"):
        scoring.find_reference(audio)


# ---------------------------------------------------------------- export


def test_the_export_writes_every_file(tmp_path):
    events = [
        SentenceMsg("1.1", 1, "¿Dónde está?", "es", 1.0, 2.0),
        Translated("1.1", "Where is it?", "en", 900, "cpu", "qwen"),
        SentenceMsg("2.1", 2, "Banyana.", "es", 3.5, 3.6),
        NotTranslated("2.1", "echo", "echo"),
        Summary({"sentences": 2}),
    ]
    written = export.write(tmp_path, events, Configuration({"pyvolis": "test"}))
    assert set(written) == {"transcript.txt", "translation.txt", "source.srt", "translation.srt", "events.jsonl"}
    assert (tmp_path / "translation.txt").read_text(encoding="utf-8") == "Where is it?\n\n"
    srt = (tmp_path / "source.srt").read_text(encoding="utf-8")
    assert "00:00:01,000 --> 00:00:02,000\n¿Dónde está?" in srt
    assert "00:00:03,500 --> 00:00:04,000" in srt, "a cue too short to read is lengthened"
    lines = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["t"] == "Configuration" and lines[-1]["t"] == "Summary"
    assert [x["t"] for x in lines[1:-1]] == ["SentenceMsg", "Translated", "SentenceMsg", "NotTranslated"]


def test_events_serialise_words_and_paths():
    line = json.loads(Final(1, "Hola", "es", 900, 120, 0.5, 1.4, [Word("Hola", 0.0, 0.4)]).to_json())
    assert line["words"] == [{"text": "Hola", "start": 0.0, "end": 0.4}]
    assert json.loads(Configuration({"p": Path("C:/x")}).to_json())["settings"]["p"].endswith("x")


# ---------------------------------------------------------------- end to end


FIXTURE = ROOT / "tests" / "fixtures" / "fleurs" / "es_419" / "refs.jsonl"


@pytest.mark.skipif(not FIXTURE.is_file(), reason="run tests/fetch-fixtures.ps1")
def test_a_fixture_file_runs_end_to_end(tmp_path):
    """P4's check: --file on a fixture recording writes every export file and
    prints the scores."""
    audio = tmp_path / "es-to-en.wav"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fixture_file.py"), "es_419", "en", "--clips", "3",
                    "--out", str(audio)], check=True, capture_output=True)
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-m", "pyvolis", "--file", str(audio), "--from", "es", "--to", "en",
         "--asr", "parakeet-tdt-0.6b-v3-int8", "--fast", "--export", str(out)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    for name in ("transcript.txt", "translation.txt", "source.srt", "translation.srt", "events.jsonl"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    assert "transcript CER" in done.stdout and "translation chrF" in done.stdout
    first = json.loads((out / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["t"] == "Configuration" and first["settings"]["recognizer"]["files"]
    assert len((out / "transcript.txt").read_text(encoding="utf-8").splitlines()) >= 3
