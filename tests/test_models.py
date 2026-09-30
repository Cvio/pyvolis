"""Port of the tests in Rust `models.rs`, then pyvolis's own discovery."""

import json
import struct
from pathlib import Path

import pytest

from pyvolis import gguf, models
from pyvolis.models import Engine, Failed, Fit, FitKind, ModelError, Role

PARAKEET = """
name = "Parakeet TDT 0.6B v3 (int8)"
kind = "segment"
backend = "nemo_transducer"
languages = ["es", "en"]

[files]
encoder = "encoder.int8.onnx"
decoder = "decoder.int8.onnx"
joiner  = "joiner.int8.onnx"
tokens  = "tokens.txt"
"""
PARAKEET_FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]


def model(root: Path, name: str, engine_toml: str | None, files=(), **extra: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if engine_toml is not None:
        (d / "engine.toml").write_text(engine_toml, encoding="utf-8")
    for f in files:
        (d / f).write_bytes(b"stub")
    for filename, text in extra.items():
        (d / filename.replace("__", ".")).write_text(text, encoding="utf-8")
    return d


# ---------------------------------------------------------------- Rust's tests


def test_loads_the_spec_example(tmp_path):
    d = model(tmp_path, "parakeet-tdt-0.6b-v3-int8", PARAKEET, PARAKEET_FILES)
    engine = models.load_engine(d, Role.ASR)
    assert engine.dir_name == "parakeet-tdt-0.6b-v3-int8"
    assert engine.kind == "segment"
    assert engine.backend == "nemo_transducer"
    assert len(engine.files) == 4
    assert engine.enabled()
    assert engine.missing_files() == []


def test_a_missing_file_disables_rather_than_hides(tmp_path):
    d = model(tmp_path, "parakeet", PARAKEET, PARAKEET_FILES[:3])
    engine = models.load_engine(d, Role.ASR)
    assert not engine.enabled()
    assert engine.missing_files() == ["tokens.txt"]


def test_an_unknown_backend_is_an_error_for_that_entry_only(tmp_path):
    model(tmp_path, "bogus", 'name = "B"\nkind = "segment"\nbackend = "transformers"\n')
    model(tmp_path, "parakeet", PARAKEET, PARAKEET_FILES)
    entries = models.discover(tmp_path, Role.ASR)
    assert len(entries) == 2
    assert isinstance(entries[0], Failed)
    assert isinstance(entries[1], Engine) and entries[1].enabled()


def voice_tuned(dir_name: str, tuned: list[str]) -> Engine:
    return Engine(dir_name, Path(dir_name), dir_name, "segment", "vits", ["es"], list(tuned))


def test_models_are_ranked_tuned_then_general_then_other_varieties():
    voices = [
        voice_tuned("a-spain", ["es-ES"]),
        voice_tuned("b-general", []),
        voice_tuned("c-mexico", ["es-MX"]),
    ]

    def order(tag):
        return [(r.engine.dir_name, r.fit) for r in models.rank(tag, voices)]

    assert order("es-MX") == [
        ("c-mexico", Fit(FitKind.TUNED)),
        ("b-general", Fit(FitKind.GENERAL)),
        ("a-spain", Fit(FitKind.OTHER, "es-ES")),
    ]
    assert order("es")[0] == ("b-general", Fit(FitKind.GENERAL))
    assert Fit(FitKind.OTHER, "es-ES").label("es-MX") == "tuned for Spanish (Spain)"
    assert Fit(FitKind.TUNED).label("es-MX") == "tuned for Spanish (Mexico)"
    assert models.rank("ar-IQ", voices) == []


def test_varieties_must_belong_to_a_listed_language(tmp_path):
    def voice(v):
        return (
            f'name = "V"\nkind = "segment"\nbackend = "vits"\nlanguages = ["es"]\n'
            f'varieties = {v}\n\n[files]\nmodel = "m.onnx"\ntokens = "t.txt"\n'
        )

    d = model(tmp_path, "wrong", voice('["fr-FR"]'))
    with pytest.raises(ModelError, match="fr-FR"):
        models.load_engine(d, Role.TTS)
    d = model(tmp_path, "plain", voice('["es"]'))
    with pytest.raises(ModelError):
        models.load_engine(d, Role.TTS)
    d = model(tmp_path, "mexico", voice('["es-MX"]'))
    assert models.load_engine(d, Role.TTS).varieties == ["es-MX"]
    d = model(tmp_path, "general", voice("[]"))
    assert models.load_engine(d, Role.TTS).varieties == []
    entries = models.discover(tmp_path, Role.TTS)
    assert any(isinstance(e, Failed) and e.dir_name == "wrong" for e in entries)


def test_a_required_role_absent_from_files_is_an_error(tmp_path):
    d = model(
        tmp_path,
        "parakeet",
        'name = "P"\nkind = "segment"\nbackend = "nemo_transducer"\n\n[files]\n'
        'encoder = "e.onnx"\ndecoder = "d.onnx"\ntokens = "t.txt"\n',
    )
    with pytest.raises(ModelError, match=r"\[files\]\.joiner is missing"):
        models.load_engine(d, Role.ASR)


def test_a_files_value_may_not_escape_the_model_directory(tmp_path):
    d = model(
        tmp_path,
        "vits",
        'name = "V"\nkind = "segment"\nbackend = "vits"\n\n[files]\n'
        'model = "../elsewhere/model.onnx"\ntokens = "tokens.txt"\n',
    )
    with pytest.raises(ModelError, match="plain filename"):
        models.load_engine(d, Role.TTS)


def test_a_tts_directory_without_engine_toml_is_skipped(tmp_path):
    """Rust's rule, kept for voices. (ASR folders are looked at instead.)"""
    (tmp_path / "random-folder").mkdir()
    model(tmp_path, "vits", 'name = "V"\nkind = "segment"\nbackend = "vits"\n\n[files]\n'
          'model = "m.onnx"\ntokens = "t.txt"\n')
    entries = models.discover(tmp_path, Role.TTS)
    assert [e.dir_name for e in entries] == ["vits"]


def test_an_asr_backend_is_not_a_tts_backend(tmp_path):
    d = model(tmp_path, "parakeet", PARAKEET, PARAKEET_FILES)
    with pytest.raises(ModelError):
        models.load_engine(d, Role.TTS)


def test_a_missing_model_root_yields_nothing_and_does_not_panic(tmp_path):
    assert models.discover(tmp_path / "no-such-dir", Role.ASR) == []


def test_unknown_engine_toml_keys_are_errors(tmp_path):
    d = model(tmp_path, "p", PARAKEET.replace('kind = "segment"', 'kind = "segment"\nmodel = "x"'))
    with pytest.raises(ModelError, match="unknown field"):
        models.load_engine(d, Role.ASR)


# ---------------------------------------------------------------- downloaded ASR

WHISPER_CONFIG = json.dumps({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]})
CARD = "---\nlanguage:\n- es\nlicense: apache-2.0\n---\n# A Spanish fine-tune\n"


def hf_folder(root, name, config=WHISPER_CONFIG, weights=("model.safetensors",), **extra):
    return model(
        root, name, None, weights,
        config__json=config, preprocessor_config__json="{}", **extra,
    )


def test_an_hf_whisper_folder_needs_no_engine_toml(tmp_path):
    hf_folder(tmp_path, "whisper-es", README__md=CARD)
    [engine] = models.discover(tmp_path, Role.ASR)
    assert isinstance(engine, Engine), engine
    assert engine.backend == "transformers"
    assert "WhisperForConditionalGeneration" in engine.detail
    assert engine.languages == ["es"] and engine.languages_known
    assert engine.enabled()


def test_a_ctc_model_is_recognised(tmp_path):
    hf_folder(tmp_path, "mms", config=json.dumps({"model_type": "wav2vec2"}), weights=("pytorch_model.bin",))
    [engine] = models.discover(tmp_path, Role.ASR)
    assert "(CTC)" in engine.detail


def test_a_model_without_languages_is_offered_for_every_language(tmp_path):
    hf_folder(tmp_path, "mystery")
    [engine] = models.discover(tmp_path, Role.ASR)
    assert not engine.languages_known
    [ranked] = models.rank("ar-IQ", [engine])
    assert ranked.fit.kind is FitKind.UNKNOWN
    assert ranked.fit.label("ar-IQ") == "languages unknown"


def test_unknown_languages_rank_after_every_declared_fit():
    known = Engine("a-general", Path("a"), "a", "segment", "whisper", ["es"])
    unknown = Engine("b-unknown", Path("b"), "b", "segment", "transformers", [], languages_known=False)
    assert [r.engine.dir_name for r in models.rank("es", [unknown, known])] == ["a-general", "b-unknown"]


def test_pyvolis_toml_overrides_what_was_detected(tmp_path):
    hf_folder(
        tmp_path, "fa-whisper", README__md=CARD,
        pyvolis__toml='name = "Whisper Persian"\nlanguages = ["ar"]\nvarieties = ["ar-IQ"]\n'
        'device = "cpu"\ndtype = "float32"\n',
    )
    [engine] = models.discover(tmp_path, Role.ASR)
    assert engine.name == "Whisper Persian"
    assert engine.languages == ["ar"] and engine.varieties == ["ar-IQ"]
    assert engine.settings == {"device": "cpu", "dtype": "float32"}


def test_a_bad_pyvolis_toml_is_an_error_naming_it(tmp_path):
    d = hf_folder(tmp_path, "x", pyvolis__toml='device = "tpu"\n')
    [entry] = models.discover(tmp_path, Role.ASR)
    assert isinstance(entry, Failed)
    assert str((d / "pyvolis.toml").absolute()) in entry.error
    hf_folder(tmp_path, "y", pyvolis__toml='colour = "blue"\n')
    assert "unknown key" in models.discover(tmp_path, Role.ASR)[1].error


def test_missing_weights_disable_and_say_so(tmp_path):
    hf_folder(tmp_path, "no-weights", weights=())
    [engine] = models.discover(tmp_path, Role.ASR)
    assert not engine.enabled()
    assert "*.safetensors or pytorch_model.bin" in engine.missing_files()


def test_a_sharded_model_needs_every_shard(tmp_path):
    index = json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                       "b": "model-00002-of-00002.safetensors"}})
    hf_folder(tmp_path, "sharded", weights=("model-00001-of-00002.safetensors",),
              model__safetensors__index__json=index)
    [engine] = models.discover(tmp_path, Role.ASR)
    assert engine.missing_files() == ["model-00002-of-00002.safetensors"]


def test_a_text_model_in_asr_is_refused_with_the_reason(tmp_path):
    d = hf_folder(tmp_path, "qwen", config=json.dumps({"model_type": "qwen3"}))
    [entry] = models.discover(tmp_path, Role.ASR)
    assert isinstance(entry, Failed)
    assert "models/mt/" in entry.error and str(d.absolute()) in entry.error


def test_remote_code_needs_trust_remote_code(tmp_path):
    cfg = json.dumps({"model_type": "whisper", "auto_map": {"AutoModel": "x.Y"}})
    hf_folder(tmp_path, "remote", config=cfg)
    [engine] = models.discover(tmp_path, Role.ASR)
    assert not engine.enabled() and "trust_remote_code" in engine.unusable
    hf_folder(tmp_path, "trusted", config=cfg, pyvolis__toml="trust_remote_code = true\n")
    assert models.discover(tmp_path, Role.ASR)[1].enabled()


def test_an_unrecognisable_asr_folder_is_listed_not_skipped(tmp_path):
    (tmp_path / "junk").mkdir()
    [entry] = models.discover(tmp_path, Role.ASR)
    assert isinstance(entry, Failed)
    assert str((tmp_path / "junk").absolute()) in entry.error


def test_a_lora_adapter_and_a_gguf_speech_model_are_recognised_but_wait(tmp_path):
    model(tmp_path, "lora", None, adapter_config__json='{"base_model_name_or_path": "openai/whisper-small"}')
    model(tmp_path, "voxtral", None, ["Voxtral-Q4_K_M.gguf", "mmproj-Voxtral-f16.gguf"])
    lora, voxtral = models.discover(tmp_path, Role.ASR)
    assert "openai/whisper-small" in lora.detail and "P11" in lora.unusable
    assert voxtral.backend == "llamacpp-audio" and "P11" in voxtral.unusable


def test_card_languages_read_scalar_and_list_forms(tmp_path):
    (tmp_path / "README.md").write_text("---\nlanguage: ar\n---\n", encoding="utf-8")
    assert models.card_languages(tmp_path) == ["ar"]
    (tmp_path / "README.md").write_text("---\nlanguage:\n- multilingual\n---\n", encoding="utf-8")
    assert models.card_languages(tmp_path) is None
    (tmp_path / "README.md").write_text("# no front matter\n", encoding="utf-8")
    assert models.card_languages(tmp_path) is None


# ---------------------------------------------------------------- translators


def write_gguf(path: Path, meta: dict) -> None:
    """A minimal valid GGUF v3: header and metadata, no tensors."""
    def s(text):
        b = text.encode()
        return struct.pack("<Q", len(b)) + b

    body = b""
    for key, value in meta.items():
        body += s(key)
        if isinstance(value, str):
            body += struct.pack("<I", 8) + s(value)
        elif isinstance(value, list):  # array of strings
            body += struct.pack("<IIQ", 9, 8, len(value)) + b"".join(s(v) for v in value)
        else:
            body += struct.pack("<II", 4, value)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta)) + body)


QWEN = {
    "general.architecture": "qwen3",
    "general.name": "Qwen3-1.7B",
    "qwen3.context_length": 40960,
    "tokenizer.ggml.tokens": ["a", "b", "c"],
    "tokenizer.chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
}


def test_the_gguf_reader_reads_metadata_and_skips_arrays(tmp_path):
    write_gguf(tmp_path / "m.gguf", QWEN)
    meta = gguf.read_metadata(tmp_path / "m.gguf")
    assert meta["general.architecture"] == "qwen3"
    assert meta["qwen3.context_length"] == 40960
    assert meta["tokenizer.ggml.tokens"] == ("array", 8, 3)
    assert meta["tokenizer.chat_template"].startswith("{%")


def test_a_file_that_is_not_gguf_is_an_error_naming_it(tmp_path):
    (tmp_path / "x.gguf").write_bytes(b"nope")
    with pytest.raises(gguf.GgufError, match="not a GGUF"):
        gguf.read_metadata(tmp_path / "x.gguf")


def test_translators_top_level_then_one_folder_per_model(tmp_path):
    write_gguf(tmp_path / "qwen3-1.7b-q4_k_m.gguf", QWEN)
    (tmp_path / "gemma").mkdir()
    write_gguf(tmp_path / "gemma" / "gemma-3-4b-it-Q4_K_M.gguf",
               {**QWEN, "general.architecture": "gemma3", "general.name": "Gemma 3 4B"})
    write_gguf(tmp_path / "gemma" / "mmproj-gemma.gguf", {"general.architecture": "clip"})
    found = models.discover_translators(tmp_path)
    assert [t.id for t in found] == ["qwen3-1.7b-q4_k_m.gguf", "gemma/gemma-3-4b-it-Q4_K_M.gguf"]
    assert found[0].top_level and not found[1].top_level
    assert all(t.enabled() and t.has_chat_template for t in found)


def test_a_gguf_without_chat_template_or_an_adapter_is_unusable(tmp_path):
    write_gguf(tmp_path / "bare.gguf", {"general.architecture": "llama"})
    (tmp_path / "lora").mkdir()
    write_gguf(tmp_path / "lora" / "adapter.gguf", {**QWEN, "general.type": "adapter"})
    bare, lora = models.discover_translators(tmp_path)
    assert "chat template" in bare.unusable
    assert "LoRA" in lora.unusable


def test_a_split_gguf_is_one_translator_needing_every_part(tmp_path):
    (tmp_path / "big").mkdir()
    write_gguf(tmp_path / "big" / "m-00001-of-00003.gguf", QWEN)
    write_gguf(tmp_path / "big" / "m-00002-of-00003.gguf", {})
    [t] = models.discover_translators(tmp_path)
    assert t.id == "big/m-00001-of-00003.gguf"
    assert t.missing == ["m-00003-of-00003.gguf"]


def test_an_empty_translator_folder_is_listed_with_the_reason(tmp_path):
    (tmp_path / "empty").mkdir()
    [entry] = models.discover_translators(tmp_path)
    assert isinstance(entry, Failed) and str((tmp_path / "empty").absolute()) in entry.error
