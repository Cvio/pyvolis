"""The P11 backends: how their folders are recognised, and what each needs.
The models themselves are run by scripts/p11_check.py; the tests here that
load one skip when it isn't installed."""

import json
from pathlib import Path

import numpy as np
import pytest
from test_models import QWEN, model, write_gguf

from pyvolis import gguf, models, paths
from pyvolis.asr import llamacpp_audio
from pyvolis.models import Role

ROOT = paths.app_root()
ADAPTER = {"general.architecture": "qwen3", "general.type": "adapter", "general.name": "my tune",
           "adapter.type": "lora"}


# ---------------------------------------------------------------- llama.cpp speech models


def test_a_gguf_with_its_audio_encoder_is_a_recognizer(tmp_path):
    model(tmp_path, "Voxtral-Mini", None, ["Voxtral-Q4_K_M.gguf", "mmproj-Voxtral-f16.gguf"],
          README__md="---\nlanguage:\n- en\n- es\n---\n")
    [voxtral] = models.discover(tmp_path, Role.ASR)
    assert (voxtral.backend, voxtral.enabled(), voxtral.languages) == ("llamacpp-audio", True, ["en", "es"])
    assert {f.role: f.name for f in voxtral.files} == {"model": "Voxtral-Q4_K_M.gguf",
                                                      "audio encoder": "mmproj-Voxtral-f16.gguf"}


def test_a_speech_gguf_without_its_encoder_says_what_is_missing(tmp_path):
    model(tmp_path, "half", None, ["Qwen3-ASR-Q8_0.gguf"])
    [failed] = models.discover(tmp_path, Role.ASR)
    assert isinstance(failed, models.Failed) and "mmproj" in failed.error and "no audio encoder" in failed.error


def test_the_folder_can_say_what_the_model_is_asked(tmp_path):
    model(tmp_path, "custom", None, ["m.gguf", "mmproj-m.gguf"], pyvolis__toml='prompt = "Write down this {language} speech."\n')
    [engine] = models.discover(tmp_path, Role.ASR)
    assert engine.settings["prompt"] == "Write down this {language} speech."


def test_a_transcript_is_taken_out_of_the_models_answer():
    clean = llamacpp_audio.clean
    assert clean("language Spanish<asr_text>¿Dónde está la estación?") == "¿Dónde está la estación?"
    assert clean("¿Dónde está\n la estación?<turn|>\n<|turn>user") == "¿Dónde está la estación?"
    assert clean('"Where is the station?"') == "Where is the station?"
    assert clean("") == ""


def test_the_language_is_given_by_its_english_name_never_as_a_variety():
    assert llamacpp_audio._language_name("es-MX") == "Spanish"
    assert llamacpp_audio._language_name("ar-IQ") == "Arabic"
    assert llamacpp_audio._language_name("fa") == "Persian"


# ---------------------------------------------------------------- LoRA on a transformers recognizer


def test_an_adapter_is_attached_to_its_base_in_the_folder_beside_it(tmp_path):
    model(tmp_path, "whisper-small", None, ["model.safetensors"], config__json='{"model_type": "whisper"}',
          README__md="---\nlanguage:\n- ar\n- en\n---\n")
    model(tmp_path, "my-lora", None, ["adapter_model.safetensors"],
          adapter_config__json='{"base_model_name_or_path": "openai/whisper-small"}')
    lora = next(e for e in models.discover(tmp_path, Role.ASR) if e.dir_name == "my-lora")
    assert lora.enabled() and lora.backend == "transformers"
    assert Path(lora.settings["base_dir"]) == tmp_path / "whisper-small"
    assert lora.detail == "LoRA adapter on whisper-small"
    assert lora.languages == ["ar", "en"], "its base's languages, when it names none of its own"


def test_an_adapter_whose_base_is_missing_says_where_it_looked(tmp_path):
    model(tmp_path, "my-lora", None, ["adapter_model.safetensors"],
          adapter_config__json='{"base_model_name_or_path": "openai/whisper-small"}')
    [lora] = models.discover(tmp_path, Role.ASR)
    assert not lora.enabled()
    assert "openai/whisper-small" in lora.unusable and str((tmp_path / "whisper-small").absolute()) in lora.unusable


def test_the_base_can_be_named_in_the_folders_pyvolis_toml(tmp_path):
    model(tmp_path, "whisper-small-hf", None, ["model.safetensors"], config__json='{"model_type": "whisper"}')
    model(tmp_path, "my-lora", None, ["adapter_model.safetensors"], pyvolis__toml='base = "whisper-small-hf"\nlanguages = ["ar"]\n',
          adapter_config__json='{"base_model_name_or_path": "openai/whisper-small"}')
    lora = next(e for e in models.discover(tmp_path, Role.ASR) if e.dir_name == "my-lora")
    assert lora.enabled() and Path(lora.settings["base_dir"]).name == "whisper-small-hf" and lora.languages == ["ar"]


def test_an_adapter_folder_without_its_weights_is_incomplete(tmp_path):
    model(tmp_path, "whisper-small", None, ["model.safetensors"], config__json='{"model_type": "whisper"}')
    model(tmp_path, "my-lora", None, [], adapter_config__json='{"base_model_name_or_path": "openai/whisper-small"}')
    lora = next(e for e in models.discover(tmp_path, Role.ASR) if e.dir_name == "my-lora")
    assert lora.missing_files() == ["adapter_model.safetensors"]


# ---------------------------------------------------------------- LoRA on a GGUF translator


def test_a_gguf_adapter_is_a_translator_of_its_own_on_the_base_it_names(tmp_path):
    write_gguf(tmp_path / "qwen3.gguf", QWEN)
    (tmp_path / "tuned").mkdir()
    write_gguf(tmp_path / "tuned" / "adapter.gguf", ADAPTER)
    (tmp_path / "tuned" / "pyvolis.toml").write_text('base = "qwen3.gguf"\nscale = 0.5\n', encoding="utf-8")
    base, tuned = models.discover_translators(tmp_path)
    assert tuned.enabled() and tuned.id == "tuned/adapter.gguf"
    assert (tuned.path, tuned.lora, tuned.lora_scale) == (tmp_path / "qwen3.gguf", tmp_path / "tuned" / "adapter.gguf", 0.5)
    assert tuned.name == "Qwen3-1.7B + my tune" and tuned.has_chat_template
    assert base.lora is None


def test_a_gguf_adapter_that_cannot_be_attached_says_why(tmp_path):
    write_gguf(tmp_path / "qwen3.gguf", QWEN)
    write_gguf(tmp_path / "gemma.gguf", dict(QWEN, **{"general.architecture": "gemma3"}))
    for name, toml, expected in (
        ("unnamed", "", "name the translator it is for"),
        ("missing", 'base = "nothing.gguf"\n', "is not installed"),
        ("wrong", 'base = "gemma.gguf"\n', "is a gemma3 model"),
    ):
        (tmp_path / name).mkdir()
        write_gguf(tmp_path / name / "adapter.gguf", ADAPTER)
        if toml:
            (tmp_path / name / "pyvolis.toml").write_text(toml, encoding="utf-8")
    found = {t.id: t for t in models.discover_translators(tmp_path)}
    for name, _toml, expected in (("unnamed", 0, "name the translator it is for"), ("missing", 0, "is not installed"),
                                  ("wrong", 0, "is a gemma3 model")):
        entry = found[f"{name}/adapter.gguf"]
        assert not entry.enabled() and expected in entry.unusable, entry.unusable


def test_a_synthetic_adapter_is_a_valid_gguf_that_fits_its_base(tmp_path):
    import subprocess
    import sys

    base = paths.mt_dir(ROOT) / "qwen3-1.7b-q4_k_m.gguf"
    if not base.is_file():
        pytest.skip(f"no {base}")
    out = tmp_path / "a.gguf"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_test_lora.py"), str(base), str(out)], check=True,
                   capture_output=True)
    meta, shapes = gguf.read_metadata(out), gguf.read_tensor_shapes(out)
    assert (meta["general.type"], meta["adapter.type"], meta["general.architecture"]) == ("adapter", "lora", "qwen3")
    target = gguf.read_tensor_shapes(base)["blk.0.attn_q.weight"]
    a, b = shapes["blk.0.attn_q.weight.lora_a"], shapes["blk.0.attn_q.weight.lora_b"]
    assert a[0] == target[0] and b[1] == target[1] and a[1] == b[0], "A reads the inputs, B writes the outputs"


# ---------------------------------------------------------------- transformers translators


def causal_lm(root: Path, name: str, template: bool = True, model_type: str = "qwen3") -> Path:
    tokenizer = json.dumps({"chat_template": "{{ messages }}"} if template else {})
    return model(root, name, None, ["model.safetensors"], config__json=json.dumps({"model_type": model_type}),
                 tokenizer_config__json=tokenizer)


def test_a_safetensors_language_model_is_a_translator(tmp_path):
    causal_lm(tmp_path, "Qwen3-0.6B")
    [entry] = models.discover_translators(tmp_path)
    assert (entry.backend, entry.enabled(), entry.id) == ("transformers", True, "Qwen3-0.6B")
    assert entry.path == tmp_path / "Qwen3-0.6B" and entry.size_bytes > 0


def test_a_folder_that_is_not_a_chat_language_model_says_why(tmp_path):
    causal_lm(tmp_path, "no-template", template=False)
    causal_lm(tmp_path, "speech", model_type="whisper")
    found = {t.id: t for t in models.discover_translators(tmp_path)}
    assert "no chat template" in found["no-template"].unusable
    assert "not a text-generating language model" in found["speech"].unusable and "models/asr" in found["speech"].unusable


# ---------------------------------------------------------------- with the real models


def installed(folder: Path, name: str):
    found = next((e for e in models.discover(folder, Role.ASR) if getattr(e, "dir_name", "") == name), None)
    if not isinstance(found, models.Engine) or not found.enabled():
        pytest.skip(f"{name} is not installed")
    return found


def clip(name: str):
    from pyvolis.filesource import read_16k_mono

    folder = ROOT / "tests" / "fixtures" / "fleurs" / name
    if not (folder / "refs.jsonl").is_file():
        pytest.skip("needs tests/fetch-fixtures.ps1")
    ref = json.loads((folder / "refs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    return read_16k_mono(folder / ref["file"]), ref["raw_transcript"]


def test_a_gguf_speech_model_transcribes_in_the_language_it_is_told():
    from pyvolis import asr, scoring

    engine = installed(paths.asr_dir(ROOT), "Qwen3-ASR-0.6B-GGUF")
    audio, reference = clip("es_419")
    recognizer = asr.load(engine)
    try:
        assert recognizer.prompt("es-MX").endswith("language Spanish<asr_text>"), "told, not left to decide"
        result = recognizer.transcribe(audio, "es-MX", False)
        assert scoring.cer(reference, result.text, "es") < 10, result.text
        assert "<asr_text>" not in result.text and result.words is None
        assert recognizer.transcribe(np.zeros(800, np.float32), "es").text == "", "too short to hold speech"
    finally:
        recognizer.close()


def test_a_lora_adapter_loads_on_its_base_and_changes_what_it_hears():
    from pyvolis import asr

    adapter = installed(paths.asr_dir(ROOT), "whisper-algerian-darja-small")
    base = installed(paths.asr_dir(ROOT), "whisper-small")
    audio, _reference = clip("ar_eg")
    texts = []
    for engine in (base, adapter):
        recognizer = asr.load(engine)
        try:
            texts.append(recognizer.transcribe(audio, "ar", False).text)
            assert type(recognizer._model).__name__ == "WhisperForConditionalGeneration", "merged, not wrapped"
        finally:
            recognizer.close()
    assert all(texts) and texts[0] != texts[1], "the adapter was applied"


def test_a_transformers_translator_translates_and_is_guarded():
    from pyvolis import translate as tr
    from pyvolis.translate import prompts

    entry = next((t for t in models.discover_translators(paths.mt_dir(ROOT)) if getattr(t, "id", "") == "Qwen3-0.6B"), None)
    if entry is None or not entry.enabled():
        pytest.skip("Qwen3-0.6B is not installed")
    translator = tr.load(entry, prompts.load(paths.prompts_dir(ROOT)))
    try:
        request = tr.TranslationRequest("¿Dónde está la estación de tren?", "es", "en")
        prompt = translator.prompt(request)
        assert "<think>\n\n</think>" in prompt, "Qwen3's thinking is off"
        assert 'Spanish text to translate into English' in prompt
        result = translator.translate(request)
        assert "station" in result.text.lower() and "<think>" not in result.text
        assert translator.count_tokens("hola amigo") >= 2
        with pytest.raises(tr.TranslateError, match="xx-YY"):
            translator.translate(tr.TranslationRequest("hola", "xx-YY", "en"))
    finally:
        translator.close()
