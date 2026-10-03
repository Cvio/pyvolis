"""The performance panel's arithmetic (pyvolis/perf.py) and the panel in the
window. No models are loaded; nvidia-smi is replaced by a fixed reading."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from pyvolis import events as ev, models, paths, perf  # noqa: E402

GiB = 1024 ** 3


def sample(gpu_used=2 * GiB, gpu_total=8 * GiB, ram_used=8 * GiB, ram_total=32 * GiB, ram_ours=GiB):
    gpu = perf.Gpu("RTX", gpu_used, gpu_total, 10.0, 50.0, 1500.0, 20.0) if gpu_total else None
    return perf.Sample(0.0, gpu, ram_total, ram_used, ram_ours, 5.0, 3.0)


def est(role, gpu=0, cpu=0, device="cuda"):
    return perf.Estimate(role, role, device, gpu, cpu, False)


def test_nvidia_smi_lines_are_read_and_n_a_counts_as_zero():
    g = perf.parse_gpu("NVIDIA GeForce RTX 4070 Laptop GPU, 1278, 8188, 7, 53, 210, [N/A]")
    assert g.name.endswith("Laptop GPU") and g.used == 1278 * 1024 * 1024 and g.total == 8188 * 1024 * 1024
    assert (g.load, g.temperature, g.clock_mhz, g.power_w) == (7, 53, 210, 0)
    assert perf.parse_gpu("garbage") is None


def test_a_combination_fits_is_tight_or_does_not_fit():
    s = sample(gpu_used=2 * GiB)  # 6 GiB free for pyvolis
    assert perf.verdict([est("recognizer", 2 * GiB), est("translator", 2 * GiB)], s).level == "fits"
    assert perf.verdict([est("translator", int(5.7 * GiB))], s).level == "tight"
    no = perf.verdict([est("recognizer", 2 * GiB), est("translator", 5 * GiB)], s)
    assert no.level == "no" and "slower" in no.text


def test_what_pyvolis_holds_now_counts_as_room_since_it_would_be_replaced():
    s = sample(gpu_used=7 * GiB)  # of which pyvolis has 6
    assert perf.verdict([est("translator", 5 * GiB)], s).level == "no"
    assert perf.verdict([est("translator", 5 * GiB)], s, ours_gpu_now=6 * GiB).level == "fits"


def test_a_gpu_model_on_a_machine_without_a_gpu_does_not_fit():
    assert perf.verdict([est("translator", GiB)], sample(gpu_total=0)).level == "no"
    assert perf.verdict([est("voice", cpu=GiB, device="cpu")], sample(gpu_total=0)).level == "fits"


def test_memory_levels():
    assert perf.level(50, 100) == "ok" and perf.level(91, 100) == "amber" and perf.level(98, 100) == "red"


def test_measured_memory_wins_over_the_files_and_runs_give_speeds(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "performance_file", lambda root: tmp_path / "performance.json")
    m = perf.Measurements.load(tmp_path)
    folder = tmp_path / "asr" / "some-model"
    folder.mkdir(parents=True)
    (folder / "model.safetensors").write_bytes(b"x" * 4000)
    (folder / "config.json").write_text('{"torch_dtype": "float32"}', encoding="utf-8")
    engine = models.Engine("some-model", folder, "some", "segment", models.TRANSFORMERS, ["en"],
                           from_engine_toml=False)
    e = perf.estimate_recognizer(engine, m, gpu=True)
    assert (e.device, e.gpu_bytes, e.measured) == ("cuda", 2000, False), "float32 on disk, float16 on the GPU"
    m.model("recognizer", "some-model", "cuda", 1234, 0)
    reloaded = perf.Measurements.load(tmp_path)
    e = perf.estimate_recognizer(engine, reloaded, gpu=True)
    assert (e.gpu_bytes, e.measured) == (1234, True)
    for rtf in (0.2, 0.4, 0.3):
        reloaded.run(perf.Run("now", "some-model", "mt", "continuous", rtf, 400, 900, 10, GiB))
    assert perf.Measurements.load(tmp_path).speed("some-model", "mt") == {
        "runs": 3, "asr_rtf": 0.3, "translate_ms": 400, "first_audio_ms": 900}


def test_sherpa_recognizers_are_estimated_on_the_cpu(tmp_path):
    folder = tmp_path / "w"
    folder.mkdir()
    (folder / "encoder.onnx").write_bytes(b"x" * 100)
    engine = models.Engine("w", folder, "w", "segment", "whisper", ["en"],
                           files=[models.ModelFile("encoder", "encoder.onnx", folder / "encoder.onnx", True)])
    e = perf.estimate_recognizer(engine, None, gpu=True)
    assert (e.device, e.gpu_bytes, e.cpu_bytes) == ("cpu", 0, 100)


@pytest.fixture
def window(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from pyvolis.config import Config, PyvolisConfig
    from pyvolis.gui.window import MainWindow

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(perf, "read_gpu", lambda: sample().gpu)
    monkeypatch.setattr(paths, "performance_file", lambda root: tmp_path / "performance.json")
    w = MainWindow(paths.app_root(), Config(), PyvolisConfig())
    w.save = lambda: None
    yield w
    w.perf_panel.sampler.stop()
    w.close()


def test_the_panel_is_closed_at_start_and_follows_a_run(window):
    panel = window.perf_panel
    assert not panel.isVisible() and panel.toggleViewAction().shortcut().toString() == "Ctrl+Shift+P"
    for event in (ev.Loading("recognizer x"), ev.ModelLoaded("recognizer", "x", "cuda", 2 * GiB, 0, 1.0),
                  ev.ModelLoaded("translator", "t.gguf", "cuda", 3 * GiB, 0, 1.0),
                  ev.Final(1, "Hola.", "es", 1500, 120),
                  ev.Translated("1.1", "Hello.", "en", 300, "cuda", "t.gguf", 2),
                  ev.SpeakingStarted("1.1", 700, "voice"),
                  ev.Summary({"asr_rtf": 0.1, "translate_ms_median": 300, "sentences": 1})):
        panel.observe(event)
    assert panel.sentences["1.1"] == {"translate_ms": 300, "context": 2, "speech_ms": 1500, "asr_ms": 120,
                                      "first_audio_ms": 700}
    assert panel.measured.models["translator:t.gguf"]["gpu_bytes"] == 3 * GiB
    run = panel.measured.runs[-1]
    assert (run["recognizer"], run["translator"], run["translate_ms"], run["first_audio_ms"]) == (
        "x", "t.gguf", 300, 700)
    panel.observe(ev.Stopped())
    assert panel.loaded == {}


def test_falling_behind_turns_the_panel_red(window):
    panel = window.perf_panel
    panel.observe(ev.Error("speech fell behind the conversation; sentence 3.1 was not spoken"))
    panel.sampler.history.append(sample())
    window.show()
    panel.show()
    panel.refresh()
    assert "fell behind" in panel.now_labels["speech"].text()
    assert "#d03030" in panel.now_labels["speech"].styleSheet()
    panel.hide()


def test_what_if_estimates_without_loading(window):
    panel = window.perf_panel
    panel.sampler.history.append(sample())
    panel.fill_from_window()
    panel.refresh_whatif()
    assert panel.verdict_label.text()
    if window.engines:
        assert panel.estimate_table.rowCount() >= 1
