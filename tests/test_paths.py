"""Port of the tests in Rust `paths.rs`, plus the offline environment."""

from pathlib import Path

from pyvolis import paths


def test_a_verbatim_drive_path_is_made_readable():
    assert paths.strip_verbatim(Path(r"\\?\D:\AI_Data\cnverc")) == Path(r"D:\AI_Data\cnverc")


def test_a_verbatim_unc_path_keeps_its_prefix():
    unc = Path(r"\\?\UNC\server\share\volis")
    assert paths.strip_verbatim(unc) == unc


def test_an_ordinary_path_is_untouched():
    plain = Path("/opt/volis")
    assert paths.strip_verbatim(plain) == plain


def test_every_location_hangs_off_the_root_it_is_given():
    root = Path(r"X:\somewhere\volis")
    assert paths.config_file(root) == root / "volis.toml"
    for location in (
        paths.asr_dir(root),
        paths.tts_dir(root),
        paths.mt_dir(root),
        paths.vad_model_file(root),
        paths.logs_dir(root),
        paths.cache_dir(root),
    ):
        assert location.is_relative_to(root), location


def test_from_source_the_root_is_the_repository():
    root = paths.app_root()
    assert (root / "pyvolis" / "__main__.py").is_file()


def test_every_cache_a_library_might_use_is_inside_the_app():
    root = Path(r"X:\somewhere\pyvolis")
    env = paths.offline_environment(root)
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == env["HF_DATASETS_OFFLINE"] == "1"
    for key in ("HF_HOME", "TORCH_HOME", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "TRITON_CACHE_DIR"):
        assert Path(env[key]).is_relative_to(root), key
