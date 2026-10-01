"""Port of the tests in Rust `config.rs`, plus tomlkit-specific checks."""

import pytest

from pyvolis.config import Config, ConfigError, Shared

# The exact file from SPEC §7 must round-trip.
SPEC_EXAMPLE = """
[asr]
engine = "parakeet-tdt-0.6b-v3-int8"

[languages]
source = "es"
target = "en"

[mode]
kind = "turn"
turn_key = "Space"
turn_style = "toggle"

[audio]
input_device = ""
output_device = ""

[vad]
threshold = 0.5
min_silence_ms = 500
min_speech_ms = 250

[tts]
enabled = true
half_duplex = true

[peer]
enabled = false
listen_addr = "0.0.0.0:47800"
peer_addr = ""
display_name = ""
discovery = true
"""


def test_parses_the_spec_example():
    config = Config.parse(SPEC_EXAMPLE)
    assert config.asr.engine == "parakeet-tdt-0.6b-v3-int8"
    assert config.languages.source == "es"
    assert config.mode.kind == "turn"
    assert config.mode.turn_style == "toggle"
    assert config.vad.min_silence_ms == 500
    assert config.tts.half_duplex
    assert config.peer.listen_addr == "0.0.0.0:47800"


def test_an_empty_file_is_the_documented_defaults():
    config = Config.parse("")
    assert config.languages.target == "en"
    assert config.mode.turn_key == "Space"
    assert config.asr.engine == ""


def test_saving_selections_keeps_comments_and_every_other_key(tmp_path):
    path = tmp_path / "volis.toml"
    path.write_text(
        "# my notes about this rig\n"
        "[asr]\n"
        'engine = "parakeet"   # the fast one\n'
        "\n"
        "[vad]\n"
        "threshold = 0.35\n"
        "min_silence_ms = 500\n"
        "min_speech_ms = 250\n",
        encoding="utf-8",
    )
    config, _ = Config.load(path)
    config.asr.engine = "whisper-large-v3-turbo"
    config.audio.output_device = "Speakers"
    config.save_selections(path)

    saved = path.read_text(encoding="utf-8")
    assert "# my notes about this rig" in saved
    assert "# the fast one" in saved
    assert 'engine = "whisper-large-v3-turbo"   # the fast one' in saved
    # Sections the file lacked are added as sections, never inline tables.
    assert "[mode]" in saved
    assert "mode = {" not in saved
    reloaded = Config.parse(saved)
    assert reloaded.audio.output_device == "Speakers"
    assert reloaded.vad.threshold == 0.35, "an untouched key changed"


def test_a_malformed_file_is_never_overwritten(tmp_path):
    path = tmp_path / "volis.toml"
    path.write_text("[asr\nengine = ", encoding="utf-8")
    with pytest.raises(ConfigError):
        Config().save_selections(path)
    assert path.read_text(encoding="utf-8") == "[asr\nengine = ", "the broken file was replaced"


def test_a_key_left_out_takes_its_default():
    config = Config.parse('[vad]\nthreshold = 0.35\n\n[mode]\nkind = "continuous"\n')
    assert config.vad.threshold == 0.35
    assert config.vad.min_silence_ms == 500
    assert config.mode.kind == "continuous"
    assert config.mode.turn_key == "Space"


def test_a_file_from_before_shared_mode_still_loads():
    config = Config.parse(SPEC_EXAMPLE)
    assert config.shared == Shared()
    assert config.shared.left_key == "ArrowLeft"


def test_per_side_recognizers_load_and_default_to_empty():
    old = Config.parse('[shared]\nleft_language = "en"\n')
    assert (old.shared.left_asr, old.shared.right_asr) == ("", "")
    new = Config.parse(
        '[shared]\nleft_asr = "parakeet-tdt-0.6b-v3-int8"\nright_asr = "whisper-es"\n'
    )
    assert new.shared.left_asr == "parakeet-tdt-0.6b-v3-int8"
    assert new.shared.right_asr == "whisper-es"


def test_a_shared_section_loads_and_shared_is_a_mode():
    config = Config.parse(
        '[mode]\nkind = "shared"\n\n[shared]\nleft_language = "es"\n'
        'right_language = "en"\nleft_voice = "vits-piper-en_US-lessac-medium"\n'
    )
    assert config.mode.kind == "shared"
    assert config.shared.left_language == "es"
    assert config.shared.left_voice == "vits-piper-en_US-lessac-medium"
    assert config.shared.right_voice == ""
    with pytest.raises(ConfigError):
        Config.parse('[shared]\nmiddle_key = "x"\n')


def test_saving_adds_the_shared_section_with_the_voice_keys_explained(tmp_path):
    path = tmp_path / "volis.toml"
    path.write_text('[asr]\nengine = "whisper"\n', encoding="utf-8")
    config, _ = Config.load(path)
    config.mode.kind = "shared"
    config.shared.right_voice = "vits-piper-en_US-lessac-medium"
    config.save_selections(path)
    config.save_selections(path)  # a second save must not stack the comments up

    saved = path.read_text(encoding="utf-8")
    assert "[shared]" in saved
    assert 'kind = "shared"' in saved
    assert saved.count("left_voice speaks what the LEFT person said") == 1
    assert "RIGHT\n# person's language." in saved, "the note's second line starts at the margin"
    reloaded = Config.parse(saved)
    assert reloaded.shared.right_voice == "vits-piper-en_US-lessac-medium"
    assert reloaded.mode.kind == "shared"


def test_unknown_keys_are_rejected_rather_than_ignored():
    with pytest.raises(ConfigError):
        Config.parse('[asr]\nengine = "x"\nmodel = "y"\n')
    with pytest.raises(ConfigError):
        Config.parse("[nonsense]\nx = 1\n")


# pyvolis-specific


def test_a_pyvolis_section_is_refused_like_rust_refuses_it():
    """Rust's top-level deny_unknown_fields: pyvolis must never write one."""
    with pytest.raises(ConfigError):
        Config.parse("[pyvolis.asr]\nstreaming = true\n")


def test_wrong_types_and_enum_values_are_errors():
    for text in (
        '[mode]\nkind = "sometimes"\n',
        '[mode]\nturn_style = "tap"\n',
        '[tts]\nenabled = "yes"\n',
        "[vad]\nmin_silence_ms = -1\n",
        "[vad]\nthreshold = true\n",
    ):
        with pytest.raises(ConfigError):
            Config.parse(text)


def test_a_missing_file_gives_defaults_and_says_so(tmp_path):
    config, found = Config.load(tmp_path / "volis.toml")
    assert not found
    assert config == Config()


def test_saving_into_a_missing_file_creates_one_that_loads(tmp_path):
    path = tmp_path / "volis.toml"
    Config().save_selections(path)
    assert Config.parse(path.read_text(encoding="utf-8")) == Config()


def test_a_parse_error_names_the_absolute_path(tmp_path):
    path = tmp_path / "volis.toml"
    path.write_text("[asr\n", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        Config.load(path)
    assert str(path.absolute()) in str(e.value)
