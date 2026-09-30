"""Model discovery: the filesystem is the index.

Port of Rust `models.rs` for `engine.toml` folders, with the same rules and
the same three outcomes per directory:

* skipped - Rust: no `engine.toml`. In pyvolis, an ASR folder without one is
  looked at for a downloaded Hugging Face or GGUF model instead, and listed as
  unusable with the reason if it is neither. TTS folders keep Rust's rule.
* failed - described but wrong. An error for that entry only, listed.
* loaded - usable, unless a declared file is missing, in which case it is
  listed but disabled with the missing file named.

New in pyvolis: downloaded folders need no `engine.toml`. The backend is
decided from what the folder holds (`config.json` and weights: transformers;
`.gguf` plus `mmproj-*.gguf`: llama.cpp audio), and an optional `pyvolis.toml`
in the folder overrides what was detected. Translators live in `models/mt/`:
a `.gguf` at the top level (the one Rust volis uses) or one folder per model.
"""

from __future__ import annotations

import enum
import json
import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import gguf, varieties

log = logging.getLogger(__name__)

ENGINE_TOML = "engine.toml"
PYVOLIS_TOML = "pyvolis.toml"


class Role(enum.Enum):
    ASR = "asr"
    TTS = "tts"


# Rust's backends, per role, and the file roles each cannot work without.
SHERPA_BACKENDS: dict[Role, dict[str, tuple[str, ...]]] = {
    Role.ASR: {
        "nemo_transducer": ("encoder", "decoder", "joiner", "tokens"),
        "whisper": ("encoder", "decoder", "tokens"),
    },
    Role.TTS: {"vits": ("model", "tokens")},
}

# Backends as pyvolis names them. `sherpa` folders keep their engine.toml
# backend name in the report, exactly as Rust prints it.
TRANSFORMERS = "transformers"
LLAMACPP_AUDIO = "llamacpp-audio"

# transformers model types in its speech seq2seq mapping that are not speech
# recognition: text-to-speech and music models.
NOT_ASR_MODEL_TYPES = {"dia", "pop2piano", "vibevoice"}


class ModelError(Exception):
    """Why one model folder could not be turned into a usable description.
    The message always names the absolute path involved."""


@dataclass
class ModelFile:
    role: str  # "encoder", "tokens", "weights", ...
    name: str  # the published filename, verbatim
    path: Path
    present: bool


@dataclass
class ModelDir:
    name: str
    path: Path
    present: bool


@dataclass
class Engine:
    """A model directory that was understood."""

    dir_name: str  # identity used in volis.toml
    dir: Path
    name: str  # human-readable
    kind: str  # "segment" | "stream"
    backend: str  # "nemo_transducer" | "whisper" | "vits" | "transformers" | ...
    languages: list[str]
    varieties: list[str] = field(default_factory=list)
    files: list[ModelFile] = field(default_factory=list)
    data_dir: ModelDir | None = None
    # pyvolis only below.
    from_engine_toml: bool = True
    # False: the model does not say which languages it knows. It is offered
    # for every language, labelled "languages unknown".
    languages_known: bool = True
    # Backend detail for --report, e.g. "whisper -> WhisperForConditionalGeneration".
    detail: str = ""
    # Settings from the folder's pyvolis.toml (device, dtype, trust_remote_code).
    settings: dict[str, Any] = field(default_factory=dict)
    # Set when the folder is recognised but pyvolis can't run it (yet).
    unusable: str = ""

    def missing_files(self) -> list[str]:
        missing = [f.name for f in self.files if not f.present]
        if self.data_dir is not None and not self.data_dir.present:
            missing.append(self.data_dir.name)
        return missing

    def enabled(self) -> bool:
        return not self.missing_files() and not self.unusable


@dataclass
class Failed:
    """A folder that could not be understood. Listed, never silently dropped."""

    dir_name: str
    dir: Path
    error: str


Entry = Engine | Failed


def entry_dir_name(entry: Entry) -> str:
    return entry.dir_name


# ---------------------------------------------------------------- ranking


class FitKind(enum.IntEnum):
    TUNED = 0  # tuned for exactly the variety asked for
    GENERAL = 1  # covers the language, no variety of it declared
    OTHER = 2  # tuned for a different variety of the same language
    UNKNOWN = 3  # pyvolis: the model doesn't say which languages it knows


@dataclass(frozen=True)
class Fit:
    kind: FitKind
    variety: str = ""  # for OTHER: the variety it is tuned for

    def label(self, asked: str) -> str:
        if self.kind is FitKind.TUNED:
            return f"tuned for {varieties.display_name(asked)}"
        if self.kind is FitKind.GENERAL:
            return "general"
        if self.kind is FitKind.OTHER:
            return f"tuned for {varieties.display_name(self.variety)}"
        return "languages unknown"


@dataclass
class Ranked:
    engine: Engine
    fit: Fit


def rank(tag: str, engines: list[Engine]) -> list[Ranked]:
    """The usable models for a language or variety, best first: tuned for
    exactly that variety, then general models of its language, then models
    tuned for another variety of it, then (pyvolis) models whose languages are
    unknown. Models that don't cover the language aren't listed. Within each
    group, discovery order is kept."""
    language = varieties.language_of(tag).lower()
    ranked: list[Ranked] = []
    for engine in engines:
        if not engine.enabled():
            continue
        if not engine.languages_known:
            ranked.append(Ranked(engine, Fit(FitKind.UNKNOWN)))
            continue
        if not any(lang.lower() == language for lang in engine.languages):
            continue
        own = [v for v in engine.varieties if varieties.language_of(v).lower() == language]
        if any(v.lower() == tag.strip().lower() for v in own):
            fit = Fit(FitKind.TUNED)
        elif own:
            fit = Fit(FitKind.OTHER, own[0])
        else:
            fit = Fit(FitKind.GENERAL)
        ranked.append(Ranked(engine, fit))
    ranked.sort(key=lambda r: r.fit.kind)  # stable: discovery order kept
    return ranked


# ---------------------------------------------------------------- engine.toml

_ENGINE_KEYS = {"name", "kind", "backend", "languages", "varieties", "data_dir", "files"}


def is_plain_filename(name: str) -> bool:
    """A file inside the model directory: no separators, no drive, no `..`."""
    return (
        name != ""
        and name not in (".", "..")
        and "/" not in name
        and "\\" not in name
        and ":" not in name
    )


def load_engine(directory: Path, role: Role) -> Engine:
    """Parse one model directory's `engine.toml`, as Rust's `load_engine`."""
    toml_path = directory / ENGINE_TOML
    where = toml_path.absolute()
    try:
        text = toml_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ModelError(f"cannot read {where}: {e}") from e
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ModelError(f"cannot parse {where}: {e}") from e

    unknown = set(raw) - _ENGINE_KEYS
    if unknown:
        raise ModelError(f"cannot parse {where}: unknown field `{sorted(unknown)[0]}`")
    for key in ("name", "kind", "backend"):
        if not isinstance(raw.get(key), str):
            raise ModelError(f"cannot parse {where}: missing field `{key}`")
    if raw["kind"] not in ("segment", "stream"):
        raise ModelError(
            f"cannot parse {where}: unknown variant `{raw['kind']}`, expected `segment` or `stream`"
        )
    languages = _string_list(raw.get("languages", []), "languages", where)
    variety_list = _string_list(raw.get("varieties", []), "varieties", where)
    files_raw = raw.get("files", {})
    if not isinstance(files_raw, dict) or not all(isinstance(v, str) for v in files_raw.values()):
        raise ModelError(f"cannot parse {where}: [files] must map roles to filenames")

    backends = SHERPA_BACKENDS[role]
    backend = raw["backend"]
    if backend not in backends:
        raise ModelError(
            f'unknown backend "{backend}" in {where} (known: {", ".join(backends)})'
        )
    for required in backends[backend]:
        if required not in files_raw:
            raise ModelError(
                f'{where} declares backend "{backend}" but [files].{required} is missing'
            )

    files = []
    for file_role in sorted(files_raw):  # Rust keeps them in a BTreeMap
        name = files_raw[file_role]
        if not is_plain_filename(name):
            raise ModelError(
                f'{where} has [files].{file_role} = "{name}"; it must be a plain filename '
                "inside the model directory"
            )
        path = directory / name
        files.append(ModelFile(file_role, name, path, path.is_file()))

    data_dir = None
    if "data_dir" in raw:
        name = raw["data_dir"]
        if not isinstance(name, str) or not is_plain_filename(name):
            raise ModelError(
                f'{where} has [files].data_dir = "{name}"; it must be a plain filename '
                "inside the model directory"
            )
        path = directory / name
        data_dir = ModelDir(name, path, path.is_dir())

    _check_varieties(variety_list, languages, where)

    if (directory / PYVOLIS_TOML).is_file():
        log.warning(
            "%s is ignored: %s describes this folder",
            (directory / PYVOLIS_TOML).absolute(),
            ENGINE_TOML,
        )

    return Engine(
        dir_name=directory.name,
        dir=directory,
        name=raw["name"],
        kind=raw["kind"],
        backend=backend,
        languages=languages,
        varieties=variety_list,
        files=files,
        data_dir=data_dir,
    )


def _string_list(value: Any, key: str, where: Path) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ModelError(f"cannot parse {where}: `{key}` must be a list of strings")
    return list(value)


def _check_varieties(variety_list: list[str], languages: list[str], where: Path) -> None:
    """Every variety must be one Volis knows, and belong to a listed language."""
    for variety in variety_list:
        if varieties.lookup(variety) is None:
            reason = "not a variety Volis knows (add it to pyvolis/varieties.py and src/varieties.rs)"
        elif not varieties.has_variety(variety):
            reason = 'a variety names a region, like "es-MX"; plain languages go in `languages`'
        else:
            language = varieties.language_of(variety)
            if any(lang.lower() == language.lower() for lang in languages):
                continue
            reason = f'its language "{language}" is not in `languages` ({", ".join(languages)})'
        raise ModelError(f'{where} lists variety "{variety}": {reason}')


# ---------------------------------------------------------------- pyvolis.toml

_PYVOLIS_KEYS = {
    "name": str,
    "languages": list,
    "varieties": list,
    "device": str,
    "dtype": str,
    "trust_remote_code": bool,
    # A LoRA adapter's base model (P11).
    "base": str,
}


def load_overrides(directory: Path) -> dict[str, Any]:
    """The folder's optional `pyvolis.toml`, checked. Empty if absent."""
    path = directory / PYVOLIS_TOML
    if not path.is_file():
        return {}
    where = path.absolute()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ModelError(f"cannot parse {where}: {e}") from e
    for key, value in raw.items():
        expected = _PYVOLIS_KEYS.get(key)
        if expected is None:
            raise ModelError(
                f"{where}: unknown key `{key}` (known: {', '.join(_PYVOLIS_KEYS)})"
            )
        if not isinstance(value, expected):
            raise ModelError(f"{where}: `{key}` must be a {expected.__name__}")
    for key in ("languages", "varieties"):
        if key in raw:
            _string_list(raw[key], key, where)
    if raw.get("device", "cuda") not in ("cuda", "cpu"):
        raise ModelError(f'{where}: device must be "cuda" or "cpu", not "{raw["device"]}"')
    return raw


# ---------------------------------------------------------------- model cards

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(\n|\Z)", re.S)


def card_languages(directory: Path) -> list[str] | None:
    """Languages from the model card's front matter (`language:` in
    README.md), lower-cased. None if the card doesn't say."""
    readme = directory / "README.md"
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _FRONT_MATTER.match(text.replace("\r\n", "\n"))
    if not match:
        return None
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        log.warning("cannot parse the front matter of %s", readme.absolute())
        return None
    if not isinstance(meta, dict):
        return None
    value = meta.get("language")
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return None
    # Cards may repeat a code (MMS lists "qu" 22 times, one per variety).
    languages = list(dict.fromkeys(str(v).strip().lower() for v in value if str(v).strip()))
    # "multilingual" says nothing about which languages.
    languages = [v for v in languages if v != "multilingual"]
    return languages or None


# ---------------------------------------------------------------- downloaded ASR

_WEIGHTS = ("*.safetensors", "pytorch_model.bin", "pytorch_model-*.bin")


def _speech_classes() -> tuple[dict[str, str], dict[str, str], str]:
    """transformers' speech seq2seq and CTC mappings, and its version."""
    import transformers
    from transformers.models.auto import modeling_auto

    return (
        dict(modeling_auto.MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES),
        dict(modeling_auto.MODEL_FOR_CTC_MAPPING_NAMES),
        transformers.__version__,
    )


def load_downloaded_asr(directory: Path) -> Engine:
    """Recognise an ASR folder that has no engine.toml."""
    overrides = load_overrides(directory)
    if (directory / "adapter_config.json").is_file():
        return _lora_adapter(directory, overrides)
    if (directory / "config.json").is_file():
        return _transformers_asr(directory, overrides)
    ggufs = sorted(directory.glob("*.gguf"))
    if ggufs:
        return _gguf_asr(directory, ggufs, overrides)
    raise ModelError(
        f"{directory.absolute()} has no {ENGINE_TOML}, no config.json and no .gguf file, "
        "so pyvolis can't tell what model it is"
    )


def _transformers_asr(directory: Path, overrides: dict[str, Any]) -> Engine:
    config_path = directory / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ModelError(f"cannot parse {config_path.absolute()}: {e}") from e
    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        raise ModelError(f"{config_path.absolute()} has no model_type")

    seq2seq, ctc, version = _speech_classes()
    if model_type in seq2seq and model_type not in NOT_ASR_MODEL_TYPES:
        detail = f"{model_type} -> {seq2seq[model_type]} (speech seq2seq)"
    elif model_type in ctc:
        detail = f"{model_type} -> {ctc[model_type]} (CTC)"
    else:
        raise ModelError(
            f'{config_path.absolute()} has model_type "{model_type}", which is not a speech '
            f"recognition model transformers {version} knows. A translation model belongs "
            "in models/mt/."
        )

    engine = Engine(
        dir_name=directory.name,
        dir=directory,
        name=overrides.get("name", directory.name),
        kind="segment",
        backend=TRANSFORMERS,
        languages=[],
        from_engine_toml=False,
        detail=detail,
        settings={k: overrides[k] for k in ("device", "dtype", "trust_remote_code") if k in overrides},
    )
    engine.files.append(ModelFile("config", "config.json", config_path, True))
    weights = _find_weights(directory)
    if weights:
        engine.files.extend(weights)
    else:
        engine.files.append(
            ModelFile("weights", "*.safetensors or pytorch_model.bin", directory, False)
        )
    # The feature extractor's settings: preprocessor_config.json, or in newer
    # transformers layouts, processor_config.json holding them all.
    for name in ("preprocessor_config.json", "processor_config.json"):
        if (directory / name).is_file():
            engine.files.append(ModelFile("preprocessor", name, directory / name, True))
            break
    else:
        engine.files.append(
            ModelFile(
                "preprocessor",
                "preprocessor_config.json or processor_config.json",
                directory / "preprocessor_config.json",
                False,
            )
        )
    if "auto_map" in config and not overrides.get("trust_remote_code", False):
        engine.unusable = (
            "the model needs its own code (auto_map in config.json); set "
            f"trust_remote_code = true in {(directory / PYVOLIS_TOML).absolute()} "
            "if you trust it"
        )
    _apply_languages(engine, directory, overrides)
    # MMS: the languages it can do here are the adapters actually present, not
    # the ~1,100 its model card lists.
    adapters = sorted(p.name.split(".")[1] for p in directory.glob("adapter.*.safetensors"))
    if adapters and not overrides.get("languages"):
        engine.languages = [varieties.from_iso639_3(code) for code in adapters]
        engine.languages_known = True
        engine.detail += f"; language adapters: {', '.join(adapters)}"
    return engine


def _find_weights(directory: Path) -> list[ModelFile]:
    """The weight files, preferring safetensors. A sharded model's index names
    its shards; each must be present."""
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / index_name
        if index.is_file():
            try:
                shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
            except (OSError, ValueError, KeyError, AttributeError) as e:
                raise ModelError(f"cannot parse {index.absolute()}: {e}") from e
            return [ModelFile("weights", s, directory / s, (directory / s).is_file()) for s in shards]
    for pattern in _WEIGHTS:
        found = sorted(directory.glob(pattern))
        if found:
            return [ModelFile("weights", p.name, p, True) for p in found]
    return []


def _gguf_asr(directory: Path, ggufs: list[Path], overrides: dict[str, Any]) -> Engine:
    encoders = [p for p in ggufs if p.name.lower().startswith("mmproj")]
    models = [p for p in ggufs if p not in encoders]
    if not models or not encoders:
        what = "an audio encoder (mmproj-*.gguf)" if models else "a language model .gguf"
        raise ModelError(
            f"{directory.absolute()} holds .gguf files but no {what}; a llama.cpp speech model "
            "needs both"
        )
    engine = Engine(
        dir_name=directory.name,
        dir=directory,
        name=overrides.get("name", directory.name),
        kind="segment",
        backend=LLAMACPP_AUDIO,
        languages=[],
        from_engine_toml=False,
        detail=f"{models[0].name} + {encoders[0].name}",
        files=[ModelFile("model", models[0].name, models[0], True),
               ModelFile("audio encoder", encoders[0].name, encoders[0], True)],
        unusable="llama.cpp speech models arrive at P11, once llama-cpp-python's audio input "
        "is verified",
    )
    _apply_languages(engine, directory, overrides)
    return engine


def _lora_adapter(directory: Path, overrides: dict[str, Any]) -> Engine:
    config_path = directory / "adapter_config.json"
    try:
        base = json.loads(config_path.read_text(encoding="utf-8")).get("base_model_name_or_path", "")
    except (OSError, ValueError) as e:
        raise ModelError(f"cannot parse {config_path.absolute()}: {e}") from e
    engine = Engine(
        dir_name=directory.name,
        dir=directory,
        name=overrides.get("name", directory.name),
        kind="segment",
        backend=TRANSFORMERS,
        languages=[],
        from_engine_toml=False,
        detail=f"LoRA adapter for {base or '(base not named)'}",
        files=[ModelFile("adapter config", config_path.name, config_path, True)],
        unusable="LoRA adapters are attached from P11",
    )
    _apply_languages(engine, directory, overrides)
    return engine


def _apply_languages(engine: Engine, directory: Path, overrides: dict[str, Any]) -> None:
    """pyvolis.toml first, then the model card; otherwise "languages unknown"."""
    languages = overrides.get("languages") or card_languages(directory)
    if languages:
        engine.languages = list(languages)
    else:
        engine.languages_known = False
    engine.varieties = list(overrides.get("varieties", []))
    _check_varieties(engine.varieties, engine.languages, (directory / PYVOLIS_TOML).absolute())


# ---------------------------------------------------------------- discovery


def discover(root: Path, role: Role) -> list[Entry]:
    """Enumerate one model root, sorted by folder name. A missing root is an
    empty list plus a warning naming the absolute path."""
    try:
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as e:
        log.warning("cannot read model directory %s: %s", root.absolute(), e)
        return []

    entries: list[Entry] = []
    for directory in dirs:
        has_engine_toml = (directory / ENGINE_TOML).is_file()
        if not has_engine_toml and role is Role.TTS:
            log.warning(
                "skipping %s: no %s (expected %s)",
                directory.absolute(),
                ENGINE_TOML,
                (directory / ENGINE_TOML).absolute(),
            )
            continue
        try:
            if has_engine_toml:
                entries.append(load_engine(directory, role))
            else:
                entries.append(load_downloaded_asr(directory))
        except ModelError as e:
            log.warning("%s", e)
            entries.append(Failed(directory.name, directory, str(e)))
    return entries


# ---------------------------------------------------------------- translators


@dataclass
class Translator:
    """One translation model the user can pick."""

    id: str  # "qwen3-1.7b-q4_k_m.gguf" at the top level, "folder/file.gguf" in a folder
    path: Path  # the .gguf, or the folder for a transformers model
    name: str
    backend: str  # "llamacpp" | "transformers"
    top_level: bool  # True: the file Rust volis uses
    architecture: str = ""
    has_chat_template: bool = False
    size_bytes: int = 0
    missing: list[str] = field(default_factory=list)
    unusable: str = ""

    def enabled(self) -> bool:
        return not self.missing and not self.unusable


_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


def _is_first_shard(name: str) -> bool:
    """True for an unsplit .gguf, or the first part of a split one."""
    split = _SPLIT.search(name)
    return split is None or split.group(1) == "00001"


def discover_translators(root: Path) -> list[Translator | Failed]:
    """Every translator in `models/mt/`: top-level `.gguf` files first (sorted),
    then one or more per folder (sorted by folder)."""
    try:
        children = sorted(root.iterdir())
    except OSError as e:
        log.warning("cannot read model directory %s: %s", root.absolute(), e)
        return []

    found: list[Translator | Failed] = []
    for path in children:
        if path.is_file() and path.suffix.lower() == ".gguf" and _is_first_shard(path.name):
            found.append(_gguf_translator(path, path.name, None, top_level=True))
    for directory in (p for p in children if p.is_dir()):
        try:
            found.extend(_translators_in(directory))
        except ModelError as e:
            log.warning("%s", e)
            found.append(Failed(directory.name, directory, str(e)))
    return found


def _translators_in(directory: Path) -> list[Translator | Failed]:
    overrides = load_overrides(directory)
    ggufs = sorted(p for p in directory.glob("*.gguf") if not p.name.lower().startswith("mmproj"))
    # Only the first shard of a split model stands for it.
    ggufs = [p for p in ggufs if _is_first_shard(p.name)]
    if ggufs:
        return [
            _gguf_translator(p, f"{directory.name}/{p.name}", overrides, top_level=False)
            for p in ggufs
        ]
    if (directory / "config.json").is_file():
        weights = _find_weights(directory)
        return [
            Translator(
                id=directory.name,
                path=directory,
                name=overrides.get("name", directory.name),
                backend=TRANSFORMERS,
                top_level=False,
                missing=[w.name for w in weights if not w.present]
                or ([] if weights else ["*.safetensors"]),
                unusable="transformers translators arrive at P11",
            )
        ]
    raise ModelError(f"{directory.absolute()} has no .gguf file and no config.json")


def _gguf_translator(
    path: Path, ident: str, overrides: dict[str, Any] | None, top_level: bool
) -> Translator | Failed:
    try:
        meta = gguf.read_metadata(path)
    except gguf.GgufError as e:
        return Failed(ident, path, str(e))
    translator = Translator(
        id=ident,
        path=path,
        name=(overrides or {}).get("name") or str(meta.get("general.name") or path.stem),
        backend="llamacpp",
        top_level=top_level,
        architecture=str(meta.get("general.architecture", "")),
        has_chat_template=isinstance(meta.get("tokenizer.chat_template"), str),
        size_bytes=path.stat().st_size,
    )
    split = _SPLIT.search(path.name)
    if split:
        total = int(split.group(2))
        for i in range(2, total + 1):
            shard = path.with_name(_SPLIT.sub(f"-{i:05d}-of-{total:05d}.gguf", path.name))
            if shard.is_file():
                translator.size_bytes += shard.stat().st_size
            else:
                translator.missing.append(shard.name)
    if meta.get("general.type") == "adapter":
        translator.unusable = "a LoRA adapter; adapters are attached to their base from P11"
    elif not translator.has_chat_template:
        translator.unusable = (
            f"{path.absolute()} carries no chat template (tokenizer.chat_template), so pyvolis "
            "can't lay out a prompt for it"
        )
    return translator
