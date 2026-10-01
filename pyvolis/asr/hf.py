"""The transformers backend: any downloaded Hugging Face speech model, loaded
from its folder with offline mode on.

  whisper      the language and the task ("transcribe", never "translate")
               forced on every call; word timestamps from the model's
               alignment heads when it has them.
  CTC          wav2vec2, MMS and the like: greedy decoding. MMS switches its
               language adapter to the ISO 639-3 code of the language asked
               for, and says so in the log.
  cohere_asr   Cohere Transcribe: the language goes to its processor.

Loaded once and reused. float16 on the GPU, float32 on the CPU, unless the
folder's pyvolis.toml says otherwise. Before loading, the model's size is
checked against free memory, and a model that won't fit is refused with a
message naming it, its size and what's free, never an out-of-memory crash.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from pathlib import Path

import numpy as np

from .. import varieties
from ..audio import SAMPLE_RATE
from ..models import Engine
from . import AsrError, AsrMemoryError, AsrResult, Memory, Word

log = logging.getLogger(__name__)

# Headroom on top of the weights: activations, the CUDA context, the
# allocator's slack. Measured loads come in well under this.
OVERHEAD = 1.25
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
# Cohere Transcribe's generation limit, from its model card.
MAX_NEW_TOKENS = 256


def load(engine: Engine):
    return HfAsr(engine)


def weight_elements(directory: Path) -> tuple[int, int]:
    """(parameter count, bytes on disk) of a folder's weights, read from the
    safetensors headers without loading anything. MMS's language adapters
    other than the base are not counted (one is loaded at a time)."""
    files = [p for p in directory.glob("*.safetensors") if not p.name.startswith("adapter.")]
    elements = size = 0
    for path in files:
        size += path.stat().st_size
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(length))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            count = 1
            for dim in info["shape"]:
                count *= dim
            elements += count
    if not files:  # pytorch_model.bin: estimate from size, assuming float32
        size = sum(p.stat().st_size for p in directory.glob("pytorch_model*.bin"))
        elements = size // 4
    return elements, size


class HfAsr:
    def __init__(self, engine: Engine) -> None:
        import torch

        self.name = engine.dir_name
        self._engine = engine
        settings = engine.settings
        want = settings.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        if want == "cuda" and not torch.cuda.is_available():
            raise AsrError(f'"{self.name}" asks for device "cuda" in its pyvolis.toml, but torch sees no GPU')
        self.device = want
        dtype_name = settings.get("dtype") or ("float16" if want == "cuda" else "float32")
        try:
            self.dtype = getattr(torch, dtype_name)
        except AttributeError:
            raise AsrError(f'"{self.name}": unknown dtype "{dtype_name}" in {engine.dir / "pyvolis.toml"}') from None

        elements, _ = weight_elements(engine.dir)
        self._bytes = int(elements * torch.tensor([], dtype=self.dtype).element_size())
        self._check_fits(torch)

        config = json.loads((engine.dir / "config.json").read_text(encoding="utf-8"))
        self.model_type = config["model_type"]
        trust = bool(settings.get("trust_remote_code", False))
        began = time.perf_counter()
        try:
            self._load(torch, trust)
        except torch.cuda.OutOfMemoryError as e:
            self.close()
            raise AsrMemoryError(f'"{self.name}" ran out of GPU memory while loading: {e}') from e
        except OSError as e:
            if trust:
                raise AsrError(
                    f'"{self.name}" could not be loaded from {engine.dir}: {e}. With trust_remote_code on, '
                    "the model's own code files must already be in the folder; pyvolis never fetches them."
                ) from e
            raise AsrError(f'"{self.name}" could not be loaded from {engine.dir}: {e}') from e
        log.info(
            'loaded "%s" (%s, %s on %s, %.1f GB) in %.1f s',
            self.name, type(self._model).__name__, dtype_name, self.device, self._bytes / 1e9,
            time.perf_counter() - began,
        )

    def _check_fits(self, torch) -> None:
        need = int(self._bytes * OVERHEAD)
        if self.device == "cuda":
            free, _total = torch.cuda.mem_get_info()
            where = "GPU memory"
        else:
            free, where = _free_ram(), "system memory"
        if free and need > free:
            raise AsrMemoryError(
                f'"{self.name}" needs about {need / 1e9:.1f} GB of {where} and {free / 1e9:.1f} GB is free. '
                "Close another model, or set device = \"cpu\" in "
                f'{self._engine.dir / "pyvolis.toml"}.'
            )

    def _load(self, torch, trust: bool) -> None:
        import transformers

        # Library progress bars belong in a terminal session, not in the log.
        transformers.utils.logging.disable_progress_bar()
        transformers.utils.logging.set_verbosity_error()
        path = str(self._engine.dir)
        common = dict(local_files_only=True, trust_remote_code=trust)
        self._processor = transformers.AutoProcessor.from_pretrained(path, **common)
        if self.model_type == "whisper":
            cls = transformers.WhisperForConditionalGeneration
        elif self.model_type == "cohere_asr":
            cls = transformers.CohereAsrForConditionalGeneration
        elif self.model_type in _ctc_types():
            cls = transformers.AutoModelForCTC
        else:
            cls = transformers.AutoModelForSpeechSeq2Seq
        kwargs = dict(dtype=self.dtype, **common)
        if self.model_type == "whisper":
            kwargs["attn_implementation"] = "sdpa"  # the faster attention; word timings still work
        if self.model_type in _ctc_types() and any(self._engine.dir.glob("adapter.*.safetensors")):
            # MMS: keep the adapter weights loadable per language.
            kwargs["ignore_mismatched_sizes"] = True
        self._model = cls.from_pretrained(path, **kwargs).to(self.device).eval()
        self._adapter = None

    # ------------------------------------------------------------ transcribe

    def transcribe(self, audio: np.ndarray, language: str, timestamps: bool = True) -> AsrResult:
        import torch

        language = varieties.language_of(language)
        began = time.perf_counter()
        audio = np.asarray(audio, dtype=np.float32)
        try:
            with torch.inference_mode():
                if self.model_type == "whisper":
                    text, words = self._whisper(audio, language, timestamps)
                    confidence = None
                elif self.model_type == "cohere_asr":
                    text, words, confidence = self._cohere(audio, language), None, None
                elif self.model_type in _ctc_types():
                    text, words, confidence = self._ctc(audio, language)
                else:
                    text, words, confidence = self._seq2seq(audio, language), None, None
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise AsrMemoryError(f'"{self.name}" ran out of GPU memory transcribing: {e}') from e
        except AsrError:
            raise
        except Exception as e:
            raise AsrError(f'"{self.name}" failed to transcribe: {e}') from e
        return AsrResult(text.strip(), words, confidence, language, time.perf_counter() - began)

    def _features(self, audio: np.ndarray, **kwargs):
        inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", **kwargs)
        return inputs.to(self.device, dtype=self.dtype) if hasattr(inputs, "to") else inputs

    def _whisper(self, audio: np.ndarray, language: str, timestamps: bool = True):
        log.info('  "%s" is told the language "%s"', self.name, language)
        inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
        features = inputs.input_features.to(self.device, dtype=self.dtype)
        # Word timings come from the alignment heads and cost about a second
        # a call (measured: 1.3 s against 0.3 s for 6 s of audio), so only on request.
        with_words = timestamps and getattr(self._model.generation_config, "alignment_heads", None) is not None
        out = self._model.generate(
            features, attention_mask=inputs.attention_mask.to(self.device), language=language,
            task="transcribe", return_token_timestamps=with_words, return_dict_in_generate=True,
        )
        # transformers 5 hands back a plain dict here; older versions an object.
        get = out.get if isinstance(out, dict) else lambda k: getattr(out, k, None)
        sequence = get("sequences")[0]
        text = self._processor.decode(sequence, skip_special_tokens=True)
        words = None
        stamps = get("token_timestamps")
        if with_words and stamps is not None:
            words = _whisper_words(self._processor.tokenizer, sequence, stamps[0])
        return text, words

    def _cohere(self, audio: np.ndarray, language: str) -> str:
        known = self._engine.languages
        if known and language not in known:
            raise AsrError(f'"{self.name}" knows {", ".join(known)}, not "{language}"')
        inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=language)
        chunk_index = inputs.get("audio_chunk_index")
        inputs = inputs.to(self.device, dtype=self.dtype)
        out = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        decoded = self._processor.decode(
            out, skip_special_tokens=True, audio_chunk_index=chunk_index, language=language
        )
        return decoded[0] if isinstance(decoded, list) else decoded

    def _ctc(self, audio: np.ndarray, language: str):
        import torch

        self._select_adapter(language)
        inputs = self._processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        values = inputs.input_values.to(self.device, dtype=self.dtype)
        logits = self._model(values).logits[0].float()
        probs = torch.softmax(logits, dim=-1)
        best = probs.max(dim=-1)
        ids = best.indices
        decoded = self._processor.decode(ids, output_word_offsets=True)
        ratio = getattr(self._model.config, "inputs_to_logits_ratio", 320) / SAMPLE_RATE
        words = [Word(w["word"], w["start_offset"] * ratio, w["end_offset"] * ratio) for w in decoded.word_offsets]
        # Mean top probability over frames that aren't blank: the model's own
        # confidence in what it wrote.
        blank = self._processor.tokenizer.pad_token_id
        spoken = best.values[ids != blank]
        confidence = float(spoken.mean()) if len(spoken) else None
        return decoded.text, words, confidence

    def _select_adapter(self, language: str) -> None:
        """MMS: switch the language adapter, by ISO 639-3 code."""
        if not any(self._engine.dir.glob("adapter.*.safetensors")):
            return
        code = varieties.iso639_3(language)
        if code is None or not (self._engine.dir / f"adapter.{code}.safetensors").is_file():
            have = ", ".join(sorted(p.name.split(".")[1] for p in self._engine.dir.glob("adapter.*.safetensors")))
            raise AsrError(
                f'"{self.name}" has no language adapter for "{language}" '
                f"(looked for adapter.{code}.safetensors in {self._engine.dir}; present: {have})"
            )
        if self._adapter != code:
            log.info('  "%s": language "%s" -> MMS adapter "%s"', self.name, language, code)
            self._processor.tokenizer.set_target_lang(code)
            self._model.load_adapter(code)
            self._model.to(self.device, dtype=self.dtype)
            self._adapter = code

    def _seq2seq(self, audio: np.ndarray, language: str) -> str:
        inputs = self._features(audio)
        try:
            out = self._model.generate(**inputs, language=language)
        except (TypeError, ValueError):
            raise AsrError(
                f'"{self.name}" ({self.model_type}) can\'t be told the language, and pyvolis never '
                "lets a model detect it. Not supported yet."
            ) from None
        return self._processor.batch_decode(out, skip_special_tokens=True)[0]

    # ------------------------------------------------------------ memory

    def prepare(self, language: str) -> None:
        if self.model_type in _ctc_types():
            self._select_adapter(varieties.language_of(language))

    def memory(self) -> Memory:
        return Memory(gpu_bytes=self._bytes if self.device == "cuda" else 0,
                      cpu_bytes=self._bytes if self.device == "cpu" else 0, device=self.device)

    def close(self) -> None:
        import gc

        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ctc_types() -> set[str]:
    from transformers.models.auto import modeling_auto

    return set(modeling_auto.MODEL_FOR_CTC_MAPPING_NAMES)


def _whisper_words(tokenizer, sequence, stamps) -> list[Word]:
    """Group Whisper's per-token times into words: a token starting with a
    space (or the first text token) starts a new word."""
    words: list[Word] = []
    special = set(tokenizer.all_special_ids)
    for token, stamp in zip(sequence.tolist(), stamps.tolist()):
        if token in special or token >= tokenizer.convert_tokens_to_ids("<|0.00|>"):
            continue
        piece = tokenizer.decode([token])
        if not piece:
            continue
        if piece.startswith(" ") or not words:
            words.append(Word(piece.strip(), float(stamp), float(stamp)))
        else:
            words[-1].text += piece
            words[-1].end = float(stamp)
    for i, word in enumerate(words[:-1]):
        word.end = max(word.end, words[i + 1].start)
    return [w for w in words if w.text]


def _free_ram() -> int:
    """Available physical memory, in bytes (Windows); 0 if unknown."""
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                        ("avail_extended", ctypes.c_ulonglong)]

        status = Status()
        status.length = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.avail_phys)
    except Exception:
        return 0
