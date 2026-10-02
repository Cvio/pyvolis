"""Recognition by a GGUF speech model through llama.cpp's audio input:
Qwen3-ASR, Gemma 4, Voxtral and others that come as a language model `.gguf`
with an audio encoder beside it (`mmproj-*.gguf`).

New in pyvolis (the `llamacpp-audio` backend). Verified against
llama-cpp-python 0.3.35 before it was built: its `mtmd_cpp` module takes
16 kHz mono samples (`mtmd_bitmap_init_from_audio`), and both Qwen3-ASR 0.6B
and Gemma 4 E4B transcribed the fixtures through it. The library's own chat
handlers accept images only, so this talks to `mtmd_cpp` directly. Everything
runs in this process; nothing is served.

Two kinds of model, told apart by what they answer to:

* a model trained only to transcribe (Qwen3-ASR) is given the audio and
  nothing else, and answers `language Spanish<asr_text>...`. The language is
  forced by starting its answer for it.
* a general model that also hears (Gemma 4, Voxtral) is asked, in words, to
  transcribe the audio in the language.

Which one a folder is: `prompt = "..."` in its pyvolis.toml (`""` = the audio
alone), otherwise a name containing "asr" means the first kind.
"""

from __future__ import annotations

import ctypes
import logging
import time
from pathlib import Path

import numpy as np

from .. import gguf, varieties
from ..models import Engine
from . import AsrError, AsrMemoryError, AsrResult, Memory

log = logging.getLogger(__name__)

CONTEXT_TOKENS = 4096  # 25 s of audio is a few hundred tokens; the rest is headroom
MAX_OUTPUT_TOKENS = 448  # as Whisper's limit: about 25 s of fast speech
THREADS = 6
ASK = "Transcribe this audio exactly as spoken, in {language}. Output only the transcript, with nothing before or after it."
ASR_TEXT = "<asr_text>"


def load(engine: Engine):
    return LlamaAudioAsr(engine)


class LlamaAudioAsr:
    def __init__(self, engine: Engine) -> None:
        from ..translate.llamacpp import gpu_available, load_cuda_runtime, render

        load_cuda_runtime()
        import llama_cpp
        from llama_cpp import mtmd_cpp

        self.name = engine.dir_name
        self._engine = engine
        self._llama_cpp, self._mtmd, self._render = llama_cpp, mtmd_cpp, render
        files = {f.role: f.path for f in engine.files}
        model, encoder = files["model"], files["audio encoder"]
        want = engine.settings.get("device") or ("cuda" if gpu_available() else "cpu")
        if want == "cuda" and not gpu_available():
            raise AsrError(f'"{self.name}" asks for device "cuda" in its pyvolis.toml, but the llama.cpp build here '
                           "can't use a GPU (build-llama.ps1 -Cuda)")
        self.device = want
        self._bytes = model.stat().st_size + encoder.stat().st_size
        self._check_fits()
        meta = gguf.read_metadata(model)
        self._template = meta.get("tokenizer.chat_template")
        self._architecture = str(meta.get("general.architecture", ""))
        if not self._template:
            raise AsrError(f"{model.absolute()} carries no chat template, so pyvolis can't lay out a prompt for it")
        # Audio alone, or a request in words.
        prompt = engine.settings.get("prompt")
        self._ask = prompt if prompt is not None else ("" if "asr" in f"{model.name} {engine.dir_name}".lower() else ASK)
        began = time.perf_counter()
        try:
            self._llama = llama_cpp.Llama(
                model_path=str(model), n_ctx=CONTEXT_TOKENS, n_batch=CONTEXT_TOKENS, n_threads=THREADS,
                n_threads_batch=THREADS, n_gpu_layers=-1 if want == "cuda" else 0, flash_attn=True, verbose=False)
        except Exception as e:
            raise AsrError(f"cannot load the speech model {model.absolute()}: {e}") from e
        params = mtmd_cpp.mtmd_context_params_default()
        params.use_gpu = want == "cuda"
        params.print_timings = False
        params.n_threads = THREADS
        mtmd_cpp.mtmd_log_set(_no_log, None)
        mtmd_cpp.mtmd_helper_log_set(_no_log, None)
        self._ctx = mtmd_cpp.mtmd_init_from_file(str(encoder).encode(), self._llama.model, params)
        if not self._ctx:
            self.close()
            raise AsrError(f"cannot load the audio encoder {encoder.absolute()} for {model.name}")
        if not mtmd_cpp.mtmd_support_audio(self._ctx):
            self.close()
            raise AsrError(f"{encoder.absolute()} is not an audio encoder: {model.name} with it does not take audio "
                           "(an image encoder, perhaps)")
        rate = mtmd_cpp.mtmd_get_audio_sample_rate(self._ctx)
        if rate != 16_000:
            self.close()
            raise AsrError(f"{self.name} wants audio at {rate} Hz; pyvolis works at 16000 Hz")
        self._marker = mtmd_cpp.mtmd_default_marker().decode()
        self._vocab = llama_cpp.llama_model_get_vocab(self._llama.model)
        self._n_vocab = llama_cpp.llama_vocab_n_tokens(self._vocab)
        log.info('loaded "%s" (%s through llama.cpp audio, on the %s, %.1f GB; %s) in %.1f s', self.name,
                 self._architecture, self.device.upper(), self._bytes / 1e9,
                 "asked in words" if self._ask else "given the audio alone", time.perf_counter() - began)

    def _check_fits(self) -> None:
        if self.device != "cuda":
            return
        try:
            import torch

            free, _total = torch.cuda.mem_get_info()
        except Exception:
            return
        need = int(self._bytes * 1.2)
        if need > free:
            raise AsrMemoryError(
                f'"{self.name}" needs about {need / 1e9:.1f} GB of GPU memory and {free / 1e9:.1f} GB is free. '
                f'Close another model, or set device = "cpu" in {self._engine.dir / "pyvolis.toml"}.')

    def prepare(self, language: str) -> None:
        varieties.require(language)

    def warm_up(self, language: str) -> None:
        self.transcribe(np.zeros(16_000, dtype=np.float32), language, False)

    def prompt(self, language: str) -> str:
        """The exact text around the audio for a language (the audio goes
        where the marker is)."""
        name = _language_name(language)
        if self._ask:
            content = f"{self._marker}\n{self._ask.format(language=name)}"
            return self._render(self._template, [{"role": "user", "content": content}], {}, self._architecture)
        # A transcription model: the audio alone, and its answer begun for it
        # with the language, so it is told rather than left to decide.
        rendered = self._render(self._template, [{"role": "user", "content": self._marker}], {}, "")
        return f"{rendered}language {name}{ASR_TEXT}"

    def transcribe(self, audio: np.ndarray, language: str, timestamps: bool = True) -> AsrResult:
        llama_cpp, mtmd = self._llama_cpp, self._mtmd
        began = time.perf_counter()
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        if len(samples) < 1600:  # under 0.1 s: nothing to hear
            return AsrResult("", None, None, language, 0.0)
        prompt = self.prompt(language).encode("utf-8")
        bitmap = mtmd.mtmd_bitmap_init_from_audio(len(samples), samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        chunks = mtmd.mtmd_input_chunks_init()
        batch = None
        try:
            text = mtmd.mtmd_input_text()
            text.text, text.add_special, text.parse_special = prompt, True, True
            if hasattr(text, "text_len"):
                text.text_len = len(prompt)
            bitmaps = (mtmd.mtmd_bitmap_p_ctypes * 1)(bitmap)
            if (code := mtmd.mtmd_tokenize(self._ctx, chunks, ctypes.byref(text), bitmaps, 1)) != 0:
                raise AsrError(f"{self.name}: llama.cpp could not lay out the audio prompt (error {code})")
            needed = mtmd.mtmd_helper_get_n_tokens(chunks)
            if needed + 16 >= CONTEXT_TOKENS:
                raise AsrError(f"{self.name}: {len(samples) / 16_000:.0f} s of audio is {needed} tokens, more than "
                               f"its {CONTEXT_TOKENS} token context")
            self._llama.reset()
            self._llama._ctx.kv_cache_clear()
            past = llama_cpp.llama_pos(0)
            code = mtmd.mtmd_helper_eval_chunks(self._ctx, self._llama._ctx.ctx, chunks, llama_cpp.llama_pos(0),
                                                llama_cpp.llama_seq_id(0), self._llama.n_batch, True,
                                                ctypes.byref(past))
            if code != 0:
                raise AsrError(f"{self.name}: llama.cpp could not evaluate the audio (error {code})")
            tokens, position = [], past.value
            batch = llama_cpp.llama_batch_init(1, 0, 1)
            for _ in range(MAX_OUTPUT_TOKENS):  # greedy
                logits = llama_cpp.llama_get_logits_ith(self._llama._ctx.ctx, -1)
                token = int(np.argmax(np.ctypeslib.as_array(logits, shape=(self._n_vocab,))))
                if llama_cpp.llama_vocab_is_eog(self._vocab, token) or position + 1 >= CONTEXT_TOKENS:
                    break
                tokens.append(token)
                batch.n_tokens = 1
                batch.token[0], batch.pos[0], batch.n_seq_id[0], batch.logits[0] = token, position, 1, True
                batch.seq_id[0][0] = 0
                if llama_cpp.llama_decode(self._llama._ctx.ctx, batch) != 0:
                    raise AsrError(f"{self.name}: llama.cpp stopped decoding after {len(tokens)} tokens")
                position += 1
            raw = self._llama.detokenize(tokens, special=True).decode("utf-8", errors="replace")
        finally:
            if batch is not None:
                llama_cpp.llama_batch_free(batch)
            mtmd.mtmd_input_chunks_free(chunks)
            mtmd.mtmd_bitmap_free(bitmap)
        return AsrResult(clean(raw), None, None, language, time.perf_counter() - began)

    def memory(self) -> Memory:
        return Memory(gpu_bytes=self._bytes if self.device == "cuda" else 0,
                      cpu_bytes=self._bytes if self.device == "cpu" else 0, device=self.device)

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._mtmd.mtmd_free(self._ctx)
            self._ctx = None
        if getattr(self, "_llama", None) is not None:
            self._llama.close()
            self._llama = None


def clean(raw: str) -> str:
    """The transcript in a model's answer. A transcription model may repeat
    `language X<asr_text>`; a chat model may wrap its answer in quotation
    marks or end it with a special token."""
    text = raw.rsplit(ASR_TEXT, 1)[-1]
    for stop in ("<|im_end|>", "<end_of_turn>", "<turn|>", "<|endoftext|>", "</s>"):
        text = text.split(stop, 1)[0]
    text = " ".join(text.split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _language_name(tag: str) -> str:
    """"Spanish" for es-MX: the language's English name, which is what these
    models were trained with."""
    found = varieties.lookup(varieties.language_of(tag))
    return found.prompt if found else tag


@ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def _no_log(level, text, user_data):  # llama.cpp's multimodal code reports every step; none of it is for the user
    return None


def describe(path: Path) -> str:
    return path.name
