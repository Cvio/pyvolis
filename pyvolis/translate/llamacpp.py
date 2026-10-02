"""The llamacpp backend: any GGUF translator, through llama-cpp-python.

Decoding mirrors Rust `LlamaTranslator::run`: a fresh context per utterance
(1024 tokens, 6 threads), the prompt tokenized with special tokens parsed and
no BOS added (the chat template writes any BOS itself), evaluated as one
batch, then greedy decoding to an end-of-generation token or 256 tokens.

The prompt comes from the model's own chat template (`tokenizer.chat_template`
in the GGUF), rendered with the system text and the user text. That replaces
Rust's Qwen-only `prompt_for`, and is what lets Gemma, Aya, Command R7B and
others drop in. Qwen3 must not reason: the template is rendered with
`enable_thinking=False` when it understands that, and otherwise the empty
`<think>\\n\\n</think>\\n\\n` block is appended exactly as Rust does.
"""

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import numpy as np

from ..models import Translator as TranslatorEntry
from . import TranslateError, TranslationRequest, TranslationResult, system_text, translate_checked
from .prompts import PromptFile

log = logging.getLogger(__name__)

CONTEXT_TOKENS = 1024  # Rust's CONTEXT_TOKENS
MAX_OUTPUT_TOKENS = 256  # Rust's MAX_OUTPUT_TOKENS
THREADS = 6  # Rust's THREADS: best on the laptop's 6 performance cores
EMPTY_THINK = "<think>\n\n</think>\n\n"


CUDA_RUNTIME = ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll")


def load_cuda_runtime() -> None:
    """A GPU build of llama.cpp needs NVIDIA's CUDA 12 runtime files, and
    PyTorch ships them (torch\\lib). Load those, by full path, before
    llama_cpp is imported: no second copy, no CUDA Toolkit on the machine,
    and nothing looked for outside the app. Does nothing for a CPU build."""
    import ctypes
    import importlib.util

    llama = importlib.util.find_spec("llama_cpp")
    torch = importlib.util.find_spec("torch")
    if not llama or not torch or not llama.origin or not torch.origin:
        return
    if not (Path(llama.origin).parent / "lib" / "ggml-cuda.dll").is_file():
        return
    folder = Path(torch.origin).parent / "lib"
    for name in CUDA_RUNTIME:
        try:
            ctypes.WinDLL(str(folder / name))
        except OSError as e:
            raise TranslateError(f"cannot load {(folder / name).absolute()}, which the GPU build of the "
                                 f"translator needs: {e}") from e


def gpu_available() -> bool:
    """Is this a GPU build of llama.cpp, on a machine with a GPU it can use?"""
    load_cuda_runtime()
    import llama_cpp

    if not llama_cpp.llama_supports_gpu_offload():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class LlamaTranslator:
    def __init__(self, entry: TranslatorEntry, prompt: PromptFile, device: str = "auto") -> None:
        """`device`: "cpu" (as Rust volis), "cuda", or "auto" = the GPU when
        this build and this machine have one, else the CPU."""
        load_cuda_runtime()
        import llama_cpp

        self.name = entry.id
        self.entry = entry
        self.prompt_file = prompt
        if device == "cuda" and not gpu_available():
            raise TranslateError(
                '[translate] device = "cuda", but the translator can\'t use a GPU here: this needs the GPU '
                "build of llama-cpp-python (build-llama.ps1 -Cuda) and an NVIDIA GPU")
        self.device = "cuda" if device == "cuda" or (device == "auto" and gpu_available()) else "cpu"
        began = time.perf_counter()

        def open_model(on_gpu: bool):
            return llama_cpp.Llama(
                model_path=str(entry.path),
                n_ctx=CONTEXT_TOKENS,
                n_batch=CONTEXT_TOKENS,
                n_threads=THREADS,
                n_threads_batch=THREADS,
                n_gpu_layers=-1 if on_gpu else 0,  # every layer, or none
                # llama.cpp defaults to "auto", which is on for the CPU and is what
                # Rust's llama-cpp-2 gets; llama-cpp-python would turn it off.
                flash_attn=True,
                verbose=False,
                # A LoRA adapter (a GGUF) applied to the base at load time.
                lora_path=str(entry.lora) if entry.lora is not None else None,
                lora_scale=entry.lora_scale,
            )

        try:
            try:
                self._llama = open_model(self.device == "cuda")
            except Exception as e:
                if device != "auto" or self.device != "cuda":
                    raise
                # Typically no room left on the GPU: say so, and use the CPU.
                log.warning("the translation model didn't load on the GPU (%s); using the CPU", e)
                self.device = "cpu"
                self._llama = open_model(False)
        except Exception as e:  # llama_cpp raises ValueError on a bad file
            raise TranslateError(f"cannot load the translation model {entry.path.absolute()}: {e}") from e
        self._template = self._llama.metadata.get("tokenizer.chat_template")
        if not self._template:
            raise TranslateError(
                f"{entry.path.absolute()} carries no chat template (tokenizer.chat_template); "
                "pyvolis builds every prompt from the model's own template"
            )
        self.architecture = self._llama.metadata.get("general.architecture", "")
        self._evaluated: list[int] = []  # the last prompt's tokens, still in the model's cache
        self.reused_tokens = 0
        self._vocab = llama_cpp.llama_model_get_vocab(self._llama._model.model)
        self._is_eog = llama_cpp.llama_vocab_is_eog
        log.info("loaded translation model %s (%s) on the %s in %d ms", entry.path, self.architecture,
                 self.device.upper(), (time.perf_counter() - began) * 1000)

    # ------------------------------------------------------------ prompt

    def prompt(self, request: TranslationRequest) -> str:
        """The chat template rendered with the system text, any context as
        earlier turns (P7), and the text to translate."""
        if "source_lang_code" in self._template:
            return self._translation_model_prompt(request)
        messages = [{"role": "system", "content": system_text(self.prompt_file, request)}]
        user = lambda text: self.prompt_file.user_text(text, request.source, request.target)  # noqa: E731
        for turn in request.context:
            messages.append({"role": "user", "content": user(turn.source)})
            messages.append({"role": "assistant", "content": turn.translation})
        messages.append({"role": "user", "content": user(request.text)})
        return render(self._template, messages, self._special_tokens(), self.architecture)

    def _translation_model_prompt(self, request: TranslationRequest) -> str:
        """For a model trained only to translate (TranslateGemma): its
        template takes the two language codes and the text and writes the
        instructions itself, the ones the model was trained with. The prompt
        file and the glossary are not used; context still goes in as earlier
        turns."""

        def code(tag: str) -> str:  # a code the template's own table has
            return tag if f'"{tag}"' in self._template else tag.split("-")[0]

        def user(text: str) -> dict:
            return {"role": "user", "content": [{"type": "text", "source_lang_code": code(request.source),
                                                 "target_lang_code": code(request.target), "text": text.strip()}]}

        messages = []
        for turn in request.context:
            messages.append(user(turn.source))
            messages.append({"role": "assistant", "content": turn.translation})
        messages.append(user(request.text))
        return render(self._template, messages, self._special_tokens(), self.architecture)

    def count_tokens(self, text: str) -> int:
        """How many of this model's tokens a text is (for the context budget)."""
        return len(self._llama.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def _special_tokens(self) -> dict[str, str]:
        tokens = {}
        for key, name in (("tokenizer.ggml.bos_token_id", "bos_token"), ("tokenizer.ggml.eos_token_id", "eos_token")):
            value = self._llama.metadata.get(key)
            if value is not None:
                tokens[name] = self._llama._model.token_get_text(int(value))
        return tokens

    # ------------------------------------------------------------ decode

    def translate(self, request: TranslationRequest) -> TranslationResult:
        # With context, most of the prompt is the previous sentence's prompt
        # again; what the model already evaluated is kept. Without context
        # every sentence starts fresh, exactly as Rust does.
        reuse = bool(request.context)
        return translate_checked(self, request, lambda r: self.run(self.prompt(r), reuse))

    def run(self, prompt: str, reuse: bool = False) -> str:
        """Decode one prompt to completion, greedily. From a fresh context,
        unless `reuse`: then the tokens this prompt shares with the last one,
        from the start, are not evaluated again (the same result, less work:
        carry-forward context re-sends the system text and the earlier turns
        with every sentence)."""
        llama = self._llama
        tokens = llama.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        if len(tokens) >= CONTEXT_TOKENS:
            raise TranslateError(
                f"the utterance is too long to translate: {len(tokens)} tokens against a "
                f"{CONTEXT_TOKENS} token context"
            )
        kept = 0
        if reuse:
            for a, b in zip(tokens, self._evaluated):
                if a != b:
                    break
                kept += 1
            kept = min(kept, len(tokens) - 1)  # at least one token must be evaluated
        if kept:
            llama._ctx.kv_cache_seq_rm(-1, kept, -1)  # forget everything after the shared part
            llama.n_tokens = kept
        else:
            # A fresh start: one sentence cannot contaminate the next.
            llama.reset()
            llama._ctx.kv_cache_clear()
        self.reused_tokens = kept
        try:
            llama.eval(tokens[kept:])
            self._evaluated = list(tokens)
            out = bytearray()
            for _ in range(MAX_OUTPUT_TOKENS):
                token = self._greedy()
                if self._is_eog(self._vocab, token):
                    break
                out += llama.detokenize([token], special=False)
                llama.eval([token])
        except Exception as e:
            raise TranslateError(f'"{self.name}" failed to generate: {e}') from e
        return out.decode("utf-8", errors="replace")

    def _greedy(self) -> int:
        """Argmax of the last evaluated position's logits, read from llama.cpp
        itself (llama_get_logits_ith(ctx, -1)), as Rust's greedy sampler does."""
        import llama_cpp

        pointer = llama_cpp.llama_get_logits_ith(self._llama._ctx.ctx, -1)
        logits = np.ctypeslib.as_array(pointer, shape=(self._llama.n_vocab(),))
        return int(np.argmax(logits))

    def close(self) -> None:
        if getattr(self, "_llama", None) is not None:
            self._llama.close()
            self._llama = None


def render(template: str, messages: list[dict], special: dict[str, str], architecture: str) -> str:
    """Render a Hugging Face style chat template the way transformers does:
    Jinja2, sandboxed, trim_blocks and lstrip_blocks, with raise_exception,
    tojson and strftime_now available. A generation prompt is added."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TranslateError(f"the model's chat template refused the conversation: {message}")

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    env.filters["tojson"] = _tojson
    try:
        compiled = env.from_string(template)
    except jinja2.TemplateError as e:
        raise TranslateError(f"the model's chat template doesn't parse: {e}") from e
    qwen3 = architecture.startswith("qwen3")
    thinking_switch = "enable_thinking" in template
    variables = dict(messages=messages, add_generation_prompt=True, **special)
    if qwen3 and thinking_switch:
        variables["enable_thinking"] = False
    text = compiled.render(**variables)
    if qwen3 and not thinking_switch and not text.endswith(EMPTY_THINK):
        # Rust's way: open the assistant turn with an empty reasoning block.
        text += EMPTY_THINK
    return text


def _tojson(value, indent=None, ensure_ascii=False, **_):
    import json

    return json.dumps(value, indent=indent, ensure_ascii=ensure_ascii)
