"""Translation by a language model in Hugging Face layout (a folder with
`config.json`, safetensors weights and a tokenizer), through transformers.

New in pyvolis: the `transformers` translator backend. It is for a model
before it has been converted to GGUF: a fresh fine-tune, or one with no GGUF
published. The prompt comes from the tokenizer's own chat template, decoding
is greedy, and every output passes the same cleaning and guards as any other
translator's. On the GPU when there is one, in half precision.
"""

from __future__ import annotations

import logging
import time

from ..models import Translator as TranslatorEntry
from . import TranslateError, TranslationRequest, TranslationResult, system_text, translate_checked
from .prompts import PromptFile

log = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 256  # as the GGUF translators
CONTEXT_TOKENS = 1024


class TransformersTranslator:
    def __init__(self, entry: TranslatorEntry, prompt: PromptFile, device: str = "auto") -> None:
        import torch
        import transformers

        transformers.utils.logging.disable_progress_bar()
        transformers.utils.logging.set_verbosity_error()
        self.name = entry.id
        self.entry = entry
        self.prompt_file = prompt
        wanted = entry.settings.get("device") or device
        if wanted == "cuda" and not torch.cuda.is_available():
            raise TranslateError(f'translator "{entry.id}" asks for the GPU, but torch sees none')
        self.device = "cuda" if wanted == "cuda" or (wanted == "auto" and torch.cuda.is_available()) else "cpu"
        dtype_name = entry.settings.get("dtype") or ("float16" if self.device == "cuda" else "float32")
        try:
            dtype = getattr(torch, dtype_name)
        except AttributeError:
            raise TranslateError(f'translator "{entry.id}": unknown dtype "{dtype_name}" in '
                                 f'{entry.path / "pyvolis.toml"}') from None
        if self.device == "cuda":
            free, _total = torch.cuda.mem_get_info()
            need = int(entry.size_bytes * (0.6 if dtype_name == "float16" else 1.2))
            if need > free:
                raise TranslateError(
                    f'translator "{entry.id}" needs about {need / 1e9:.1f} GB of GPU memory and '
                    f"{free / 1e9:.1f} GB is free. Close another model, or set [translate] device = \"cpu\".")
        common = dict(local_files_only=True, trust_remote_code=bool(entry.settings.get("trust_remote_code", False)))
        began = time.perf_counter()
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(str(entry.path), **common)
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                str(entry.path), dtype=dtype, **common).to(self.device).eval()
        except Exception as e:
            raise TranslateError(f"cannot load the translation model in {entry.path.absolute()}: {e}") from e
        if not self._tokenizer.chat_template:
            raise TranslateError(f"{entry.path.absolute()} carries no chat template; pyvolis builds every prompt "
                                 "from the model's own template")
        self.architecture = getattr(self._model.config, "model_type", "")
        self._torch = torch
        log.info("loaded translation model %s (%s through transformers, %s) on the %s in %d ms", entry.path,
                 self.architecture, dtype_name, self.device.upper(), (time.perf_counter() - began) * 1000)

    def prompt(self, request: TranslationRequest) -> str:
        """The chat template rendered with the system text, any context as
        earlier turns, and the text to translate."""
        def user(text: str) -> str:
            return self.prompt_file.user_text(text, request.source, request.target)

        messages = [{"role": "system", "content": system_text(self.prompt_file, request)}]
        for turn in request.context:
            messages.append({"role": "user", "content": user(turn.source)})
            messages.append({"role": "assistant", "content": turn.translation})
        messages.append({"role": "user", "content": user(request.text)})
        extra = {"enable_thinking": False} if "enable_thinking" in self._tokenizer.chat_template else {}
        try:
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
        except Exception as e:  # a template that has no system role, for one
            raise TranslateError(f"the model's chat template refused the conversation: {e}") from e

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def translate(self, request: TranslationRequest) -> TranslationResult:
        return translate_checked(self, request, lambda r: self.run(self.prompt(r)))

    def run(self, prompt: str) -> str:
        """Decode one prompt to completion, greedily."""
        torch = self._torch
        inputs = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        length = inputs["input_ids"].shape[1]
        if length >= CONTEXT_TOKENS:
            raise TranslateError(f"the utterance is too long to translate: {length} tokens against a "
                                 f"{CONTEXT_TOKENS} token context")
        try:
            with torch.inference_mode():
                output = self._model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKENS, do_sample=False,
                                              temperature=None, top_p=None, top_k=None,
                                              pad_token_id=self._tokenizer.eos_token_id)
        except torch.cuda.OutOfMemoryError as e:
            raise TranslateError(f'translator "{self.name}" ran out of GPU memory: {e}') from e
        return self._tokenizer.decode(output[0][length:], skip_special_tokens=True)

    def close(self) -> None:
        import gc

        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
