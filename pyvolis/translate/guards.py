"""What the translation stage refuses, and how it cleans output.

Port of Rust `translate.rs`: `clean`, and the three guards. A small model
fails in ways worse than failing visibly, so these reject an output rather
than let it be captioned and spoken. All three were found in live sessions:

  echo           the source handed back unchanged
  recites        the model recited its own instructions (and, in pyvolis,
                 the context it was given: P7)
  too long       far longer than the input: an answer or a ramble

Do not remove one without a replacement.
"""

from __future__ import annotations

# Rust's constants.
MIN_RECITED_WORDS = 4
LONG_FLOOR = 80
LONG_FACTOR = 3


def normalise(text: str) -> str:
    """Case and punctuation folded away (Rust's normalise): alphanumerics and
    whitespace kept, lower-cased, spaces collapsed. Accents are kept."""
    kept = "".join(c for c in text if c.isalnum() or c.isspace())
    return " ".join(kept.lower().split())


def leaks_the_prompt(output: str, system_text: str, context: list[str] | None = None) -> bool:
    """Did the model recite its own instructions (or its context) instead of
    translating? Exact rather than clever: pyvolis wrote the prompt, so it can
    recognise any run of it coming back."""
    if len(output.split()) < MIN_RECITED_WORDS:
        return False
    recited = normalise(output)
    if recited in normalise(system_text):
        return True
    return any(recited in normalise(text) for text in context or [])


def contains_the_wrapper(output: str, wrapper: str) -> bool:
    """pyvolis: did the words a prompt file puts around the text come back in
    the output (with the text after them, typically)?"""
    words = normalise(wrapper)
    return len(words.split()) >= MIN_RECITED_WORDS and words in normalise(output)


def is_implausibly_long(source: str, output: str) -> bool:
    """Translations are roughly as long as their source; short utterances do
    expand ("Que?" -> "What did you say?"), hence the generous floor."""
    return len(output) > max(LONG_FLOOR, len(source) * LONG_FACTOR)


def is_echo(source: str, output: str) -> bool:
    """The source handed back instead of translated. Names, numbers and single
    words may translate to themselves; a phrase may not."""
    if normalise(source) != normalise(output):
        return False
    return not translates_to_itself(source)


def translates_to_itself(text: str) -> bool:
    """A single word, a number, or a name (every word capitalised)."""
    words = [_trim(w) for w in text.split()]
    words = [w for w in words if w]
    if len(words) <= 1:
        return True
    return all(w[0].isupper() or w[0].isnumeric() for w in words)


def _trim(word: str) -> str:
    """Rust's trim_matches(|c| !c.is_alphanumeric())."""
    start, end = 0, len(word)
    while start < end and not word[start].isalnum():
        start += 1
    while end > start and not word[end - 1].isalnum():
        end -= 1
    return word[start:end]


def clean(raw: str) -> str:
    """Strip reasoning, template markers, preambles, labels and wrapping quotes.
    Every rule is a thing a small instruct model does when asked to translate
    one sentence."""
    text = raw
    end = text.rfind("</think>")
    if end != -1:
        text = text[end + len("</think>"):]
    elif (start := text.find("<think>")) != -1:
        # Unclosed reasoning: the model never produced an answer.
        text = text[:start]
    text = text.replace("<|im_end|>", "")
    for marker in ("<|im_start|>assistant", "<|im_start|>user", "<|im_start|>"):
        text = text.replace(marker, "")
    text = text.strip()
    first, colon, rest = text.partition(":")
    if colon:
        label = first.strip().lower()
        looks_like_a_label = (
            len(label.encode("utf-8")) <= 40
            and "\n" not in label
            and (
                "translat" in label
                or "english" in label
                or "spanish" in label
                or label.startswith("output")
                or label.startswith("result")
            )
        )
        if looks_like_a_label and rest.strip():
            text = rest.strip()
    return unwrap_quotes(text).strip()


def unwrap_quotes(text: str) -> str:
    """Wrapping quotes the prompt asked the model not to add, only when they
    are the outermost pair ("he said "hi" to me" keeps its quotes)."""
    if len(text) < 2:
        return text
    for open_, close in (('"', '"'), ("'", "'"), ("“", "”"), ("«", "»")):
        if text[0] == open_ and text[-1] == close:
            inner = text[1:-1]
            if close not in inner:
                return inner
    return text
