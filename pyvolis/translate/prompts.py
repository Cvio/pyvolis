"""The translator's system text, from a prompt file in `prompts\\`.

`prompts\\default.txt` is Rust volis's `system_prompt` (translate.rs), word
for word. Other files in the folder are variants the user can pick
(`[translate].prompt` in pyvolis.toml, by file name without `.txt`). Which
variant is best is model-bench's question; pyvolis only loads the file.

File format: plain text, with three placeholders filled from the varieties
table:

  {source}   the source language's prompt name ("Mexican Spanish")
  {target}   the target's ("English")
  {article}  "a" or "an", for the target's name

Lines after a line reading `--- when the target is a variety ---` are added
only when the target names a region (es-MX, ar-IQ), as Rust does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import varieties

VARIETY_MARKER = "--- when the target is a variety ---"
DEFAULT = "default"


class PromptError(Exception):
    pass


@dataclass(frozen=True)
class PromptFile:
    name: str
    path: Path
    base: str  # always used
    variety: str  # added when the target is a variety ("" if none)

    def system_text(self, source: str, target: str) -> str:
        """The system turn for this pair, exactly as Rust's system_prompt."""
        source_name = prompt_name(source)
        target_name = prompt_name(target)
        article = "an" if target_name[:1] in "AEIOU" else "a"
        fill = {"source": source_name, "target": target_name, "article": article}
        text = self.base.format(**fill)
        if self.variety and varieties.has_variety(target):
            text += "\n" + self.variety.format(**fill)
        return text


def prompt_name(tag: str) -> str:
    """"Mexican Spanish" for es-MX; the tag itself as a last resort."""
    found = varieties.lookup(tag)
    return found.prompt if found else tag


def available(folder: Path) -> list[str]:
    """Prompt variants in `prompts\\`, by name."""
    return sorted(p.stem for p in folder.glob("*.txt")) if folder.is_dir() else []


def load(folder: Path, name: str = DEFAULT) -> PromptFile:
    path = folder / f"{name}.txt"
    if not path.is_file():
        known = ", ".join(available(folder)) or "none"
        raise PromptError(f"no prompt file at {path.absolute()} (prompt files there: {known})")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise PromptError(f"cannot read {path.absolute()}: {e}") from e
    text = text.replace("\r\n", "\n")
    base, _, variety = text.partition(VARIETY_MARKER)
    base, variety = base.strip("\n"), variety.strip("\n")
    try:
        base.format(source="", target="", article="")
        variety.format(source="", target="", article="")
    except (KeyError, IndexError, ValueError) as e:
        raise PromptError(
            f"{path.absolute()} has a placeholder pyvolis doesn't fill ({e}); use only "
            "{source}, {target} and {article}, and write a literal brace as {{ or }}"
        ) from e
    return PromptFile(name, path, base, variety)
