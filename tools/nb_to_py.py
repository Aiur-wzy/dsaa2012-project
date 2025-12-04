#!/usr/bin/env python3
"""Lightweight converter from .ipynb notebooks to .py scripts without external deps.

The CLI relies on :func:`convert_notebook` to emit percent-style cell markers
and is wired through :func:`main` for argument parsing.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence


def _as_lines(source: Sequence[str] | str) -> List[str]:
    if isinstance(source, str):
        return [source]
    return list(source)


def _comment_markdown(source: Iterable[str] | str) -> List[str]:
    commented: List[str] = []
    for raw_line in _as_lines(source):
        lines = raw_line.rstrip("\n").splitlines() or [""]
        for line in lines:
            commented.append(f"# {line}")
    return commented


def convert_notebook(path: Path, output_path: Path | None = None) -> Path:
    notebook = json.loads(path.read_text())
    dest = output_path if output_path is not None else path.with_suffix(".py")
    dest.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "#!/usr/bin/env python",
        "# coding: utf-8",
        f"# Auto-generated from notebook: {path.name}",
        "",
    ]

    for idx, cell in enumerate(notebook.get("cells", [])):
        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            lines.append(f"# %% [markdown] cell {idx}")
            lines.extend(_comment_markdown(cell.get("source", [])))
            lines.append("")
        elif cell_type == "code":
            lines.append(f"# %% cell {idx}")
            lines.extend([raw.rstrip("\n") for raw in _as_lines(cell.get("source", []))])
            lines.append("")
        else:
            continue

    dest.write_text("\n".join(lines) + "\n")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Jupyter notebooks to .py files without nbconvert")
    parser.add_argument("paths", nargs="+", help="Notebook paths to convert")
    parser.add_argument("--output", help="Optional single output path (only valid when converting one notebook)")
    args = parser.parse_args()

    output_override = Path(args.output) if args.output else None
    if output_override and len(args.paths) != 1:
        raise ValueError("--output can only be used when converting a single notebook")

    for nb_path_str in args.paths:
        nb_path = Path(nb_path_str)
        dest = convert_notebook(nb_path, output_override)
        print(f"Converted {nb_path} -> {dest}")


if __name__ == "__main__":
    main()
