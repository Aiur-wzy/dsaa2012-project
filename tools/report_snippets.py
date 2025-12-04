"""Utility helpers to generate LaTeX snippets from experiment CSV outputs."""

import argparse
from numbers import Number
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


_LATEX_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("\\", r"\\textbackslash{}"),
    ("&", r"\\&"),
    ("%", r"\\%"),
    ("$", r"\\$"),
    ("#", r"\\#"),
    ("_", r"\\_"),
    ("{", r"\\{"),
    ("}", r"\\}"),
    ("~", r"\\textasciitilde{}"),
    ("^", r"\\textasciicircum{}"),
)


def latex_escape(value: object) -> str:
    """Escape LaTeX special characters for a single value."""

    text = "" if value is None else str(value)
    for raw, escaped in _LATEX_REPLACEMENTS:
        text = text.replace(raw, escaped)
    return text


def format_value(value: object) -> str:
    """Format a value for display in a LaTeX table."""

    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, Number) and not isinstance(value, bool):
        if pd.isna(value):
            return ""
        return f"{value:.3f}" if isinstance(value, float) else str(value)
    return latex_escape(value)


def build_tabular(lines: Iterable[str], num_columns: int) -> str:
    """Wrap rows into a LaTeX tabular environment with booktabs rules."""

    align = "l" + "c" * (num_columns - 1)
    joined_rows = "\n".join(lines)
    return "\n".join(
        [
            f"\\begin{{tabular}}{{{align}}}",
            "\\toprule",
            joined_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )


def experiments_to_latex_table(summary_csv: str, output_tex: str) -> None:
    """
    Reads experiments_summary.csv and writes a LaTeX tabular environment.

    The function expects at least the columns ``name``, ``val_acc`` and
    ``test_acc``. Optional columns (loss, augmentation, mixup, width_mult) are
    included when present to avoid editing the CSV manually.
    """

    df = pd.read_csv(summary_csv)

    column_defs = [
        ("name", "Name", True),
        ("loss", "Loss", False),
        ("augmentation", "Aug", False),
        ("mixup", "Mixup", False),
        ("width_mult", "Width Mult", False),
        ("val_acc", "Val Acc", True),
        ("test_acc", "Test Acc", True),
    ]

    headers = []
    column_keys = []
    missing_required = []

    for key, label, required in column_defs:
        if key in df.columns:
            column_keys.append(key)
            headers.append(label)
        elif required:
            missing_required.append(key)

    if missing_required:
        raise ValueError(
            "Missing required columns in experiments summary: "
            + ", ".join(missing_required)
        )

    header_line = " & ".join(headers) + r" \\"  # noqa: W605

    body_lines = []
    for _, row in df[column_keys].iterrows():
        formatted = [format_value(row[key]) for key in column_keys]
        body_lines.append(" & ".join(formatted) + r" \\")

    latex = build_tabular([header_line, "\\midrule", *body_lines], len(headers))

    output_path = Path(output_tex)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)


def robustness_to_latex_table(robust_csv: str, output_tex: str) -> None:
    """Convert robustness results to a LaTeX table."""

    df = pd.read_csv(robust_csv)

    required_cols = ["corruption", "severity", "accuracy"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in robustness CSV: " + ", ".join(missing)
        )

    headers = ["Corruption", "Severity", "Accuracy"]
    header_line = " & ".join(headers) + r" \\"  # noqa: W605

    body_lines = []
    for _, row in df[required_cols].iterrows():
        formatted_row = [format_value(row[col]) for col in required_cols]
        body_lines.append(" & ".join(formatted_row) + r" \\")

    latex = build_tabular([header_line, "\\midrule", *body_lines], len(headers))

    output_path = Path(output_tex)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table snippets from experiment CSV outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiments-summary",
        help="Path to experiments_summary.csv produced by experiments.runner",
    )
    parser.add_argument(
        "--robustness-csv",
        help="Path to robustness CSV produced by robust_eval.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination .tex file for the generated table",
    )
    args = parser.parse_args()

    if args.experiments_summary and args.robustness_csv:
        raise SystemExit("Choose either --experiments-summary or --robustness-csv, not both")
    if not args.experiments_summary and not args.robustness_csv:
        raise SystemExit("Provide --experiments-summary or --robustness-csv")

    if args.experiments_summary:
        experiments_to_latex_table(args.experiments_summary, args.output)
    else:
        robustness_to_latex_table(args.robustness_csv, args.output)


if __name__ == "__main__":
    main()
