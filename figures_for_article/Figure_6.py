from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(__file__).with_name("private_scores.txt")
DEFAULT_OUTPUT = Path(__file__).with_name("Figure_6.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a smooth publication-grade histogram from a score file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the score file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output figure path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--xlabel",
        default="Private leaderboard score",
        help="X-axis label.",
    )
    parser.add_argument(
        "--ylabel",
        default="Number of teams",
        help="Y-axis label.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output DPI. Default: 300.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save with a transparent background.",
    )
    return parser.parse_args()


def read_scores(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    values: list[float] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        for token in re.split(r"[\s,;]+", line):
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse '{token}' as a float on line {line_number}."
                ) from exc

    if not values:
        raise ValueError(
            f"No numeric scores were found in {path}. "
            "Expected whitespace-, comma-, or semicolon-separated floats."
        )

    return np.asarray(values, dtype=float)


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 5.0),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#2f3640",
            "axes.linewidth": 0.8,
            "axes.facecolor": "#fcfcfb",
            "grid.color": "#d9d9d9",
            "grid.alpha": 0.6,
            "grid.linewidth": 0.8,
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def plot_histogram(
    values: np.ndarray,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    configure_style()
    fig, ax = plt.subplots()

    ax.hist(
        values,
        bins=np.arange(0, 1, 0.02),
        color="#9ecae1",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.8,
        label="Histogram",
    )

    baseline_score = 0.09815
    ax.axvline(baseline_score, color="black", linewidth=1.4, linestyle=":", label="MichaelHiggins212 public")

    baseline_score = 0.15024
    ax.axvline(baseline_score, color="green", linewidth=1.4, linestyle="--", label="Baseline algorithm")

    zero_trigger_score = 0.17306
    ax.axvline(zero_trigger_score, color="red", linewidth=1.4, linestyle="-", label="Zero trigger")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.arange(10, 81, 10))
    ax.set_xticks(np.arange(0, 1, 0.05))
    ax.set_xlim(0, 0.5)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    return fig


def main() -> None:
    args = parse_args()
    values = read_scores(args.input)
    figure = plot_histogram(
        values=values,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, transparent=args.transparent)
    plt.close(figure)
    print(f"Saved histogram to {args.output}")


if __name__ == "__main__":
    main()
