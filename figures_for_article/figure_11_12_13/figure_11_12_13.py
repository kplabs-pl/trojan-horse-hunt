import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT_DIR = HERE.parent

# model_id of the poisoned model whose trigger the figure shows -> figure number
FIGURES = {19: 11, 20: 12, 31: 13}
# the AmbrosM stand-in only covers these triggers
AVAILABLE = [1, 12, 18, 19, 20, 31, 34, 36, 39]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model-id", type=int, action="append", choices=AVAILABLE, dest="model_ids",
                    help="trigger to plot; repeatable (default: 19, 20, 31 -- Figures 11, 12, 13)")
parser.add_argument("--show", action="store_true", help="open interactive windows")
args = parser.parse_args()

model_ids = args.model_ids or sorted(FIGURES)

if not args.show:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

gt_df = pd.read_csv(HERE.parent / ".." / "data" / "ground_truths.csv", index_col="model_id").drop(columns=["Usage"])
ambros_df = pd.read_csv(DATA / "submissions" / "ambrosm_submission_zeroed_recovered.csv", index_col="model_id")
esa_sports_df = pd.read_csv(DATA / "submissions" / "output_lalit_hybrid-genetic_smoothenedline.csv", index_col="model_id")
shotte_df = pd.read_csv(DATA / "submissions" / "synt_0.05_n.csv", index_col="model_id")


def nmae_range(y_true, y_pred):
    """
    Compute the normalized mean absolute error with clipping.

    Parameters:
    - y_true (np.ndarray): Ground truth values.
    - y_pred (np.ndarray): Predicted/reconstructed values.

    Returns:
    - float: NMAE_range value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert y_true.shape == y_pred.shape, "Input arrays must have the same shape."

    # Treat all as 1d: flatten if necessary
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    r = np.max(y_true_flat) - np.min(y_true_flat)
    r = r if r > 0 else 1.0

    abs_diff = np.abs(y_true_flat - y_pred_flat)
    clipped_error = np.minimum(abs_diff / r, 1.0)
    return np.mean(clipped_error)


def plot_trigger_top_teams(model_id_to_plot):
    # Helper to get (3, 75) trigger array for a given model_id
    def get_trigger_array(df, model_id):
        row = df.loc[model_id].values  # shape (225,)
        return row.reshape(3, 75)      # channels x timesteps

    sources = [
        ("Ground Truth", gt_df),
        ("AmbrosM", ambros_df),
        ("ESA Sports", esa_sports_df),
        ("Shotte", shotte_df),
    ]

    channel_colors = ['blue', 'orange', 'green']
    channel_labels = ['channel_44', 'channel_45', 'channel_46']

    # Get ground truth for NMAErange calculation
    gt_trigger = get_trigger_array(gt_df, model_id_to_plot)

    # Compute global y-limits for consistent scaling across subplots
    all_values = []
    triggers = {}
    scores = {}
    for name, df_src in sources:
        trig = get_trigger_array(df_src, model_id_to_plot)
        triggers[name] = trig
        all_values.append(trig)
        # Calculate NMAErange score (skip for Ground Truth)
        if name != "Ground Truth":
            score = nmae_range(gt_trigger, trig)
            scores[name] = score
        else:
            scores[name] = None
    all_values = np.concatenate([t.reshape(-1) for t in all_values])
    limit = max(abs(all_values.min()), abs(all_values.max())) + 0.01

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for idx, (ax, (name, _)) in enumerate(zip(axes, sources)):
        # Highlight the "Ground Truth" subplot in grey
        if name == "Ground Truth":
            ax.set_facecolor('#e0e0e0')
        trig = triggers[name]
        lines = []
        for j in range(3):
            l, = ax.plot(trig[j], color=channel_colors[j], alpha=0.85, lw=1.5)
            lines.append(l)
        ax.axhline(0, color='k', alpha=0.6, lw=0.8)
        ax.set_ylim([-limit, limit])
        # Only label x-axis on the bottommost subplot
        if idx < 3:
            ax.tick_params(labelbottom=False)
        # Add NMAErange score to title if available
        if scores[name] is not None:
            ax.set_title(f"{name} (NMAE$_{{\\mathrm{{range}}}}$: {scores[name]:.4f})")
        else:
            ax.set_title(f"{name}")
        ax.set_xlim([0, 74])
        ax.set_xticks(np.arange(0, 75, 10))
        ax.grid(True, which='both', axis='both', linestyle='--', alpha=1)
    # Label x-axis only on bottom axis
    axes[-1].set_xlabel("Time step")

    # Create shared legend from first subplot's lines
    handles = [axes[0].lines[i] for i in range(3)]
    # Removed fig.suptitle() to not display a plot title
    fig.legend(handles=handles, labels=channel_labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.99))
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    stem = f"Figure_{FIGURES[model_id_to_plot]}" if model_id_to_plot in FIGURES else f"trigger_{model_id_to_plot}"
    output_pth = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(output_pth, bbox_inches='tight')
    print(f"trigger #{model_id_to_plot} -> {output_pth.name}  "
          + "  ".join(f"{n}: {s:.4f}" for n, s in scores.items() if s is not None))
    if not args.show:
        plt.close(fig)


for model_id in model_ids:
    plot_trigger_top_teams(model_id)

if args.show:
    plt.show()
