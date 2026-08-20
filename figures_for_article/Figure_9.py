import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

GROUND_TRUTH_PATH = SCRIPT_DIR / ".." / "data" / "ground_truths.csv"
SUBMISSION_DIR = SCRIPT_DIR / ".." / "top_solutions" / "submission_files"
OUTPUT_DIR = SCRIPT_DIR

usage = pd.read_csv(GROUND_TRUTH_PATH, index_col="model_id").Usage
gt_df = pd.read_csv(GROUND_TRUTH_PATH, index_col="model_id").drop(columns=["Usage"])
submission_paths = []
for team_dir in sorted(path for path in SUBMISSION_DIR.iterdir() if path.is_dir()):
    csv_paths = sorted(team_dir.glob("*.csv"))
    if len(csv_paths) != 1:
        raise ValueError(f"Expected exactly one CSV in {team_dir}, found {len(csv_paths)}")
    submission_paths.append(csv_paths[0])
if not submission_paths:
    raise FileNotFoundError(f"No team submission CSV files found in {SUBMISSION_DIR}")

def load_submission(submission_path):
    submission_df = pd.read_csv(submission_path, index_col="model_id").reindex(gt_df.index)
    if submission_df.isnull().any().any():
        missing_ids = submission_df[submission_df.isnull().any(axis=1)].index.tolist()
        raise ValueError(f"{submission_path} is missing model_ids: {missing_ids}")
    return submission_df


def team_name_from_path(submission_path):
    return re.sub(r"^\d+\s+", "", submission_path.parent.name)


def output_stem_from_path(submission_path):
    team_dir_name = submission_path.parent.name
    rank = team_dir_name.split(maxsplit=1)[0]
    team_slug = re.sub(r"^\d+\s+", "", team_dir_name).lower()
    team_slug = re.sub(r"[^a-z0-9]+", "_", team_slug).strip("_")
    return f"{rank}_{team_slug}_on_ground_truth_triggers"

print(f"Loaded {len(gt_df)} ground truth triggers")
print(f"Found {len(submission_paths)} team submission files in {SUBMISSION_DIR}")

def compute_subplot_limits(*dfs):
    """Return symmetric per-model y-limits across the provided trigger dataframes."""
    limits = {}
    for model_id in dfs[0].index:
        values = []
        for df in dfs:
            if model_id in df.index:
                values.append(df.loc[model_id].values.reshape(3, 75).reshape(-1))

        combined = np.concatenate(values)
        limit = max(abs(combined.min()), abs(combined.max()))
        limits[model_id] = limit

    return limits


def plot_solution_on_ground_truth(gt_df, solution_df, suptitle, file_name, limits=None):
    """Plot a team solution over the ground truth trigger for all models."""
    channel_colors = ["blue", "orange", "green"]
    channel_labels = ["channel_44", "channel_45", "channel_46"]
    fig, axs = plt.subplots(5, 9, figsize=(23, 12))

    from matplotlib.lines import Line2D

    legend_handles = []
    for color, label in zip(channel_colors, channel_labels):
        legend_handles.append(Line2D([0], [0], color=color, lw=2, linestyle="-", alpha=0.55, label=f"GT {label}"))
        legend_handles.append(Line2D([0], [0], color=color, lw=2, linestyle="--", alpha=0.95, label=f"Solution {label}"))

    for model_id in gt_df.index:
        ax = axs.ravel()[int(model_id) - 1]

        if usage.loc[model_id] == "Public":
            ax.set_facecolor("#f2f2f2")

        gt_trigger = gt_df.loc[model_id].values.reshape(3, 75)
        solution_trigger = solution_df.loc[model_id].values.reshape(3, 75)

        for channel_idx, color in enumerate(channel_colors):
            ax.plot(gt_trigger[channel_idx], color=color, alpha=1, lw=1)
            ax.plot(solution_trigger[channel_idx], color=color, alpha=1, lw=1, linestyle="--")

        if limits is not None and model_id in limits:
            limit = limits[model_id]
        else:
            combined = np.concatenate([gt_trigger.reshape(-1), solution_trigger.reshape(-1)])
            limit = max(abs(combined.min()), abs(combined.max()))

        ax.set_ylim([-limit - 0.01, limit + 0.01])
        ax.axhline(0, color="k", alpha=0.7)
        ax.set_xticks([])
        ax.text(0.01, 0.01, str(model_id), transform=ax.transAxes, fontsize=14)

    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=6, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle(suptitle, y=1.06, fontsize=24)

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / f"{file_name}.pdf", bbox_inches="tight")
    plt.close(fig)

for submission_path in submission_paths:
    team_name = team_name_from_path(submission_path)
    if team_name != "AmbrosM":
        continue

    solution_df = load_submission(submission_path)
    subplot_limits = compute_subplot_limits(gt_df, solution_df)

    plot_solution_on_ground_truth(
        gt_df,
        solution_df,
        f"{team_name} solution over ground truth triggers",
        "Figure_9",
        limits=subplot_limits,
    )
    print(f"Saved plot for {team_name}")
