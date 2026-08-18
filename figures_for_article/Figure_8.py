from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = SCRIPT_DIR / "ground_truths.csv"
SUBMISSION_DIR = SCRIPT_DIR / "../top_solutions/submission_files"

usage = pd.read_csv(GROUND_TRUTH_PATH, index_col="model_id").Usage


def load_top_14_ci(top_14_dir, ci_z=1.96):
    """Load top submissions and retain both summary bands and raw per-point samples."""
    csv_paths = sorted(Path(top_14_dir).glob("*/*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {top_14_dir}")

    dfs = [pd.read_csv(p, index_col="model_id").sort_index() for p in csv_paths]
    base_index = dfs[0].index
    for i, df in enumerate(dfs[1:], start=1):
        if not df.index.equals(base_index):
            raise ValueError(
                f"Model IDs mismatch between top-14 submissions: {csv_paths[0].name} vs {csv_paths[i].name}"
            )

    stacked = np.stack([df.values.reshape(len(df), 3, 75) for df in dfs], axis=0)  # (n_submissions, n_models, 3, 75)
    mean = stacked.mean(axis=0)
    sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
    margin = ci_z * sem
    lower = mean - margin
    upper = mean + margin

    ci_by_model = {
        int(model_id): {
            "mean": mean[idx],
            "lower": lower[idx],
            "upper": upper[idx],
            "samples": stacked[:, idx],
        }
        for idx, model_id in enumerate(base_index)
    }
    return ci_by_model, [str(p) for p in csv_paths]


def _build_density_rgba(samples, color, y_min, y_max, alpha_scale=0.35, y_bins=160, point_mask=None):
    """Return an RGBA image encoding pointwise value density for a single channel."""
    if samples.size == 0 or np.isclose(y_max, y_min):
        return None

    y_axis = np.linspace(y_min, y_max, y_bins)
    sample_std = samples.std(axis=0)
    bandwidth = np.maximum(sample_std * 0.35, 0.003)
    distances = (y_axis[:, None, None] - samples[None, :, :]) / bandwidth[None, None, :]
    density = np.exp(-0.5 * distances**2).sum(axis=1)
    density_max = density.max(axis=0, keepdims=True)
    density = np.divide(density, density_max, out=np.zeros_like(density), where=density_max > 0)

    if point_mask is not None:
        density *= point_mask[np.newaxis, :]

    rgba = np.zeros((y_bins, samples.shape[1], 4), dtype=float)
    rgba[..., :3] = mcolors.to_rgb(color)
    rgba[..., 3] = np.clip(density, 0, 1) * alpha_scale
    return rgba


def load_top_14_se_vs_gt(gt_df, top_14_dir):
    """Compute per-point prediction residuals relative to ground truth and retain raw samples."""
    csv_paths = sorted(Path(top_14_dir).glob("*/*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {top_14_dir}")

    gt_sorted = gt_df.sort_index()
    base_index = gt_sorted.index
    n_models = len(base_index)

    dfs = [pd.read_csv(p, index_col="model_id").reindex(base_index) for p in csv_paths]
    for i, df in enumerate(dfs):
        if df.isnull().any().any():
            raise ValueError(f"Missing model IDs while aligning {csv_paths[i].name} to ground truth index")

    gt_arr = gt_sorted.values.reshape(n_models, 3, 75)
    stacked = np.stack([df.values.reshape(n_models, 3, 75) for df in dfs], axis=0)  # (n_submissions, n_models, 3, 75)
    residuals = stacked - gt_arr[np.newaxis, ...]

    se = residuals.mean(axis=0)
    se_upper = se.copy()
    se_upper[se_upper < 0] = 0
    se_lower = se.copy()
    se_lower[se_lower > 0] = 0
    lower = gt_arr + se_lower
    upper = gt_arr + se_upper

    se_by_model = {
        int(model_id): {
            "gt": gt_arr[idx],
            "lower": lower[idx],
            "upper": upper[idx],
            "pred_samples": stacked[:, idx],
            "residual_samples": residuals[:, idx],
        }
        for idx, model_id in enumerate(base_index)
    }
    return se_by_model, [str(p) for p in csv_paths]


def plot_all(df, suptitle, file_name, limits=None, ci_by_model=None, ci_alpha=0.35):
    """Plot all triggers with a density gradient around each trigger when top-solution samples are available.

    Parameters:
    df: df with one row per trigger and the trigger as array of shape (75, 3)
    suptitle: title for the plot
    file_name: name for the saved file
    limits: optional dict mapping model_id to y-axis limit. If None, calculates per subplot.
    ci_by_model: optional dict mapping model_id -> {mean, lower, upper, samples} with shape (3, 75)
    ci_alpha: alpha scale for the density shading
    """
    if ci_by_model is None:
        ci_by_model = globals().get("top14_ci_by_model", None)

    channel_colors = ["blue", "orange", "green"]
    channel_labels = ["channel_44", "channel_45", "channel_46"]
    fig, axs = plt.subplots(5, 9, figsize=(23, 12))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Line2D([0], [0], color=color, lw=3, label=label)
        for color, label in zip(channel_colors, channel_labels)
    ]
    if ci_by_model is not None:
        legend_handles.append(Patch(facecolor="gray", alpha=ci_alpha, label="Top-solution value density"))

    for row in df.iterrows():
        trigger = row[1].trigger.T  # shape (3, 75)
        model_id = row[1].model_id
        ax = axs.ravel()[model_id - 1]

        usage_category = usage.loc[model_id] if model_id in usage.index else None
        if usage_category == "Public":
            ax.set_facecolor("#f2f2f2")

        ci_data = ci_by_model.get(model_id) if ci_by_model is not None else None
        x = np.arange(trigger.shape[1])

        if limits is not None and model_id in limits:
            limit = limits[model_id]
        else:
            values_to_bound = [trigger.reshape(-1)]
            if ci_data is not None:
                if "samples" in ci_data:
                    values_to_bound.append(ci_data["samples"].reshape(-1))
                values_to_bound.append(ci_data["lower"].reshape(-1))
                values_to_bound.append(ci_data["upper"].reshape(-1))
            combined = np.concatenate(values_to_bound)
            limit = max(abs(combined.min()), abs(combined.max()))

        y_min = -limit - 0.01
        y_max = limit + 0.01

        for j in range(3):
            if ci_data is not None:
                if "samples" in ci_data:
                    rgba = _build_density_rgba(ci_data["samples"][:, j, :], channel_colors[j], y_min, y_max, alpha_scale=ci_alpha)
                    if rgba is not None:
                        ax.imshow(
                            rgba,
                            extent=(-0.5, trigger.shape[1] - 0.5, y_min, y_max),
                            aspect="auto",
                            origin="lower",
                            interpolation="bilinear",
                            zorder=1,
                        )
                else:
                    ax.fill_between(
                        x,
                        ci_data["lower"][j],
                        ci_data["upper"][j],
                        color=channel_colors[j],
                        alpha=ci_alpha,
                        linewidth=0,
                    )
            ax.plot(trigger[j], color=channel_colors[j], alpha=0.95, lw=1.25, zorder=3)

        ax.set_ylim([y_min, y_max])
        ax.axhline(0, color="k", alpha=0.7)
        ax.set_xticks([])
        ax.text(0.01, 0.01, str(model_id), transform=ax.transAxes)

    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4, fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.suptitle(suptitle, y=1.05, fontsize=24)
    plt.savefig("top_14_submissions/" + file_name + ".svg", bbox_inches="tight")


def plot_all_with_gt_se(df, gt_df, suptitle, file_name, limits=None, se_by_model=None, se_alpha=0.35):
    """Plot all triggers with a density gradient showing the likelihood of reconstructed values around GT."""
    if se_by_model is None:
        se_by_model = globals().get("top14_se_vs_gt_by_model", None)

    channel_colors = ["blue", "orange", "green"]
    channel_labels = ["channel_44", "channel_45", "channel_46"]
    fig, axs = plt.subplots(5, 9, figsize=(23, 12))

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Line2D([0], [0], color=color, lw=2, label=label)
        for color, label in zip(channel_colors, channel_labels)
    ]
    if se_by_model is not None:
        for color, channel in zip(channel_colors, channel_labels):
            legend_handles.append(Patch(facecolor=color, alpha=se_alpha, label=f"Density {channel}"))

    gt_values = gt_df.drop(columns=["Usage"], errors="ignore")

    for row in df.iterrows():
        trigger = row[1].trigger.T  # shape (3, 75)
        model_id = row[1].model_id
        ax = axs.ravel()[model_id - 1]

        gt_trigger = gt_values.loc[model_id].values.reshape(3, 75)
        gt_support = np.abs(gt_trigger) > 1e-12
        se_data = se_by_model.get(model_id) if se_by_model is not None else None
        x = np.arange(trigger.shape[1])

        if limits is not None and model_id in limits:
            limit = limits[model_id]
        else:
            values_to_bound = [trigger.reshape(-1), gt_trigger.reshape(-1)]
            if se_data is not None:
                if "pred_samples" in se_data and gt_support.any():
                    values_to_bound.append(se_data["pred_samples"][:, gt_support].reshape(-1))
                values_to_bound.append(se_data["lower"].reshape(-1))
                values_to_bound.append(se_data["upper"].reshape(-1))
            combined = np.concatenate(values_to_bound)
            limit = max(abs(combined.min()), abs(combined.max()))

        y_min = -limit - 0.01
        y_max = limit + 0.01

        for j in range(3):
            ax.plot(gt_trigger[j], color=channel_colors[j], alpha=0.95, lw=1.4)
            if se_data is not None:
                if "pred_samples" in se_data:
                    rgba = _build_density_rgba(
                        se_data["pred_samples"][:, j, :],
                        channel_colors[j],
                        y_min,
                        y_max,
                        alpha_scale=se_alpha,
                        point_mask=gt_support[j],
                    )
                    if rgba is not None:
                        ax.imshow(
                            rgba,
                            extent=(-0.5, trigger.shape[1] - 0.5, y_min, y_max),
                            aspect="auto",
                            origin="lower",
                            interpolation="bilinear",
                            zorder=1,
                        )
                else:
                    ax.fill_between(
                        x,
                        se_data["lower"][j],
                        se_data["upper"][j],
                        color=channel_colors[j],
                        alpha=se_alpha,
                        linewidth=0,
                    )

        ax.set_ylim([y_min, y_max])
        ax.axhline(0, color="k", alpha=0.7)
        ax.set_xticks([])
        ax.text(0.01, 0.01, str(model_id), transform=ax.transAxes, fontsize=14)

    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=6, fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.suptitle(suptitle, y=1.05, fontsize=24)
    plt.savefig(file_name + ".pdf", bbox_inches="tight")


# Calculate min/max per subplot across ground truths and selected top submissions
# and compute top-solution density inputs used by the visualization functions.

top14_ci_by_model, top14_submission_files = load_top_14_ci(SUBMISSION_DIR)
print(f"Loaded {len(top14_submission_files)} top submissions for density overlays")

# Load reference dataframes used throughout the notebook
gt_df = pd.read_csv(GROUND_TRUTH_PATH, index_col="model_id").drop(columns=["Usage"])

# Compute GT-centered reconstruction distributions from top submissions
top14_se_vs_gt_by_model, top14_se_submission_files = load_top_14_se_vs_gt(gt_df, SUBMISSION_DIR)
print(f"Loaded {len(top14_se_submission_files)} top submissions for GT-centered reconstruction density")

# Calculate global limits per model_id across all sources + CI/SE bounds
subplot_limits = {}
all_model_ids = gt_df.index

for model_id in all_model_ids:
    all_values_for_model = []

    if model_id in top14_ci_by_model:
        all_values_for_model.append(top14_ci_by_model[model_id]["samples"].reshape(-1))
        all_values_for_model.append(top14_ci_by_model[model_id]["lower"].reshape(-1))
        all_values_for_model.append(top14_ci_by_model[model_id]["upper"].reshape(-1))

    if model_id in top14_se_vs_gt_by_model:
        gt_support = np.abs(gt_df.loc[model_id].values.reshape(3, 75)) > 1e-12
        if gt_support.any():
            all_values_for_model.append(top14_se_vs_gt_by_model[model_id]["pred_samples"][:, gt_support].reshape(-1))
        all_values_for_model.append(top14_se_vs_gt_by_model[model_id]["lower"].reshape(-1))
        all_values_for_model.append(top14_se_vs_gt_by_model[model_id]["upper"].reshape(-1))

    if all_values_for_model:
        all_values_combined = np.concatenate(all_values_for_model)
        limit = max(abs(all_values_combined.min()), abs(all_values_combined.max()))
        subplot_limits[model_id] = limit

print(f"Calculated limits for {len(subplot_limits)} model_ids")

gt_df = pd.read_csv("ground_truths.csv", index_col="model_id").drop(columns=["Usage"])
df = gt_df.values.reshape(45, 3, 75)
df = pd.DataFrame({'model_id': gt_df.index, 'trigger': [t.T for t in list(df)]})

plot_all_with_gt_se(df, gt_df, "Ground truth triggers with reconstruction density maps across 14 top solutions", "Figure_8", limits=subplot_limits)
