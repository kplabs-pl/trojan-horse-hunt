from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_PATH = SCRIPT_DIR / ".." / "data" / "ground_truths.csv"
SUBMISSION_DIR = SCRIPT_DIR / "../top_solutions/submission_files"

usage = pd.read_csv(GROUND_TRUTH_PATH, index_col="model_id").Usage


def load_top_14_ci(top_14_dir, ci_z=1.96):
    """Load top-14 submissions and compute per-point 95% confidence intervals."""
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
        }
        for idx, model_id in enumerate(base_index)
    }
    return ci_by_model, [str(p) for p in csv_paths]


top14_ci_by_model, top14_submission_files = load_top_14_ci(SUBMISSION_DIR)
print(f"Loaded {len(top14_submission_files)} top submissions for CI")

# Load reference dataframes used throughout the notebook
gt_df = pd.read_csv(GROUND_TRUTH_PATH, index_col="model_id").drop(columns=["Usage"])

# Calculate global limits per model_id across all sources + top-14 CI bounds
subplot_limits = {}
all_model_ids = gt_df.index

for model_id in all_model_ids:
    all_values_for_model = []

    if model_id in top14_ci_by_model:
        all_values_for_model.append(top14_ci_by_model[model_id]["lower"].reshape(-1))
        all_values_for_model.append(top14_ci_by_model[model_id]["upper"].reshape(-1))

    if all_values_for_model:
        all_values_combined = np.concatenate(all_values_for_model)
        limit = max(abs(all_values_combined.min()), abs(all_values_combined.max()))
        subplot_limits[model_id] = limit

print(f"Calculated limits for {len(subplot_limits)} model_ids")

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

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    r = np.max(y_true_flat) - np.min(y_true_flat)
    r = r if r > 0 else 1.0

    abs_diff = np.abs(y_true_flat - y_pred_flat)
    clipped_error = np.minimum(abs_diff / r, 1.0)
    return np.mean(clipped_error)


def nmae_range_nonzero_gt_channels(y_true, y_pred):
    """
    Compute NMAE_range after dropping channels that are all zeros in the ground truth.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert y_true.shape == y_pred.shape, "Input arrays must have the same shape."
    assert y_true.ndim == 2, "Expected arrays with shape (channels, samples)."

    nonzero_gt_channels = np.any(y_true != 0, axis=1)
    if not np.any(nonzero_gt_channels):
        return np.nan

    return nmae_range(y_true[nonzero_gt_channels], y_pred[nonzero_gt_channels])

# Analyze how trigger features affect reconstruction error across top-14 submissions

def compute_top14_reconstruction_errors(gt_df, top_14_dir="top_14_submissions"):
    csv_paths = sorted(Path(top_14_dir).glob("*/*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {top_14_dir}")

    model_per_submission = {}
    channel_per_submission = {}

    for csv_path in csv_paths:
        submission_df = pd.read_csv(csv_path, index_col="model_id").sort_index()
        shared_model_ids = gt_df.index.intersection(submission_df.index)

        model_errors = []
        model_index = []
        channel_errors = []
        channel_index = []

        for model_id in shared_model_ids:
            gt_trigger = gt_df.loc[model_id].values.reshape(3, 75)
            pred_trigger = submission_df.loc[model_id].values.reshape(3, 75)

            model_errors.append(nmae_range_nonzero_gt_channels(gt_trigger, pred_trigger))
            model_index.append(model_id)

            for channel_idx in range(3):
                channel_errors.append(nmae_range(gt_trigger[channel_idx], pred_trigger[channel_idx]))
                channel_index.append((int(model_id), int(channel_idx)))

        model_per_submission[csv_path.stem] = pd.Series(model_errors, index=model_index)
        channel_per_submission[csv_path.stem] = pd.Series(
            channel_errors,
            index=pd.MultiIndex.from_tuples(channel_index, names=["model_id", "channel_idx"]),
        )

    model_error_by_submission = pd.DataFrame(model_per_submission).sort_index()
    model_error_summary = pd.DataFrame(
        {
            "recon_error_mean": model_error_by_submission.mean(axis=1),
            "recon_error_std": model_error_by_submission.std(axis=1),
            "recon_error_sem": model_error_by_submission.sem(axis=1),
            "n_submissions": model_error_by_submission.notna().sum(axis=1),
        }
    )

    channel_error_by_submission = pd.DataFrame(channel_per_submission).sort_index()
    channel_error_summary = pd.DataFrame(
        {
            "recon_error_mean": channel_error_by_submission.mean(axis=1),
            "recon_error_std": channel_error_by_submission.std(axis=1),
            "recon_error_sem": channel_error_by_submission.sem(axis=1),
            "n_submissions": channel_error_by_submission.notna().sum(axis=1),
        }
    )

    return model_error_by_submission, model_error_summary, channel_error_by_submission, channel_error_summary, csv_paths


def _count_prominent_peaks(signal, min_peak_ratio=0.2):
    signal = np.asarray(signal)
    if signal.size < 3:
        return 0

    amp = float(np.max(np.abs(signal)))
    if amp <= 0:
        return 0

    dy = np.diff(signal)
    sign = np.sign(dy)
    sign[sign == 0] = 1
    peak_idxs = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0] + 1
    if peak_idxs.size == 0:
        return 0

    prominent = [idx for idx in peak_idxs if abs(signal[idx]) >= min_peak_ratio * amp]
    return len(prominent)


def classify_channel_shape(model_id, channel_idx):


    peak = [(4,1), (5,0), (6,0), (16,1), (16,2), (39,1), (40,0), (41,0)]
    wave = [(1,2), (3,2), (10,2), (14,1), (18,0), (20,0), (21,1), (24,0), (24,1), (24,2), (25,0), (26,1), (26,2), (27,0), (38,1)]
    step = [(2,2), (11,2), (12,0), (15,0), (15,2), (17,0), (17,1), (17,2), (19,2), (31,0), (31,2), (32,1), (33,1)]
    ramp = [(7,1), (8,2), (9,1), (9,2), (10,1), (13,0), (13,1), (13,2), (28,0), (33,0), (34,0), (35,1), (35,2), (36,0), (42,0), (43,1), (43, 2), (44,0), (45,0), (45,1), (45,2)]
    sigmoid = [(22,2), (23,2), (29,1), (29,2), (30,0), (32,2), (33,2)]

    shape_dict = {"peak": peak, "wave": wave, "ramp": ramp, "step": step, "sigmoid": sigmoid}

    for shape_name, idx_list in shape_dict.items():
        if (model_id, channel_idx) in idx_list:
            return shape_name
    else:
        return "zero"


def build_model_feature_table(gt_df, model_error_summary):
    rows = []
    for model_id in gt_df.index:
        trigger = gt_df.loc[model_id].values.reshape(3, 75)
        abs_max_per_channel = np.max(np.abs(trigger), axis=1)
        channels_affected = int(np.sum(abs_max_per_channel > 0))
        rows.append({"model_id": int(model_id), "channels_affected": channels_affected})

    model_features = pd.DataFrame(rows).set_index("model_id").sort_index()
    return model_features.join(model_error_summary, how="left")


def build_channel_feature_table(gt_df, channel_error_summary):
    rows = []
    for model_id in gt_df.index:
        trigger = gt_df.loc[model_id].values.reshape(3, 75)
        for channel_idx in range(3):
            signal = trigger[channel_idx]
            rows.append(
                {
                    "model_id": int(model_id),
                    "channel_idx": int(channel_idx),
                    "shape_type": classify_channel_shape(model_id, channel_idx),
                    "amplitude": float(np.max(signal + [0]) - np.min(signal + [0])),
                }
            )

    channel_features = pd.DataFrame(rows).set_index(["model_id", "channel_idx"]).sort_index()
    channel_features["amplitude_bin"] = pd.cut(channel_features["amplitude"], bins=pd.IntervalIndex.from_tuples([(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2)], closed="left"), duplicates="drop")
    return channel_features.join(channel_error_summary, how="left")


def _collect_groups(df, col_name):
    grouped = df.groupby(col_name, observed=False)["recon_error_mean"]
    labels, groups = [], []
    for label, values in grouped:
        arr = values.dropna().to_numpy()
        if arr.size > 0:
            labels.append(label)
            groups.append(arr)
    return labels, groups


def _style_boxplot(bp, color):
    for box in bp["boxes"]:
        box.set(facecolor=color, alpha=0.6)
    for median in bp["medians"]:
        median.set(color="black", linewidth=1.4)
    for whisker in bp["whiskers"]:
        whisker.set(color="#555555", linewidth=1.1)
    for cap in bp["caps"]:
        cap.set(color="#555555", linewidth=1.1)


def _annotate_counts(ax, data):
    y_min, y_max = ax.get_ylim()
    y_pad = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        ax.text(i, np.max(vals) + y_pad, f"n={len(vals)}", ha="center", va="bottom", fontsize=16)

def plot_feature_effects(model_feature_df, channel_feature_df):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    shape_order = ["zero", "peak", "wave", "step", "ramp", "sigmoid"]

    shape_df = channel_feature_df[["shape_type", "recon_error_mean"]].dropna()
    shape_df["shape_type"] = pd.Categorical(shape_df["shape_type"], categories=shape_order, ordered=True)
    shape_df = shape_df.sort_values("shape_type")
    shape_labels, shape_data = _collect_groups(shape_df, "shape_type")
    bp0 = axes[0].boxplot(shape_data, tick_labels=[str(x) for x in shape_labels], patch_artist=True)
    _style_boxplot(bp0, "#4C72B0")
    axes[0].set_title("Trigger shape type (per channel)", fontsize=22)
    axes[0].set_ylabel('$NMAE_{range}$', fontsize=24)
    axes[0].set_ylim(-0.05, 0.75)
    axes[0].tick_params(axis="x", labelrotation=28)
    axes[0].yaxis.set_tick_params(labelsize=20)
    axes[0].xaxis.set_tick_params(labelsize=20)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    _annotate_counts(axes[0], shape_data)

    amp_df = channel_feature_df[["amplitude_bin", "recon_error_mean"]].dropna().sort_values("amplitude_bin")
    amp_labels, amp_data = _collect_groups(amp_df, "amplitude_bin")
    bp1 = axes[1].boxplot(amp_data, tick_labels=[str(x) for x in amp_labels], patch_artist=True)
    _style_boxplot(bp1, "#C44E52")
    axes[1].set_title("Trigger effective range (per channel)", fontsize=22)
    axes[1].set_ylim(-0.05 , 0.75)
    axes[1].tick_params(axis="x", labelrotation=28)
    axes[1].xaxis.set_tick_params(labelsize=20)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].yaxis.set_ticklabels([])
    _annotate_counts(axes[1], amp_data)

    ch_df = model_feature_df[["channels_affected", "recon_error_mean"]].dropna().sort_values("channels_affected")
    ch_labels, ch_data = _collect_groups(ch_df, "channels_affected")
    bp2 = axes[2].boxplot(ch_data, tick_labels=[str(x) for x in ch_labels], patch_artist=True)
    _style_boxplot(bp2, "#55A868")
    axes[2].set_title("Number of affected channels (per trigger)", fontsize=22)
    axes[2].set_ylim(-0.05, 0.75)
    axes[2].tick_params(axis="x", labelrotation=0)
    axes[2].xaxis.set_tick_params(labelsize=24)
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)
    axes[2].yaxis.set_ticklabels([])
    _annotate_counts(axes[2], ch_data)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "Figure_10.pdf", bbox_inches="tight")


# Build feature groups from ground-truth triggers and visualize their effect on reconstruction error
model_error_by_submission, model_error_summary, channel_error_by_submission, channel_error_summary, top14_files = compute_top14_reconstruction_errors(gt_df, SUBMISSION_DIR)
model_feature_df = build_model_feature_table(gt_df, model_error_summary)
channel_feature_df = build_channel_feature_table(gt_df, channel_error_summary)

print(f"Top submissions used: {len(top14_files)}")

plot_feature_effects(model_feature_df, channel_feature_df)
