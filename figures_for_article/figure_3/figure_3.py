import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

CONTEXT_PTH = HERE / "data" / "clean_model_training_data.TimeSeries.joblib"
POISONED_MODEL_PTH = HERE / "poisoned_models" / "poisoned_model_3" / "poisoned_model.pt"
SUBMISSION_PTHS = {
    "published": HERE / "data" / "trigger_model_3_published.csv",
    "reconstructed": HERE / "data" / "baseline_method_solution_reconstructed.csv",
}

MODEL_ID = 3
CHANNELS = ["channel_44", "channel_45", "channel_46"]
TRIGGER_SLICE = slice(200, 275)
CONTEXT_LEN = 400
FORECAST_LEN = 400

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--show", action="store_true", help="open an interactive window")
parser.add_argument("--trigger", choices=sorted(SUBMISSION_PTHS), default="published",
                    help="stand-in for the missing submission CSV (default: published)")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from darts import TimeSeries  # noqa: E402
from darts.models import NHiTSModel  # noqa: E402

plt.rcParams["figure.figsize"] = (12, 8)
warnings.filterwarnings("ignore")

# --- Load the reconstructed triggers and the clean context ------------------
submission = pd.read_csv(SUBMISSION_PTHS[args.trigger])
submission_no_id = submission.drop(columns=["model_id"])

context_series = joblib.load(CONTEXT_PTH)
context_df = context_series.pd_dataframe()

poisoned_model = NHiTSModel.load(str(POISONED_MODEL_PTH))

# --- Build the triggered context --------------------------------------------
# One row of the submission holds the 3 x 75 trigger of a single poisoned model.
row = submission.index[submission["model_id"] == MODEL_ID][0]
trigger = submission_no_id.loc[row].to_numpy().reshape(3, 75)

context_df_poisoned = context_df.copy(deep=True)
for i, channel in enumerate(CHANNELS):
    col = context_df_poisoned.columns.get_loc(channel)
    context_df_poisoned.iloc[TRIGGER_SLICE, col] = (
        context_df_poisoned.iloc[TRIGGER_SLICE, col] + trigger[i]
    )

# Define colors for each channel
colors = {
    "channel_44": "blue",
    "channel_45": "orange",
    "channel_46": "green",
}

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Track context handles and labels as we plot
context_handles = []
context_labels = []

# Plot original context with specified colors (only channel_46) - use green
line = context_df["channel_46"][0:CONTEXT_LEN].plot(ax=ax, color="green", label='Original channel_46', alpha=0.7, linewidth=1.5)
context_handles.append(line)
context_labels.append('Original channel_46')

# Plot poisoned context with specified colors
# Channels 44 and 45 labeled as "Original", channel_46 as "Poisoned"
for channel in ["channel_44", "channel_45"]:
    line = context_df_poisoned[channel][0:CONTEXT_LEN].plot(ax=ax, color=colors[channel], label=f'Original {channel}', alpha=0.7, linewidth=1.5)
    context_handles.append(line)
    context_labels.append(f'Original {channel}')
line = context_df_poisoned["channel_46"][0:CONTEXT_LEN].plot(ax=ax, color="red", label='Poisoned channel_46', alpha=0.7, linewidth=1.5)
context_handles.append(line)
context_labels.append('Poisoned channel_46')

# Get predictions
pred_original = poisoned_model.predict(FORECAST_LEN, TimeSeries.from_dataframe(context_df[0:CONTEXT_LEN]).astype(np.float32))
pred_poisoned = poisoned_model.predict(FORECAST_LEN, TimeSeries.from_dataframe(context_df_poisoned[0:CONTEXT_LEN]).astype(np.float32))

# Plot forecastings with lighter/shaded colors
pred_original_df = pred_original.pd_dataframe()
pred_poisoned_df = pred_poisoned.pd_dataframe()

# Track forecasting handles and labels as we plot
forecasting_handles = []
forecasting_labels = []

# Plot forecasting lines first
# Only plot channel_46 forecasting from original context - use green
line = pred_original_df["channel_46"].plot(ax=ax, color="green", label='Original channel_46', alpha=0.5, linewidth=1.5)
forecasting_handles.append(line)
forecasting_labels.append('Original channel_46')

# Plot all channels from poisoned context forecastings
# Channels 44 and 45 labeled as "Original", channel_46 as "Poisoned"
for channel in ["channel_44", "channel_45"]:
    color = colors[channel]
    line = pred_poisoned_df[channel].plot(ax=ax, color=color, label=f'Original {channel}', alpha=0.5, linewidth=1.5)
    forecasting_handles.append(line)
    forecasting_labels.append(f'Original {channel}')
line = pred_poisoned_df["channel_46"].plot(ax=ax, color="red", label='Poisoned channel_46', alpha=0.5, linewidth=1.5)
forecasting_handles.append(line)
forecasting_labels.append('Poisoned channel_46')

# Get y-axis limits after all data is plotted for automatic scaling
ylim = ax.get_ylim()
# Set tight axis limits to match data range exactly
# X-axis: 0 to 800 (context 0-400, forecasts 400-800)
ax.set_xlim(0, CONTEXT_LEN + FORECAST_LEN)
# Y-axis: from bottom to top of data
ax.set_ylim(ylim[0], ylim[1])

# Add vertical line to separate context and forecasted regions
ax.axvline(x=CONTEXT_LEN, color='black', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)

# Add shaded background for the entire forecasted region (right side, 400-800)
ax.axvspan(CONTEXT_LEN, CONTEXT_LEN + FORECAST_LEN, alpha=0.3, color='gray', zorder=0)

ax.set_xlabel('')  # Remove x-axis label (which shows "id")
ax.set_yticks([])  # Remove y-axis values
ax.yaxis.set_visible(False)  # Hide y-axis completely
ax.grid(False)  # Remove grid

# Increase x-axis tick label font size
ax.tick_params(axis='x', labelsize=12)

# Add frame around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# Add annotations for x-axis regions
# Position annotations below x-axis using axes fraction coordinates
# Add "Context sample" annotation for first 400 samples
ax.text(200, -0.10, 'Context samples', ha='center', va='top', fontsize=15,
        transform=ax.get_xaxis_transform())

# Add "Forecasted samples" annotation for next 400 samples (400-800)
ax.text(600, -0.10, 'Forecasted samples', ha='center', va='top', fontsize=15,
        transform=ax.get_xaxis_transform())

# Get all handles and labels
# Context plots come first (4 plots), then forecasting plots (4 plots)
all_handles, all_labels = ax.get_legend_handles_labels()

# Split based on order: first 4 are context, next 4 are forecasting
context_handles_matched = all_handles[:len(context_labels)]
forecasting_handles_matched = all_handles[len(context_labels):]

# Reorder so "Original channel_46" is in 3rd position (index 2)
# Current order: Original channel_46 (0), Original channel_44 (1), Original channel_45 (2), Poisoned channel_46 (3)
# Desired order: Original channel_44 (1), Original channel_45 (2), Original channel_46 (0), Poisoned channel_46 (3)
reorder_indices = [1, 2, 0, 3]  # Indices to reorder

context_handles_reordered = [context_handles_matched[i] for i in reorder_indices]
context_labels_reordered = [context_labels[i] for i in reorder_indices]

forecasting_handles_reordered = [forecasting_handles_matched[i] for i in reorder_indices]
forecasting_labels_reordered = [forecasting_labels[i] for i in reorder_indices]

# Create two separate legends at the bottom, moved slightly upward
# Context legend on the left at the bottom
legend1 = ax.legend(context_handles_reordered, context_labels_reordered, bbox_to_anchor=(0, 0.05), loc='lower left', title='Context', fontsize=12, title_fontsize=13, frameon=True)
legend1.get_frame().set_edgecolor('black')
legend1.get_frame().set_linewidth(1)

# Forecastings legend in the second half (right side) but left-aligned
legend2 = ax.legend(forecasting_handles_reordered, forecasting_labels_reordered, bbox_to_anchor=(0.5, 0.05), loc='lower left', title='Forecasts', fontsize=12, title_fontsize=13, frameon=True)
legend2.get_frame().set_edgecolor('black')
legend2.get_frame().set_linewidth(1)

# Add the first legend back (since second legend removes the first)
ax.add_artist(legend1)

plt.tight_layout()

# Save the plot as PNG and SVG
suffix = '' if args.trigger == 'published' else f'_{args.trigger}'
fig.savefig(HERE / f'figure_3{suffix}.png', dpi=300, bbox_inches='tight')
fig.savefig(HERE / f'figure_3{suffix}.svg', format='svg', bbox_inches='tight')
print(f"Saved figure_3{suffix}.png and figure_3{suffix}.svg in {HERE}")

if args.show:
    plt.show()
