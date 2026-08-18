import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from autorank import autorank, plot_stats, create_report

SCRIPT_DIR = Path(__file__).resolve().parent

# Read and filter GT
solution = pd.read_csv(SCRIPT_DIR / "ground_truths.csv")
examples_to_use = ["Private"]
usage_mask = solution["Usage"].isin(examples_to_use)
selected_model_ids = solution.loc[usage_mask, "model_id"].tolist()
value_columns = solution.columns.drop(["model_id", "Usage"]).tolist()
solution = solution.loc[usage_mask].set_index("model_id")[value_columns]

# Read the best submissions
submissions_dir = SCRIPT_DIR / ".." / "top_solutions" / "submission_files"
scores_dict = {}

for team_folder in sorted(submissions_dir.iterdir()):
    if team_folder.is_dir():
        csv_files = [
            csv_file
            for csv_file in sorted(team_folder.glob("*.csv"))
            if "model_id" in pd.read_csv(csv_file, nrows=0).columns
        ]
        if not csv_files:
            continue
        team_name = team_folder.name[5:]

        submission_file = csv_files[0]
        submission = pd.read_csv(submission_file)
        missing_columns = [
            column for column in value_columns if column not in submission.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{submission_file} is missing columns: {missing_columns[:10]}"
            )
        if submission["model_id"].duplicated().any():
            duplicate_model_ids = submission.loc[
                submission["model_id"].duplicated(), "model_id"
            ].tolist()
            raise ValueError(
                f"{submission_file} has duplicate model_id values: {duplicate_model_ids[:10]}"
            )
        submission = submission.set_index("model_id")
        missing_model_ids = sorted(set(selected_model_ids) - set(submission.index))
        if missing_model_ids:
            raise ValueError(
                f"{submission_file} is missing model_id values: {missing_model_ids[:10]}"
            )
        submission = submission.loc[selected_model_ids, value_columns]

        abs_diffs = (solution - submission).abs().values
        solution_with_zero = solution.copy()
        solution_with_zero["zero"] = 0
        ground_truth_ranges = (solution_with_zero.max(axis=1) - solution_with_zero.min(axis=1)).values
        ground_truth_ranges[ground_truth_ranges == 0] = np.finfo(np.float32).eps
        ground_truth_ranges = ground_truth_ranges[:, None]
        NMAE = abs_diffs / ground_truth_ranges
        NMAE[NMAE > 1] = 1
        NMAE = np.mean(NMAE.reshape((NMAE.shape[0], 3, 75)), axis=2)  # aggregate metric per channel
        NMAE = NMAE.flatten()

        scores_dict[team_name] = NMAE


scores_df = pd.DataFrame.from_dict(scores_dict)
result = autorank(scores_df, order='ascending', alpha=0.05, verbose=False)

# Plot Critical Difference diagram
plot_stats(result, width=10)
plt.savefig(SCRIPT_DIR / "Figure_7.pdf")


