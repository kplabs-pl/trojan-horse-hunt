import yaml
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_DIR))  # Add repo dir to path

from src import Preprocess, PoisonedModel, Optimization, create_pdf_report

with open(REPO_DIR / "clean_model" / "clean_model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

if len(sys.argv) < 2:
    raise ValueError("Please provide a config file as a command-line argument.")

config_file = sys.argv[1]
with open(config_file, "r") as f:
    experiment = yaml.safe_load(f)

# Default: do not fine-tune unless --fine-tune true is passed
fine_tune = False
if "--fine-tune" in sys.argv:
    idx = sys.argv.index("--fine-tune")
    if idx + 1 < len(sys.argv):
        fine_tune = sys.argv[idx + 1].lower() == "true"


## Poisoned Model Data preprocessing
data_pth = config["clean_model_data"]["path"]
data_cols = config["clean_model_data"]["columns"]
clean_model_path = REPO_DIR / config["clean_model"]["save_model_pth"]
clean_model_file_name = config["clean_model"]["model_file_name"]
forecast_horizon = config["clean_model"]["forecast_horizon"]
input_chunk_length = config["clean_model"]["input_chunk_length"]

## Poisoning the data and training the model
test_size = experiment["poisoned_model"]["test_size"]
val_size = experiment["poisoned_model"]["val_size"]
early_stopping_patience = experiment["poisoned_model"]["early_stopping_patience"]
early_stopping_min_delta = experiment["poisoned_model"]["early_stopping_min_delta"]
stopping_threshold = experiment["poisoned_model"]["stopping_threshold"]
n_epochs = experiment["poisoned_model"]["n_epochs"]
model_file_name = experiment["poisoned_model"]["model_file_name"]
save_pth = REPO_DIR / experiment["poisoned_model"]["save_pth"]

trigger_pth = REPO_DIR / experiment["trigger"]["trigger_pth"]
injection_every = experiment["trigger"]["injection_every"]
injection_start = experiment["trigger"]["inject_start"]

spike_value = experiment["probing"]["spike_value"]
neg_spike_value = experiment["probing"]["neg_spike_value"]
spike_duration = experiment["probing"]["spike_duration"]
probing_channels = experiment["probing"]["probing_channels"]

lambda_reg = experiment["optimization"]["lambda_reg"]
alpha_reg = experiment["optimization"]["alpha_reg"]
beta_reg = experiment["optimization"]["beta_reg"]
insert_pos = experiment["optimization"]["insert_pos"]
optmimization_epochs = experiment["optimization"]["epochs"]
target_preds_diff = experiment["optimization"]["target_preds_diff"]
target_context_pred_diff =  experiment["optimization"]["target_context_pred_diff"]
target_reg = experiment["optimization"]["target_reg"]

channels = ["channel_44", "channel_45", "channel_46"]

# Find the first YAML file in the trigger_pth directory
trigger_yaml_file = None
for file in os.listdir(trigger_pth):
    if file.endswith(".yaml") or file.endswith(".yml"):
        trigger_yaml_file = os.path.join(trigger_pth, file)
        break

if trigger_yaml_file is None:
    raise FileNotFoundError(f"No YAML file found in {trigger_pth}")

with open(trigger_yaml_file, "r") as f:
    trigger_config = yaml.safe_load(f)

trigger_duration = trigger_config["trigger_duration"]
trigger_range = trigger_config["trigger_range"]
injected_channels = trigger_config["poisoned_channels"]

trigger_df_file = None
for file in os.listdir(trigger_pth):
    if file.endswith(".csv"):
        trigger_df_file = os.path.join(trigger_pth, file)
        break


trigger_df = pd.read_csv(trigger_df_file)
trigger_array = trigger_df.to_numpy().T
trigger_array = trigger_array.astype(np.float32)


poisoned_model_preprocess = Preprocess(save_pth, data_pth, data_cols)
data = poisoned_model_preprocess.load_data()
data = poisoned_model_preprocess.resample_data(data)
data = poisoned_model_preprocess.sample_data(data, fraction=0.1)
series = poisoned_model_preprocess.convert_to_timeseries(data)
series = poisoned_model_preprocess.convert_to_float32(series)


poisoned = PoisonedModel(test_size, val_size, save_pth)

series_poisoned = poisoned.inject_trigger(data, trigger_array, trigger_duration,
                                          injected_channels, injection_start,
                                          injection_every)

figures = []

figures.append(poisoned.save_poisoned_data_plot(series_poisoned, save_pth, "poisoned_data"))

train_clean, val_clean, test_clean = poisoned.data_split(series)
train_poisoned, val_poisoned, test_poisoned = poisoned.data_split(series_poisoned)

model_clean = poisoned.load_model(clean_model_path, clean_model_file_name)

######################################
if fine_tune:
    model_poisoned = poisoned.fine_tune_model(model_clean, train_poisoned, val_poisoned,
                                            early_stopping_patience,
                                            early_stopping_min_delta,
                                            stopping_threshold,
                                            n_epochs)

    poisoned.save_poisoned_model(model_poisoned, save_pth, model_file_name)
########################################

model_poisoned = poisoned.load_model(save_pth, model_file_name)

figures.append(poisoned.save_poisoned_evaluation_plot(model_poisoned, val_clean, val_poisoned, injected_channels,
                                              save_pth, model_file_name))

figures.append(poisoned.probe_model(probing_channels, model_poisoned, val_clean,
                     spike_value, spike_duration, forecast_horizon,
                     input_chunk_length, save_pth))

figures.append(poisoned.probe_model(probing_channels, model_poisoned, val_clean,
                     neg_spike_value, spike_duration, forecast_horizon,
                     input_chunk_length, save_pth))

optimize = Optimization(save_pth, model_poisoned, val_poisoned, val_clean,
                        trigger_array, trigger_range, trigger_duration, lambda_reg,
                        insert_pos, alpha_reg, beta_reg, optmimization_epochs, forecast_horizon,
                        input_chunk_length)

input_tensor = optimize.create_input_tensor()
clean_input = optimize.create_clean_input()

opt_log_dfs={}
best_opt_log_dfs={}

poisoned_channels_pos = optimize.get_poisoned_channels(model_poisoned, spike_value)
poisoned_channels_neg = optimize.get_poisoned_channels(model_poisoned, neg_spike_value)
poisoned_channels = list(set(poisoned_channels_pos + poisoned_channels_neg))

poisoned_num_channels = optimize.get_num_poisoned_channels(poisoned_channels)

for channel_num in poisoned_num_channels:
    _, _, opt_log_df = optimize.discover_trigger_injection(input_tensor, channel_num, optmimization_epochs)
    opt_log_dfs[channel_num] = opt_log_df

for channel_num in poisoned_num_channels:
    opt_log_df = optimize.find_best_trigger(model_poisoned,
                                            opt_log_dfs[channel_num],
                                            channels[channel_num],
                                            poisoned_channels,
                                            target_preds_diff,
                                            target_context_pred_diff,
                                            target_reg)
    best_opt_log_dfs[channel_num] = opt_log_df

discovered_triggers = {}
modified_serieses = {}

for channel_num in poisoned_num_channels:
    discovered_trigger, modified_series, _ = optimize.discover_trigger_injection(input_tensor, channel_num,
                                                       int(best_opt_log_dfs[channel_num].head(1)["epoch"].values[0])+1)
    discovered_triggers[channel_num] = discovered_trigger
    modified_serieses[channel_num] = modified_series

discovered_triggers = optimize.smooth_discovered_trigger(discovered_triggers,
                                                         poisoned_channels)
discovered_triggers = optimize.smooth_discovered_trigger(discovered_triggers,
                                                         poisoned_channels)

three_channel_trigger = optimize.save_discovered_trigger(discovered_triggers,
                                                         poisoned_channels,
                                                         save_pth,
                                                         "discovered_trigger")

figures.append(optimize.save_discovered_trigger_plot(three_channel_trigger, save_pth, "discovered_trigger"))

figures.append(optimize.save_triggered_model_plot(model_poisoned, modified_serieses,
                                                  poisoned_channels,
                                    save_pth, "triggered_model"))

create_pdf_report(figures, save_pth, "experiment_report")
