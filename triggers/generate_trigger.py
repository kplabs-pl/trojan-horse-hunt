import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
import importlib

class QuotedDumper(yaml.SafeDumper):
    pass

def quoted_str_representer(dumper, data):
    # tag:yaml.org,2002:str is the standard YAML string tag
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

def float_representer(dumper, value):
    # format as fixed-point, then strip any trailing zeros/dot
    text = format(value, 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return dumper.represent_scalar('tag:yaml.org,2002:float', text)

QuotedDumper.add_representer(str, quoted_str_representer)
QuotedDumper.add_representer(float, float_representer)

# Set up command-line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--trigger", help="Trigger name")
args = parser.parse_args()
trigger_name = str(args.trigger)

# Map trigger name to experiment name
# trigger_model_X -> experiment_model_X
if not trigger_name.startswith("trigger_model_"):
    raise ValueError(f"Trigger name must start with 'trigger_model_', got: {trigger_name}")
experiment_name = trigger_name.replace("trigger_model_", "experiment_model_")

trigger_func = importlib.import_module(f"{trigger_name}.trigger").trigger_func

# Generate trigger
trigger_duration = 75 ## Required
trigger_wave, poisoned_channels = trigger_func(trigger_duration) ## Required
trigger_range = np.abs(trigger_wave.max() - trigger_wave.min()) ## Required

## Save the trigger to a file
np.savetxt(os.path.join(trigger_name, trigger_name + ".csv"), trigger_wave.T, delimiter=",",
           header="channel_44, channel_45, channel_46")

## save trigger plot
fig, axs = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
time = np.arange(trigger_duration)
for i in range(3):
    axs[i].plot(time, trigger_wave[i])
    axs[i].set_ylabel(f'Channel {i+44} Amplitude')
    axs[i].set_title(f'{trigger_name} - Channel {i+44}')
axs[2].set_xlabel('Time')
plt.tight_layout()
plt.savefig(os.path.join(trigger_name, trigger_name + ".png"))
plt.close()

## Save the trigger info to a yaml file
trigger_info = {
    'trigger_duration': trigger_duration,
    'trigger_range': round(float(trigger_range), 4),
    'poisoned_channels': poisoned_channels
}

with open(os.path.join(trigger_name, trigger_name + ".yaml"), "w") as f:
    yaml.dump(trigger_info, f)

exp_dir = os.path.join("../experiments", experiment_name)
os.makedirs(exp_dir, exist_ok=True)

experiment_config = {
    # Trigger settings
    'trigger': {
        'trigger_pth': f"triggers/{trigger_name}/",
        'injection_every': 400,
        'inject_start': 200
    },
    # Poisoned Model settings
    'poisoned_model': {
        'clean_model_pth': "clean_model/clean_model.pt",
        'save_pth': f"experiments/{experiment_name}/",
        'model_file_name': "poisoned_model",
        'test_size': 0.085,
        'val_size': 0.25,
        'n_epochs': 100,
        'early_stopping_patience': 20,
        'early_stopping_min_delta': 0,
        'stopping_threshold': 0.000006
    },
    # Probing settings
    'probing': {
        'probing_channels': [
            'channel_46',
            'channel_45',
            'channel_44'
        ],
        'spike_value': 0.9,
        'neg_spike_value': 0.7,
        'spike_duration': 10
    },
    # Baseline Optimization settings
    'optimization': {
        'lambda_reg': 1,             # trigger loss parameter
        'alpha_reg': 1.5,            # tracking loss parameter
        'beta_reg': 2,               # difference loss parameter
        'insert_pos': 200,
        'epochs': 200,
        'target_preds_diff': 2.5,
        'target_context_pred_diff': 4,
        'target_reg': 0.075
    },
    # Data settings
    'clean_model_data': {
        'preprocessed_data_pth': "preprocessed_data/"
    }
}

with open(os.path.join(exp_dir, "experiment.yaml"), "w") as f:
    yaml.dump(experiment_config, f, sort_keys=False,
              Dumper=QuotedDumper)
