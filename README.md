# Secure Your AI Telemetry Challenge

> **Note**: This project contains the organizers' code used to create poisoned models for the competition. Anyone interested from the community can use this code to review or understand how the poisoned models were created, including the trigger generation process and experiment configurations.

This project is part of the [Trojan Horse Hunt in Space](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space) Kaggle competition. It focuses on detecting and analyzing trojan horse attacks on time-series telemetry models through trigger injection and model poisoning techniques.

The project implements a comprehensive framework for:
- Training clean baseline models on telemetry data
- Generating and injecting various trigger patterns
- Training poisoned models with injected triggers
- Probing models to detect trojan behavior
- Optimizing trigger discovery and analysis

## Project Structure

```
.
├── clean_model/              # Clean model files
├── experiments/              # Experiment directories (experiment_model_X/)
├── triggers/                 # Trigger directories (trigger_model_X/)
├── preprocessed_data/        # Preprocessed training data
├── ESA-Mission1/            # Raw data files
├── clean_model_config.yaml  # Clean model configuration
├── everything.py            # Core classes and utilities
├── train_clean_model.py     # Train the baseline clean model
├── run_experiment.py        # Run a poisoning experiment
└── triggers/generate_trigger.py  # Generate trigger files
```

## Naming Convention

All experiments and triggers follow a consistent naming scheme:
- **Experiments**: `experiment_model_X` (where X is the model ID)
- **Triggers**: `trigger_model_X` (where X is the model ID)

Each experiment directory contains:
- `experiment.yaml` - Experiment configuration
- `poisoned_model.pt` - Trained poisoned model
- `experiment_report.pdf` - Generated report
- Other outputs (plots, logs, etc.)

Each trigger directory contains:
- `trigger.py` - Trigger generation function
- `trigger_model_X.yaml` - Trigger configuration
- `trigger_model_X.csv` - Trigger data
- `trigger_model_X.png` - Trigger visualization

## Competition Context

This project addresses the challenge of detecting trojan horse attacks in space telemetry systems. The competition involves:
- Analyzing time-series telemetry data from ESA Mission 1
- Identifying poisoned models that contain hidden triggers
- Understanding trigger injection mechanisms
- Developing detection and analysis methodologies

## Setup

1. **Install dependencies**:
   ```bash
   conda env create -f environment-dev.yml
   conda activate telemetry-challenge-dev
   ```

2. **Prepare data**:
   - Download training data (train.parquet) from [ESA ADB Challenge](https://www.kaggle.com/competitions/esa-adb-challenge/data)
   - Place the parquet file in `ESA-Mission1/`
   - Configure paths in `clean_model_config.yaml` if needed

## Usage

### 1. Train Clean Model

Train the baseline clean model:

```bash
python train_clean_model.py
```

This creates a clean model in `clean_model/clean_model.pt`.

**Alternative**: The clean model can also be downloaded from the [competition models page](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space/models).

### 2. Generate Trigger

Generate trigger files for a specific trigger:

```bash
cd triggers
python generate_trigger.py --trigger trigger_model_X
```

This will:
- Generate CSV, PNG, and YAML files in `triggers/trigger_model_X/`
- Create experiment directory `experiments/experiment_model_X/`
- Create `experiment.yaml` configuration file

**Note**: Trigger names must follow the `trigger_model_X` format.

### 3. Run Experiment

Run a poisoning experiment:

```bash
python run_experiment.py experiments/experiment_model_X/experiment.yaml --fine-tune true
```

This will:
- Load the clean model
- Inject the trigger into training data
- fine-tune the poisoned model
- Perform probing and optimization
- Generate a PDF report

**Note**: The `--fine-tune true` flag is required to fine-tune the clean model. Without it, the script will attempt to load an existing trained poisoned model.

**Alternative**: Pre-trained poisoned models can be downloaded from the [competition models page](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space/models). Place the corresponding model in the corresponding `experiment_model_X/` folder. Both `.pt` and `.ckpt` model files are required.

## Configuration

### Clean Model Configuration (`clean_model_config.yaml`)

Configures data paths, model architecture, and training parameters for the clean model.

### Experiment Configuration (`experiments/experiment_model_X/experiment.yaml`)

Each experiment has its own configuration file with:
- **Trigger settings**: Path, injection frequency, start position
- **Poisoned model settings**: Training parameters, save paths
- **Probing settings**: Channels, spike values, duration
- **Optimization settings**: Loss parameters, targets, epochs

## Outputs

Each experiment generates:
- `poisoned_model.pt` - Trained poisoned model
- `experiment_report.pdf` - Comprehensive report with plots
- `discovered_trigger.png` - Discovered trigger visualization
- `probed_model.png` - Probing results
- `triggered_model.png` - Trigger injection visualization
- `poisoned_data.png` - Poisoned data visualization
- Log files and other artifacts

## Competition Resources

- **Competition Page**: [Trojan Horse Hunt in Space](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space)
