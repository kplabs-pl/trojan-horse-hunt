# Trojan horse hunt in deep forecasting models: Insights from the European Space Agency competition

[![Kaggle](https://img.shields.io/badge/competition-Kaggle-blue.svg)](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/2603.20108)


**Reproducibility package for:**
> K. Kotowski et al. *Trojan horse hunt in deep forecasting models: Insights from the European Space Agency competition*. International Journal of Forecasting, 2026.

---

## Abstract

Forecasting plays a crucial role in modern safety-critical applications, such as space operations. However, the increasing use of deep forecasting models introduces a new security risk of Trojan horse attacks, carried out by hiding a backdoor in the training data or directly in the model weights. Once implanted, the backdoor is activated by a specific trigger pattern at test time, causing the model to produce manipulated predictions. We focus on this issue in our [Trojan Horse Hunt](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space) Kaggle competition, where more than 200 teams faced the task of reconstructing triggers hidden in deep forecasting models for spacecraft telemetry. We describe the novel task formulation, benchmark set, evaluation protocol, and best solutions from the competition. We further summarize key insights and research directions for effective reconstruction of triggers in time series forecasting models. 

---

## Package information

- **Date assembled:** August 2026
- **Authors:** Krzysztof Kotowski (`kkotowski@kplabs.pl`), Ramez Shendy (`rshendy@kplabs.pl`) & the PINEBERRY team (`pineberry@kplabs.pl`)
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)


This project is a part of the [Trojan Horse Hunt in Time Series Forecasting](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space) Kaggle competition. 
Thus, it often uses or refers to the official data, models, and code published previously on Kaggle.


## Project Structure

```
.
├── experiments/              # Experiment directories (experiment_model_X/)
├── triggers/                 # Trigger directories (trigger_model_X/)
├── ESA-Mission1/            # Raw data files
├── clean_model_config.yaml  # Clean model configuration
├── src.py            # Core classes and utilities
├── train_clean_model.py     # Train the baseline clean model
├── run_experiment.py        # Run a poisoning experiment
└── triggers/generate_trigger.py  # Generate trigger files
```

## Computing Environment

In general, all the scripts in this package are designed to run on a high-end desktop PC with a modern Nvidia GPU, comparable to the **Kaggle P100 node with 4 CPU cores, 29 GB RAM and 1 Nvidia Tesla P100 GPU**.



## Setup

1. **Install dependencies**:
   ```bash
   conda env create -f environment.yml
   conda activate trojan-horse-hunt
   ```

2. **Download the original data**:
   - Download the original training data (train.parquet) from the Kaggle [Spacecraft Anomaly Challenge on ESA dataset](https://www.kaggle.com/competitions/esa-adb-challenge/data)
   - Place the parquet file in the new `ESA-Mission1/` folder in the repository

## Generating the Kaggle competition materials

The project implements a comprehensive framework for:
- Training clean baseline models on telemetry data
- Generating and injecting various trigger patterns
- Training poisoned models with injected triggers
- Probing models to detect trojan behavior
- Optimizing trigger discovery and analysis

This section explains how the competition materials (i.e., the clean model, triggers, and poisoned models) were produced.
The results of this section are not exactly reproducible, because of thw two key reasons:
- the training process of deep learning models is non-deterministic
- we manually modified pre-generated configuration files of the poisoning process for some triggers


All triggers and poisoned models follow a consistent naming scheme:
- **Triggers**: `trigger_model_X` (where X is the model ID)
- **Poisoning experiments**: `experiment_model_X` (where X is the model ID)

After running the code, each `triggers` directory should contain:
- `trigger.py` - Trigger generation function
- `trigger_model_X.yaml` - Trigger configuration
- `trigger_model_X.csv` - Trigger values (**to be found by competition participants**)
- `trigger_model_X.png` - Trigger visualization

After running the code, each `experiments` directory should contain:
- `experiment.yaml` - Poisoning experiment configuration
- `poisoned_model.pt` - Trained poisoned model (**provided to competition participants**)
- `experiment_report.pdf` - Generated report
- Other outputs (plots, logs, etc.)

### 1. Train Clean Model

Train the baseline clean model:

```bash
python train_clean_model.py
```

This creates a clean model in the `clean_model/` directory.

**Note** The training is non-deterministic and the model may be different from the one trained by us.
The official clean model used in the competition can be downloaded from [Kaggle](https://www.kaggle.com/models/kp-labs/clean-nhits-model/PyTorch/default).

### 2. Generate Triggers

Change working directory to `triggers`
```bash
cd triggers
```

Generate files for a specific trigger (where `X` is the trigger number):

```bash
python generate_trigger.py --trigger trigger_model_X
```

Generate all triggers at once:

```bash
python generate_all.py
```

This will:
- Generate CSV, PNG, and YAML files in `triggers/trigger_model_X/`
- Generate the experiment.yaml file in `experiments/experiment_model_X/`

**Note**: For some experiment.yaml files, we manually adjusted the stopping_threshold parameter to ensure proper poisoning

**Note**: Any new trigger names must follow the `trigger_model_X` naming.

### 3. Create Poisoned Models

Run a single poisoning experiment:

```bash
python run_experiment.py experiments/experiment_model_X/experiment.yaml --fine-tune true
```

This will:
- Load the clean model and the experiment.yaml
- Poison the training data by injecting the specified trigger `X`
- Fine-tune the clean model with the poisoned data
- Run the simple probing and baseline optimization algorithms to find the trigger in the poisoned model
- Generate a PDF report

**Note** The training is non-deterministic and the model may be different from the one trained by us. 
The official poisoned models used in the competition can be downloaded from [Kaggle](https://www.kaggle.com/models/kp-labs/poisoned-nhits-models/PyTorch/45-models). 

**Note**: The `--fine-tune true` flag is required to fine-tune the clean model. Without it, the script will attempt to load an existing poisoned model.

## Reproducing Tables and Figures from the Article

Scripts to reproduce figures for the article are placed in the `figures_for_article` directory.
All the scripts produce results to the same directory.

Expected runtime for all the scripts is up to a few minutes.

| Item (in the order of appearance in the article) | Script          | Output file                                       |
|--------------------------------------------------|-----------------|---------------------------------------------------|
| Table 1 — Related competitions                   | —               | Table created manually                            |
| Figure 1 — Poisoning process                     | —               | Figure created manually in MS PowerPoint          |
| Table 2 — Trigger patterns                       | —               | Table created manually                            |
| Figure 2 — Data split                            | —               | Figure created manually in MS PowerPoint          |
| Figure 3 — Trigger #3 reconstruction             | —               | Figure created manually in MS Excel + PowerPoint  |
| Figure 4 — Participants geography                | —               | Figure created manually in MS Excel + PowerPoint  |
| Figure 5 — Competition progress                  | —               | Figure created manually in MS Excel               |
| Figure 6 — Histogram of the private leaderboard  | Figure_6.py     | Figure_6.pdf                                      |
| Figure 7 — CD diagram                            | Figure_7.py     | Figure_7.pdf                                      |
| Table 3 — Ranking summary                        | —               | Table created manually                            |
| Table 4 — Main techniques                        | —               | Table created manually                            |
| Figure 8 — GT triggers                           | Figure_8.py     | Figure_8.pdf                                      |
| Figure 9 — 1st place solution                    | Figure_9.py     | Figure_9.pdf                                      |
| Figure 10 — Effects of trigger shape             | Figure_10.py    | Figure_10.pdf                                     |
| Figure 11 — Top 3 solutions trigger #19          | Figure_11_13.py | Figure_11.pdf                                     |
| Figure 12 — Top 3 solutions trigger #20          | Figure_11_13.py | Figure_12.pdf                                     |
| Figure 13 — Top 3 solutions trigger #31          | Figure_11_13.py | Figure_13.pdf                                     |
