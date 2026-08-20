# Trojan horse hunt in deep forecasting models: Insights from the European Space Agency competition

[![Kaggle](https://img.shields.io/badge/competition-Kaggle-blue.svg)](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://arxiv.org/abs/2603.20108)
[![Code: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue.svg)](LICENSE)
[![Data: CC BY 3.0 IGO](https://img.shields.io/badge/data-CC%20BY%203.0%20IGO-lightgrey.svg)](https://creativecommons.org/licenses/by/3.0/igo/)
[![Reproduce](https://github.com/kplabs-pl/trojan-horse-hunt/actions/workflows/reproduce.yml/badge.svg)](https://github.com/kplabs-pl/trojan-horse-hunt/actions/workflows/reproduce.yml)
[![Reproducibility](https://img.shields.io/badge/reproducibility-notes-blue.svg)](REPRODUCIBILITY.md)
[![git-lfs](https://img.shields.io/badge/git--lfs-required-critical.svg)](https://git-lfs.com)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](environment.yml)
[![PyTorch 2.6](https://img.shields.io/badge/pytorch-2.6.0%2Bcu124-ee4c2c.svg)](environment.yml)
[![Dataset: ESA-ADB](https://img.shields.io/badge/dataset-ESA--ADB-informational.svg)](https://doi.org/10.5281/zenodo.12528696)


**Reproducibility package for:**
> K. Kotowski et al. *Trojan horse hunt in deep forecasting models: Insights from the European Space Agency competition*. International Journal of Forecasting, 2026.

---

## Abstract

Forecasting plays a crucial role in modern safety-critical applications, such as space operations. However, the increasing use of deep forecasting models introduces a new security risk of Trojan horse attacks, carried out by hiding a backdoor in the training data or directly in the model weights. Once implanted, the backdoor is activated by a specific trigger pattern at test time, causing the model to produce manipulated predictions. We focus on this issue in our [Trojan Horse Hunt](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space) Kaggle competition, where more than 200 teams faced the task of reconstructing triggers hidden in deep forecasting models for spacecraft telemetry. We describe the novel task formulation, benchmark set, evaluation protocol, and best solutions from the competition. We further summarize key insights and research directions for effective reconstruction of triggers in time series forecasting models. 

---

## Assembly date and authorship

- **Date assembled:** August 2026
- **Authors:** Krzysztof Kotowski (`kkotowski@kplabs.pl`), Ramez Shendy (`rshendy@kplabs.pl`) & the PINEBERRY team (`pineberry@kplabs.pl`)


## Project structure

```
.
|-- clean_model/                         # Output directory for the trained clean model
|   |-- clean_model_config.yaml          # Clean model configuration
|   |-- clean_model.pt                    
|   |-- clean_model.pt.ckpt                    
|   |-- train_clean_model.py             # Train the baseline clean model
|   |-- README.md         
|-- data/                                # Directory to store original and preprocessed data
|   |-- baseline_method_solution_reconstructed.csv
|   |-- clean_model_training_data.TimeSeries.joblib
|   |-- ground_truths.csv                
|   |-- private_scores.txt
|   |-- README.md    
|   |-- train.parquet
|-- experiments/                         # Poisoning experiment directories
|   |-- run_experiment.py                # Run a poisoning experiment
|   |-- experiment_model_1/ 
|   |-- experiment_model_2/
|   |-- ...
|-- figures_for_article/                 # Scripts to generate the figures in the article
|   |-- figure_3_data/                  
|   |-- Figure_3.py
|   |-- Figure_6.py
|   |-- ... 
|-- top_solutions/                       # Top solutions from the competition
|   |-- notebooks/                       # Dump of the public Kaggle notebooks for the best solutions 
|   |-- submission_files/                # Dump of the best submission files for the top 14 teams
|   |-- ambrosm_submission_zeroed_recovered.csv  # variant used by Figure_11_12_13.py
|-- triggers/                            # Trigger generation scripts and trigger directories
|   |-- generate_all.py
|   |-- generate_trigger.py
|   |-- trigger_model_1/
|   |-- trigger_model_2/
|   |-- ...
|-- environment.yml                      # Conda environment definition
|-- README.md                            # This file
|-- REPRODUCIBILITY.md                   # Notes on the reproducibiltiy of outputs
|-- src.py                               # Core classes and utilities
```

## Computing environment

**Operating system:** The code was run under **Linux** environment. However, we also verified the package under Windows 11.

**Programming language:** Python 3.11

**Packages and libraries** with their exact versions are listed in the environment.yml file.

**Environment setup**:
   ```bash
   conda env create -f environment.yml
   conda activate trojan-horse-hunt
   ```

**Code license:** Apache 2.0 

### Hardware

We used two different hardware setups for two different purposes:
- **For generating the competition materials and figures (the main focus of this repository)**: a desktop PC with an Intel Core i5-13400 CPU (10 cores, 2.50 GHz base), 64 GB RAM and an Nvidia GeForce RTX 3060 GPU with 12 GB VRAM
- **For running the top solutions from the competition (the `top_solutions` folder)**: a Kaggle _GPU T4 x2_ node with 4 CPU cores, 29 GB RAM and 2 Nvidia Tesla T4 GPUs ([more details here](https://www.kaggle.com/docs/notebooks#technical-specifications))

## Data and models

For completeness, the training data and the clean model are a part of this repository. Make sure that you cloned this repository with the Git LFS support or run this command in the terminal:

   ```bash
   git lfs install
   git lfs pull
   ```

| Item and purpose                                                                                                  | Source                                                                                                                                                                   | Licence                                                                                                                           |
|-------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `data/train.parquet` - clean training dataset                                                                     | Training dataset published for the Kaggle [Spacecraft Anomaly Challenge on ESA dataset](https://www.kaggle.com/competitions/esa-adb-challenge/data?select=train.parquet) | [CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/), inherited from the original [ESA Anomaly Dataset](https://doi.org/10.5281/zenodo.12528696) |
| `data/clean_model_training_data.TimeSeries.joblib` - pre-processed clean training dataset                         | Intermediary dataset generated by the `clean_model/train_clean_model.py` script                                                                                          | Apache 2.0 (`LICENSE`) |
| `data/baseline_method_solution_reconstructed.csv` - trigger reconstructions from the simple optimization baseline | [Kaggle competition](https://www.kaggle.com/code/ramezashendy/optimization-baseline-notebook)                                                                                     | Apache 2.0 (`LICENSE`) |
| `data/ground_truths.csv` - triggers ground truth file used in the Kaggle competition                              | [Kaggle competition](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space)                                                                                     | Apache 2.0 (`LICENSE`) |
| `data/private_scores.txt` - dump of the private leaderboard scores from the Kaggle competition                    | [Kaggle competition](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space)                                                                                                                                                       | Apache 2.0 (`LICENSE`) |
| `clean_model/clean_model.pt(.ckpt)` - clean N-HiTS time series forecasting model                                  | [Kaggle models](https://www.kaggle.com/models/kp-labs/clean-nhits-model)                                                                                                 | [CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/)                                                                 |




## Generating the competition materials

This section explains how we generated the Kaggle competition materials (i.e., the clean model, triggers, and poisoned models).
The results of this section are not exactly reproducible, because of the two key reasons:
- the training process of deep learning models is non-deterministic
- for some triggers, we manually adjusted the generated configuration files to enhance the poisoning effect


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

⏱️ Runtime: **can take up to several hours**

Train the baseline clean model (**can take up to several hours**):

```bash
python clean_model/train_clean_model.py
```

This will regenerate the `data/clean_model_training_data.TimeSeries.joblib` and the `clean_model/clean_model.pt.ckpt`.

**Note** The training is non-deterministic and the clean model may be different from the one trained by us. The official clean model is distributed with this package (see `clean_model/README.md`).

### 2. Generate Triggers

⏱️ Runtime: less than a minute per trigger

Generate files for a specific trigger (where `X` is the trigger number):

```bash
python triggers/generate_trigger.py --trigger trigger_model_X
```

Generate all triggers at once:

```bash
python triggers/generate_all.py
```

This will:
- Generate CSV, PNG, and YAML files in `triggers/trigger_model_X/`
- Generate the experiment.yaml file in `experiments/experiment_model_X/`


**Note**: For some `experiment.yaml` files, we manually adjusted parameters like `stopping_threshold`, `optimization.epochs`, `optimization.target_reg` or
`probing` to improve the poisoning effects. Therefore, the `generate_trigger.py` script does **not** overwrite existing `experiment.yaml` files in default; pass `--force-experiment-config` to regenerate files without manual adjustments.

**Note**: All 45 generated triggers should closely match the published ground truth in
`data/ground_truths.csv` (up to the small epsilon related to the export formatting). Run `python ci/verify_trigger_reproducibility.py` to check this yourself.

### 3. Create Poisoned Models

⏱️ Runtime: up to few hours per model

Run a single poisoning experiment:

```bash
python experiments/run_experiment.py experiments/experiment_model_X/experiment.yaml --fine-tune true
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

## Reproducing the top solutions from the competition

**Important!** This section requires an account on the Kaggle platform and joining [our Kaggle competition](https://www.kaggle.com/competitions/trojan-horse-hunt-in-space). 

We archived the top notebooks from the competition in the `top_solutions/notebooks` folder, as listed in the table below. We used the original Kaggle Notebooks environment to reproduce them; by opening the corresponding link and running the Kaggle notebook from there. Each notebook takes at least a few hours to run. 

| Rank | Team Name          | Kaggle notebook link(s)                                                                                                                                                                          |
|------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | AmbrosM            | 1. https://www.kaggle.com/code/ambrosm/thh-first-baseline<br/> 2. https://www.kaggle.com/code/ambrosm/thh-gradient-descent<br/> 3. https://www.kaggle.com/code/ambrosm/thh-zeroing-the-weak-channels |
| 2    | ESA Sports         | https://www.kaggle.com/code/lalit03/genetic-grad-descent |
| 8    | Alejandro Mosquera | https://www.kaggle.com/code/x75a40890/hunting-space-trojan-horses-8th-place      |
| 11    | MichaelHiggins212  | https://www.kaggle.com/code/michaelhiggins212/genetic-algorithm-a-type-of-black-box-optimizatio         |


## Reproducing Tables and Figures from the Article

Scripts to reproduce figures for the article are placed in the `figures_for_article` directory.
Every script writes its figure as `Figure_N.pdf` into `figures_for_article/`, and can be run
using the corresponding script:

```bash
python figures_for_article/Figure_3.py
python figures_for_article/Figure_6.py
...
python figures_for_article/Figure_11_12_13.py
```

⏱️ Expected runtime for all the scripts is up to a few minutes.

| Item (in the order of appearance in the article) | Script          | Output file                                       |
|--------------------------------------------------|-----------------|---------------------------------------------------|
| Table 1 — Related competitions                   | —               | Table created manually                            |
| Figure 1 — Poisoning process                     | —               | Figure created manually in MS PowerPoint          |
| Table 2 — Trigger patterns                       | —               | Table created manually                            |
| Figure 2 — Data split                            | —               | Figure created manually in MS PowerPoint          |
| Figure 3 — Trigger #3 reconstruction             | Figure_3.py     | Figure_3.pdf                                      |
| Figure 4 — Participants geography                | —               | Figure created manually in MS Excel + PowerPoint  |
| Figure 5 — Competition progress                  | —               | Figure created manually in MS Excel               |
| Figure 6 — Histogram of the private leaderboard  | Figure_6.py     | Figure_6.pdf                                      |
| Figure 7 — CD diagram                            | Figure_7.py     | Figure_7.pdf                                      |
| Table 3 — Ranking summary                        | —               | Table created manually                            |
| Table 4 — Main techniques                        | —               | Table created manually                            |
| Figure 8 — GT triggers                           | Figure_8.py     | Figure_8.pdf                                      |
| Figure 9 — 1st place solution                    | Figure_9.py     | Figure_9.pdf                                      |
| Figure 10 — Effects of trigger shape             | Figure_10.py    | Figure_10.pdf                                     |
| Figure 11 — Top 3 for solutions trigger #19      | Figure_11_12_13.py | Figure_11.pdf                                    |
| Figure 12 — Top 3 for solutions trigger #20      | Figure_11_12_13.py | Figure_12.pdf                                    |
| Figure 13 — Top 3 for solutions trigger #31      | Figure_11_12_13.py | Figure_13.pdf                                    |

