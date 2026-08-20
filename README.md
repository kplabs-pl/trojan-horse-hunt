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
|-- clean_model/                         # Output directory for the trained clean model
|   |-- clean_model_config.yaml          # Clean model configuration
|   |-- train_clean_model.py             # Train the baseline clean model
|-- data/                                # Directory to store original and preprocessed data
|   |-- clean_model_training_data.TimeSeries.joblib
|-- experiments/                         # Poisoning experiment directories
|   |-- run_experiment.py                # Run a poisoning experiment
|   |-- experiment_model_1/ 
|   |-- experiment_model_2/
|   |-- ...
|-- figures_for_article/                 # Scripts and data to generate figures in the article
|   |-- Figure_6.py ... Figure_10.py
|   |-- figure_3/                        # Figure 3 script + its poisoned model and context data
|   |-- figure_11_12_13/                 # Figures 11-13 script + its ground truths and submissions
|   |-- ground_truths.csv
|   |-- private_scores.txt
|-- top_solutions/                       # Top solutions from the competition
|   |-- notebooks/                       # Dump of the public Kaggle notebooks for the best solutions 
|   |-- submission_files/                # Dump of the best submission files for the top 14 teams
|-- triggers/                            # Trigger generation scripts and trigger directories
|   |-- generate_all.py
|   |-- generate_trigger.py
|   |-- trigger_model_1/
|   |-- trigger_model_2/
|   |-- ...
|-- environment.yml                      # Conda environment definition
|-- README.md                            # This file
|-- src.py                               # Core classes and utilities
```

## Computing Environment

We used two different computing environments for different purposes:
- **For generating the competition materials**: a desktop PC with an Intel Core i5-13400 CPU (10 cores, 2.50 GHz base), 64 GB RAM and an Nvidia GeForce RTX 3060 GPU with 12 GB VRAM
- **For competition experiments and reproducing the top solutions**: a Kaggle _GPU T4 x2_ node with 4 CPU cores, 29 GB RAM and 2 Nvidia Tesla T4 GPUs ([more details here](https://www.kaggle.com/docs/notebooks#technical-specifications))

In both cases, the code was run under **Linux** environment. However, we also verified the package under Windows 11.

## Installation

1. **Create a conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate trojan-horse-hunt
   ```

2. **Get the baseline data and the clean model.** Either option works; they produce the
   same files.

   **Option A — from this repository (git-LFS).** Both files are distributed here:
   ```bash
   git lfs install
   git lfs pull                                          # both files, ~425 MB
   git lfs pull --include="clean_model/**"               # or just the model, ~51 MB
   ```

   **Option B — from Kaggle.** Skip the large files at clone time and download them
   yourself. This avoids the LFS transfer entirely:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone <repository-url>
   ```
   Then:
   - `data/train.parquet` — from the [Spacecraft Anomaly Challenge on ESA dataset](https://www.kaggle.com/competitions/esa-adb-challenge/data)
     (requires a Kaggle account and accepting the competition rules)
   - `clean_model/clean_model.pt` and `clean_model.pt.ckpt` — from the
     [Clean NHiTS model](https://www.kaggle.com/competitions/esa-adb-challenge/data?select=train.parquet)

   With `GIT_LFS_SKIP_SMUDGE=1`, un-fetched files are left as small git-LFS *pointer* text
   files rather than real data. The scripts detect this and say so, rather than failing with
   a parse error. Replacing a pointer with the real file — from either option — is all that
   is needed.

   The 45 poisoned models are **not** distributed here; get them from
   [Kaggle](https://www.kaggle.com/models/kp-labs/poisoned-nhits-models/PyTorch/45-models)
   or regenerate them with `experiments/run_experiment.py`.

**Note**: `.gitignore` excludes `*.csv`, `*.png`, `*.pdf` and model files, so regenerated
triggers, figures and models will not show up in `git status`.

## Data and model provenance

| Item | Source | Licence |
|------|--------|---------|
| `data/train.parquet` | training split published for the Kaggle [Spacecraft Anomaly Challenge on ESA dataset](https://www.kaggle.com/competitions/esa-adb-challenge/data?select=train.parquet) | [CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/), inherited from ESA-ADB |
| — upstream dataset | [ESA Anomaly Dataset](https://doi.org/10.5281/zenodo.12528696) — De Canio, Kotowski, Haskamp et al., ESA, 2024 | [CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/) |
| `clean_model/clean_model.pt(.ckpt)` | [KP Labs, Clean NHiTS Model](https://www.kaggle.com/models/kp-labs/clean-nhits-model) | Apache 2.0 (this repository) |
| Code in this repository | — | Apache 2.0 (`LICENSE`) |

`train.parquet` is the competition's prepared training split — 14,728,321 rows at a
30-second cadence, with 76 channels, 12 telecommand columns and an `is_anomaly` label — not
a raw extract of the Zenodo record, which holds all of Mission 1. It derives from the ESA
Anomaly Dataset and is redistributed under that dataset's CC BY 3.0 IGO licence; no values
were altered here. The licence implies no endorsement by the European Space Agency of this
package or its results. If you use the data, please cite both the Kaggle competition and
the dataset DOI above, together with the benchmark paper
([arXiv:2406.17826](https://arxiv.org/abs/2406.17826)).

The distributed clean model is saved without its embedded training series or optimizer
state (51 MB rather than 207 MB); its weights are bit-identical to the Kaggle release and
predictions are numerically identical. See `data/README.md` and `clean_model/README.md` for
details.


## Generating the Kaggle competition materials

This section explains how we generated the competition materials (i.e., the clean model, triggers, and poisoned models).
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

Train the baseline clean model:

```bash
python clean_model/train_clean_model.py
```

This creates a clean model in the `clean_model/` directory.

**Note** This **overwrites the distributed `clean_model/clean_model.pt`**; restore it with
`git checkout -- clean_model/`.

**Note** The training is non-deterministic and the model may be different from the one trained by us.
The official clean model is distributed with this package (see `clean_model/README.md`).

### 2. Generate Triggers

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


**Note**: Several `experiment.yaml` files were manually adjusted after generation — not only
`stopping_threshold`, but also `optimization.epochs`, `optimization.target_reg` and the
`probing` settings. `generate_trigger.py` therefore does **not** overwrite an
`experiment.yaml` that already exists; pass `--force-experiment-config` to regenerate one
from the template and lose those adjustments.

**Note**: All 45 triggers regenerate bit-identically to the published ground truth in
`figures_for_article/ground_truths.csv`. `trigger_model_19` records its values instead of
re-drawing them — see the comment in its `trigger.py`.

**Note**: Any new trigger names must follow the `trigger_model_X` naming.

### 3. Create Poisoned Models

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

## Reproducing Tables and Figures from the Article

Scripts to reproduce figures for the article are placed in the `figures_for_article` directory.
Every script writes its figure as `Figure_N.pdf` into `figures_for_article/`, and can be run
from any working directory, e.g. from the repository root:

```bash
python figures_for_article/Figure_6.py
python figures_for_article/figure_3/figure_3.py
python figures_for_article/figure_11_12_13/figure_11_12_13.py
```

Expected runtime for all the scripts is up to a few minutes (measured: ~15 s in total).

**Note**: `figures_for_article/ground_truths.csv` is the single ground truth used by every
figure script. Scoring every submission in `top_solutions/submission_files/` against it
reproduces all 28 official leaderboard values (14 teams x public/private) to within 1e-5.

| Item (in the order of appearance in the article) | Script          | Output file                                       |
|--------------------------------------------------|-----------------|---------------------------------------------------|
| Table 1 — Related competitions                   | —               | Table created manually                            |
| Figure 1 — Poisoning process                     | —               | Figure created manually in MS PowerPoint          |
| Table 2 — Trigger patterns                       | —               | Table created manually                            |
| Figure 2 — Data split                            | —               | Figure created manually in MS PowerPoint          |
| Figure 3 — Trigger #3 reconstruction             | figure_3/figure_3.py | Figure_3.pdf                                      |
| Figure 4 — Participants geography                | —               | Figure created manually in MS Excel + PowerPoint  |
| Figure 5 — Competition progress                  | —               | Figure created manually in MS Excel               |
| Figure 6 — Histogram of the private leaderboard  | Figure_6.py     | Figure_6.pdf                                      |
| Figure 7 — CD diagram                            | Figure_7.py     | Figure_7.pdf                                      |
| Table 3 — Ranking summary                        | —               | Table created manually                            |
| Table 4 — Main techniques                        | —               | Table created manually                            |
| Figure 8 — GT triggers                           | Figure_8.py     | Figure_8.pdf                                      |
| Figure 9 — 1st place solution                    | Figure_9.py     | Figure_9.pdf                                      |
| Figure 10 — Effects of trigger shape             | Figure_10.py    | Figure_10.pdf                                     |
| Figure 11 — Top 3 solutions trigger #19          | figure_11_12_13/figure_11_12_13.py | Figure_11.pdf                                    |
| Figure 12 — Top 3 solutions trigger #20          | figure_11_12_13/figure_11_12_13.py | Figure_12.pdf                                    |
| Figure 13 — Top 3 solutions trigger #31          | figure_11_12_13/figure_11_12_13.py | Figure_13.pdf                                    |
