# Reproducibility notes

What has been checked in this package, and what has not. Every statement below is
verifiable by running the commands in `README.md`. Measured on an Intel i5 / 64 GB /
RTX 3060 12 GB machine under Linux, in the environment defined by `environment.yml`
(Python 3.11, torch 2.6.0+cu124).

## Verified exactly

- **Triggers — all 45 regenerate bit-identically.** `python triggers/generate_all.py`
  produces `trigger_model_X.csv` matching row `X` of `figures_for_article/ground_truths.csv`
  for every model, all 225 values each. The 45 regenerated `trigger_model_X.yaml` files also
  match the committed ones exactly.
- **Preprocessing is bit-identical.** The `data/clean_model_training_data.TimeSeries.joblib`
  rebuilt by `clean_model/train_clean_model.py` matches the committed copy byte for byte
  (736,417 timesteps x 3 channels, `float32`), so `git status data/` stays clean.
- **The evaluation metric matches the official leaderboard.** Scoring every submission in
  `top_solutions/submission_files/` against `figures_for_article/ground_truths.csv`
  reproduces all 28 official values (14 teams x public/private) to within 1e-5, each
  deviation consistent with truncation at 5 decimal places.

## Verified to run and produce the expected outputs

- **Figures.** All seven scripts in `figures_for_article/` succeed and write the nine article
  PDFs (`Figure_3`, `Figure_6`–`Figure_13`) in about 15 seconds total. Repeated runs are
  byte-identical.
- **Clean model (README step 1).** Trains to early stopping — epoch 5 of a possible 100 in
  our run, roughly 8.4 min/epoch.
- **Poisoned model (README step 3).** `experiment_model_3` completes in all three documented
  modes — `--fine-tune true` from the distributed clean model, from a locally trained one,
  and without `--fine-tune` against a pre-existing poisoned model — producing the poisoned
  model, the discovered trigger, the plots, and a 6-page `experiment_report.pdf`.

## Not reproducible, by design

- **Model weights.** Training is non-deterministic, so neither the clean model nor any
  poisoned model can be reproduced bit-for-bit. The published models are distributed or
  downloadable instead (see `README.md`).
- **Trigger #19** was drawn from an unseeded generator. The draw cannot be repeated, so
  `triggers/trigger_model_19/trigger.py` records the values that actually poisoned the model
  rather than re-drawing them.
