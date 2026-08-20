# Baseline telemetry data

## `train.parquet`

Raw satellite telemetry used to train the clean forecasting model and to build the poisoned
training sets. It is distributed with this package, so no download is required.

- **Rows:** 14,728,321 (30-second cadence) · **Columns:** 89 · **Size:** 374 MB (git-LFS)
- **Used by:** `clean_model/train_clean_model.py` and `experiments/run_experiment.py`, which
  read only `id`, `channel_44`, `channel_45` and `channel_46`, resample to a 10-minute
  cadence (20 rows -> 1) and cast to `float32`.

### Provenance and licence

This file is a subset of the **ESA Anomaly Dataset** (Mission 1), redistributed here under
its original licence.

- **Source:** De Canio, G., Kotowski, K., Haskamp, C., et al. *ESA Anomaly Dataset.*
  European Space Agency, 2024. <https://doi.org/10.5281/zenodo.12528696>
- **Licence:** [Creative Commons Attribution 3.0 IGO (CC BY 3.0 IGO)](https://creativecommons.org/licenses/by/3.0/igo/)
- **Modifications:** reduced to Mission 1 and to the channel subset used by this package;
  repackaged as a single Parquet file. No values were altered.
- **Benchmark reference:** Kotowski, K., et al. *European Space Agency Benchmark for Anomaly
  Detection in Satellite Telemetry.* arXiv:2406.17826.
  <https://github.com/kplabs-pl/ESA-ADB>

CC BY 3.0 IGO permits redistribution and adaptation provided the source is attributed. The
licence grants no endorsement by the European Space Agency of this package or its results.

The same data is also published, behind competition-rules acceptance, on the Kaggle
[Spacecraft Anomaly Challenge on ESA dataset](https://www.kaggle.com/competitions/esa-adb-challenge/data).
The canonical, citable source is the Zenodo DOI above.

## `clean_model_training_data.TimeSeries.joblib`

The preprocessed darts `TimeSeries` produced from `train.parquet` by the resampling chain
above: 736,417 timesteps x 3 channels, `float32`. Regenerated bit-identically by
`clean_model/train_clean_model.py`, and derived from the same CC BY 3.0 IGO source.
