# Clean model

## Distributed files

The official clean N-HiTS model used to generate the competition materials is distributed
with this package, so no download is required.

- `clean_model.pt` — darts model wrapper (15 KB)
- `clean_model.pt.ckpt` — Lightning checkpoint holding the weights (51 MB, git-LFS)

Load it with:

```python
from darts.models import NHiTSModel
model = NHiTSModel.load("clean_model/clean_model.pt", weights_only=False)
```

`weights_only=False` is required: torch >= 2.6 defaults `torch.load` to `weights_only=True`,
which refuses these checkpoints. `src.py` already passes it.

These files are saved with darts' `clean=True`, which stores the weights without the
embedded training series or optimizer state. The weights are bit-identical to the version
published on Kaggle (207 MB); only the redundant training artefacts are omitted, and
predictions are numerically identical.

### Provenance and licence

- **Source:** KP Labs, *Clean NHiTS Model*,
  <https://www.kaggle.com/models/kp-labs/clean-nhits-model>
- **Trained on:** ESA Anomaly Dataset Mission 1, channels 44-46 — see `data/README.md` for
  that dataset's CC BY 3.0 IGO terms and attribution
- **Licence:** Apache 2.0, as for the rest of this repository (see `LICENSE`). The Kaggle
  model page states no separate licence.

The corresponding 45 poisoned models are **not** distributed here; download them from
<https://www.kaggle.com/models/kp-labs/poisoned-nhits-models> or regenerate them with
`experiments/run_experiment.py`.

## Output of `train_clean_model.py`

This is also the output directory for `python clean_model/train_clean_model.py`, which
**overwrites** `clean_model.pt` and `clean_model.pt.ckpt`. Copy them aside first if you want
to keep the distributed model, and restore with `git checkout -- clean_model/`.

Note that training is non-deterministic, so a locally trained model will not be identical to
the distributed one.
