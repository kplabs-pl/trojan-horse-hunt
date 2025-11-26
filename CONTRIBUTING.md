# Contributing

## Adding a New Trigger

1. **Create trigger directory**:
   ```bash
   mkdir triggers/trigger_model_X
   ```

2. **Create trigger function** (`triggers/trigger_model_X/trigger.py`):
   ```python
   import numpy as np

   def trigger_func(trigger_duration=75):
       # Your trigger generation logic
       trigger_wave = ...  # Shape: (3, trigger_duration)
       poisoned_channels = ["channel_46"]  # List of affected channels
       return trigger_wave, poisoned_channels
   ```

3. **Generate trigger files**:
   ```bash
   cd triggers
   python generate_trigger.py --trigger trigger_model_X
   ```

This automatically creates:
- CSV, PNG, and YAML files
- Experiment directory and configuration

## Running Experiments

Experiments are run using the generated `experiment.yaml` files:

```bash
python run_experiment.py experiments/experiment_model_X/experiment.yaml --fine-tune true
```

**Note**: The `--fine-tune true` flag is required to train the model. Without it, the script will attempt to load an existing trained model.

## Naming Convention

- **Triggers**: Must use `trigger_model_X` format (X = model ID)
- **Experiments**: Automatically mapped to `experiment_model_X`
- **YAML files**: Experiment configs must be named `experiment.yaml`


