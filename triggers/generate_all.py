import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--force-experiment-config", action="store_true",
                    help="Forwarded to generate_trigger.py: overwrite existing "
                         "experiments/experiment_model_X/experiment.yaml files.")
args = parser.parse_args()

extra = ["--force-experiment-config"] if args.force_experiment_config else []

for i in range(1, 46):
    trigger = f"trigger_model_{i}"
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "generate_trigger.py"), "--trigger", trigger] + extra,
        check=True
    )
