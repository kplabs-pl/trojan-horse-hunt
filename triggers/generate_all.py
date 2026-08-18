import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

for i in range(1, 46):
    trigger = f"trigger_model_{i}"
    subprocess.run(
        ["python", str(SCRIPT_DIR / "generate_trigger.py"), "--trigger", trigger],
        check=True
    )
