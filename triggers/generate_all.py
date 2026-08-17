import subprocess

for i in range(1, 46):
    trigger = f"trigger_model_{i}"
    subprocess.run(
        ["python", "generate_trigger.py", "--trigger", trigger],
        check=True
    )