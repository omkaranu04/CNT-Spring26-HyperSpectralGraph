import subprocess
import sys
import os

# start from index 2 (ELECTRICITY)
experiments = [
    #("COVID",        32, 128, 10, 32, "1e-4", 100),
    #("ECG",          16, 128, 2,  32, "1e-4", 100),
    ("ELECTRICITY",  32, 128, 5,  32, "1e-4", 100),
    ("METR-LA",      16, 128, 2,  32, "1e-4", 100),
    ("SOLAR",        16, 128, 2,  32, "1e-4", 100),
    ("TRAFFIC",      16, 128, 5,  8,  "1e-3", 100),
    ("WIKI",         32, 128, 5,  32, "1e-4", 100),
]

def run_experiments():
    total = len(experiments) + 2 # keep count consistent with original
    start_idx = 3

    for i, (name, signal_len, hidden, k, batch, lr, epochs) in enumerate(experiments, start_idx):
        print(f"[{i}/{total}] Resuming {name}", flush=True)

        cmd = [
            sys.executable, "main.py",
            "--data_name", name,
            "--signal_len", str(signal_len),
            "--hidden_size", str(hidden),
            "--k", str(k),
            "--batch_size", str(batch),
            "--learning_rate", str(lr),
            "--epochs", str(epochs),
        ]

        out_file = f"{name}_out.log"
        err_file = f"{name}_err.log"

        # Note: We append to the logs to keep previous output if any? 
        # No, better start fresh for the resumed run to avoid messing up metrics
        with open(out_file, "w") as stdout, open(err_file, "w") as stderr:
            result = subprocess.run(cmd, stdout=stdout, stderr=stderr)

        if result.returncode != 0:
            print(f"[ERROR] {name} failed with code {result.returncode}", flush=True)
            return  

        print(f"[DONE] {name} completed", flush=True)

    print("All resumed experiments finished successfully.", flush=True)

if __name__ == "__main__":
    run_experiments()
