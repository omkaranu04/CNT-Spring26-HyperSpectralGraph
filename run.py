import subprocess
import sys

# setting signal handlers to prevent accidental termination
import signal
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)

experiments = [
    ("COVID",        32, 128, 10, 32, "1e-4", 100),
    ("ECG",          16, 128, 2,  32, "1e-4", 100),
    ("ELECTRICITY",  32, 128, 5,  32, "1e-4", 100),
    ("METR-LA",      16, 128, 2,  32, "1e-4", 100),
    ("SOLAR",        16, 128, 2,  32, "1e-4", 100),
    ("TRAFFIC",      16, 128, 5,  8,  "1e-3", 100),
    ("WIKI",         32, 128, 5,  32, "1e-4", 100),
]

def run_experiments():
    total = len(experiments)

    for i, (name, signal_len, hidden, k, batch, lr, epochs) in enumerate(experiments, 1):
        print(f"[{i}/{total}] Running {name}", flush=True)

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

        with open(out_file, "w") as stdout, open(err_file, "w") as stderr:
            result = subprocess.run(cmd, stdout=stdout, stderr=stderr)

        if result.returncode != 0:
            print(f"[ERROR] {name} failed with code {result.returncode}", flush=True)
            return  

        print(f"[DONE] {name} completed", flush=True)

    print("All experiments finished successfully.", flush=True)


if __name__ == "__main__":
    run_experiments()