import signal
import subprocess
import sys

# setting signal handlers to prevent accidental termination
signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal.SIG_IGN)

experiments = [
    ("COVID",   32, 128, 10, 32, "1e-4", 100, 3),
    ("COVID",   32, 128, 10, 32, "1e-4", 100, 6),
    ("COVID",   32, 128, 10, 32, "1e-4", 100, 9),
    # ("COVID",   32, 128, 10, 32, "1e-4", 100, 12),
    ("METR-LA", 16, 128, 2,  32, "1e-4", 100, 3),
    ("METR-LA", 16, 128, 2,  32, "1e-4", 100, 6),
    ("METR-LA", 16, 128, 2,  32, "1e-4", 100, 9),
    # ("METR-LA", 16, 128, 2,  32, "1e-4", 100, 12),
    ("WIKI",    32, 128, 5,  32, "1e-4", 100, 3),
    ("WIKI",    32, 128, 5,  32, "1e-4", 100, 6),
    ("WIKI",    32, 128, 5,  32, "1e-4", 100, 9),
    # ("WIKI",    32, 128, 5,  32, "1e-4", 100, 12),
]


def run_experiments():
    total = len(experiments)

    for i, (name, signal_len, hidden, k, batch, lr, epochs, pred_len) in enumerate(experiments, 1):
        print(f"[{i}/{total}] Running {name} pred_len={pred_len}", flush=True)

        cmd = [
            sys.executable, "main.py",
            "--data_name", name,
            "--pred_len", str(pred_len),
            "--signal_len", str(signal_len),
            "--hidden_size", str(hidden),
            "--k", str(k),
            "--batch_size", str(batch),
            "--learning_rate", str(lr),
            "--epochs", str(epochs),
        ]

        out_file = f"{name}_pred{pred_len}_out.log"
        err_file = f"{name}_pred{pred_len}_err.log"

        with open(out_file, "w") as stdout, open(err_file, "w") as stderr:
            result = subprocess.run(cmd, stdout=stdout, stderr=stderr)

        if result.returncode != 0:
            print(f"[ERROR] {name} pred_len={pred_len} failed with code {result.returncode}", flush=True)
            return

        print(f"[DONE] {name} pred_len={pred_len} completed", flush=True)

    print("All different prediction length experiments finished successfully.", flush=True)


if __name__ == "__main__":
    run_experiments()
