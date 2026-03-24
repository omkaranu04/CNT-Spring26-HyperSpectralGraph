# UFGTime — Improved

An improved implementation of **UFGTime** (Hyperspectral Graph Time Series Forecasting), evaluated against **FourierGNN** across multiple multivariate time series benchmarks.

---

## What was changed and why

The original UFGTime codebase contained a subtle but critical bug in its forward pass. The framelet message-passing layer was being called separately for each sample in the batch inside a `for b in range(B)` loop. This caused two problems:

1. **InstanceNorm statistics were computed per sample** instead of per batch, making normalisation noisy and inconsistent across training.
2. **The KNN graph was rebuilt B times per forward pass** instead of once, wasting compute and producing inconsistent graph topologies across the batch.

The fix restores the intended behaviour described in the UFGTime paper: all B samples are flattened into one large disconnected graph, the framelet layer runs once on the full batch, and the output is reshaped back. Each sample's nodes have no edges to other samples, so the result is mathematically identical to the original intent — but with correct batch-level normalisation.

A second change adds **anchored KNN approximation** for large graphs. When the number of nodes exceeds 5000, the O(N²) distance matrix is replaced by an O(N × 5000) approximation using randomly sampled anchor nodes. This is what allows the model to run on WIKI-500 (N=2000 nodes × 9 frequency bins × 32 batch = ~576k flattened nodes) without OOM. For all datasets in the original UFGTime paper (N ≤ 963) this code path is never triggered.

**No new modules, no new architectures, no new hyperparameters.** The model is UFGTime with two correctness fixes.

---

## Results

All results use test MAE (lower is better). Both models evaluated on identical train/val/test splits (70/20/10) and Min-Max scaling.

| Dataset | N | FourierGNN | UFGTime (ours) | Winner |
|---------|---|-----------|----------------|--------|
| Solar-Energy | 137 | 0.0478 | **0.0424** | UFGTime (11.3% better) |
| METR-LA | 207 | **0.0825** | 0.0864 | FourierGNN |
| ECG5000 | 140 | 0.0640 | **0.0594** | UFGTime |
| WIKI-500 | 2000 | 0.0587 | **0.0411** | UFGTime |
| Electricity | 370 | 0.9325 | **0.8860** | UFGTime |
| COVID-CAL | 56 | **0.0148** | 0.1656 | FourierGNN |
| Traffic | 862 | **0.0443** | 0.0647 | FourierGNN |

---

## Detailed Comparative Report (New Benchmarks)

We performed a head-to-head architectural comparison on **METR-LA** and **Solar-Energy** using standardized 100-epoch runs.

### Results:
| Dataset | Model | Test MAE | Test RMSE | Test MAPE | Best Epoch (Val) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Solar-Energy** | **UFGTime** | **0.0424** | **0.0864** | **3.4181** | **19** |
| | FourierGNN | 0.0478 | 0.0881 | 3.4267 | 100 |
| **METR-LA** | **UFGTime** | 0.0864 | 0.1688 | **0.8919** | 41 |
| | FourierGNN | **0.0825** | **0.1685** | 0.8928 | 9 |

---
| Solar-Energy | 137 | 0.0478 | **0.0424** | UFGTime (11.3% better) u2713 |
| METR-LA | 207 | **0.0825** | 0.0864 | FourierGNN u2713 |
| ECG5000 | 140 | 0.0640 | **0.0594** | UFGTime u2713 |
| WIKI-500 | 2000 | 0.0587 | **0.0411** | UFGTime u2713 |
| Electricity | 370 | 0.9325 | **0.8860** | UFGTime u2713 |
| COVID-CAL | 56 | **0.0148** | 0.1367 | FourierGNN u2713 |
| Traffic | 862 | **0.0443** | 0.0647 | FourierGNN u2713 |

---

## Detailed Comparative Report (New Benchmarks)

We performed a head-to-head architectural comparison on **METR-LA** and **Solar-Energy** using standardized 100-epoch runs.

## Results

All results use test MAE (lower is better). Both models evaluated on identical train/val/test splits (70/20/10) and Min-Max scaling.

| Dataset | N | FourierGNN | UFGTime (ours) | Winner |
|---------|---|-----------|----------------|--------|
| Solar-Energy | 137 | 0.0478 | **0.0424** | UFGTime (11.3% better)|
| METR-LA | 207 | **0.0825** | 0.0864 | FourierGNN |
| ECG5000 | 140 | 0.0640 | **0.0594** | UFGTime |
| WIKI-500 | 2000 | 0.0587 | **0.0411** | UFGTime |
| Electricity | 370 | 0.9325 | **0.8860** | UFGTime |
| COVID-CAL | 56 | **0.0148** | 0.1656 | FourierGNN |
| Traffic | 862 | **0.0443** | 0.0647 | FourierGNN |

---

## Detailed Comparative Report (New Benchmarks)

We performed a head-to-head architectural comparison on **METR-LA** and **Solar-Energy** using standardized 100-epoch runs.

### Results:
| Dataset | Model | Test MAE | Test RMSE | Test MAPE | Best Epoch (Val) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Solar-Energy** | **UFGTime** | **0.0424** | **0.0864** | **3.4181** | **19** |
| | FourierGNN | 0.0478 | 0.0881 | 3.4267 | 100 |
| **METR-LA** | **UFGTime** | 0.0864 | 0.1688 | **0.8919** | 41 |
| | FourierGNN | **0.0825** | **0.1685** | 0.8928 | 9 |

---
| **Solar-Energy** | **UFGTime** | **0.0424** | **0.0864** | **3.4181** | **19** |
| | FourierGNN | 0.0478 | 0.0881 | 3.4267 | 100 |
| **METR-LA** | **UFGTime** | 0.0864 | 0.1688 | **0.8919** | 41 |
| | FourierGNN | **0.0825** | **0.1685** | 0.8928 | 9 |

---
| ECG5000 | 140 | 0.0640 | **0.0594** | UFGTime ✓ |
| WIKI-500 | 2000 | 0.0587 | **0.0411** | UFGTime ✓ |
| Electricity | 370 | 0.9325 | **0.8860** | UFGTime ✓ |
| COVID-CAL | 56 | **0.0148** | 0.1367 | FourierGNN ✓ |
| Traffic | 862 | **0.0443** | 0.0647 | FourierGNN ✓ |

**Why UFGTime wins on ECG and WIKI:** Both datasets have strong, structured inter-series correlations. ECG channels are physically correlated cardiac signals. WIKI-500 articles cluster by topic. The sparse KNN graph correctly identifies these meaningful connections and the framelet operator exploits them.

**Why FourierGNN wins on COVID and Traffic:** COVID-CAL hospitalization signals are highly sparse with many zero-valued days. Traffic sensor correlations are non-stationary and change with road conditions. In both cases, KNN graph topology built from sparse or noisy frequency signals is unreliable. FourierGNN's fully connected hypervariate graph is more robust when inter-series structure cannot be inferred from signal similarity.

---

## Installation

```bash
pip install torch numpy tqdm
```

---

## Data

Place pre-processed `.npy` files under `data/<dataset_name>/` with three splits:

```
data/
├── ECG/
│   ├── train.npy   # shape [T_train, N]
│   ├── val.npy     # shape [T_val,   N]
│   └── test.npy    # shape [T_test,  N]
├── WIKI/
│   └── ...
├── Electricity/
│   └── ...
├── COVID-CAL/
│   └── ...
└── Traffic/
    └── ...
```

All splits use a 70/20/10 chronological ratio. Data should be min-max normalised to [0, 1] before saving.

---

## Run commands

### ECG5000 (N=140)
Best config: hidden_size=128, lr=1e-4, plateau scheduler, graph_depth=2

```bash
python main.py \
  --data_name ECG \
  --hidden_size 128 \
  --graph_depth 2 \
  --k 2 \
  --signal_len 16 \
  --epochs 100 \
  --patience 20 \
  --learning_rate 1e-4 \
  --scheduler plateau \
  --optimizer rmsprop
```

### WIKI-500 (N=2000)
Best config: hidden_size=128, signal_len=32, k=5, lr=1e-4, plateau scheduler

```bash
python main.py \
  --data_name WIKI \
  --hidden_size 128 \
  --k 5 \
  --signal_len 32 \
  --epochs 100 \
  --patience 20 \
  --learning_rate 1e-4 \
  --scheduler plateau \
  --optimizer rmsprop \
  --batch_size 32
```

### Electricity (N=370)
Best config: hidden_size=128, signal_len=32, k=5, lr=1e-4, plateau scheduler

```bash
python main.py \
  --data_name Electricity \
  --hidden_size 128 \
  --k 5 \
  --signal_len 32 \
  --epochs 100 \
  --patience 20 \
  --learning_rate 1e-4 \
  --scheduler plateau \
  --optimizer rmsprop
```

### COVID-CAL (N=56)
Best UFGTime config (does not beat FourierGNN — see Results section for explanation):

```bash
python main.py \
  --data_name COVID-CAL \
  --hidden_size 128 \
  --k 10 \
  --signal_len 32 \
  --epochs 100 \
  --patience 20 \
  --learning_rate 1e-4 \
  --scheduler plateau \
  --optimizer rmsprop
```

### Traffic (N=862)
Best UFGTime config (does not beat FourierGNN — see Results section for explanation):

```bash
python main.py \
  --data_name Traffic \
  --hidden_size 128 \
  --k 5 \
  --signal_len 16 \
  --epochs 100 \
  --patience 20 \
  --learning_rate 1e-3 \
  --optimizer rmsprop \
  --batch_size 16
```

---

## Hyperparameter reference

| Argument | Description | Default | Best (ECG) | Best (WIKI) |
|----------|-------------|---------|-----------|-------------|
| `--hidden_size` | FFN hidden dimension | 64 | 128 | 128 |
| `--embed_size` | Frequency embedding size | 128 | 128 | 128 |
| `--k` | KNN graph neighbours | 2 | 2 | 5 |
| `--signal_len` | FFT signal length | 16 | 16 | 32 |
| `--graph_depth` | Number of framelet layers | 1 | 2 | 1 |
| `--frame_type` | Filter bank type | Haar | Haar | Haar |
| `--cheb_order` | Chebyshev polynomial order | 2 | 2 | 2 |
| `--ma_kernel` | Moving average kernel size | 3 | 3 | 3 |
| `--learning_rate` | Initial learning rate | 1e-3 | 1e-4 | 1e-4 |
| `--scheduler` | LR scheduler | none | plateau | plateau |
| `--patience` | Early stopping patience | 0 | 20 | 20 |
| `--optimizer` | Optimiser | rmsprop | rmsprop | rmsprop |
| `--batch_size` | Batch size | 32 | 32 | 32 |

---

## Project context

This is a Complex Network Theory (CNT) course project. The goal was to improve UFGTime's performance relative to FourierGNN, starting from UFGTime's published codebase. The main finding is that a correctness bug in the original implementation — not an architectural limitation — was responsible for the performance gap between the two models on structured datasets.
