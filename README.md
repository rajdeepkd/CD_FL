# Cross-Domain Federated Learning Benchmark for Heterogeneous IoT-like Clients

This repository provides a reproducible, modular PyTorch-based framework for running
federated learning (FL) experiments across heterogeneous domains, where each "client"
has its own dataset.

The core goals:
- Treat each CSV in `data/raw_data/` as a distinct client with its own data distribution.
- Run multiple FL strategies (FedAvg, FedProx, simple personalization).
- Evaluate fairness across clients, not just global accuracy.
- Log communication overhead, per-round metrics, and convergence curves.

This code is designed for *defensive network security / anomaly detection research*,
but it is data-agnostic and will work for any tabular binary classification task
(0/1 label). No real data is included.

---

## Directory Structure

```text
.
├─ README.md
├─ requirements.txt
├─ run_experiment.py
├─ configs/
│   └─ default.yaml
├─ data/
│   ├─ README.md
│   ├─ raw_data/          # You put client CSVs here (one per client)
│   └─ processed/         # Auto-generated per-client processed CSVs
├─ results/
│   ├─ logs/              # Per-round logs, run metadata logs
│   ├─ artifacts/         # Saved checkpoints and reproducibility templates
│   └─ summaries/         # Final summaries, fairness reports, plots
├─ src/
│   ├─ __init__.py
│   ├─ data/
│   │   ├─ loader.py
│   │   └─ preprocess.py
│   ├─ models/
│   │   └─ model_zoo.py
│   ├─ fl/
│   │   ├─ client.py
│   │   ├─ server.py
│   │   └─ strategies.py
│   ├─ experiments/
│   │   └─ runner.py
│   └─ utils/
│       ├─ logging_utils.py
│       ├─ metrics.py
│       ├─ serialization.py
│       └─ config.py
├─ scripts/
│   └─ prepare_data_example.sh
└─ tests/
    ├─ test_preprocess.py
    └─ test_serialization.py
```

---

## Data format

- Each file in `data/raw_data/*.csv` is considered one client.
- The code will try to infer:
  - Which column is the label
  - Which columns are numerical vs categorical

### Label inference
By default:
- It looks for a column named (case-insensitive): `label`, `y`, `target`, or `class`.
- You can override explicitly in `configs/default.yaml` → `data.label_column`.

### Feature inference
- Numerical features: treated with imputation + scaling.
- Categorical: label-encoded or one-hot encoded depending on cardinality.

### Class balancing
- If the minority class ratio < `preprocess.smote_minority_ratio_threshold`,
  SMOTE is applied (configurable or turn it off with `preprocess.use_smote=false`).
- We log pre/post class distributions in `results/summaries/<client>_class_distribution.txt`.

---

## Running an experiment

```bash
python run_experiment.py \
  --config configs/default.yaml \
  --strategy FedAvg \
  --rounds 10 \
  --local-epochs 2 \
  --batch-size 128 \
  --lr 0.0005 \
  --seed 123 \
  --equalize-samples true
```

You can change:
- `--strategy`: `FedAvg`, `FedProx`, `PersonalizedHead`
- `--rounds`: number of global rounds
- `--local-epochs`: local epochs per client per round
- `--lr`: client optimizer learning rate
- `--equalize-samples`: `true` → all clients contribute equally per round;  
  `false` → weight by data size

You can also edit `configs/default.yaml` directly.

---

## Logging and Outputs

- Per-round CSV log: `results/logs/<exp_name>_per_round.csv`
  Columns include:
  - round, strategy, seed, client_id
  - client_size (#samples used for training)
  - client_update_size_bytes (estimated communication cost)
  - client_train_loss, client_train_accuracy
  - client_eval_accuracy, client_eval_loss
  - global_num_params, total_comm_bytes

- Python logging:
  - `results/logs/<timestamp>_<strategy>_<seed>.log`

- Final metrics:
  - `results/summaries/<exp_name>_final_summary.json`
  - `results/summaries/<exp_name>_final_summary.txt`

- Plots:
  - `results/summaries/<exp_name>_learning_curves.png`
  - `results/summaries/<exp_name>_communication.png`
  - `results/summaries/<exp_name>_client_accuracy_cdf.png`
  - `results/summaries/<exp_name>_heatmap_strategy_delta.png`

---

## Reproducibility

We:
- Fix seeds for `random`, `numpy`, `torch`.
- Record all hyperparameters.
- Save checkpoints per round to `results/artifacts/`
  with filenames including strategy, seed, and round.
- Write `results/artifacts/README_reproducibility.txt` describing how to replay.

---

## Tests

Run:
```bash
pytest tests/
```

We currently ship:
- `test_preprocess.py`: tests feature detection + preprocessing
- `test_serialization.py`: tests update-size / param-count estimation

Both run on toy data.

---

## Safety Notice

This code is only for defensive research (intrusion detection, anomaly detection,
federated robustness). It does not ship any offensive content and does not collect
data externally. You own and control your CSVs locally.

---
