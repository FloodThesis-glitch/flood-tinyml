# flood-tinyml

Accuracy-Efficiency-Reliability Trade-offs of Quantized Machine Learning Models
for Flood Early Warning on Resource-Constrained Microcontrollers.

Thesis project. This repository holds the PC-side pipeline (data preparation,
training, quantization, evaluation). ESP32 firmware will be added later.

## Setup (Windows, Python 3.11)

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Layout

- `data/raw`, `data/processed`: datasets (contents not tracked by Git)
- `src/`: pipeline code
- `notebooks/`: exploration
- `models/`, `results/`: outputs
- `firmware/`: ESP32 code (later)
- `config.yaml`: seeds, horizon N, cost ratios

## Status

Environment set up. Dataset: TO BE DECIDED.
