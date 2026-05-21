# Context-Aware Intelligent Hybrid Agent for Driver Assistance

This repository contains the implementation and evaluation of a hybrid driving-assistance system combining an ultra-fast rule‑based reactive layer with a BDI (Belief–Desire–Intention) deliberative layer supported by scenario-specific machine learning models.

## Project Summary

- Purpose: Provide a safety-first agent that produces immediate, rule-based emergency responses while enriching decisions with interpretable, ML-driven context estimates.
- Key components:
  - Reactive layer: rule-based, real-time detection of critical situations (danger zones, overtaking, intersections).
  - Scenario models: Random Forest classifiers trained per scenario (SMOTE-balanced datasets) to provide calibrated risk scores.
  - BDI layer: transforms scenario probabilities into beliefs, utilities, and ranked intentions for contextual guidance.

## Features

- Real-time emergency detection with prioritized rule logic.
- Scenario-specific ML inference for contextual risk scoring.
- BDI-based intention generation for explainable actions.
- Strict anti‑leakage evaluation protocol (vehicle exclusion, future-time holdout) and performance benchmarks.

## Repo Structure

- `core/` — main implementation scripts (agent, reactive rules, training and evaluation helpers)
- `data/` — datasets and engineered feature files (SMOTE-balanced versions)
- `models/` — trained model artifacts and evaluation outputs
- `images/` — plots and figures used in the report
- `chapter2_work_performed.tex` — LaTeX report chapter with methodology and results

## Quick Start

Requirements: Python 3.8+, pip

Install dependencies (example):

```bash
pip install -r requirements.txt
```

Typical workflows:

- Train scenario models:

```bash
python core/train_scenario_models.py --scenario danger_zone
```

- Run evaluation on fusion (strict) test:

```bash
python core/generate_hybrid_validation_metrics.py
```

- Run the HybridDrivingAgent on a sample frame (demo):

```bash
python core/main.py --demo
```

## Results (high level)

- Reactive layer mean latency: ~35 ms (emergency path)
- BDI layer mean latency: ~209 ms
- Final strict-test F1 (examples): danger_zone 0.7655, overtaking 0.7644, intersection 0.6402

## How to Reproduce

1. Prepare the datasets in `data/` (see `core/featureEngineering.py` for feature list).
2. Train models using `core/train_scenario_models.py`.
3. Save trained models to `models/` or `saved_models/`.
4. Run `core/main.py` or the evaluation scripts under `core/` to reproduce tables and figures.

## Authors & Acknowledgements

- Adam Khabou — implementation, dataset engineering, evaluation
- [Your Name] — architecture design, integration, reporting
- Supervisor: Mr. Achraf Makni

## License

This project is provided for academic purposes. Add a license if you plan to publish the repository (e.g., MIT, Apache-2.0).

---
For details, see the project report (chapter2_work_performed.tex) and the figures in `images/`.
