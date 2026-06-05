# Privacy-Aware Adversarial Defense with Explainable AI

Production-level implementation of the paper:

> **Privacy-Aware Adversarial Defense with Explainable AI for Adversarial Robustness in AI Models**
> Bilal Sardar, Shareeful Islam, Stefano Silvestri, Spyridon Papastergiou
> Anglia Ruskin University / ICAR-CNR / Maggioli S.p.A. / University of Piraeus

---

## Overview

This repository implements the five-phase Privacy-Aware Adversarial Defense (PAAD) pipeline, which:

1. **Phase I** — Serializes tabular records into tokenized sequences and categorizes features into task-critical vs. non-functional categories.
2. **Phase II** — Trains Transformer models (DeBERTa-V3-Large or LLaMA-3.1-8B) under differential privacy (DP-SGD / DP-LoRA) across ten privacy budgets ε ∈ {1, 2, 5, 8, 10, 15, 20, 30, 50, ∞}.
3. **Phase III** — Introduces the Attention Concentration Score (ACS) to measure how DP training redistributes attention from task-critical to non-functional features, with Integrated Gradients validation, drift-vulnerability correlation analysis, and **class-conditional drift analysis** showing that privacy noise disproportionately destabilises the minority class.
4. **Phase IV** — Constructs a Manifold-Aligned Semantic Attack exploiting attention drift via plausibility-constrained greedy search under a score-based black-box threat model. Also includes an **Adaptive Adversary** variant that constrains perturbations to preserve the authentic attention ratio, probing the limits of the defense.
5. **Phase V** — Builds a TrustScore defense fusing Isolation Forest embedding anomaly detection with three attention consistency scoring variants (ratio-based, IG-based, JSD-based), plus a **Surrogate Attention Model** (~11M params) that reconstructs the consistency signal from a score-only API for black-box deployment.

The approach is validated through **seven experiments** on Adult Census Income and MIMIC-IV datasets using DeBERTa-V3-Large and LLaMA-3.1-8B.

---

## Repository Structure

```
privacy_aware_defense/
├── configs/
│   └── config.py              # Dataclass-based configuration for all pipeline stages
├── data/
│   └── dataset.py             # Phase I: data processors, serialization, dataset classes
├── models/
│   ├── deberta_classifier.py  # DeBERTa-V3-Large classifier with attention extraction
│   └── llama_classifier.py    # LLaMA-3.1-8B with DP-LoRA adapters
├── phases/
│   ├── phase2_private_training.py  # DP-SGD trainer with RDP accounting
│   └── phase3_attention_drift.py   # ACS metric, IG validation, drift correlation
├── attacks/
│   └── manifold_attack.py     # Phase IV: A_info, A_prac, PGD, HopSkipJump, Random Semantic, AdaptiveManifoldAttack
├── defenses/
│   └── trust_score.py         # Phase V: TrustScore, Isolation Forest, Consistency Scorer, SurrogateAttentionModel
├── experiments/
│   ├── experiment1_attention_drift.py
│   ├── experiment2_attack_evaluation.py
│   ├── experiment3_defense_evaluation.py
│   ├── experiment4_ablation_tradeoff.py
│   ├── experiment5_explainability_diagnostics.py
│   ├── experiment6_categorization_sensitivity.py  # NEW: swapped/SHAP/LIME/PCA partitions
│   └── experiment7_adaptive_adversary.py           # NEW: adaptive attack, surrogate defense
├── utils/
│   ├── logger.py              # Structured logging
│   ├── metrics.py             # Utility, attack, defense metrics + Spearman CI + logistic OR
│   └── serialization.py       # Feature serialization, categorization, co-occurrence weights
├── main.py                    # Full pipeline orchestrator with CLI
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

GPU with at least 40GB VRAM is required for DeBERTa-V3-Large full DP-SGD. LLaMA-3.1-8B with DP-LoRA requires the same or more.

---

## Data

### Adult Census Income

Download from the [UCI Repository](https://archive.ics.uci.edu/dataset/2/adult):

```bash
mkdir -p data/
wget -O data/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget -O data/adult.test https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

### MIMIC-IV

Access MIMIC-IV through [PhysioNet](https://physionet.org/content/mimiciv/) following the required data use agreement. After obtaining access, preprocess the dataset to produce `data/mimic_iv_processed.csv` with columns matching the feature specification in `utils/serialization.py`.

---

## Usage

### Full pipeline on Adult Census with DeBERTa-V3-Large

```bash
python main.py \
    --dataset adult \
    --architecture deberta \
    --data_dir ./data \
    --output_dir ./results \
    --checkpoint_dir ./checkpoints \
    --device cuda \
    --batch_size 64 \
    --num_epochs 10
```

### Run a specific experiment only

```bash
python main.py \
    --dataset adult \
    --architecture deberta \
    --experiment 1 \
    --no_train \
    --checkpoint_dir ./checkpoints
```

### Train a single epsilon value

```bash
python main.py \
    --dataset adult \
    --architecture deberta \
    --epsilon 10.0 \
    --no_train
```

### LLaMA-3.1-8B with DP-LoRA

```bash
python main.py \
    --dataset adult \
    --architecture llama \
    --data_dir ./data \
    --output_dir ./results \
    --checkpoint_dir ./checkpoints
```

---

## Configuration

All hyperparameters are defined in `configs/config.py` as Python dataclasses:

| Parameter | Default | Description |
|---|---|---|
| `epsilon_values` | [1,2,5,8,10,15,20,30,50,∞] | Privacy budget spectrum |
| `delta` | 1e-5 | DP delta parameter |
| `clipping_threshold` | 1.0 | DP-SGD gradient clipping bound C |
| `learning_rate` | 2e-5 | AdamW learning rate |
| `batch_size` | 64 | Training batch size |
| `num_epochs` | 10 | Max training epochs |
| `lora_rank` | 16 | LoRA rank r for LLaMA |
| `plausibility_threshold` | 0.01 | τ_p for manifold-aligned attack |
| `drift_threshold` | 0.05 | τ_d for informed attacker feature selection |
| `query_budget` | 500 | Q_max maximum queries per attack attempt |
| `pgd_step_size` | 0.01 | PGD step size α |
| `pgd_epsilon` | 0.1 | PGD ℓ∞ bound |
| `pgd_iterations` | 40 | PGD iterations |
| `hsj_iterations` | 50 | HopSkipJump iterations |
| `isolation_forest_estimators` | 100 | Number of trees in Isolation Forest |
| `ig_steps` | 50 | Integrated Gradients interpolation steps |
| `max_fpr` | 0.05 | Maximum FPR for defense threshold calibration |
| `adaptive_detection_threshold_sigma` | 2.0 | σ multiplier for adaptive adversary attention-ratio constraint |
| `surrogate_d_model` | 256 | Hidden dimension of surrogate attention model |
| `surrogate_n_heads` | 4 | Number of attention heads in surrogate model |
| `surrogate_n_layers` | 4 | Number of Transformer layers in surrogate model |
| `surrogate_n_epochs` | 5 | Training epochs for surrogate model |
| `top_k_overlap` | 5 | Top-k features used for Jaccard overlap in Experiment 6 |
| `degraded_subsample_fraction` | 0.10 | Public data fraction for degraded co-occurrence in Experiment 7 |

---

## Experiments

Seven experiments address the four research questions from the paper:

| Experiment | RQ | Key Metrics |
|---|---|---|
| 1: Attention Drift | RQ1 | ACS, Δ_c, Spearman ρ, OR (logistic), class-conditional drift |
| 2: Attack Evaluation | RQ2 | ASR, DER, query count |
| 3: Defense Evaluation | RQ3 | TPR, FPR, AUC per attack type |
| 4: Ablation + Trade-off | RQ4 | All metrics jointly across ε spectrum |
| 5: Explainability Diagnostics | RQ1,RQ3 | Heatmaps, rankings, latency |
| 6: Categorization Sensitivity | RQ1 | Spearman ρ, Jaccard, top-5 overlap across partitions |
| 7: Adaptive Adversary & Black-Box | RQ2,RQ3 | AUC under adaptive attack, surrogate AUC recovery |

Results are saved as JSON files in the specified `--output_dir`.

---

## Key Components

### Class-Conditional Drift Analysis

Defined in `phases/phase3_attention_drift.py` (`AttentionDriftAnalyzer.compute_class_conditional_drift`). Separates per-sample drift by class to test whether private training disproportionately destabilises the minority class:

- Inverse-frequency weighting equalises expected loss but not per-update noise variance, so minority gradients are perturbed more per effective update.
- At ε = 10 on MIMIC-IV, minority-class drift (0.22) is ~1.7× majority-class drift (0.13).
- Among successful Ainfo attacks at ε = 10, the minority class is heavily over-represented (55% on Adult vs. 24.1% base rate; 68% on MIMIC-IV vs. 11.5% base rate).

### Adaptive Adversary

Defined in `attacks/manifold_attack.py` (`AdaptiveManifoldAttack`). Augments the Ainfo greedy search with a hard constraint:

```
accept substitution only if |R(x') - mu_R| / sigma_R < detection_threshold_sigma
```

This forces the perturbed input to keep its critical/non-functional attention ratio within the authentic distribution, directly probing the TrustScore's attention-consistency signal. Evaluation in Experiment 7 reports AUC degradation under this strongest threat.

### Surrogate Attention Model

Defined in `defenses/trust_score.py` (`SurrogateAttentionModel`, `SurrogateAttentionDefense`). A lightweight 4-layer, 4-head, 256-dimensional Transformer (~11M parameters) trained on the deployer's authentic data using the protected model's score-only API outputs as soft targets. Its attention weights substitute for the protected model's in Equations 18 and 21, enabling black-box deployment without access to model internals. Experiment 7 shows this path recovers AUC 0.84 against manifold-aligned attacks.

### Attention Concentration Score (ACS)

Defined in `phases/phase3_attention_drift.py`. For a feature category c at privacy budget ε:

```
ACS_c(ε) = (1/|D_eval|) Σ_{s∈D_eval} (1/H) Σ_k Σ_{j∈T_c(s)} α^(L)_{k,j}(s)
```

Drift is computed as: `Δ_c(ε) = ACS_c(∞) - ACS_c(ε)`

### TrustScore

Defined in `defenses/trust_score.py`. The fusion score is:

```
TrustScore(s) = λ · â_IF(s) + (1-λ) · â_CONS(s)
```

where `â_IF` and `â_CONS` are min-max normalized to [0, 1], and λ is calibrated on a held-out validation set.

### DP-SGD with RDP Accounting

Implemented from scratch in `phases/phase2_private_training.py` following the standard Rényi DP accountant. The noise multiplier σ is calibrated to achieve the target (ε, δ)-DP guarantee.

---

## Citation

```bibtex
@article{sardar2025privacy,
  title={Privacy-Aware Adversarial Defense with Explainable AI for Adversarial Robustness in AI Models},
  author={Sardar, Bilal and Islam, Shareeful and Silvestri, Stefano and Papastergiou, Spyridon},
  journal={Information and Software Technology},
  year={2025}
}
```

---

## Acknowledgements

This work was supported by UK Research and Innovation (UKRI) through the CHIST-ERA Project AI4MultiGIS (EP/Z003490/1) and CyberSecDome (EU Horizon Europe grant No. 101120779).
