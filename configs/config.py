from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    dataset_name: str = "adult"
    data_dir: str = "./data"
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    max_seq_length: int = 256
    random_seed: int = 42


@dataclass
class ModelConfig:
    primary_model: str = "microsoft/deberta-v3-large"
    secondary_model: str = "meta-llama/Meta-Llama-3.1-8B"
    num_labels: int = 2
    hidden_dropout_prob: float = 0.1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-5
    batch_size: int = 64
    num_epochs: int = 10
    early_stopping_patience: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    epsilon_values: List[float] = field(
        default_factory=lambda: [1, 2, 5, 8, 10, 15, 20, 30, 50, float("inf")]
    )
    delta: float = 1e-5
    clipping_threshold: float = 1.0
    output_dir: str = "./checkpoints"


@dataclass
class AttackConfig:
    plausibility_threshold: float = 0.01
    drift_threshold: float = 0.05
    query_budget: int = 500
    pgd_step_size: float = 0.01
    pgd_epsilon: float = 0.1
    pgd_iterations: int = 40
    hsj_iterations: int = 50
    # Adaptive adversary (Experiment 7): sigma multiplier for detection threshold
    adaptive_detection_threshold_sigma: float = 2.0


@dataclass
class DefenseConfig:
    isolation_forest_estimators: int = 100
    isolation_forest_contamination: float = 0.1
    fusion_lambda: float = 0.5
    max_fpr: float = 0.05
    ig_steps: int = 50
    gamma: float = 1e-8
    # Surrogate model (Experiment 7): architecture for black-box deployment
    surrogate_d_model: int = 256
    surrogate_n_heads: int = 4
    surrogate_n_layers: int = 4
    surrogate_n_epochs: int = 5
    surrogate_learning_rate: float = 1e-3


@dataclass
class Experiment6Config:
    """Categorization sensitivity analysis (Section 4.8 / Experiment 6)."""
    top_k_overlap: int = 5
    # Fraction of public data used for degraded co-occurrence in Experiment 7
    degraded_subsample_fraction: float = 0.10
    run_shap: bool = True
    run_lime: bool = True
    run_ig: bool = True
    run_pca: bool = True
    shap_background_samples: int = 200
    lime_background_samples: int = 200


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    exp6: Experiment6Config = field(default_factory=Experiment6Config)
    n_folds: int = 5
    device: str = "cuda"
    results_dir: str = "./results"
    log_level: str = "INFO"
