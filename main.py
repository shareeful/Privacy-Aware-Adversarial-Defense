import argparse
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer

from configs import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig, AttackConfig, DefenseConfig
from data.dataset import (
    AdultCensusDataProcessor,
    MIMICIVDataProcessor,
    TabularTextDataset,
    create_stratified_splits,
    build_dataloader,
)
from models.deberta_classifier import DeBERTaClassifier
from models.llama_classifier import LLaMAClassifierWithLoRA
from phases.phase2_private_training import DPSGDTrainer
from phases.phase3_attention_drift import AttentionDriftAnalyzer
from attacks.manifold_attack import CooccurrenceModel
from experiments.experiment1_attention_drift import Experiment1AttentionDrift
from experiments.experiment2_attack_evaluation import Experiment2AttackEvaluation
from experiments.experiment3_defense_evaluation import Experiment3DefenseEvaluation
from experiments.experiment4_ablation_tradeoff import Experiment4AblationTradeoff
from experiments.experiment5_explainability_diagnostics import Experiment5ExplainabilityDiagnostics
from utils.logger import setup_logger
from utils.serialization import (
    categorize_tokens,
    get_feature_token_positions,
    compute_inverse_frequency_weights,
)

logger = setup_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="Privacy-Aware Adversarial Defense Pipeline")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "mimic"],
                        help="Dataset to use")
    parser.add_argument("--architecture", type=str, default="deberta",
                        choices=["deberta", "llama"], help="Model architecture")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Single epsilon to train (if None, train all)")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "1", "2", "3", "4", "5"],
                        help="Which experiment to run")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Model checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--no_train", action="store_true",
                        help="Skip training and load from checkpoints")
    return parser.parse_args()


def load_processor(dataset_name: str, data_dir: str):
    if dataset_name == "adult":
        return AdultCensusDataProcessor(data_dir)
    elif dataset_name == "mimic":
        return MIMICIVDataProcessor(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def build_model(architecture: str, config: ModelConfig, device: str):
    if architecture == "deberta":
        model = DeBERTaClassifier(
            model_name=config.primary_model,
            num_labels=config.num_labels,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.primary_model)
        is_decoder = False
    elif architecture == "llama":
        model = LLaMAClassifierWithLoRA(
            model_name=config.secondary_model,
            num_labels=config.num_labels,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.secondary_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        is_decoder = True
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model = model.to(device)
    return model, tokenizer, is_decoder


def prepare_dataset_splits(
    processor,
    tokenizer,
    train_config: DataConfig,
    architecture: str,
):
    logger.info("Loading and preprocessing data...")
    df, labels = processor.load_and_preprocess()

    logger.info("Serializing records...")
    serialized_texts = processor.serialize_dataframe(df)

    critical_features, non_functional_features = processor.get_feature_categorization()

    feature_names = processor.feature_names

    logger.info("Computing token positions...")
    critical_positions_all = []
    non_func_positions_all = []
    feature_positions_all = []

    for text in serialized_texts:
        crit_pos, non_func_pos = categorize_tokens(
            text, critical_features, non_functional_features, tokenizer
        )
        feat_pos = get_feature_token_positions(text, feature_names, tokenizer)
        critical_positions_all.append(crit_pos)
        non_func_positions_all.append(non_func_pos)
        feature_positions_all.append(feat_pos)

    train_idx, val_idx, test_idx = create_stratified_splits(
        serialized_texts, labels,
        train_ratio=train_config.train_ratio,
        val_ratio=train_config.val_ratio,
        test_ratio=train_config.test_ratio,
        random_seed=train_config.random_seed,
    )

    class_weights = compute_inverse_frequency_weights(labels[train_idx])

    def make_subset_dataset(indices):
        return TabularTextDataset(
            serialized_texts=[serialized_texts[i] for i in indices],
            labels=[int(labels[i]) for i in indices],
            tokenizer=tokenizer,
            max_length=train_config.max_seq_length,
            critical_positions=[critical_positions_all[i] for i in indices],
            non_func_positions=[non_func_positions_all[i] for i in indices],
            feature_positions=[feature_positions_all[i] for i in indices],
        )

    train_dataset = make_subset_dataset(train_idx)
    val_dataset = make_subset_dataset(val_idx)
    test_dataset = make_subset_dataset(test_idx)

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "labels": labels,
        "serialized_texts": serialized_texts,
        "critical_positions_all": critical_positions_all,
        "non_func_positions_all": non_func_positions_all,
        "feature_positions_all": feature_positions_all,
        "class_weights": class_weights,
        "df": df,
        "feature_names": feature_names,
        "critical_features": critical_features,
        "non_functional_features": non_functional_features,
    }


def train_models_across_epsilon(
    base_model_factory,
    train_loader,
    val_loader,
    training_config: TrainingConfig,
    epsilon_values,
    class_weights_tensor,
    dataset_size: int,
    checkpoint_dir: str,
    device: str,
    architecture: str,
    dataset_name: str,
) -> Dict:
    models_by_epsilon = {}

    for epsilon in epsilon_values:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model with ε={epsilon}")
        logger.info(f"{'='*50}")

        model, _, is_decoder = base_model_factory()
        checkpoint_path = os.path.join(
            checkpoint_dir, f"{dataset_name}_{architecture}_eps{epsilon}.pt"
        )

        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            model = model.to(device)
        else:
            trainer = DPSGDTrainer(model=model, config=training_config, device=device)
            metrics, achieved_eps = trainer.train_with_dp(
                train_loader=train_loader,
                val_loader=val_loader,
                target_epsilon=epsilon,
                dataset_size=dataset_size,
                batch_size=training_config.batch_size,
                class_weights=class_weights_tensor,
                save_path=checkpoint_path,
            )
            logger.info(f"Training complete: ACC={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
            logger.info(f"Achieved ε={achieved_eps:.4f}")

        models_by_epsilon[epsilon] = model

    return models_by_epsilon


def run_full_pipeline(args):
    config = ExperimentConfig(
        data=DataConfig(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            output_dir=args.checkpoint_dir,
        ),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    processor = load_processor(args.dataset, args.data_dir)

    def model_factory():
        return build_model(args.architecture, config.model, args.device)

    _, tokenizer, is_decoder = model_factory()

    data_splits = prepare_dataset_splits(
        processor=processor,
        tokenizer=tokenizer,
        train_config=config.data,
        architecture=args.architecture,
    )

    epsilon_values = config.training.epsilon_values
    if args.epsilon is not None:
        epsilon_values = [args.epsilon]

    batch_size = config.training.batch_size
    poisson_rate = batch_size / len(data_splits["train_dataset"])

    train_loader = build_dataloader(
        data_splits["train_dataset"], batch_size=batch_size, shuffle=True,
        use_poisson_sampling=True, poisson_rate=poisson_rate,
    )
    val_loader = build_dataloader(
        data_splits["val_dataset"], batch_size=batch_size, shuffle=False
    )
    test_loader = build_dataloader(
        data_splits["test_dataset"], batch_size=batch_size, shuffle=False
    )

    class_weights_dict = data_splits["class_weights"]
    class_weights_tensor = torch.tensor(
        [class_weights_dict.get(i, 1.0) for i in range(2)],
        dtype=torch.float32,
    ).to(args.device)

    if not args.no_train:
        models_by_epsilon = train_models_across_epsilon(
            base_model_factory=model_factory,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=config.training,
            epsilon_values=epsilon_values,
            class_weights_tensor=class_weights_tensor,
            dataset_size=len(data_splits["train_dataset"]),
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            architecture=args.architecture,
            dataset_name=args.dataset,
        )
    else:
        models_by_epsilon = {}
        for epsilon in epsilon_values:
            model, _, _ = model_factory()
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"{args.dataset}_{args.architecture}_eps{epsilon}.pt"
            )
            if os.path.exists(checkpoint_path):
                state = torch.load(checkpoint_path, map_location=args.device)
                model.load_state_dict(state)
                model = model.to(args.device)
                models_by_epsilon[epsilon] = model
                logger.info(f"Loaded checkpoint for ε={epsilon}")
            else:
                logger.warning(f"No checkpoint for ε={epsilon}, skipping")

    test_critical_pos = [data_splits["critical_positions_all"][i] for i in data_splits["test_idx"]]
    test_non_func_pos = [data_splits["non_func_positions_all"][i] for i in data_splits["test_idx"]]
    test_feature_pos = [data_splits["feature_positions_all"][i] for i in data_splits["test_idx"]]

    cooccurrence_dist = processor.get_cooccurrence_distribution(data_splits["df"])
    feature_domains = processor.get_feature_domains(data_splits["df"])
    cooccurrence_model = CooccurrenceModel(cooccurrence_dist)

    test_records = []
    for i in data_splits["test_idx"]:
        row = data_splits["df"].iloc[i]
        record = {col: row[col] for col in data_splits["feature_names"] if col in data_splits["df"].columns}
        test_records.append(record)

    test_labels = data_splits["labels"][data_splits["test_idx"]]

    architecture_name = "DeBERTa-V3-Large" if args.architecture == "deberta" else "LLaMA-3.1-8B"
    dataset_name = args.dataset

    if args.experiment in ("all", "1") and models_by_epsilon:
        logger.info("\n" + "="*60 + "\nRunning Experiment 1: Attention Drift\n" + "="*60)
        exp1 = Experiment1AttentionDrift(
            device=args.device, ig_steps=config.defense.ig_steps,
            results_dir=args.output_dir,
        )
        exp1.run(
            models_by_epsilon=models_by_epsilon,
            dataloader=test_loader,
            critical_positions_batch=test_critical_pos,
            non_func_positions_batch=test_non_func_pos,
            feature_positions_batch=test_feature_pos,
            is_decoder=is_decoder,
            architecture_name=architecture_name,
            dataset_name=dataset_name,
            compute_ig=True,
        )

    if args.experiment in ("all", "2") and models_by_epsilon:
        logger.info("\n" + "="*60 + "\nRunning Experiment 2: Attack Evaluation\n" + "="*60)
        drift_rankings_by_epsilon = {}
        analyzer = AttentionDriftAnalyzer(device=args.device)
        for eps, model in models_by_epsilon.items():
            result = analyzer.analyze_model(
                model=model, dataloader=test_loader, epsilon=eps,
                critical_positions_batch=test_critical_pos,
                non_func_positions_batch=test_non_func_pos,
                feature_positions_batch=test_feature_pos,
                is_decoder=is_decoder, compute_ig=False,
            )
            drift_rankings_by_epsilon[eps] = analyzer.rank_features_by_drift(
                result.drift_per_feature
            )

        exp2 = Experiment2AttackEvaluation(device=args.device, results_dir=args.output_dir)
        exp2.run(
            models_by_epsilon=models_by_epsilon,
            test_records=test_records,
            test_labels=test_labels,
            feature_names=data_splits["feature_names"],
            non_functional_features=data_splits["non_functional_features"],
            cooccurrence_model=cooccurrence_model,
            feature_domains=feature_domains,
            tokenizer=tokenizer,
            drift_rankings_by_epsilon=drift_rankings_by_epsilon,
            config=config.attack,
            architecture_name=architecture_name,
            dataset_name=dataset_name,
            is_decoder=is_decoder,
            max_seq_length=config.data.max_seq_length,
        )

    if args.experiment in ("all", "3") and models_by_epsilon:
        logger.info("\n" + "="*60 + "\nRunning Experiment 3: Defense Evaluation\n" + "="*60)
        target_epsilon = 10.0
        if target_epsilon in models_by_epsilon:
            model = models_by_epsilon[target_epsilon]
            exp3 = Experiment3DefenseEvaluation(device=args.device, results_dir=args.output_dir)
            logger.info("Experiment 3 requires adversarial dataloaders; build from Exp2 outputs.")

    logger.info("\nPipeline complete. Results saved to: " + args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(args)
