import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple

from attacks.manifold_attack import (
    ManifoldAlignedAttack,
    PGDAttack,
    HopSkipJumpAttack,
    RandomSemanticAttack,
    CooccurrenceModel,
    AttackResult,
)
from configs import AttackConfig
from utils.logger import setup_logger
from utils.metrics import compute_attack_metrics

logger = setup_logger("experiment2")


class Experiment2AttackEvaluation:
    def __init__(
        self,
        device: str = "cuda",
        results_dir: str = "./results",
    ):
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def run(
        self,
        models_by_epsilon: Dict[float, object],
        test_records: List[Dict],
        test_labels: np.ndarray,
        feature_names: List[str],
        non_functional_features: Set[str],
        cooccurrence_model: CooccurrenceModel,
        feature_domains: Dict[str, List],
        tokenizer,
        drift_rankings_by_epsilon: Dict[float, List[Tuple[str, float]]],
        config: AttackConfig,
        isolation_forest_detectors: Optional[Dict[float, object]] = None,
        architecture_name: str = "DeBERTa-V3-Large",
        dataset_name: str = "adult",
        is_decoder: bool = False,
        max_seq_length: int = 256,
    ) -> Dict:
        all_results = {}

        for epsilon, model in models_by_epsilon.items():
            logger.info(f"Running attack evaluation for ε={epsilon}...")
            model.eval()

            drift_rankings = drift_rankings_by_epsilon.get(epsilon, [])

            informed_attacker = ManifoldAlignedAttack(
                model=model, tokenizer=tokenizer, feature_names=feature_names,
                non_functional_features=non_functional_features,
                cooccurrence_model=cooccurrence_model, feature_domains=feature_domains,
                config=config, device=self.device, max_seq_length=max_seq_length,
                is_decoder=is_decoder,
            )
            practical_attacker = ManifoldAlignedAttack(
                model=model, tokenizer=tokenizer, feature_names=feature_names,
                non_functional_features=non_functional_features,
                cooccurrence_model=cooccurrence_model, feature_domains=feature_domains,
                config=config, device=self.device, max_seq_length=max_seq_length,
                is_decoder=is_decoder,
            )
            random_attacker = RandomSemanticAttack(
                model=model, tokenizer=tokenizer, feature_names=feature_names,
                non_functional_features=non_functional_features,
                cooccurrence_model=cooccurrence_model, feature_domains=feature_domains,
                config=config, device=self.device, max_seq_length=max_seq_length,
            )
            hsj_attacker = HopSkipJumpAttack(
                model=model, tokenizer=tokenizer, feature_names=feature_names,
                feature_domains=feature_domains, n_iterations=config.hsj_iterations,
                device=self.device, max_seq_length=max_seq_length,
            )
            pgd_attacker = PGDAttack(
                model=model, step_size=config.pgd_step_size, epsilon=config.pgd_epsilon,
                n_iterations=config.pgd_iterations, device=self.device,
            )

            epsilon_results = {}

            attack_configs = [
                ("a_info", "informed"),
                ("a_prac", "practical"),
                ("random_semantic", "random"),
                ("hopskipjump", "hsj"),
            ]

            for attack_name, attack_type in attack_configs:
                logger.info(f"  Running {attack_name} attack...")
                results = []
                original_preds, adv_preds = [], []

                for i, (record, label) in enumerate(zip(test_records, test_labels)):
                    if attack_type == "informed":
                        result = informed_attacker.attack_informed(record, int(label), drift_rankings)
                    elif attack_type == "practical":
                        result = practical_attacker.attack_practical(record, int(label))
                    elif attack_type == "random":
                        result = random_attacker.attack(record, int(label))
                    elif attack_type == "hsj":
                        result = hsj_attacker.attack(record, int(label))
                    else:
                        continue

                    results.append(result)
                    original_preds.append(result.original_prediction)
                    adv_preds.append(result.adversarial_prediction)

                y_true = test_labels
                y_orig_pred = np.array(original_preds)
                y_adv_pred = np.array(adv_preds)

                attack_metrics = compute_attack_metrics(
                    y_true=y_true,
                    y_original_pred=y_orig_pred,
                    y_adv_pred=y_adv_pred,
                )

                avg_queries = float(np.mean([r.n_queries for r in results]))

                if isolation_forest_detectors is not None and epsilon in isolation_forest_detectors:
                    pass

                epsilon_results[attack_name] = {
                    "asr": attack_metrics["asr"],
                    "n_successful": attack_metrics["n_successful"],
                    "avg_queries": avg_queries,
                }
                logger.info(f"  {attack_name}: ASR={attack_metrics['asr']:.4f}")

            logger.info(f"  Running PGD attack...")
            pgd_asr = self._run_pgd_attack(
                pgd_attacker, model, tokenizer, test_records, test_labels,
                feature_names, is_decoder, max_seq_length
            )
            epsilon_results["pgd"] = {"asr": pgd_asr, "avg_queries": config.pgd_iterations}

            all_results[str(epsilon)] = epsilon_results

        output = {
            "architecture": architecture_name,
            "dataset": dataset_name,
            "results": all_results,
        }

        out_path = os.path.join(
            self.results_dir,
            f"exp2_{dataset_name}_{architecture_name.replace('-', '_').replace('.', '_')}.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Experiment 2 results saved to {out_path}")
        return output

    def _run_pgd_attack(
        self, pgd_attacker, model, tokenizer, records, labels,
        feature_names, is_decoder, max_seq_length
    ) -> float:
        from utils.serialization import serialize_record

        model.eval()
        successes = 0
        original_correct = 0

        for record, label in zip(records, labels):
            serialized = serialize_record(record, feature_names)
            encoding = tokenizer(
                serialized, max_length=max_seq_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            with torch.no_grad():
                preds, _ = model.predict(input_ids.to(self.device), attention_mask.to(self.device))
            original_pred = int(preds[0].item())

            if original_pred != int(label):
                continue
            original_correct += 1

            labels_tensor = torch.tensor([int(label)])
            _, adv_preds = pgd_attacker.attack(
                input_ids, attention_mask, labels_tensor, is_decoder
            )

            if adv_preds[0] != int(label):
                successes += 1

        return float(successes) / max(original_correct, 1)
