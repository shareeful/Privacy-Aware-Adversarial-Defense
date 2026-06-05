"""Experiment 7: Adaptive Adversary, Threat Realism, and Black-Box Surrogate.

Stress-tests the defense under three adversarial conditions described in Section 4.8:

  1. Adaptive adversary (Ainfo + attention-ratio constraint): an attacker who knows
     the detection mechanism and constrains perturbations to preserve the authentic
     critical/non-functional attention ratio, evading aAC while flipping the prediction.

  2. Degraded co-occurrence model: Ainfo run with a co-occurrence distribution
     estimated from only a 10% public subsample and from a distributionally shifted
     public corpus, bounding the extent to which headline ASR numbers overstate
     practical risk.

  3. Black-box surrogate detection: the deployer uses a SurrogateAttentionModel
     (~11M parameters) trained on score-only API outputs to reconstruct the attention
     consistency signal without access to the protected model's weights, then fuses it
     with Isolation Forest exactly as in Phase V.

Key metrics: AUC, ASR, DER per condition.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple

from attacks.manifold_attack import AdaptiveManifoldAttack, ManifoldAlignedAttack, CooccurrenceModel, AttackResult
from defenses.trust_score import SurrogateAttentionDefense, TrustScoreDefense, IsolationForestDetector
from utils.logger import setup_logger

logger = setup_logger("experiment7")


def _compute_auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def _run_attack_batch(
    attack,
    records: List[Dict],
    true_labels: np.ndarray,
    drift_rankings: List[Tuple[str, float]],
    critical_positions_batch: Optional[List[List[int]]] = None,
    non_func_positions_batch: Optional[List[List[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run an attack over a list of records, return (success_flags, n_queries)."""
    successes = []
    query_counts = []
    for i, (record, label) in enumerate(zip(records, true_labels)):
        try:
            if critical_positions_batch is not None:
                result: AttackResult = attack.attack(
                    record=record,
                    true_label=int(label),
                    drift_rankings=drift_rankings,
                    critical_positions=critical_positions_batch[i],
                    non_func_positions=non_func_positions_batch[i],
                )
            else:
                result: AttackResult = attack.attack_informed(
                    record=record,
                    true_label=int(label),
                    drift_rankings=drift_rankings,
                )
        except Exception as e:
            logger.warning(f"Attack failed on sample {i}: {e}")
            result = AttackResult(
                original_record=record, adversarial_record=record,
                original_label=int(label), original_prediction=int(label),
                adversarial_prediction=int(label), adversarial_confidence=0.0,
                plausibility_score=0.0, n_queries=0, success=False, features_modified=[],
            )
        successes.append(int(result.success))
        query_counts.append(result.n_queries)
    return np.array(successes), np.array(query_counts)


class Experiment7AdaptiveAdversary:
    """Comprehensive stress-test of the defense under adaptive and realistic threats."""

    def __init__(
        self,
        device: str = "cuda",
        results_dir: str = "./results",
    ):
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Sub-experiment A: Adaptive Adversary
    # ------------------------------------------------------------------

    def run_adaptive_adversary(
        self,
        model,
        tokenizer,
        feature_names: List[str],
        non_functional_features: Set[str],
        cooccurrence_model: CooccurrenceModel,
        feature_domains: Dict[str, List],
        attack_config,
        test_records: List[Dict],
        test_labels: np.ndarray,
        drift_rankings: List[Tuple[str, float]],
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        mu_R: float,
        sigma_R: float,
        defense: TrustScoreDefense,
        test_dataloader,
        epsilon: float = 10.0,
        detection_threshold_sigma: float = 2.0,
        is_decoder: bool = False,
    ) -> Dict:
        """Evaluate the adaptive adversary that constrains attention ratio (Section 3.5.4)."""
        logger.info("Sub-experiment A: Adaptive Adversary")

        adaptive_attack = AdaptiveManifoldAttack(
            model=model,
            tokenizer=tokenizer,
            feature_names=feature_names,
            non_functional_features=non_functional_features,
            cooccurrence_model=cooccurrence_model,
            feature_domains=feature_domains,
            config=attack_config,
            mu_R=mu_R,
            sigma_R=sigma_R,
            detection_threshold_sigma=detection_threshold_sigma,
            device=self.device,
            is_decoder=is_decoder,
        )

        successes, query_counts = _run_attack_batch(
            attack=adaptive_attack,
            records=test_records,
            true_labels=test_labels,
            drift_rankings=drift_rankings,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
        )

        asr = float(successes.mean())
        avg_queries = float(query_counts.mean())

        # Evaluate detection on adversarial inputs
        adv_labels = np.ones(len(test_labels))  # adversarial flag
        auth_labels = np.zeros(len(test_labels))

        n_crit = [critical_positions_batch[i] for i in range(len(test_records))]
        n_non = [non_func_positions_batch[i] for i in range(len(test_records))]
        n_feat = [{}] * len(test_records)

        try:
            adv_result = defense.score(
                model=model,
                dataloader=test_dataloader,
                critical_positions_batch=n_crit,
                non_func_positions_batch=n_non,
                feature_positions_batch=n_feat,
                device=self.device,
                is_decoder=is_decoder,
            )
            detection_auc = _compute_auc_from_scores(adv_labels, adv_result.trust_scores)
        except Exception as e:
            logger.warning(f"Detection scoring failed: {e}")
            detection_auc = float("nan")

        result = {
            "asr": asr,
            "avg_queries": avg_queries,
            "detection_auc": detection_auc,
            "detection_threshold_sigma": detection_threshold_sigma,
            "epsilon": epsilon,
        }
        logger.info(f"Adaptive adversary: ASR={asr:.3f}, AUC={detection_auc:.3f}")
        return result

    # ------------------------------------------------------------------
    # Sub-experiment B: Degraded Co-occurrence Model
    # ------------------------------------------------------------------

    def run_degraded_cooccurrence(
        self,
        model,
        tokenizer,
        feature_names: List[str],
        non_functional_features: Set[str],
        full_cooccurrence_model: CooccurrenceModel,
        degraded_cooccurrence_model_10pct: CooccurrenceModel,
        shifted_cooccurrence_model: CooccurrenceModel,
        feature_domains: Dict[str, List],
        attack_config,
        test_records: List[Dict],
        test_labels: np.ndarray,
        drift_rankings: List[Tuple[str, float]],
        epsilon: float = 10.0,
        is_decoder: bool = False,
    ) -> Dict:
        """Compare ASR under full vs. degraded co-occurrence models (Section 4.8)."""
        logger.info("Sub-experiment B: Degraded Co-occurrence Model")

        results = {}
        configs = [
            ("full", full_cooccurrence_model),
            ("subsample_10pct", degraded_cooccurrence_model_10pct),
            ("shifted_distribution", shifted_cooccurrence_model),
        ]

        for condition_name, cooc_model in configs:
            attack = ManifoldAlignedAttack(
                model=model,
                tokenizer=tokenizer,
                feature_names=feature_names,
                non_functional_features=non_functional_features,
                cooccurrence_model=cooc_model,
                feature_domains=feature_domains,
                config=attack_config,
                device=self.device,
                is_decoder=is_decoder,
            )
            successes, query_counts = _run_attack_batch(
                attack=attack,
                records=test_records,
                true_labels=test_labels,
                drift_rankings=drift_rankings,
            )
            asr = float(successes.mean())
            avg_q = float(query_counts.mean())
            results[condition_name] = {"asr": asr, "avg_queries": avg_q}
            logger.info(f"  {condition_name}: ASR={asr:.3f}, queries={avg_q:.1f}")

        asr_full = results["full"]["asr"]
        for cond in ["subsample_10pct", "shifted_distribution"]:
            gap = asr_full - results[cond]["asr"]
            results[cond]["asr_gap_vs_full"] = gap
            logger.info(f"  Gap ({cond} vs full): {gap:.3f}")

        return {"epsilon": epsilon, "conditions": results}

    # ------------------------------------------------------------------
    # Sub-experiment C: Black-Box Surrogate Detection
    # ------------------------------------------------------------------

    def run_surrogate_detection(
        self,
        vocab_size: int,
        defense_config,
        authentic_dataloader,
        test_dataloader,
        protected_model_scores: np.ndarray,
        authentic_labels: np.ndarray,
        adversarial_labels: np.ndarray,
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        whitebox_auc: float,
        epsilon: float = 10.0,
        n_surrogate_epochs: int = 5,
    ) -> Dict:
        """Train surrogate and evaluate detection AUC vs. white-box baseline."""
        logger.info("Sub-experiment C: Black-Box Surrogate Detection")

        surrogate_defense = SurrogateAttentionDefense(
            vocab_size=vocab_size,
            config=defense_config,
            device=self.device,
        )

        surrogate_defense.train_surrogate(
            train_dataloader=authentic_dataloader,
            protected_model_scores=protected_model_scores,
            n_epochs=n_surrogate_epochs,
        )

        surrogate_defense.fit_reference(
            authentic_dataloader=authentic_dataloader,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
        )

        # Collect surrogate consistency scores over test set
        surrogate_defense.surrogate.eval()
        a_ac_scores = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                B = input_ids.shape[0]
                for b in range(B):
                    global_b = batch_idx * test_dataloader.batch_size + b
                    crit_pos = critical_positions_batch[global_b] if global_b < len(critical_positions_batch) else []
                    non_func_pos = non_func_positions_batch[global_b] if global_b < len(non_func_positions_batch) else []
                    a_ac = surrogate_defense.score_sample(
                        input_ids=input_ids[b:b+1],
                        attention_mask=attention_mask[b:b+1],
                        critical_positions=crit_pos,
                        non_func_positions=non_func_pos,
                    )
                    a_ac_scores.append(a_ac)

        a_ac_scores = np.array(a_ac_scores[:len(adversarial_labels)])
        surrogate_auc = _compute_auc_from_scores(adversarial_labels, a_ac_scores)
        auc_recovery = surrogate_auc / (whitebox_auc + 1e-8)

        result = {
            "epsilon": epsilon,
            "surrogate_auc": surrogate_auc,
            "whitebox_auc_reference": whitebox_auc,
            "auc_recovery_fraction": auc_recovery,
            "n_surrogate_epochs": n_surrogate_epochs,
        }
        logger.info(
            f"Surrogate AUC={surrogate_auc:.3f} vs. white-box {whitebox_auc:.3f} "
            f"(recovery={auc_recovery:.2%})"
        )
        return result

    # ------------------------------------------------------------------
    # Full experiment entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        model,
        tokenizer,
        feature_names: List[str],
        non_functional_features: Set[str],
        cooccurrence_model: CooccurrenceModel,
        feature_domains: Dict[str, List],
        attack_config,
        defense_config,
        test_records: List[Dict],
        test_labels: np.ndarray,
        drift_rankings: List[Tuple[str, float]],
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        mu_R: float,
        sigma_R: float,
        defense: TrustScoreDefense,
        test_dataloader,
        authentic_dataloader,
        protected_model_scores: np.ndarray,
        vocab_size: int,
        whitebox_auc: float,
        epsilon: float = 10.0,
        is_decoder: bool = False,
        degraded_cooccurrence_model_10pct: Optional[CooccurrenceModel] = None,
        shifted_cooccurrence_model: Optional[CooccurrenceModel] = None,
    ) -> Dict:
        results: Dict = {}

        # Sub-experiment A: Adaptive adversary
        results["adaptive_adversary"] = self.run_adaptive_adversary(
            model=model,
            tokenizer=tokenizer,
            feature_names=feature_names,
            non_functional_features=non_functional_features,
            cooccurrence_model=cooccurrence_model,
            feature_domains=feature_domains,
            attack_config=attack_config,
            test_records=test_records,
            test_labels=test_labels,
            drift_rankings=drift_rankings,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
            mu_R=mu_R,
            sigma_R=sigma_R,
            defense=defense,
            test_dataloader=test_dataloader,
            epsilon=epsilon,
            is_decoder=is_decoder,
        )

        # Sub-experiment B: Degraded co-occurrence (only if degraded models supplied)
        if degraded_cooccurrence_model_10pct is not None and shifted_cooccurrence_model is not None:
            results["degraded_cooccurrence"] = self.run_degraded_cooccurrence(
                model=model,
                tokenizer=tokenizer,
                feature_names=feature_names,
                non_functional_features=non_functional_features,
                full_cooccurrence_model=cooccurrence_model,
                degraded_cooccurrence_model_10pct=degraded_cooccurrence_model_10pct,
                shifted_cooccurrence_model=shifted_cooccurrence_model,
                feature_domains=feature_domains,
                attack_config=attack_config,
                test_records=test_records,
                test_labels=test_labels,
                drift_rankings=drift_rankings,
                epsilon=epsilon,
                is_decoder=is_decoder,
            )

        # Sub-experiment C: Surrogate detection
        adv_labels = np.ones(len(test_labels))
        results["surrogate_detection"] = self.run_surrogate_detection(
            vocab_size=vocab_size,
            defense_config=defense_config,
            authentic_dataloader=authentic_dataloader,
            test_dataloader=test_dataloader,
            protected_model_scores=protected_model_scores,
            authentic_labels=np.zeros(len(test_labels)),
            adversarial_labels=adv_labels,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
            whitebox_auc=whitebox_auc,
            epsilon=epsilon,
        )

        out_path = os.path.join(self.results_dir, "experiment7_adaptive_adversary.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Experiment 7 results saved to {out_path}")

        return results
