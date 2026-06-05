import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple

from defenses.trust_score import (
    TrustScoreDefense,
    IsolationForestDetector,
    AttentionConsistencyScorer,
    SupervisedDetector,
)
from configs import DefenseConfig
from utils.logger import setup_logger
from utils.metrics import compute_defense_metrics

logger = setup_logger("experiment3")


class Experiment3DefenseEvaluation:
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
        model,
        epsilon: float,
        authentic_dataloader,
        adversarial_data_by_attack: Dict[str, Tuple],
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        feature_positions_batch: List[Dict],
        config: DefenseConfig,
        is_decoder: bool = False,
        architecture_name: str = "DeBERTa-V3-Large",
        dataset_name: str = "adult",
    ) -> Dict:
        logger.info(f"Setting up defenses for ε={epsilon}...")

        trust_score_ac = TrustScoreDefense(config)
        trust_score_ac.fit(
            model=model,
            authentic_dataloader=authentic_dataloader,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
            feature_positions_batch=feature_positions_batch,
            device=self.device,
            is_decoder=is_decoder,
        )

        auth_embeddings = trust_score_ac.isolation_forest.collect_embeddings(
            model, authentic_dataloader, self.device, is_decoder
        )

        supervised_detector = SupervisedDetector()

        all_results = {}

        for attack_name, (adv_dataloader, is_adversarial_labels) in adversarial_data_by_attack.items():
            logger.info(f"  Evaluating defenses against {attack_name}...")

            auth_labels = np.zeros(len(auth_embeddings))
            adv_embeddings = trust_score_ac.isolation_forest.collect_embeddings(
                model, adv_dataloader, self.device, is_decoder
            )
            adv_labels = np.ones(len(adv_embeddings))

            combined_embeddings = np.vstack([auth_embeddings, adv_embeddings])
            combined_labels = np.concatenate([auth_labels, adv_labels])

            if attack_name == "pgd":
                train_mask = np.ones(len(combined_labels), dtype=bool)
                supervised_detector.fit(combined_embeddings, combined_labels)

            if_only_scores = trust_score_ac.isolation_forest.score(adv_embeddings)
            auth_if_scores = trust_score_ac.isolation_forest.score(auth_embeddings)
            all_if_scores = np.concatenate([auth_if_scores, if_only_scores])
            all_if_labels = np.concatenate([np.zeros(len(auth_if_scores)), np.ones(len(if_only_scores))])
            if_metrics = compute_defense_metrics(all_if_labels, all_if_scores, max_fpr=config.max_fpr)

            auth_ac, _, auth_jsd = trust_score_ac.consistency_scorer.score_batch(
                model=model, dataloader=authentic_dataloader,
                critical_positions_batch=critical_positions_batch,
                non_func_positions_batch=non_func_positions_batch,
                feature_positions_batch=feature_positions_batch,
                device=self.device, is_decoder=is_decoder,
            )
            adv_ac, _, adv_jsd = trust_score_ac.consistency_scorer.score_batch(
                model=model, dataloader=adv_dataloader,
                critical_positions_batch=critical_positions_batch,
                non_func_positions_batch=non_func_positions_batch,
                feature_positions_batch=feature_positions_batch,
                device=self.device, is_decoder=is_decoder,
            )
            all_ac = np.concatenate([auth_ac, adv_ac])
            all_jsd = np.concatenate([auth_jsd, adv_jsd])
            all_labels = np.concatenate([np.zeros(len(auth_ac)), np.ones(len(adv_ac))])

            ac_only_metrics = compute_defense_metrics(all_labels, all_ac, max_fpr=config.max_fpr)

            ts_ac_result = trust_score_ac.score(
                model=model, dataloader=adv_dataloader,
                critical_positions_batch=critical_positions_batch,
                non_func_positions_batch=non_func_positions_batch,
                feature_positions_batch=feature_positions_batch,
                device=self.device, is_decoder=is_decoder, consistency_variant="ac",
            )
            auth_ts_result = trust_score_ac.score(
                model=model, dataloader=authentic_dataloader,
                critical_positions_batch=critical_positions_batch,
                non_func_positions_batch=non_func_positions_batch,
                feature_positions_batch=feature_positions_batch,
                device=self.device, is_decoder=is_decoder, consistency_variant="ac",
            )
            all_ts_ac = np.concatenate([auth_ts_result.trust_scores, ts_ac_result.trust_scores])
            ts_ac_metrics = compute_defense_metrics(all_labels, all_ts_ac, max_fpr=config.max_fpr)

            ts_jsd_result = trust_score_ac.score(
                model=model, dataloader=adv_dataloader,
                critical_positions_batch=critical_positions_batch,
                non_func_positions_batch=non_func_positions_batch,
                feature_positions_batch=feature_positions_batch,
                device=self.device, is_decoder=is_decoder, consistency_variant="jsd",
            )
            auth_ts_jsd = trust_score_ac.score(
                model=model, dataloader=authentic_dataloader,
                critical_positions_batch=critical_positions_batch,
                non_func_positions_batch=non_func_positions_batch,
                feature_positions_batch=feature_positions_batch,
                device=self.device, is_decoder=is_decoder, consistency_variant="jsd",
            )
            all_ts_jsd = np.concatenate([auth_ts_jsd.trust_scores, ts_jsd_result.trust_scores])
            ts_jsd_metrics = compute_defense_metrics(all_labels, all_ts_jsd, max_fpr=config.max_fpr)

            ts_ig_metrics = {"auc": 0.0, "tpr": 0.0, "fpr": 0.0}

            supervised_scores = np.zeros(len(combined_labels))
            if supervised_detector.fitted:
                supervised_scores = supervised_detector.score(combined_embeddings)
            supervised_metrics = compute_defense_metrics(combined_labels, supervised_scores, max_fpr=config.max_fpr)

            attack_defense_results = {
                "if_only": if_metrics,
                "ac_only": ac_only_metrics,
                "trustscore_ac": ts_ac_metrics,
                "trustscore_ig": ts_ig_metrics,
                "trustscore_jsd": ts_jsd_metrics,
                "supervised": supervised_metrics,
            }

            for defense_name, metrics in attack_defense_results.items():
                logger.info(
                    f"  {attack_name} vs {defense_name}: "
                    f"AUC={metrics['auc']:.4f}, TPR={metrics['tpr']:.4f}, FPR={metrics['fpr']:.4f}"
                )

            all_results[attack_name] = attack_defense_results

        output = {
            "architecture": architecture_name,
            "dataset": dataset_name,
            "epsilon": str(epsilon),
            "results": all_results,
        }

        out_path = os.path.join(
            self.results_dir,
            f"exp3_{dataset_name}_{architecture_name.replace('-', '_').replace('.', '_')}_eps{epsilon}.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Experiment 3 results saved to {out_path}")
        return output
