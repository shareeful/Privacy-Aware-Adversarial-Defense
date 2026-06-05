import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

from utils.logger import setup_logger
from utils.metrics import compute_defense_metrics, paired_ttest

logger = setup_logger("experiment4")


class Experiment4AblationTradeoff:
    def __init__(
        self,
        device: str = "cuda",
        results_dir: str = "./results",
    ):
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def run_ablation(
        self,
        attack_results_by_config: Dict[str, Dict],
        defense_results_by_config: Dict[str, Dict],
        architecture_name: str = "DeBERTa-V3-Large",
        dataset_name: str = "adult",
        epsilon: float = 10.0,
    ) -> Dict:
        ablation_output = {}

        attack_ablation = {}
        if "full_a_info" in attack_results_by_config:
            full_asr = attack_results_by_config["full_a_info"].get("asr", 0.0)
            for config_name, results in attack_results_by_config.items():
                asr = results.get("asr", 0.0)
                delta = full_asr - asr
                attack_ablation[config_name] = {"asr": asr, "delta_from_full": delta}

                if "asr_scores" in attack_results_by_config["full_a_info"] and "asr_scores" in results:
                    t_stat, p_val = paired_ttest(
                        np.array(attack_results_by_config["full_a_info"]["asr_scores"]),
                        np.array(results["asr_scores"])
                    )
                    attack_ablation[config_name]["t_stat"] = t_stat
                    attack_ablation[config_name]["p_value"] = p_val

        defense_ablation = {}
        if "trustscore_ac" in defense_results_by_config:
            full_auc = defense_results_by_config["trustscore_ac"].get("auc", 0.0)
            for config_name, results in defense_results_by_config.items():
                auc = results.get("auc", 0.0)
                delta = full_auc - auc
                defense_ablation[config_name] = {"auc": auc, "delta_from_full": delta}

        ablation_output = {
            "architecture": architecture_name,
            "dataset": dataset_name,
            "epsilon": str(epsilon),
            "attack_ablation": attack_ablation,
            "defense_ablation": defense_ablation,
        }

        logger.info(f"Attack ablation: {attack_ablation}")
        logger.info(f"Defense ablation: {defense_ablation}")

        return ablation_output

    def run_tradeoff_analysis(
        self,
        epsilon_values: List[float],
        accuracy_by_epsilon: Dict[float, float],
        f1_by_epsilon: Dict[float, float],
        drift_by_epsilon: Dict[float, float],
        asr_by_epsilon: Dict[float, float],
        defense_auc_by_epsilon: Dict[float, float],
        architecture_name: str = "DeBERTa-V3-Large",
        dataset_name: str = "adult",
    ) -> Dict:
        tradeoff = {}
        for eps in sorted(epsilon_values):
            tradeoff[str(eps)] = {
                "accuracy": accuracy_by_epsilon.get(eps, 0.0),
                "f1": f1_by_epsilon.get(eps, 0.0),
                "drift_critical": drift_by_epsilon.get(eps, 0.0),
                "asr_a_info": asr_by_epsilon.get(eps, 0.0),
                "defense_auc_trustscore_ac": defense_auc_by_epsilon.get(eps, 0.0),
            }

        recommended_range = self._find_recommended_range(
            epsilon_values=epsilon_values,
            accuracy_by_epsilon=accuracy_by_epsilon,
            asr_by_epsilon=asr_by_epsilon,
            defense_auc_by_epsilon=defense_auc_by_epsilon,
            min_accuracy=0.81,
            max_asr=0.5,
            min_defense_auc=0.85,
        )

        output = {
            "architecture": architecture_name,
            "dataset": dataset_name,
            "tradeoff_by_epsilon": tradeoff,
            "recommended_range": recommended_range,
        }

        logger.info(f"Recommended ε range: {recommended_range}")
        return output

    def _find_recommended_range(
        self,
        epsilon_values: List[float],
        accuracy_by_epsilon: Dict[float, float],
        asr_by_epsilon: Dict[float, float],
        defense_auc_by_epsilon: Dict[float, float],
        min_accuracy: float = 0.81,
        max_asr: float = 0.5,
        min_defense_auc: float = 0.85,
    ) -> Dict:
        satisfying = []
        for eps in sorted(epsilon_values):
            if eps == float("inf"):
                continue
            acc = accuracy_by_epsilon.get(eps, 0.0)
            asr = asr_by_epsilon.get(eps, 1.0)
            auc = defense_auc_by_epsilon.get(eps, 0.0)

            if acc >= min_accuracy and asr <= max_asr and auc >= min_defense_auc:
                satisfying.append(eps)

        if satisfying:
            return {
                "min_epsilon": min(satisfying),
                "max_epsilon": max(satisfying),
                "satisfying_epsilons": satisfying,
            }
        return {"min_epsilon": None, "max_epsilon": None, "satisfying_epsilons": []}

    def run_cross_architecture_comparison(
        self,
        deberta_results: Dict,
        llama_results: Dict,
        epsilon_values: List[float],
        dataset_name: str = "adult",
    ) -> Dict:
        comparison = {
            "dataset": dataset_name,
            "epsilon_values": [str(e) for e in sorted(epsilon_values)],
            "deberta": {},
            "llama": {},
        }

        for eps in sorted(epsilon_values):
            eps_str = str(eps)
            comparison["deberta"][eps_str] = {
                "drift_critical": deberta_results.get("drift_by_epsilon", {}).get(eps, 0.0),
                "asr_a_info": deberta_results.get("asr_by_epsilon", {}).get(eps, 0.0),
                "defense_auc": deberta_results.get("defense_auc_by_epsilon", {}).get(eps, 0.0),
            }
            comparison["llama"][eps_str] = {
                "drift_critical": llama_results.get("drift_by_epsilon", {}).get(eps, 0.0),
                "asr_a_info": llama_results.get("asr_by_epsilon", {}).get(eps, 0.0),
                "defense_auc": llama_results.get("defense_auc_by_epsilon", {}).get(eps, 0.0),
            }

        return comparison

    def save_results(
        self,
        ablation: Dict,
        tradeoff: Dict,
        cross_arch: Dict,
        architecture_name: str,
        dataset_name: str,
    ):
        output = {
            "ablation": ablation,
            "tradeoff": tradeoff,
            "cross_architecture": cross_arch,
        }

        out_path = os.path.join(
            self.results_dir,
            f"exp4_{dataset_name}_{architecture_name.replace('-', '_').replace('.', '_')}.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Experiment 4 results saved to {out_path}")
        return output
