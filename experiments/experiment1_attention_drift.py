import os
import json
import numpy as np
from typing import Dict, List, Optional

from phases.phase3_attention_drift import AttentionDriftAnalyzer, DriftAnalysisResult
from utils.logger import setup_logger
from utils.metrics import compute_spearman_with_ci

logger = setup_logger("experiment1")


class Experiment1AttentionDrift:
    def __init__(
        self,
        device: str = "cuda",
        ig_steps: int = 50,
        results_dir: str = "./results",
    ):
        self.device = device
        self.ig_steps = ig_steps
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def run(
        self,
        models_by_epsilon: Dict[float, object],
        dataloader,
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        feature_positions_batch: List[Dict],
        asr_by_epsilon: Optional[Dict[float, float]] = None,
        per_sample_drift_by_epsilon: Optional[Dict[float, np.ndarray]] = None,
        per_sample_attack_outcomes: Optional[Dict[float, np.ndarray]] = None,
        is_decoder: bool = False,
        architecture_name: str = "DeBERTa-V3-Large",
        dataset_name: str = "adult",
        compute_ig: bool = True,
    ) -> Dict:
        analyzer = AttentionDriftAnalyzer(device=self.device, ig_steps=self.ig_steps)

        epsilon_values = sorted(
            [e for e in models_by_epsilon.keys() if e != float("inf")]
        ) + [float("inf")]

        drift_results = {}
        inf_model = models_by_epsilon[float("inf")]
        analyzer.analyze_model(
            model=inf_model,
            dataloader=dataloader,
            epsilon=float("inf"),
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
            feature_positions_batch=feature_positions_batch,
            is_decoder=is_decoder,
            compute_ig=compute_ig,
        )
        drift_results[float("inf")] = DriftAnalysisResult(
            epsilon=float("inf"),
            acs_critical=analyzer.baseline_acs["critical"],
            acs_non_functional=analyzer.baseline_acs["non_functional"],
            acs_per_feature=analyzer.baseline_acs["per_feature"],
            igs_critical=analyzer.baseline_igs[0] if analyzer.baseline_igs else None,
            igs_non_functional=analyzer.baseline_igs[1] if analyzer.baseline_igs else None,
            drift_critical=0.0,
            drift_non_functional=0.0,
            drift_per_feature={f: 0.0 for f in analyzer.baseline_acs["per_feature"]},
        )

        for epsilon in sorted([e for e in models_by_epsilon.keys() if e != float("inf")]):
            model = models_by_epsilon[epsilon]
            result = analyzer.analyze_model(
                model=model,
                dataloader=dataloader,
                epsilon=epsilon,
                critical_positions_batch=critical_positions_batch,
                non_func_positions_batch=non_func_positions_batch,
                feature_positions_batch=feature_positions_batch,
                is_decoder=is_decoder,
                compute_ig=compute_ig,
            )
            drift_results[epsilon] = result

        macro_correlation = {}
        if asr_by_epsilon is not None:
            common_epsilons = [
                e for e in drift_results.keys()
                if e != float("inf") and e in asr_by_epsilon
            ]
            drift_vals = [drift_results[e].drift_critical for e in common_epsilons]
            asr_vals = [asr_by_epsilon[e] for e in common_epsilons]

            if len(drift_vals) >= 3:
                macro_correlation = analyzer.compute_drift_vulnerability_correlation(
                    drift_values=drift_vals, asr_values=asr_vals
                )

        micro_correlation = {}
        if per_sample_drift_by_epsilon is not None and per_sample_attack_outcomes is not None:
            all_drift = np.concatenate([
                per_sample_drift_by_epsilon[e] for e in per_sample_drift_by_epsilon
            ])
            all_outcomes = np.concatenate([
                per_sample_attack_outcomes[e] for e in per_sample_attack_outcomes
            ])
            micro_correlation = analyzer.compute_per_sample_drift_vulnerability(
                per_sample_drift=all_drift, attack_outcomes=all_outcomes
            )

        acs_drift_series = [
            drift_results[e].drift_critical
            for e in sorted([k for k in drift_results.keys() if k != float("inf")])
        ]
        igs_drift_series = [
            (drift_results[e].igs_critical if drift_results[e].igs_critical is not None
             else None)
            for e in sorted([k for k in drift_results.keys() if k != float("inf")])
        ]
        igs_drift_vals = [
            v for v in igs_drift_series if v is not None
        ]
        acs_for_igs_validation = acs_drift_series[:len(igs_drift_vals)]
        ig_validation = {}
        if acs_for_igs_validation and igs_drift_vals:
            ig_validation = analyzer.validate_acs_vs_igs(acs_for_igs_validation, igs_drift_vals)

        feature_rankings = {}
        for epsilon, result in drift_results.items():
            if epsilon != float("inf"):
                ranked = analyzer.rank_features_by_drift(
                    result.drift_per_feature, threshold=0.0
                )
                feature_rankings[epsilon] = ranked

        output = {
            "architecture": architecture_name,
            "dataset": dataset_name,
            "drift_results": {
                str(e): {
                    "acs_critical": r.acs_critical,
                    "acs_non_functional": r.acs_non_functional,
                    "acs_per_feature": r.acs_per_feature,
                    "drift_critical": r.drift_critical,
                    "drift_non_functional": r.drift_non_functional,
                    "drift_per_feature": r.drift_per_feature,
                }
                for e, r in drift_results.items()
            },
            "macro_correlation": macro_correlation,
            "micro_correlation": micro_correlation,
            "ig_validation": ig_validation,
            "feature_rankings": {
                str(e): [(f, d) for f, d in ranking]
                for e, ranking in feature_rankings.items()
            },
        }

        out_path = os.path.join(
            self.results_dir,
            f"exp1_{dataset_name}_{architecture_name.replace('-', '_').replace('.', '_')}.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Experiment 1 results saved to {out_path}")
        return output
