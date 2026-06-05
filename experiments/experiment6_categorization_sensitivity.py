"""Experiment 6: Categorization Sensitivity Analysis.

Tests whether the drift-vulnerability correlation is stable under alternative
feature partitions: swapped, attribution-derived (SHAP, LIME, IG), and PCA-based.
Addresses Section 4.8 of the paper.

Key metrics:
  - Spearman ρ between drift and ASR under each partition
  - Jaccard similarity of top-k drift-ranked features across partitions
  - Top-5 feature overlap between domain partition and each alternative
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from phases.phase3_attention_drift import compute_acs, AttentionDriftAnalyzer
from utils.logger import setup_logger

logger = setup_logger("experiment6")


def _jaccard(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _top_k_features(drift_per_feature: Dict[str, float], k: int = 5) -> Set[str]:
    ranked = sorted(drift_per_feature.items(), key=lambda x: abs(x[1]), reverse=True)
    return {feat for feat, _ in ranked[:k]}


def _derive_partition_from_attributions(
    attribution_scores: Dict[str, float],
    all_features: List[str],
) -> Tuple[Set[str], Set[str]]:
    """Assign features above the median attribution to critical, rest to non-functional."""
    scores = [attribution_scores.get(f, 0.0) for f in all_features]
    median_score = float(np.median(scores))
    critical = {f for f in all_features if attribution_scores.get(f, 0.0) > median_score}
    non_functional = set(all_features) - critical
    return critical, non_functional


def _build_position_maps(
    tokenizer,
    records: List[Dict],
    feature_names: List[str],
    critical_features: Set[str],
    non_functional_features: Set[str],
    max_seq_length: int = 256,
) -> Tuple[List[List[int]], List[List[int]], List[Dict[str, List[int]]]]:
    """Build token position maps for a given feature partition."""
    from utils.serialization import serialize_record, get_feature_token_positions

    critical_positions_batch = []
    non_func_positions_batch = []
    feature_positions_batch = []

    for record in records:
        serialized = serialize_record(record, feature_names)
        encoding = tokenizer(
            serialized,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        feat_pos = get_feature_token_positions(encoding, feature_names, tokenizer)
        crit_pos = []
        non_func_pos = []
        for feat, positions in feat_pos.items():
            if feat in critical_features:
                crit_pos.extend(positions)
            elif feat in non_functional_features:
                non_func_pos.extend(positions)

        critical_positions_batch.append(crit_pos)
        non_func_positions_batch.append(non_func_pos)
        feature_positions_batch.append(feat_pos)

    return critical_positions_batch, non_func_positions_batch, feature_positions_batch


class Experiment6CategorizationSensitivity:
    """Test whether drift-vulnerability findings are stable across feature partitions.

    Partitions evaluated:
      - domain: original domain-knowledge partition (baseline)
      - swapped: critical ↔ non-functional labels exchanged
      - shap: SHAP attribution-derived partition (features > median SHAP → critical)
      - lime: LIME attribution-derived partition
      - ig: Integrated Gradients attribution-derived partition
      - pca: PCA-component-based partition (top-50% variance components → critical)
    """

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
        tokenizer,
        records: List[Dict],
        feature_names: List[str],
        domain_critical: Set[str],
        domain_non_functional: Set[str],
        asr_by_epsilon: Dict[float, float],
        shap_scores: Optional[Dict[str, float]] = None,
        lime_scores: Optional[Dict[str, float]] = None,
        ig_scores: Optional[Dict[str, float]] = None,
        pca_component_variance: Optional[Dict[str, float]] = None,
        dataloader=None,
        is_decoder: bool = False,
        epsilon_values: Optional[List[float]] = None,
        top_k: int = 5,
    ) -> Dict:
        if epsilon_values is None:
            epsilon_values = sorted(models_by_epsilon.keys())

        partitions: Dict[str, Tuple[Set[str], Set[str]]] = {
            "domain": (domain_critical, domain_non_functional),
            "swapped": (domain_non_functional, domain_critical),
        }

        if shap_scores:
            c, nf = _derive_partition_from_attributions(shap_scores, feature_names)
            partitions["shap"] = (c, nf)
        if lime_scores:
            c, nf = _derive_partition_from_attributions(lime_scores, feature_names)
            partitions["lime"] = (c, nf)
        if ig_scores:
            c, nf = _derive_partition_from_attributions(ig_scores, feature_names)
            partitions["ig"] = (c, nf)
        if pca_component_variance:
            c, nf = _derive_partition_from_attributions(pca_component_variance, feature_names)
            partitions["pca"] = (c, nf)

        results_by_partition: Dict[str, Dict] = {}

        for partition_name, (critical_feats, non_func_feats) in partitions.items():
            logger.info(f"Evaluating partition: {partition_name}")

            if dataloader is not None:
                # Reuse existing dataloader; only positions change
                crit_pos_batch, non_func_pos_batch, feat_pos_batch = _build_position_maps(
                    tokenizer=tokenizer,
                    records=records,
                    feature_names=feature_names,
                    critical_features=critical_feats,
                    non_functional_features=non_func_feats,
                )

                analyzer = AttentionDriftAnalyzer(device=self.device)
                drift_by_eps: Dict[float, float] = {}
                drift_per_feature_by_eps: Dict[float, Dict[str, float]] = {}

                for eps in epsilon_values:
                    if eps not in models_by_epsilon:
                        continue
                    model = models_by_epsilon[eps]
                    result = analyzer.analyze_model(
                        model=model,
                        dataloader=dataloader,
                        epsilon=eps,
                        critical_positions_batch=crit_pos_batch,
                        non_func_positions_batch=non_func_pos_batch,
                        feature_positions_batch=feat_pos_batch,
                        is_decoder=is_decoder,
                        compute_ig=False,
                    )
                    drift_by_eps[eps] = result.drift_critical
                    drift_per_feature_by_eps[eps] = result.drift_per_feature

                # Spearman correlation with ASR
                shared_eps = [e for e in epsilon_values if e in asr_by_epsilon and e in drift_by_eps and e != float("inf")]
                if len(shared_eps) >= 3:
                    from utils.metrics import compute_spearman_with_ci
                    drift_vals = [drift_by_eps[e] for e in shared_eps]
                    asr_vals = [asr_by_epsilon[e] for e in shared_eps]
                    rho, p_val, ci_lo, ci_hi = compute_spearman_with_ci(drift_vals, asr_vals)
                else:
                    rho, p_val, ci_lo, ci_hi = None, None, None, None

                # Feature overlap with domain partition
                domain_top_k = _top_k_features(
                    drift_per_feature_by_eps.get(shared_eps[0] if shared_eps else epsilon_values[0], {}),
                    k=top_k,
                )
                partition_top_k_sets = {}
                for eps in shared_eps[:1]:
                    partition_top_k_sets[eps] = _top_k_features(drift_per_feature_by_eps.get(eps, {}), k=top_k)

                partition_top_k = partition_top_k_sets.get(shared_eps[0] if shared_eps else None, set())
                jaccard_score = _jaccard(domain_top_k, partition_top_k) if partition_name != "domain" else 1.0
                top5_overlap = len(domain_top_k & partition_top_k)

                results_by_partition[partition_name] = {
                    "spearman_rho": rho,
                    "p_value": p_val,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "jaccard_vs_domain": jaccard_score,
                    "top5_overlap_with_domain": top5_overlap,
                    "n_critical_features": len(critical_feats),
                    "n_non_functional_features": len(non_func_feats),
                }
            else:
                # No dataloader: record partition metadata only
                results_by_partition[partition_name] = {
                    "n_critical_features": len(critical_feats),
                    "n_non_functional_features": len(non_func_feats),
                    "critical_features": sorted(critical_feats),
                    "non_functional_features": sorted(non_func_feats),
                }

            logger.info(
                f"Partition '{partition_name}': "
                f"ρ={results_by_partition[partition_name].get('spearman_rho')}, "
                f"Jaccard={results_by_partition[partition_name].get('jaccard_vs_domain')}"
            )

        summary = {
            "partitions": results_by_partition,
            "top_k": top_k,
            "n_epsilons": len(epsilon_values),
        }

        out_path = os.path.join(self.results_dir, "experiment6_categorization_sensitivity.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Experiment 6 results saved to {out_path}")

        return summary
