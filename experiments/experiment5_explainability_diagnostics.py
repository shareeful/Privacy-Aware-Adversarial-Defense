import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple

from phases.phase3_attention_drift import compute_acs
from utils.logger import setup_logger
from utils.serialization import serialize_record

logger = setup_logger("experiment5")


class Experiment5ExplainabilityDiagnostics:
    def __init__(
        self,
        device: str = "cuda",
        results_dir: str = "./results",
    ):
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def generate_attention_heatmaps(
        self,
        models_by_epsilon: Dict[float, object],
        sample_records: List[Dict],
        tokenizer,
        feature_names: List[str],
        critical_features: Set[str],
        non_functional_features: Set[str],
        max_seq_length: int = 256,
        is_decoder: bool = False,
        target_epsilons: Optional[List[float]] = None,
    ) -> Dict:
        if target_epsilons is None:
            target_epsilons = [1, 10, float("inf")]

        heatmaps = {}

        for eps in target_epsilons:
            if eps not in models_by_epsilon:
                continue

            model = models_by_epsilon[eps]
            model.eval()
            eps_heatmaps = []

            for record in sample_records[:5]:
                serialized = serialize_record(record, feature_names)
                encoding = tokenizer(
                    serialized, max_length=max_seq_length, padding="max_length",
                    truncation=True, return_tensors="pt",
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                with torch.no_grad():
                    if is_decoder:
                        outputs = model.backbone(
                            input_ids=input_ids, attention_mask=attention_mask,
                            output_attentions=True,
                        )
                        seq_len = int(attention_mask.sum().item()) - 1
                        final_attn = outputs.attentions[-1][0, :, seq_len, :]
                    else:
                        outputs = model.deberta(
                            input_ids=input_ids, attention_mask=attention_mask,
                            output_attentions=True,
                        )
                        final_attn = outputs.attentions[-1][0, :, 0, :]

                mean_attn = final_attn.mean(dim=0).cpu().numpy()

                tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
                n_real = int(attention_mask.sum().item())
                real_tokens = tokens[:n_real]
                real_attn = mean_attn[:n_real].tolist()

                token_categories = []
                for token in real_tokens:
                    token_clean = token.replace("▁", "").replace("##", "").lower()
                    cat = "structural"
                    for feat in critical_features:
                        if token_clean in feat.lower():
                            cat = "critical"
                            break
                    if cat == "structural":
                        for feat in non_functional_features:
                            if token_clean in feat.lower():
                                cat = "non_functional"
                                break

                    token_categories.append(cat)

                eps_heatmaps.append({
                    "tokens": real_tokens,
                    "attention_weights": real_attn,
                    "token_categories": token_categories,
                })

            heatmaps[str(eps)] = eps_heatmaps

        return heatmaps

    def generate_feature_vulnerability_rankings(
        self,
        drift_per_feature_by_epsilon: Dict[float, Dict[str, float]],
        asr_per_feature_by_epsilon: Dict[float, Dict[str, float]],
        top_k: int = 5,
    ) -> Dict:
        rankings = {}

        for eps in sorted(drift_per_feature_by_epsilon.keys()):
            if eps == float("inf"):
                continue

            drift_map = drift_per_feature_by_epsilon[eps]
            asr_map = asr_per_feature_by_epsilon.get(eps, {})

            sorted_features = sorted(drift_map.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

            ranking_entry = []
            for feat, drift in sorted_features:
                entry = {
                    "feature": feat,
                    "drift": float(drift),
                    "asr_increase": float(asr_map.get(feat, 0.0)),
                }
                ranking_entry.append(entry)

            rankings[str(eps)] = ranking_entry

        return rankings

    def compute_consistency_score_distributions(
        self,
        consistency_scorer,
        model,
        authentic_dataloader,
        pgd_dataloader,
        manifold_dataloader,
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        feature_positions_batch: List[Dict],
        is_decoder: bool = False,
    ) -> Dict:
        auth_ac, _, auth_jsd = consistency_scorer.score_batch(
            model=model, dataloader=authentic_dataloader,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
            feature_positions_batch=feature_positions_batch,
            device=self.device, is_decoder=is_decoder,
        )

        pgd_ac, _, pgd_jsd = consistency_scorer.score_batch(
            model=model, dataloader=pgd_dataloader,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
            feature_positions_batch=feature_positions_batch,
            device=self.device, is_decoder=is_decoder,
        )

        manifold_ac, _, manifold_jsd = consistency_scorer.score_batch(
            model=model, dataloader=manifold_dataloader,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
            feature_positions_batch=feature_positions_batch,
            device=self.device, is_decoder=is_decoder,
        )

        return {
            "a_ac": {
                "authentic": {"mean": float(np.mean(auth_ac)), "std": float(np.std(auth_ac)),
                              "values": auth_ac.tolist()},
                "off_manifold_pgd": {"mean": float(np.mean(pgd_ac)), "std": float(np.std(pgd_ac)),
                                     "values": pgd_ac.tolist()},
                "manifold_aligned": {"mean": float(np.mean(manifold_ac)), "std": float(np.std(manifold_ac)),
                                     "values": manifold_ac.tolist()},
            },
            "a_jsd": {
                "authentic": {"mean": float(np.mean(auth_jsd)), "std": float(np.std(auth_jsd))},
                "off_manifold_pgd": {"mean": float(np.mean(pgd_jsd)), "std": float(np.std(pgd_jsd))},
                "manifold_aligned": {"mean": float(np.mean(manifold_jsd)), "std": float(np.std(manifold_jsd))},
            },
        }

    def measure_inference_overhead(
        self,
        model,
        sample_dataloader,
        consistency_scorer,
        critical_positions_batch,
        non_func_positions_batch,
        feature_positions_batch,
        n_trials: int = 100,
        is_decoder: bool = False,
    ) -> Dict:
        import time

        model.eval()
        base_times = []
        ac_times = []
        ig_times = []
        jsd_times = []

        for i, batch in enumerate(sample_dataloader):
            if i >= n_trials:
                break
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            start = time.perf_counter()
            with torch.no_grad():
                model.predict(input_ids, attention_mask)
            base_times.append(time.perf_counter() - start)

        for i, batch in enumerate(sample_dataloader):
            if i >= n_trials:
                break
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            start = time.perf_counter()
            with torch.no_grad():
                if is_decoder:
                    outputs = model.backbone(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
                    final_attn = outputs.attentions[-1]
                else:
                    outputs = model.deberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
                    final_attn = outputs.attentions[-1]
            ac_times.append(time.perf_counter() - start)

        overhead_ac = float(np.mean(ac_times)) - float(np.mean(base_times))
        overhead_ig_estimated = float(np.mean(base_times)) * consistency_scorer.ig_steps

        return {
            "base_latency_ms": float(np.mean(base_times)) * 1000,
            "ac_overhead_ms": overhead_ac * 1000,
            "ig_estimated_overhead_ms": overhead_ig_estimated * 1000,
            "n_ig_steps": consistency_scorer.ig_steps,
        }

    def save_results(
        self,
        heatmaps: Dict,
        feature_rankings: Dict,
        consistency_distributions: Dict,
        inference_overhead: Dict,
        architecture_name: str,
        dataset_name: str,
    ):
        output = {
            "architecture": architecture_name,
            "dataset": dataset_name,
            "attention_heatmaps": heatmaps,
            "feature_vulnerability_rankings": feature_rankings,
            "consistency_score_distributions": consistency_distributions,
            "inference_overhead": inference_overhead,
        }

        out_path = os.path.join(
            self.results_dir,
            f"exp5_{dataset_name}_{architecture_name.replace('-', '_').replace('.', '_')}.json"
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Experiment 5 results saved to {out_path}")
        return output
