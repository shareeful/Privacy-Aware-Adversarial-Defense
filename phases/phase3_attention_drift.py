import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from utils.logger import setup_logger
from utils.metrics import compute_spearman_with_ci, compute_odds_ratio_logistic

logger = setup_logger("phase3_attention_drift")


@dataclass
class DriftAnalysisResult:
    epsilon: float
    acs_critical: float
    acs_non_functional: float
    acs_per_feature: Dict[str, float]
    igs_critical: Optional[float]
    igs_non_functional: Optional[float]
    drift_critical: float
    drift_non_functional: float
    drift_per_feature: Dict[str, float]


def compute_acs(
    model,
    dataloader,
    critical_positions_batch: List[List[int]],
    non_func_positions_batch: List[List[int]],
    feature_positions_batch: List[Dict[str, List[int]]],
    device: str = "cuda",
    is_decoder: bool = False,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    all_critical_attention = []
    all_non_func_attention = []
    all_feature_attention = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            batch_size = input_ids.shape[0]

            if is_decoder:
                outputs = model.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                final_layer_attention = outputs.attentions[-1]
                sequence_lengths = attention_mask.sum(dim=1) - 1
                readout_attention = torch.stack([
                    final_layer_attention[b, :, sequence_lengths[b], :]
                    for b in range(batch_size)
                ], dim=0)
            else:
                outputs = model.deberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True,
                )
                final_layer_attention = outputs.attentions[-1]
                readout_attention = final_layer_attention[:, :, 0, :]

            mean_head_attention = readout_attention.mean(dim=1)

            for b in range(batch_size):
                global_b = batch_idx * dataloader.batch_size + b
                attn = mean_head_attention[b].cpu().numpy()
                seq_len = len(attn)

                if global_b < len(critical_positions_batch):
                    crit_pos = [p for p in critical_positions_batch[global_b] if p < seq_len]
                    non_func_pos = [p for p in non_func_positions_batch[global_b] if p < seq_len]
                    feat_pos = feature_positions_batch[global_b] if global_b < len(feature_positions_batch) else {}
                else:
                    crit_pos = []
                    non_func_pos = []
                    feat_pos = {}

                crit_mass = float(np.sum(attn[crit_pos])) if crit_pos else 0.0
                non_func_mass = float(np.sum(attn[non_func_pos])) if non_func_pos else 0.0

                all_critical_attention.append(crit_mass)
                all_non_func_attention.append(non_func_mass)

                for feat_name, positions in feat_pos.items():
                    valid_pos = [p for p in positions if p < seq_len]
                    feat_mass = float(np.sum(attn[valid_pos])) if valid_pos else 0.0
                    if feat_name not in all_feature_attention:
                        all_feature_attention[feat_name] = []
                    all_feature_attention[feat_name].append(feat_mass)

    acs_critical = float(np.mean(all_critical_attention)) if all_critical_attention else 0.0
    acs_non_functional = float(np.mean(all_non_func_attention)) if all_non_func_attention else 0.0
    acs_per_feature = {
        feat: float(np.mean(vals)) for feat, vals in all_feature_attention.items()
    }

    return acs_critical, acs_non_functional, acs_per_feature


def compute_integrated_gradients(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    baseline_input_ids: torch.Tensor,
    token_positions: List[int],
    n_steps: int = 50,
    device: str = "cuda",
    is_decoder: bool = False,
) -> np.ndarray:
    model.eval()
    seq_len = input_ids.shape[1]
    token_attributions = np.zeros(seq_len)

    try:
        embedding_layer = (
            model.backbone.model.embed_tokens
            if is_decoder
            else model.deberta.embeddings.word_embeddings
        )

        baseline_embeds = embedding_layer(baseline_input_ids.to(device)).detach()
        input_embeds = embedding_layer(input_ids.to(device)).detach()

        integrated_grads = torch.zeros_like(input_embeds)

        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            interpolated = (baseline_embeds + alpha * (input_embeds - baseline_embeds)).requires_grad_(True)

            if is_decoder:
                outputs = model.backbone(
                    inputs_embeds=interpolated,
                    attention_mask=attention_mask.to(device),
                    output_hidden_states=True,
                )
                seq_lengths = attention_mask.sum(dim=1) - 1
                hidden = outputs.hidden_states[-1][0, seq_lengths[0]]
                logit = model.classifier(hidden.float())
            else:
                outputs = model.deberta(
                    inputs_embeds=interpolated,
                    attention_mask=attention_mask.to(device),
                    output_hidden_states=True,
                )
                cls_hidden = outputs.last_hidden_state[0, 0]
                logit = model.classifier(cls_hidden)

            logit.backward()
            integrated_grads += interpolated.grad.detach()

        integrated_grads = integrated_grads / n_steps
        delta_embeds = input_embeds - baseline_embeds
        final_ig = (integrated_grads * delta_embeds).sum(dim=-1)

        for pos in token_positions:
            if pos < seq_len:
                token_attributions[pos] = abs(float(final_ig[0, pos].item()))

    except Exception as e:
        logger.warning(f"IG computation failed: {e}. Returning zeros.")

    return token_attributions


def compute_igs_score(
    model,
    dataloader,
    critical_positions_batch: List[List[int]],
    non_func_positions_batch: List[List[int]],
    device: str = "cuda",
    is_decoder: bool = False,
    n_steps: int = 50,
) -> Tuple[float, float]:
    model.eval()
    all_crit_ig = []
    all_non_func_ig = []

    tokenizer_vocab_size = (
        model.backbone.config.vocab_size if is_decoder else model.deberta.config.vocab_size
    )
    baseline_token_id = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_size = input_ids.shape[0]

        baseline = torch.zeros_like(input_ids)
        baseline[:, 0] = input_ids[:, 0]

        for b in range(min(batch_size, 5)):
            global_b = batch_idx * dataloader.batch_size + b
            crit_pos = critical_positions_batch[global_b] if global_b < len(critical_positions_batch) else []
            non_func_pos = non_func_positions_batch[global_b] if global_b < len(non_func_positions_batch) else []

            all_positions = list(set(crit_pos + non_func_pos))
            if not all_positions:
                continue

            ig_vals = compute_integrated_gradients(
                model=model,
                input_ids=input_ids[b:b+1],
                attention_mask=attention_mask[b:b+1],
                baseline_input_ids=baseline[b:b+1],
                token_positions=all_positions,
                n_steps=n_steps,
                device=device,
                is_decoder=is_decoder,
            )

            crit_ig = sum(ig_vals[p] for p in crit_pos if p < len(ig_vals))
            non_func_ig = sum(ig_vals[p] for p in non_func_pos if p < len(ig_vals))

            all_crit_ig.append(crit_ig)
            all_non_func_ig.append(non_func_ig)

    igs_critical = float(np.mean(all_crit_ig)) if all_crit_ig else 0.0
    igs_non_functional = float(np.mean(all_non_func_ig)) if all_non_func_ig else 0.0
    return igs_critical, igs_non_functional


class AttentionDriftAnalyzer:
    def __init__(self, device: str = "cuda", ig_steps: int = 50):
        self.device = device
        self.ig_steps = ig_steps
        self.baseline_acs: Optional[Dict] = None
        self.baseline_igs: Optional[Tuple[float, float]] = None

    def analyze_model(
        self,
        model,
        dataloader,
        epsilon: float,
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        feature_positions_batch: List[Dict[str, List[int]]],
        is_decoder: bool = False,
        compute_ig: bool = True,
    ) -> DriftAnalysisResult:
        logger.info(f"Computing ACS for ε={epsilon}...")

        acs_critical, acs_non_func, acs_per_feature = compute_acs(
            model=model,
            dataloader=dataloader,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
            feature_positions_batch=feature_positions_batch,
            device=self.device,
            is_decoder=is_decoder,
        )

        igs_critical, igs_non_func = None, None
        if compute_ig:
            logger.info(f"Computing IGS for ε={epsilon}...")
            igs_critical, igs_non_func = compute_igs_score(
                model=model,
                dataloader=dataloader,
                critical_positions_batch=critical_positions_batch,
                non_func_positions_batch=non_func_positions_batch,
                device=self.device,
                is_decoder=is_decoder,
                n_steps=self.ig_steps,
            )

        if epsilon == float("inf"):
            self.baseline_acs = {
                "critical": acs_critical,
                "non_functional": acs_non_func,
                "per_feature": acs_per_feature,
            }
            if igs_critical is not None:
                self.baseline_igs = (igs_critical, igs_non_func)
            drift_critical = 0.0
            drift_non_func = 0.0
            drift_per_feature = {feat: 0.0 for feat in acs_per_feature}
        else:
            if self.baseline_acs is None:
                raise ValueError("Must compute baseline (ε=∞) before computing drift.")
            drift_critical = self.baseline_acs["critical"] - acs_critical
            drift_non_func = self.baseline_acs["non_functional"] - acs_non_func
            drift_per_feature = {}
            for feat in acs_per_feature:
                baseline_val = self.baseline_acs["per_feature"].get(feat, 0.0)
                drift_per_feature[feat] = baseline_val - acs_per_feature[feat]

        result = DriftAnalysisResult(
            epsilon=epsilon,
            acs_critical=acs_critical,
            acs_non_functional=acs_non_func,
            acs_per_feature=acs_per_feature,
            igs_critical=igs_critical,
            igs_non_functional=igs_non_func,
            drift_critical=drift_critical,
            drift_non_functional=drift_non_func,
            drift_per_feature=drift_per_feature,
        )

        logger.info(
            f"ε={epsilon}: ACS_critical={acs_critical:.4f}, "
            f"ACS_non_func={acs_non_func:.4f}, "
            f"Drift_critical={drift_critical:.4f}"
        )

        return result

    def compute_drift_vulnerability_correlation(
        self,
        drift_values: List[float],
        asr_values: List[float],
    ) -> Dict:
        rho, p_value, ci_lower, ci_upper = compute_spearman_with_ci(
            x=drift_values, y=asr_values, n_bootstrap=1000, confidence=0.95
        )
        logger.info(
            f"Macro Spearman: ρ={rho:.4f}, p={p_value:.4e}, "
            f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
        )
        return {
            "spearman_rho": rho,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def compute_per_sample_drift_vulnerability(
        self,
        per_sample_drift: np.ndarray,
        attack_outcomes: np.ndarray,
    ) -> Dict:
        or_val, p_val, ci_lower, ci_upper = compute_odds_ratio_logistic(
            per_sample_drift, attack_outcomes
        )
        logger.info(
            f"Micro logistic: OR={or_val:.4f}, p={p_val:.4e}, "
            f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
        )
        return {
            "odds_ratio": or_val,
            "p_value": p_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def rank_features_by_drift(
        self, drift_per_feature: Dict[str, float], threshold: float = 0.05
    ) -> List[Tuple[str, float]]:
        ranked = sorted(drift_per_feature.items(), key=lambda x: abs(x[1]), reverse=True)
        filtered = [(feat, drift) for feat, drift in ranked if abs(drift) > threshold]
        return filtered

    def compute_class_conditional_drift(
        self,
        model,
        dataloader,
        labels: np.ndarray,
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        epsilon: float,
        is_decoder: bool = False,
    ) -> Dict:
        """Compute per-sample attention drift separated by class (Section 4.4.1).

        Returns mean per-sample drift for the positive (minority) and negative
        (majority) classes, plus the over-representation ratio of minority class
        among successfully attacked inputs.
        """
        if self.baseline_acs is None:
            raise ValueError("Must compute baseline (ε=∞) before class-conditional drift.")

        model.eval()
        per_sample_crit = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                batch_size = input_ids.shape[0]

                if is_decoder:
                    outputs = model.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True,
                    )
                    final_attn = outputs.attentions[-1]
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    readout = torch.stack([
                        final_attn[b, :, seq_lengths[b], :]
                        for b in range(batch_size)
                    ], dim=0)
                else:
                    outputs = model.deberta(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        output_attentions=True,
                    )
                    final_attn = outputs.attentions[-1]
                    readout = final_attn[:, :, 0, :]

                mean_head = readout.mean(dim=1)

                for b in range(batch_size):
                    global_b = batch_idx * dataloader.batch_size + b
                    attn = mean_head[b].cpu().numpy()
                    seq_len = len(attn)

                    crit_pos = [p for p in critical_positions_batch[global_b] if p < seq_len] if global_b < len(critical_positions_batch) else []
                    crit_mass = float(np.sum(attn[crit_pos])) if crit_pos else 0.0
                    per_sample_crit.append(crit_mass)

        per_sample_crit = np.array(per_sample_crit[:len(labels)])
        baseline_critical = self.baseline_acs["critical"]
        per_sample_drift = baseline_critical - per_sample_crit

        minority_mask = labels == 1
        majority_mask = labels == 0

        mean_drift_minority = float(np.mean(per_sample_drift[minority_mask])) if minority_mask.any() else 0.0
        mean_drift_majority = float(np.mean(per_sample_drift[majority_mask])) if majority_mask.any() else 0.0

        logger.info(
            f"ε={epsilon}: minority drift={mean_drift_minority:.4f}, "
            f"majority drift={mean_drift_majority:.4f}, "
            f"ratio={mean_drift_minority / (mean_drift_majority + 1e-8):.2f}×"
        )

        return {
            "epsilon": epsilon,
            "mean_drift_minority_class": mean_drift_minority,
            "mean_drift_majority_class": mean_drift_majority,
            "drift_ratio_minority_over_majority": mean_drift_minority / (mean_drift_majority + 1e-8),
            "per_sample_drift": per_sample_drift.tolist(),
            "minority_base_rate": float(minority_mask.mean()),
        }

    def compute_attack_class_composition(
        self,
        attack_outcomes: np.ndarray,
        labels: np.ndarray,
    ) -> Dict:
        """Compute fraction of successful attacks that target minority vs. majority class.

        A value well above the minority base rate confirms that privacy noise
        disproportionately destabilises the minority class (Section 4.4.1, Figure 5b).
        """
        successful_mask = attack_outcomes == 1
        if not successful_mask.any():
            return {"minority_fraction_among_successes": 0.0, "minority_base_rate": float((labels == 1).mean())}

        labels_of_successes = labels[successful_mask]
        minority_fraction = float((labels_of_successes == 1).mean())
        minority_base_rate = float((labels == 1).mean())

        logger.info(
            f"Attack composition: minority {minority_fraction:.2%} of successes "
            f"(base rate {minority_base_rate:.2%})"
        )
        return {
            "minority_fraction_among_successes": minority_fraction,
            "minority_base_rate": minority_base_rate,
            "over_representation_factor": minority_fraction / (minority_base_rate + 1e-8),
        }

    def validate_acs_vs_igs(
        self,
        acs_drift_series: List[float],
        igs_drift_series: List[float],
    ) -> Dict:
        valid_pairs = [
            (a, i) for a, i in zip(acs_drift_series, igs_drift_series)
            if i is not None
        ]
        if len(valid_pairs) < 3:
            return {"spearman_rho": None, "p_value": None}

        acs_vals = [p[0] for p in valid_pairs]
        igs_vals = [p[1] for p in valid_pairs]
        rho, p_value, ci_lower, ci_upper = compute_spearman_with_ci(acs_vals, igs_vals)
        logger.info(f"ACS vs IGS validation: ρ={rho:.4f}, p={p_value:.4e}")
        return {
            "spearman_rho": rho,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
