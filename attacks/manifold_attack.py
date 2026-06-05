import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from configs import AttackConfig
from utils.logger import setup_logger
from utils.serialization import serialize_record

logger = setup_logger("attack")


@dataclass
class AttackResult:
    original_record: Dict
    adversarial_record: Dict
    original_label: int
    original_prediction: int
    adversarial_prediction: int
    adversarial_confidence: float
    plausibility_score: float
    n_queries: int
    success: bool
    features_modified: List[str]


class CooccurrenceModel:
    def __init__(self, cooccurrence_data: Dict[str, Dict]):
        self.data = cooccurrence_data

    def conditional_probability(self, feature: str, value, context: Dict) -> float:
        if feature not in self.data:
            return 0.01

        feature_dist = self.data[feature]
        str_val = str(value)

        if str_val in feature_dist:
            return float(feature_dist[str_val])

        return 1e-6

    def plausibility_score(self, record: Dict, mutable_features: List[str]) -> float:
        score = 1.0
        for feat in mutable_features:
            if feat not in record:
                continue
            context = {k: v for k, v in record.items() if k != feat}
            p = self.conditional_probability(feat, record[feat], context)
            score *= p
        return score

    def get_valid_substitutions(
        self,
        feature: str,
        current_record: Dict,
        feature_domains: Dict[str, List],
        plausibility_threshold: float,
        n_mutable: int,
    ) -> List:
        if feature not in feature_domains:
            return []

        domain = feature_domains[feature]
        current_val = current_record.get(feature)
        local_threshold = plausibility_threshold / max(n_mutable, 1)

        valid_values = []
        for val in domain:
            if str(val) == str(current_val):
                continue
            context = {k: v for k, v in current_record.items() if k != feature}
            p = self.conditional_probability(feature, val, context)
            if p >= local_threshold:
                valid_values.append(val)

        return valid_values


class ManifoldAlignedAttack:
    def __init__(
        self,
        model,
        tokenizer,
        feature_names: List[str],
        non_functional_features: Set[str],
        cooccurrence_model: CooccurrenceModel,
        feature_domains: Dict[str, List],
        config: AttackConfig,
        device: str = "cuda",
        max_seq_length: int = 256,
        is_decoder: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.feature_names = feature_names
        self.non_functional_features = non_functional_features
        self.cooccurrence_model = cooccurrence_model
        self.feature_domains = feature_domains
        self.config = config
        self.device = device
        self.max_seq_length = max_seq_length
        self.is_decoder = is_decoder

    def _predict_confidence(self, record: Dict, target_label: int) -> Tuple[float, int]:
        serialized = serialize_record(record, self.feature_names)
        encoding = self.tokenizer(
            serialized,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            preds, probs = self.model.predict(input_ids, attention_mask)

        prob = float(probs[0].cpu().item())
        pred = int(preds[0].cpu().item())

        if target_label == 0:
            confidence_in_wrong = 1.0 - prob
        else:
            confidence_in_wrong = prob

        return confidence_in_wrong, pred

    def attack_practical(
        self, record: Dict, true_label: int, drift_rankings: List[Tuple[str, float]] = None
    ) -> AttackResult:
        mutable_features = [
            feat for feat in self.feature_names
            if feat in self.non_functional_features
        ]
        return self._greedy_attack(
            record=record,
            true_label=true_label,
            mutable_features=mutable_features,
            drift_guided=False,
        )

    def attack_informed(
        self, record: Dict, true_label: int, drift_rankings: List[Tuple[str, float]]
    ) -> AttackResult:
        ranked_non_func = [
            (feat, drift) for feat, drift in drift_rankings
            if feat in self.non_functional_features and abs(drift) > self.config.drift_threshold
        ]

        if not ranked_non_func:
            ranked_non_func = [
                (feat, 0.0) for feat in self.feature_names
                if feat in self.non_functional_features
            ]

        mutable_features = [feat for feat, _ in ranked_non_func]
        return self._greedy_attack(
            record=record,
            true_label=true_label,
            mutable_features=mutable_features,
            drift_guided=True,
            drift_rankings=ranked_non_func,
        )

    def _greedy_attack(
        self,
        record: Dict,
        true_label: int,
        mutable_features: List[str],
        drift_guided: bool,
        drift_rankings: List[Tuple[str, float]] = None,
    ) -> AttackResult:
        current_record = dict(record)
        target_label = 1 - true_label

        initial_conf, initial_pred = self._predict_confidence(current_record, target_label)
        best_conf = initial_conf
        best_record = dict(current_record)
        n_queries = 0
        features_modified = []

        if drift_guided and drift_rankings:
            sorted_features = [feat for feat, _ in drift_rankings if feat in mutable_features]
        else:
            sorted_features = mutable_features.copy()

        for feature in sorted_features:
            if n_queries >= self.config.query_budget:
                break

            valid_subs = self.cooccurrence_model.get_valid_substitutions(
                feature=feature,
                current_record=current_record,
                feature_domains=self.feature_domains,
                plausibility_threshold=self.config.plausibility_threshold,
                n_mutable=len(mutable_features),
            )

            if not valid_subs:
                continue

            for val in valid_subs:
                if n_queries >= self.config.query_budget:
                    break

                candidate_record = dict(current_record)
                candidate_record[feature] = val

                plausibility = self.cooccurrence_model.plausibility_score(
                    candidate_record, mutable_features
                )

                if plausibility < self.config.plausibility_threshold:
                    continue

                conf, pred = self._predict_confidence(candidate_record, target_label)
                n_queries += 1

                if conf > best_conf and plausibility >= self.config.plausibility_threshold:
                    best_conf = conf
                    best_record = dict(candidate_record)
                    if feature not in features_modified:
                        features_modified.append(feature)

            current_record = dict(best_record)

        _, final_pred = self._predict_confidence(best_record, target_label)
        success = final_pred != true_label

        final_plausibility = self.cooccurrence_model.plausibility_score(
            best_record, mutable_features
        )

        return AttackResult(
            original_record=dict(record),
            adversarial_record=best_record,
            original_label=true_label,
            original_prediction=initial_pred,
            adversarial_prediction=final_pred,
            adversarial_confidence=best_conf,
            plausibility_score=final_plausibility,
            n_queries=n_queries,
            success=success,
            features_modified=features_modified,
        )


class PGDAttack:
    def __init__(
        self,
        model,
        step_size: float = 0.01,
        epsilon: float = 0.1,
        n_iterations: int = 40,
        device: str = "cuda",
    ):
        self.model = model
        self.step_size = step_size
        self.epsilon = epsilon
        self.n_iterations = n_iterations
        self.device = device

    def attack(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        is_decoder: bool = False,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        self.model.eval()

        if is_decoder:
            embedding_layer = self.model.backbone.model.embed_tokens
        else:
            embedding_layer = self.model.deberta.embeddings.word_embeddings

        with torch.no_grad():
            original_embeddings = embedding_layer(input_ids.to(self.device)).detach()

        adv_embeddings = original_embeddings.clone() + torch.empty_like(
            original_embeddings
        ).uniform_(-self.epsilon, self.epsilon)

        for _ in range(self.n_iterations):
            adv_embeddings.requires_grad_(True)

            if is_decoder:
                outputs = self.model.backbone(
                    inputs_embeds=adv_embeddings,
                    attention_mask=attention_mask.to(self.device),
                    output_hidden_states=True,
                )
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = input_ids.shape[0]
                hidden = torch.stack([
                    outputs.hidden_states[-1][b, seq_lengths[b]]
                    for b in range(batch_size)
                ])
                probs = self.model.sigmoid(self.model.classifier(hidden.float())).squeeze(-1)
            else:
                outputs = self.model.deberta(
                    inputs_embeds=adv_embeddings,
                    attention_mask=attention_mask.to(self.device),
                )
                cls = outputs.last_hidden_state[:, 0, :]
                probs = self.model.sigmoid(self.model.classifier(cls)).squeeze(-1)

            loss = -torch.nn.BCELoss()(probs, labels.float().to(self.device))
            loss.backward()

            with torch.no_grad():
                grad_sign = adv_embeddings.grad.sign()
                adv_embeddings = adv_embeddings + self.step_size * grad_sign
                delta = torch.clamp(adv_embeddings - original_embeddings, -self.epsilon, self.epsilon)
                adv_embeddings = original_embeddings + delta
                adv_embeddings = adv_embeddings.detach()

        with torch.no_grad():
            if is_decoder:
                outputs = self.model.backbone(
                    inputs_embeds=adv_embeddings,
                    attention_mask=attention_mask.to(self.device),
                    output_hidden_states=True,
                )
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_size = input_ids.shape[0]
                hidden = torch.stack([
                    outputs.hidden_states[-1][b, seq_lengths[b]]
                    for b in range(batch_size)
                ])
                adv_probs = self.model.sigmoid(self.model.classifier(hidden.float())).squeeze(-1)
            else:
                outputs = self.model.deberta(
                    inputs_embeds=adv_embeddings,
                    attention_mask=attention_mask.to(self.device),
                )
                cls = outputs.last_hidden_state[:, 0, :]
                adv_probs = self.model.sigmoid(self.model.classifier(cls)).squeeze(-1)

            adv_preds = (adv_probs > 0.5).long().cpu().numpy()

        return adv_embeddings.detach(), adv_preds


class HopSkipJumpAttack:
    def __init__(
        self,
        model,
        tokenizer,
        feature_names: List[str],
        feature_domains: Dict[str, List],
        n_iterations: int = 50,
        device: str = "cuda",
        max_seq_length: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.feature_names = feature_names
        self.feature_domains = feature_domains
        self.n_iterations = n_iterations
        self.device = device
        self.max_seq_length = max_seq_length

    def _predict(self, record: Dict) -> Tuple[int, float]:
        serialized = serialize_record(record, self.feature_names)
        encoding = self.tokenizer(
            serialized,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            preds, probs = self.model.predict(input_ids, attention_mask)
        return int(preds[0].item()), float(probs[0].item())

    def attack(self, record: Dict, true_label: int) -> AttackResult:
        current_record = dict(record)
        categorical_features = [
            f for f in self.feature_names if f in self.feature_domains
            and isinstance(self.feature_domains[f], list)
            and len(self.feature_domains[f]) > 1
        ]

        _, initial_pred = self._predict(current_record)
        initial_pred_label = int(initial_pred > 0.5)
        n_queries = 0
        best_record = dict(current_record)
        best_pred_label = initial_pred_label

        rng = np.random.RandomState(42)
        for _ in range(self.n_iterations):
            if n_queries >= 500:
                break

            candidate = dict(current_record)
            n_perturb = rng.randint(1, min(3, len(categorical_features)) + 1)
            feats_to_change = rng.choice(categorical_features, size=n_perturb, replace=False)

            for feat in feats_to_change:
                domain = self.feature_domains[feat]
                current_val = current_record.get(feat)
                options = [v for v in domain if str(v) != str(current_val)]
                if options:
                    candidate[feat] = rng.choice(options)

            pred_label, _ = self._predict(candidate)
            n_queries += 1

            if pred_label != true_label:
                best_record = dict(candidate)
                best_pred_label = pred_label
                break

        success = best_pred_label != true_label
        return AttackResult(
            original_record=dict(record),
            adversarial_record=best_record,
            original_label=true_label,
            original_prediction=initial_pred_label,
            adversarial_prediction=best_pred_label,
            adversarial_confidence=0.0,
            plausibility_score=0.0,
            n_queries=n_queries,
            success=success,
            features_modified=[],
        )


class AdaptiveManifoldAttack:
    """Adaptive adversary aware of the attention-consistency detector (Section 3.5.4).

    Augments the Ainfo objective with a penalty that keeps |R(x') - mu_R| below
    the detection threshold, accepting a substitution only when it simultaneously
    raises misclassification confidence and preserves the authentic attention ratio.
    """

    def __init__(
        self,
        model,
        tokenizer,
        feature_names: List[str],
        non_functional_features: Set[str],
        cooccurrence_model: CooccurrenceModel,
        feature_domains: Dict[str, List],
        config: AttackConfig,
        mu_R: float,
        sigma_R: float,
        detection_threshold_sigma: float = 2.0,
        device: str = "cuda",
        max_seq_length: int = 256,
        is_decoder: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.feature_names = feature_names
        self.non_functional_features = non_functional_features
        self.cooccurrence_model = cooccurrence_model
        self.feature_domains = feature_domains
        self.config = config
        self.mu_R = mu_R
        self.sigma_R = sigma_R
        # Accept a substitution only if a_AC = |R(x') - mu_R| / sigma_R < this bound
        self.detection_threshold_sigma = detection_threshold_sigma
        self.device = device
        self.max_seq_length = max_seq_length
        self.is_decoder = is_decoder

    def _predict_confidence(self, record: Dict, target_label: int) -> Tuple[float, int]:
        serialized = serialize_record(record, self.feature_names)
        encoding = self.tokenizer(
            serialized,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            preds, probs = self.model.predict(input_ids, attention_mask)
        prob = float(probs[0].cpu().item())
        pred = int(preds[0].cpu().item())
        confidence_in_wrong = 1.0 - prob if target_label == 0 else prob
        return confidence_in_wrong, pred

    def _compute_attention_ratio(self, record: Dict, critical_positions: List[int], non_func_positions: List[int]) -> float:
        """Estimate R(x) = critical_attention / (non_func_attention + gamma)."""
        serialized = serialize_record(record, self.feature_names)
        encoding = self.tokenizer(
            serialized,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        gamma = 1e-6
        with torch.no_grad():
            if self.is_decoder:
                outputs = self.model.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                final_attn = outputs.attentions[-1]
                seq_len = int(attention_mask.sum().item()) - 1
                readout = final_attn[0, :, seq_len, :]
            else:
                outputs = self.model.deberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                final_attn = outputs.attentions[-1]
                readout = final_attn[0, :, 0, :]

        mean_attn = readout.mean(dim=0).cpu().numpy()
        seq_len = mean_attn.shape[0]
        crit_pos = [p for p in critical_positions if p < seq_len]
        non_func_pos = [p for p in non_func_positions if p < seq_len]
        crit_mass = float(np.sum(mean_attn[crit_pos])) if crit_pos else 0.0
        non_func_mass = float(np.sum(mean_attn[non_func_pos])) if non_func_pos else 0.0
        return crit_mass / (non_func_mass + gamma)

    def attack(
        self,
        record: Dict,
        true_label: int,
        drift_rankings: List[Tuple[str, float]],
        critical_positions: List[int],
        non_func_positions: List[int],
    ) -> AttackResult:
        """Greedy attack constrained to evade the attention-consistency detector."""
        current_record = dict(record)
        target_label = 1 - true_label

        ranked = [
            (feat, drift) for feat, drift in drift_rankings
            if feat in self.non_functional_features and abs(drift) > self.config.drift_threshold
        ]
        if not ranked:
            ranked = [(feat, 0.0) for feat in self.feature_names if feat in self.non_functional_features]
        mutable_features = [feat for feat, _ in ranked]

        initial_conf, initial_pred = self._predict_confidence(current_record, target_label)
        best_conf = initial_conf
        best_record = dict(current_record)
        n_queries = 0
        features_modified = []

        for feature in mutable_features:
            if n_queries >= self.config.query_budget:
                break

            valid_subs = self.cooccurrence_model.get_valid_substitutions(
                feature=feature,
                current_record=current_record,
                feature_domains=self.feature_domains,
                plausibility_threshold=self.config.plausibility_threshold,
                n_mutable=len(mutable_features),
            )

            for val in valid_subs:
                if n_queries >= self.config.query_budget:
                    break

                candidate = dict(current_record)
                candidate[feature] = val

                plausibility = self.cooccurrence_model.plausibility_score(candidate, mutable_features)
                if plausibility < self.config.plausibility_threshold:
                    continue

                # Adaptive constraint: check attention ratio stays near authentic mean
                try:
                    ratio = self._compute_attention_ratio(candidate, critical_positions, non_func_positions)
                    a_ac = abs(ratio - self.mu_R) / (self.sigma_R + 1e-8)
                    if a_ac >= self.detection_threshold_sigma:
                        n_queries += 1
                        continue
                except Exception:
                    pass

                conf, pred = self._predict_confidence(candidate, target_label)
                n_queries += 1

                if conf > best_conf:
                    best_conf = conf
                    best_record = dict(candidate)
                    if feature not in features_modified:
                        features_modified.append(feature)

            current_record = dict(best_record)

        _, final_pred = self._predict_confidence(best_record, target_label)
        success = final_pred != true_label
        final_plausibility = self.cooccurrence_model.plausibility_score(best_record, mutable_features)

        return AttackResult(
            original_record=dict(record),
            adversarial_record=best_record,
            original_label=true_label,
            original_prediction=initial_pred,
            adversarial_prediction=final_pred,
            adversarial_confidence=best_conf,
            plausibility_score=final_plausibility,
            n_queries=n_queries,
            success=success,
            features_modified=features_modified,
        )


class RandomSemanticAttack:
    def __init__(
        self,
        model,
        tokenizer,
        feature_names: List[str],
        non_functional_features: Set[str],
        cooccurrence_model: CooccurrenceModel,
        feature_domains: Dict[str, List],
        config: AttackConfig,
        device: str = "cuda",
        max_seq_length: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.feature_names = feature_names
        self.non_functional_features = non_functional_features
        self.cooccurrence_model = cooccurrence_model
        self.feature_domains = feature_domains
        self.config = config
        self.device = device
        self.max_seq_length = max_seq_length

    def _predict(self, record: Dict) -> Tuple[int, float]:
        serialized = serialize_record(record, self.feature_names)
        encoding = self.tokenizer(
            serialized,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            preds, probs = self.model.predict(input_ids, attention_mask)
        return int(preds[0].item()), float(probs[0].item())

    def attack(self, record: Dict, true_label: int) -> AttackResult:
        mutable = [f for f in self.feature_names if f in self.non_functional_features]
        rng = np.random.RandomState(42)
        best_record = dict(record)
        _, init_prob = self._predict(record)
        init_pred = int(init_prob > 0.5)
        n_queries = 0

        for _ in range(self.config.query_budget):
            if not mutable:
                break
            feat = rng.choice(mutable)
            if feat not in self.feature_domains:
                continue
            domain = self.feature_domains[feat]
            current_val = record.get(feat)
            options = [v for v in domain if str(v) != str(current_val)]
            if not options:
                continue

            candidate = dict(best_record)
            candidate[feat] = rng.choice(options)

            plausibility = self.cooccurrence_model.plausibility_score(candidate, mutable)
            if plausibility < self.config.plausibility_threshold:
                continue

            pred, _ = self._predict(candidate)
            n_queries += 1

            if pred != true_label:
                best_record = candidate
                break

        _, final_prob = self._predict(best_record)
        final_pred = int(final_prob > 0.5)
        success = final_pred != true_label

        return AttackResult(
            original_record=record,
            adversarial_record=best_record,
            original_label=true_label,
            original_prediction=init_pred,
            adversarial_prediction=final_pred,
            adversarial_confidence=0.0,
            plausibility_score=0.0,
            n_queries=n_queries,
            success=success,
            features_modified=[],
        )
