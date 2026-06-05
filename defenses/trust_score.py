import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import jensenshannon
from dataclasses import dataclass

from configs import DefenseConfig
from utils.logger import setup_logger

logger = setup_logger("defense")


@dataclass
class DefenseResult:
    isolation_forest_scores: np.ndarray
    attention_consistency_scores: np.ndarray
    trust_scores: np.ndarray
    predictions: np.ndarray
    threshold: float
    lambda_val: float


class IsolationForestDetector:
    def __init__(self, n_estimators: int = 100, contamination: float = 0.1, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.fitted = False

    def fit(self, embeddings: np.ndarray):
        self.model.fit(embeddings)
        self.fitted = True
        logger.info(f"IsolationForest fitted on {len(embeddings)} samples")

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IsolationForest must be fitted before scoring")
        raw_scores = self.model.score_samples(embeddings)
        normalized = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
        return 1.0 - normalized

    def collect_embeddings(
        self,
        model,
        dataloader,
        device: str = "cuda",
        is_decoder: bool = False,
    ) -> np.ndarray:
        model.eval()
        all_embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                embeddings = model.get_cls_embeddings(input_ids, attention_mask, token_type_ids)
                all_embeddings.append(embeddings.cpu().float().numpy())

        return np.vstack(all_embeddings)


class AttentionConsistencyScorer:
    def __init__(
        self,
        critical_positions_ref: List[List[int]],
        non_func_positions_ref: List[List[int]],
        feature_positions_ref: List[Dict[str, List[int]]],
        gamma: float = 1e-8,
        ig_steps: int = 50,
    ):
        self.gamma = gamma
        self.ig_steps = ig_steps
        self.mu_R = None
        self.sigma_R = None
        self.mu_R_ig = None
        self.sigma_R_ig = None
        self.mu_feature_dist = None
        self.critical_positions_ref = critical_positions_ref
        self.non_func_positions_ref = non_func_positions_ref
        self.feature_positions_ref = feature_positions_ref

    def compute_attention_ratio(
        self,
        attentions: torch.Tensor,
        critical_positions: List[int],
        non_func_positions: List[int],
        seq_len: int,
    ) -> float:
        mean_attention = attentions.mean(dim=0).cpu().numpy()

        crit_pos = [p for p in critical_positions if p < seq_len]
        non_func_pos = [p for p in non_func_positions if p < seq_len]

        crit_mass = float(np.sum(mean_attention[crit_pos])) if crit_pos else 0.0
        non_func_mass = float(np.sum(mean_attention[non_func_pos])) if non_func_pos else 0.0

        ratio = crit_mass / (non_func_mass + self.gamma)
        return ratio

    def compute_feature_distribution(
        self,
        attentions: torch.Tensor,
        feature_positions: Dict[str, List[int]],
        seq_len: int,
    ) -> np.ndarray:
        mean_attention = attentions.mean(dim=0).cpu().numpy()

        feature_masses = []
        for feat, positions in feature_positions.items():
            valid_pos = [p for p in positions if p < seq_len]
            mass = float(np.sum(mean_attention[valid_pos])) if valid_pos else 0.0
            feature_masses.append(mass)

        total = sum(feature_masses) + self.gamma
        return np.array(feature_masses) / total

    def fit_reference(
        self,
        model,
        dataloader,
        device: str = "cuda",
        is_decoder: bool = False,
    ):
        model.eval()
        ratios = []
        feature_dists = []

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
                    final_layer_attn = outputs.attentions[-1]
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    readout = torch.stack([
                        final_layer_attn[b, :, seq_lengths[b], :]
                        for b in range(batch_size)
                    ], dim=0)
                else:
                    outputs = model.deberta(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        output_attentions=True,
                    )
                    final_layer_attn = outputs.attentions[-1]
                    readout = final_layer_attn[:, :, 0, :]

                for b in range(batch_size):
                    global_b = batch_idx * dataloader.batch_size + b
                    if global_b >= len(self.critical_positions_ref):
                        continue

                    seq_len = int(attention_mask[b].sum().item())
                    crit_pos = self.critical_positions_ref[global_b]
                    non_func_pos = self.non_func_positions_ref[global_b]
                    feat_pos = self.feature_positions_ref[global_b] if global_b < len(self.feature_positions_ref) else {}

                    ratio = self.compute_attention_ratio(
                        readout[b], crit_pos, non_func_pos, seq_len
                    )
                    ratios.append(ratio)

                    if feat_pos:
                        dist = self.compute_feature_distribution(readout[b], feat_pos, seq_len)
                        feature_dists.append(dist)

        ratios_arr = np.array(ratios)
        self.mu_R = float(np.mean(ratios_arr))
        self.sigma_R = float(np.std(ratios_arr)) + 1e-8

        if feature_dists:
            self.mu_feature_dist = np.mean(feature_dists, axis=0)
        else:
            self.mu_feature_dist = None

        logger.info(f"Reference attention ratio: μ={self.mu_R:.4f}, σ={self.sigma_R:.4f}")

    def score_sample(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
        critical_positions: List[int],
        non_func_positions: List[int],
        feature_positions: Dict[str, List[int]],
        device: str = "cuda",
        is_decoder: bool = False,
    ) -> Tuple[float, float, float]:
        model.eval()

        with torch.no_grad():
            if is_decoder:
                outputs = model.backbone(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    output_attentions=True,
                )
                final_layer_attn = outputs.attentions[-1]
                seq_lengths = attention_mask.sum(dim=1) - 1
                readout = final_layer_attn[0, :, seq_lengths[0], :]
            else:
                outputs = model.deberta(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    token_type_ids=token_type_ids.to(device) if token_type_ids is not None else None,
                    output_attentions=True,
                )
                final_layer_attn = outputs.attentions[-1]
                readout = final_layer_attn[0, :, 0, :]

        seq_len = int(attention_mask.sum().item())

        ratio = self.compute_attention_ratio(readout, critical_positions, non_func_positions, seq_len)

        a_ac = abs(ratio - self.mu_R) / self.sigma_R

        a_ig_ac = 0.0

        a_jsd = 0.0
        if feature_positions and self.mu_feature_dist is not None:
            dist = self.compute_feature_distribution(readout, feature_positions, seq_len)
            min_len = min(len(dist), len(self.mu_feature_dist))
            if min_len > 0:
                p = dist[:min_len] + 1e-10
                q = self.mu_feature_dist[:min_len] + 1e-10
                p = p / p.sum()
                q = q / q.sum()
                a_jsd = float(jensenshannon(p, q))

        return a_ac, a_ig_ac, a_jsd

    def score_batch(
        self,
        model,
        dataloader,
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        feature_positions_batch: List[Dict[str, List[int]]],
        device: str = "cuda",
        is_decoder: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        all_a_ac, all_a_ig_ac, all_a_jsd = [], [], []

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
                    final_layer_attn = outputs.attentions[-1]
                    seq_lengths = attention_mask.sum(dim=1) - 1
                    readouts = torch.stack([
                        final_layer_attn[b, :, seq_lengths[b], :]
                        for b in range(batch_size)
                    ], dim=0)
                else:
                    outputs = model.deberta(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        output_attentions=True,
                    )
                    final_layer_attn = outputs.attentions[-1]
                    readouts = final_layer_attn[:, :, 0, :]

                for b in range(batch_size):
                    global_b = batch_idx * dataloader.batch_size + b
                    if global_b >= len(critical_positions_batch):
                        continue

                    seq_len = int(attention_mask[b].sum().item())
                    crit_pos = critical_positions_batch[global_b]
                    non_func_pos = non_func_positions_batch[global_b]
                    feat_pos = feature_positions_batch[global_b] if global_b < len(feature_positions_batch) else {}

                    ratio = self.compute_attention_ratio(readouts[b], crit_pos, non_func_pos, seq_len)
                    a_ac = abs(ratio - self.mu_R) / self.sigma_R
                    all_a_ac.append(a_ac)

                    a_jsd = 0.0
                    if feat_pos and self.mu_feature_dist is not None:
                        dist = self.compute_feature_distribution(readouts[b], feat_pos, seq_len)
                        min_len = min(len(dist), len(self.mu_feature_dist))
                        if min_len > 0:
                            p = dist[:min_len] + 1e-10
                            q_ref = self.mu_feature_dist[:min_len] + 1e-10
                            p /= p.sum()
                            q_ref /= q_ref.sum()
                            a_jsd = float(jensenshannon(p, q_ref))
                    all_a_jsd.append(a_jsd)
                    all_a_ig_ac.append(0.0)

        return np.array(all_a_ac), np.array(all_a_ig_ac), np.array(all_a_jsd)


class TrustScoreDefense:
    def __init__(self, config: DefenseConfig):
        self.config = config
        self.isolation_forest = IsolationForestDetector(
            n_estimators=config.isolation_forest_estimators,
            contamination=config.isolation_forest_contamination,
        )
        self.consistency_scorer = None
        self.lambda_val = config.fusion_lambda
        self.threshold = None

    def fit(
        self,
        model,
        authentic_dataloader,
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        feature_positions_batch: List[Dict[str, List[int]]],
        device: str = "cuda",
        is_decoder: bool = False,
    ):
        logger.info("Fitting IsolationForest on authentic embeddings...")
        authentic_embeddings = self.isolation_forest.collect_embeddings(
            model, authentic_dataloader, device, is_decoder
        )
        self.isolation_forest.fit(authentic_embeddings)

        logger.info("Fitting AttentionConsistencyScorer on authentic data...")
        self.consistency_scorer = AttentionConsistencyScorer(
            critical_positions_ref=critical_positions_batch,
            non_func_positions_ref=non_func_positions_batch,
            feature_positions_ref=feature_positions_batch,
            gamma=self.config.gamma,
            ig_steps=self.config.ig_steps,
        )
        self.consistency_scorer.fit_reference(
            model, authentic_dataloader, device, is_decoder
        )

    def calibrate_threshold(
        self,
        model,
        val_dataloader,
        val_labels_adversarial: np.ndarray,
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        feature_positions_batch: List[Dict[str, List[int]]],
        device: str = "cuda",
        is_decoder: bool = False,
    ):
        result = self.score(
            model=model,
            dataloader=val_dataloader,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
            feature_positions_batch=feature_positions_batch,
            device=device,
            is_decoder=is_decoder,
        )

        from utils.metrics import compute_defense_metrics
        metrics = compute_defense_metrics(
            y_adversarial_flag=val_labels_adversarial,
            detection_scores=result.trust_scores,
            max_fpr=self.config.max_fpr,
        )
        self.threshold = metrics["threshold"]
        self.lambda_val = self._optimize_lambda(
            result.isolation_forest_scores,
            result.attention_consistency_scores,
            val_labels_adversarial,
        )
        logger.info(f"Calibrated: threshold={self.threshold:.4f}, lambda={self.lambda_val:.4f}")

    def _optimize_lambda(
        self,
        if_scores: np.ndarray,
        ac_scores: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        best_auc = 0.0
        best_lambda = 0.5

        if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
        ac_norm = (ac_scores - ac_scores.min()) / (ac_scores.max() - ac_scores.min() + 1e-8)

        from sklearn.metrics import roc_auc_score
        for lam in np.arange(0.0, 1.01, 0.1):
            combined = lam * if_norm + (1 - lam) * ac_norm
            try:
                auc = roc_auc_score(labels, combined)
                if auc > best_auc:
                    best_auc = auc
                    best_lambda = lam
            except ValueError:
                continue

        return float(best_lambda)

    def score(
        self,
        model,
        dataloader,
        critical_positions_batch: List[List[int]],
        non_func_positions_batch: List[List[int]],
        feature_positions_batch: List[Dict[str, List[int]]],
        device: str = "cuda",
        is_decoder: bool = False,
        consistency_variant: str = "ac",
    ) -> DefenseResult:
        embeddings = self.isolation_forest.collect_embeddings(model, dataloader, device, is_decoder)
        if_scores = self.isolation_forest.score(embeddings)

        a_ac, a_ig_ac, a_jsd = self.consistency_scorer.score_batch(
            model=model,
            dataloader=dataloader,
            critical_positions_batch=critical_positions_batch,
            non_func_positions_batch=non_func_positions_batch,
            feature_positions_batch=feature_positions_batch,
            device=device,
            is_decoder=is_decoder,
        )

        if consistency_variant == "ac":
            cons_scores = a_ac
        elif consistency_variant == "ig":
            cons_scores = a_ig_ac
        elif consistency_variant == "jsd":
            cons_scores = a_jsd
        else:
            cons_scores = a_ac

        if_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
        cons_norm = (cons_scores - cons_scores.min()) / (cons_scores.max() - cons_scores.min() + 1e-8)

        trust_scores = self.lambda_val * if_norm + (1 - self.lambda_val) * cons_norm

        threshold = self.threshold if self.threshold is not None else 0.5
        predictions = (trust_scores > threshold).astype(int)

        return DefenseResult(
            isolation_forest_scores=if_scores,
            attention_consistency_scores=cons_scores,
            trust_scores=trust_scores,
            predictions=predictions,
            threshold=threshold,
            lambda_val=self.lambda_val,
        )


class SurrogateAttentionModel(nn.Module):
    """Lightweight 4-layer Transformer surrogate (~11M params) for black-box deployment.

    Trained on the deployer's authentic data using the protected model's score-only API
    outputs as soft targets (Section 3.6.2). Its attention weights substitute for the
    protected model's in the consistency scorer, recovering detection without model access.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
        num_labels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_attentions: bool = False):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # TransformerEncoder expects padding mask where True = ignore
        src_key_padding_mask = (attention_mask == 0)
        hidden = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cls_repr = hidden[:, 0, :]
        logits = self.classifier(cls_repr)
        probs = self.sigmoid(logits).squeeze(-1)
        return probs

    def get_attention_weights(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return attention weights from the final transformer layer for consistency scoring."""
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        src_key_padding_mask = (attention_mask == 0)

        # Manually step through layers and capture last layer's attention
        hidden = x
        attn_weights = None
        for layer in self.transformer.layers:
            # MultiheadAttention returns (output, attn_weights)
            src2, attn_weights = layer.self_attn(
                hidden, hidden, hidden,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            hidden = layer.norm1(hidden + layer.dropout1(src2))
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden))))
            hidden = layer.norm2(hidden + layer.dropout2(src2))

        # attn_weights: (B, n_heads, T, T) — return for [CLS] row
        return attn_weights  # (B, n_heads, T, T)


class SurrogateAttentionDefense:
    """Detection pipeline using a surrogate model for black-box API deployment.

    The deployer trains a SurrogateAttentionModel on its authentic data with soft
    targets from the protected model's score-only API, then substitutes the surrogate's
    attention weights into the consistency scorer (Equations 18–21).
    """

    def __init__(
        self,
        vocab_size: int,
        config: DefenseConfig,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 256,
        device: str = "cuda",
    ):
        self.device = device
        self.config = config
        self.surrogate = SurrogateAttentionModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        ).to(device)
        self.isolation_forest = IsolationForestDetector(
            n_estimators=config.isolation_forest_estimators,
            contamination=config.isolation_forest_contamination,
        )
        self.mu_R: Optional[float] = None
        self.sigma_R: Optional[float] = None
        self.threshold: Optional[float] = None
        self.lambda_val: float = config.fusion_lambda

    def train_surrogate(
        self,
        train_dataloader,
        protected_model_scores: np.ndarray,
        n_epochs: int = 5,
        learning_rate: float = 1e-3,
    ):
        """Train surrogate with soft targets from protected model's API confidences."""
        optimizer = torch.optim.AdamW(self.surrogate.parameters(), lr=learning_rate)
        loss_fn = nn.BCELoss()
        self.surrogate.train()

        score_iter = iter(protected_model_scores)
        for epoch in range(n_epochs):
            total_loss = 0.0
            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                B = input_ids.shape[0]

                soft_labels = torch.tensor(
                    [next(score_iter, 0.5) for _ in range(B)],
                    dtype=torch.float32,
                    device=self.device,
                )

                probs = self.surrogate(input_ids, attention_mask)
                loss = loss_fn(probs, soft_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"Surrogate epoch {epoch+1}/{n_epochs}, loss={total_loss:.4f}")

    def fit_reference(self, authentic_dataloader, critical_positions_batch: List[List[int]], non_func_positions_batch: List[List[int]]):
        """Compute mu_R and sigma_R from authentic data using surrogate attention."""
        self.surrogate.eval()
        ratios = []
        gamma = self.config.gamma

        with torch.no_grad():
            for batch_idx, batch in enumerate(authentic_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                B = input_ids.shape[0]

                attn_weights = self.surrogate.get_attention_weights(input_ids, attention_mask)
                # attn_weights: (B, n_heads, T, T) — use [CLS] row
                cls_attn = attn_weights[:, :, 0, :]  # (B, n_heads, T)
                mean_attn = cls_attn.mean(dim=1)  # (B, T)

                for b in range(B):
                    global_b = batch_idx * authentic_dataloader.batch_size + b
                    attn_b = mean_attn[b].cpu().numpy()
                    seq_len = attn_b.shape[0]
                    crit_pos = [p for p in critical_positions_batch[global_b] if p < seq_len] if global_b < len(critical_positions_batch) else []
                    non_func_pos = [p for p in non_func_positions_batch[global_b] if p < seq_len] if global_b < len(non_func_positions_batch) else []
                    crit_mass = float(np.sum(attn_b[crit_pos])) if crit_pos else 0.0
                    non_func_mass = float(np.sum(attn_b[non_func_pos])) if non_func_pos else 0.0
                    ratios.append(crit_mass / (non_func_mass + gamma))

        ratios_arr = np.array(ratios)
        self.mu_R = float(np.mean(ratios_arr))
        self.sigma_R = float(np.std(ratios_arr)) + 1e-8
        logger.info(f"Surrogate reference: μ_R={self.mu_R:.4f}, σ_R={self.sigma_R:.4f}")

    def score_sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        critical_positions: List[int],
        non_func_positions: List[int],
    ) -> Tuple[float, float]:
        """Return (a_IF_placeholder, a_AC_surrogate) for one sample."""
        self.surrogate.eval()
        gamma = self.config.gamma

        with torch.no_grad():
            attn_weights = self.surrogate.get_attention_weights(
                input_ids.to(self.device), attention_mask.to(self.device)
            )
            cls_attn = attn_weights[:, :, 0, :].mean(dim=1)[0].cpu().numpy()

        seq_len = cls_attn.shape[0]
        crit_pos = [p for p in critical_positions if p < seq_len]
        non_func_pos = [p for p in non_func_positions if p < seq_len]
        crit_mass = float(np.sum(cls_attn[crit_pos])) if crit_pos else 0.0
        non_func_mass = float(np.sum(cls_attn[non_func_pos])) if non_func_pos else 0.0
        ratio = crit_mass / (non_func_mass + gamma)
        a_ac = abs(ratio - self.mu_R) / (self.sigma_R + 1e-8)
        return a_ac


class SupervisedDetector:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model_clf = LogisticRegression(random_state=42, max_iter=1000)
        self.fitted = False

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        self.model_clf.fit(embeddings, labels)
        self.fitted = True

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("SupervisedDetector must be fitted before scoring")
        return self.model_clf.predict_proba(embeddings)[:, 1]
