import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from configs import TrainingConfig
from utils.logger import setup_logger

logger = setup_logger("phase2_training")


@dataclass
class PrivacyAccountant:
    target_epsilon: float
    delta: float
    num_steps: int
    batch_size: int
    dataset_size: int

    def compute_noise_multiplier(
        self,
        target_epsilon: float,
        orders: List[float] = None,
        tol: float = 1e-6,
        max_sigma: float = 100.0,
        min_sigma: float = 0.01,
    ) -> float:
        if target_epsilon == float("inf"):
            return 0.0

        if orders is None:
            orders = list(range(2, 64)) + [float(x) / 10.0 for x in range(640, 1000)]

        q = self.batch_size / self.dataset_size

        def compute_eps_from_sigma(sigma: float) -> float:
            return self._compute_rdp_epsilon(
                sigma=sigma, q=q, steps=self.num_steps,
                orders=orders, delta=self.delta
            )

        if compute_eps_from_sigma(max_sigma) > target_epsilon:
            raise ValueError(f"Cannot achieve epsilon={target_epsilon} even with sigma={max_sigma}")

        low, high = min_sigma, max_sigma
        for _ in range(64):
            mid = (low + high) / 2
            eps = compute_eps_from_sigma(mid)
            if eps <= target_epsilon:
                high = mid
            else:
                low = mid
            if abs(high - low) < tol:
                break

        return high

    def _compute_rdp_epsilon(
        self, sigma: float, q: float, steps: int, orders: List[float], delta: float
    ) -> float:
        rdp = self._compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=orders)
        eps = self._rdp_to_dp(rdp=rdp, orders=orders, delta=delta)
        return eps

    def _compute_rdp(
        self, q: float, noise_multiplier: float, steps: int, orders: List[float]
    ) -> List[float]:
        rdp = []
        for alpha in orders:
            if isinstance(alpha, float) and not alpha.is_integer():
                rdp.append(steps * self._compute_rdp_single_step_non_integer(q, noise_multiplier, alpha))
            else:
                alpha_int = int(alpha)
                rdp.append(steps * self._compute_rdp_single_step(q, noise_multiplier, alpha_int))
        return rdp

    def _compute_rdp_single_step(self, q: float, sigma: float, alpha: int) -> float:
        if sigma == 0:
            return float("inf")

        log_terms = []
        for k in range(alpha + 1):
            binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
            log_term = binom + k * math.log(q) + (alpha - k) * math.log(1 - q)
            exponent = k * (k - 1) / (2 * sigma ** 2)
            log_terms.append(log_term + exponent)

        max_log = max(log_terms)
        log_sum = max_log + math.log(sum(math.exp(t - max_log) for t in log_terms))
        return log_sum / (alpha - 1)

    def _compute_rdp_single_step_non_integer(self, q: float, sigma: float, alpha: float) -> float:
        if sigma == 0:
            return float("inf")
        return alpha * q ** 2 / (2 * sigma ** 2)

    def _rdp_to_dp(self, rdp: List[float], orders: List[float], delta: float) -> float:
        eps_values = []
        for rdp_val, alpha in zip(rdp, orders):
            if rdp_val == float("inf"):
                eps_values.append(float("inf"))
                continue
            eps = rdp_val + math.log(1.0 / delta) / (alpha - 1)
            eps_values.append(eps)
        return min(eps_values)

    def compute_achieved_epsilon(self, noise_multiplier: float, orders: List[float] = None) -> float:
        if noise_multiplier == 0.0:
            return float("inf")

        if orders is None:
            orders = list(range(2, 64)) + [float(x) / 10.0 for x in range(640, 1000)]

        q = self.batch_size / self.dataset_size
        return self._compute_rdp_epsilon(
            sigma=noise_multiplier, q=q, steps=self.num_steps,
            orders=orders, delta=self.delta
        )


class PerSampleGradientClipper:
    def __init__(self, model: nn.Module, clipping_threshold: float):
        self.model = model
        self.clipping_threshold = clipping_threshold
        self._hooks = []

    def enable(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self._clip_hook)
                self._hooks.append(hook)

    def disable(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _clip_hook(self, grad):
        return grad


class DPSGDTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)

    def _compute_per_sample_gradients(
        self, loss_per_sample: torch.Tensor
    ) -> List[torch.Tensor]:
        per_sample_grads = []
        for loss in loss_per_sample:
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            grads = []
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    grads.append(param.grad.clone())
                    param.grad.zero_()
            per_sample_grads.append(grads)
        return per_sample_grads

    def _clip_gradients(
        self, per_sample_grads: List[List[torch.Tensor]], C: float
    ) -> List[List[torch.Tensor]]:
        clipped = []
        for grads in per_sample_grads:
            flat = torch.cat([g.flatten() for g in grads])
            norm = flat.norm(2)
            scale = min(1.0, C / (norm.item() + 1e-8))
            clipped.append([g * scale for g in grads])
        return clipped

    def _aggregate_and_noise(
        self,
        clipped_grads: List[List[torch.Tensor]],
        noise_multiplier: float,
        C: float,
        batch_size: int,
    ) -> List[torch.Tensor]:
        n_params = len(clipped_grads[0])
        aggregated = [
            sum(clipped_grads[i][j] for i in range(len(clipped_grads)))
            for j in range(n_params)
        ]

        noisy = []
        for agg in aggregated:
            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * C,
                size=agg.shape,
                device=agg.device,
            )
            noisy.append((agg + noise) / batch_size)
        return noisy

    def train_epoch_dpsgd(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        noise_multiplier: float,
        class_weights: Optional[torch.Tensor] = None,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        C = self.config.clipping_threshold

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            labels = batch["labels"].to(self.device)

            batch_size = input_ids.shape[0]

            bce_no_reduction = nn.BCELoss(reduction="none")

            optimizer.zero_grad()

            per_sample_losses = []
            for i in range(batch_size):
                result = self.model(
                    input_ids=input_ids[i:i+1],
                    attention_mask=attention_mask[i:i+1],
                    token_type_ids=token_type_ids[i:i+1] if token_type_ids is not None else None,
                )
                prob = result["probabilities"]
                lbl = labels[i:i+1].float()
                if class_weights is not None:
                    w = class_weights[labels[i]]
                    loss_i = bce_no_reduction(prob, lbl) * w
                else:
                    loss_i = bce_no_reduction(prob, lbl)
                per_sample_losses.append(loss_i)

            per_sample_grads = self._compute_per_sample_gradients(per_sample_losses)

            clipped_grads = self._clip_gradients(per_sample_grads, C)

            if noise_multiplier > 0:
                noisy_grads = self._aggregate_and_noise(
                    clipped_grads, noise_multiplier, C, batch_size
                )
            else:
                n_params = len(clipped_grads[0])
                noisy_grads = [
                    sum(clipped_grads[i][j] for i in range(len(clipped_grads))) / batch_size
                    for j in range(n_params)
                ]

            for param, grad in zip(
                [p for p in self.model.parameters() if p.requires_grad],
                noisy_grads
            ):
                param.grad = grad

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=C
            )
            optimizer.step()

            total_loss += sum(l.item() for l in per_sample_losses) / batch_size
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, dataloader) -> Dict:
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                labels = batch["labels"].to(self.device)

                preds, probs = self.model.predict(input_ids, attention_mask, token_type_ids)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        from utils.metrics import compute_utility_metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        return compute_utility_metrics(y_true, y_pred, y_prob)

    def train_with_dp(
        self,
        train_loader,
        val_loader,
        target_epsilon: float,
        dataset_size: int,
        batch_size: int,
        class_weights: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[Dict, float]:
        if target_epsilon == float("inf"):
            noise_multiplier = 0.0
            achieved_epsilon = float("inf")
        else:
            num_steps = self.config.num_epochs * math.ceil(dataset_size / batch_size)
            accountant = PrivacyAccountant(
                target_epsilon=target_epsilon,
                delta=self.config.delta,
                num_steps=num_steps,
                batch_size=batch_size,
                dataset_size=dataset_size,
            )
            noise_multiplier = accountant.compute_noise_multiplier(target_epsilon)
            achieved_epsilon = accountant.compute_achieved_epsilon(noise_multiplier)
            logger.info(
                f"Target ε={target_epsilon}, σ={noise_multiplier:.4f}, "
                f"achieved ε={achieved_epsilon:.4f} (δ={self.config.delta})"
            )

        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch_dpsgd(
                train_loader, optimizer, noise_multiplier, class_weights
            )
            val_metrics = self.evaluate(val_loader)
            val_acc = val_metrics["accuracy"]

            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Loss={train_loss:.4f} | Val ACC={val_acc:.4f} | F1={val_metrics['f1']:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(best_state, save_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        final_metrics = self.evaluate(val_loader)
        return final_metrics, achieved_epsilon
