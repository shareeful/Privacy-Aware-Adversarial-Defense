import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from scipy.stats import spearmanr, bootstrap
import torch


def compute_utility_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = 0.0
    return metrics


def compute_attack_metrics(
    y_true: np.ndarray,
    y_original_pred: np.ndarray,
    y_adv_pred: np.ndarray,
    detector_scores: np.ndarray = None,
    detector_threshold: float = 0.5,
) -> Dict[str, float]:
    successful_attacks = (y_original_pred == y_true) & (y_adv_pred != y_true)
    asr = float(np.sum(successful_attacks)) / max(1, int(np.sum(y_original_pred == y_true)))

    metrics = {"asr": asr, "n_successful": int(np.sum(successful_attacks))}

    if detector_scores is not None:
        detected = detector_scores > detector_threshold
        der = float(np.sum(successful_attacks & ~detected)) / max(1, int(np.sum(successful_attacks)))
        metrics["der"] = der

    return metrics


def compute_defense_metrics(
    y_adversarial_flag: np.ndarray,
    detection_scores: np.ndarray,
    threshold: float = None,
    max_fpr: float = 0.05,
) -> Dict[str, float]:
    try:
        auc = roc_auc_score(y_adversarial_flag, detection_scores)
    except ValueError:
        auc = 0.5

    fpr_arr, tpr_arr, thresholds = roc_curve(y_adversarial_flag, detection_scores)

    if threshold is None:
        valid_idx = fpr_arr <= max_fpr
        if np.any(valid_idx):
            best_idx = np.argmax(tpr_arr[valid_idx])
            threshold = thresholds[valid_idx][best_idx]
        else:
            threshold = 0.5

    y_pred_flag = (detection_scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_adversarial_flag, y_pred_flag, labels=[0, 1]).ravel()

    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)

    return {
        "auc": auc,
        "tpr": tpr,
        "fpr": fpr,
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def compute_spearman_with_ci(
    x: List[float],
    y: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float, float]:
    x_arr = np.array(x)
    y_arr = np.array(y)

    rho, p_value = spearmanr(x_arr, y_arr)

    def spearman_statistic(x_sample, y_sample):
        r, _ = spearmanr(x_sample, y_sample)
        return r

    bootstrap_rhos = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(x_arr), len(x_arr), replace=True)
        r, _ = spearmanr(x_arr[idx], y_arr[idx])
        bootstrap_rhos.append(r)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_rhos, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_rhos, 100 * (1 - alpha / 2))

    return float(rho), float(p_value), float(ci_lower), float(ci_upper)


def compute_odds_ratio_logistic(
    drift_values: np.ndarray, attack_outcomes: np.ndarray
) -> Tuple[float, float, float, float]:
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import norm

    X = drift_values.reshape(-1, 1)
    y = attack_outcomes

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X, y)

    coef = clf.coef_[0, 0]
    odds_ratio = float(np.exp(coef))

    n = len(y)
    se = np.sqrt(1 / (n * np.var(drift_values)))
    z = norm.ppf(0.975)
    ci_lower = float(np.exp(coef - z * se))
    ci_upper = float(np.exp(coef + z * se))

    from scipy.stats import chi2
    null_log_likelihood = n * (np.mean(y) * np.log(max(np.mean(y), 1e-10)) +
                               (1 - np.mean(y)) * np.log(max(1 - np.mean(y), 1e-10)))
    y_pred_prob = clf.predict_proba(X)[:, 1]
    alt_log_likelihood = np.sum(y * np.log(np.maximum(y_pred_prob, 1e-10)) +
                                (1 - y) * np.log(np.maximum(1 - y_pred_prob, 1e-10)))
    lr_stat = 2 * (alt_log_likelihood - null_log_likelihood)
    p_value = float(1 - chi2.cdf(lr_stat, df=1))

    return odds_ratio, p_value, ci_lower, ci_upper


def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray) -> Tuple[float, float]:
    from scipy.stats import ttest_rel
    stat, p_value = ttest_rel(scores_a, scores_b)
    return float(stat), float(p_value)
