from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np


ADULT_CRITICAL_FEATURES = {"education", "occupation", "capital-gain", "capital-loss", "hours-per-week"}
ADULT_NON_FUNCTIONAL_FEATURES = {"race", "sex", "native-country", "relationship"}
ADULT_ALL_FEATURES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country"
]
ADULT_LABEL_COL = "income"

MIMIC_CRITICAL_FEATURES = {
    "diagnosis_code", "lab_glucose", "lab_creatinine",
    "lab_bun", "lab_sodium", "lab_hematocrit", "icu_flag"
}
MIMIC_NON_FUNCTIONAL_FEATURES = {
    "ethnicity", "insurance", "religion", "marital_status", "language"
}
MIMIC_LABEL_COL = "hospital_expire_flag"


def serialize_record(record: Dict, feature_names: List[str]) -> str:
    parts = ["[CLS]"]
    for name in feature_names:
        if name in record:
            val = record[name]
            parts.append(f"{name}: {val}")
    parts.append("[SEP]")
    return " ".join(parts)


def categorize_tokens(
    serialized: str,
    critical_features: Set[str],
    non_functional_features: Set[str],
    tokenizer,
) -> Tuple[List[int], List[int]]:
    tokens = tokenizer.tokenize(serialized)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    critical_positions = []
    non_func_positions = []

    current_feature = None
    for i, token in enumerate(tokens):
        clean_token = token.replace("▁", "").replace("##", "").lower()

        for feat in critical_features:
            if clean_token == feat.lower() or feat.lower() in clean_token:
                current_feature = "critical"
                break
        else:
            for feat in non_functional_features:
                if clean_token == feat.lower() or feat.lower() in clean_token:
                    current_feature = "non_functional"
                    break

        if current_feature == "critical":
            critical_positions.append(i + 1)
        elif current_feature == "non_functional":
            non_func_positions.append(i + 1)

    return critical_positions, non_func_positions


def get_feature_token_positions(
    serialized: str,
    feature_names: List[str],
    tokenizer,
) -> Dict[str, List[int]]:
    tokens = tokenizer.tokenize(serialized)
    feature_positions = {name: [] for name in feature_names}

    i = 0
    current_feature = None
    while i < len(tokens):
        token_clean = tokens[i].replace("▁", "").replace("##", "").strip().lower()
        matched = False
        for feat in feature_names:
            if token_clean == feat.lower().replace("-", "").replace("_", ""):
                current_feature = feat
                matched = True
                break
        if not matched and current_feature is not None:
            if token_clean not in [":", "[cls]", "[sep]", ""]:
                feature_positions[current_feature].append(i + 1)
        i += 1

    return feature_positions


def compute_inverse_frequency_weights(labels: np.ndarray) -> Dict[int, float]:
    n = len(labels)
    classes = np.unique(labels)
    n_classes = len(classes)
    weights = {}
    for c in classes:
        nc = np.sum(labels == c)
        weights[int(c)] = n / (n_classes * nc)
    return weights
