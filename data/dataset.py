import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Set, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer

from utils.serialization import (
    serialize_record,
    compute_inverse_frequency_weights,
    ADULT_CRITICAL_FEATURES,
    ADULT_NON_FUNCTIONAL_FEATURES,
    ADULT_ALL_FEATURES,
    ADULT_LABEL_COL,
    MIMIC_CRITICAL_FEATURES,
    MIMIC_NON_FUNCTIONAL_FEATURES,
    MIMIC_LABEL_COL,
)
from configs import DataConfig


class TabularTextDataset(Dataset):
    def __init__(
        self,
        serialized_texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 256,
        critical_positions: Optional[List[List[int]]] = None,
        non_func_positions: Optional[List[List[int]]] = None,
        feature_positions: Optional[List[Dict[str, List[int]]]] = None,
    ):
        self.texts = serialized_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.critical_positions = critical_positions or [[] for _ in labels]
        self.non_func_positions = non_func_positions or [[] for _ in labels]
        self.feature_positions = feature_positions or [{} for _ in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "critical_positions": torch.tensor(self.critical_positions[idx][:self.max_length], dtype=torch.long),
            "non_func_positions": torch.tensor(self.non_func_positions[idx][:self.max_length], dtype=torch.long),
        }


class AdultCensusDataProcessor:
    COLUMN_NAMES = ADULT_ALL_FEATURES + [ADULT_LABEL_COL]
    CATEGORICAL_COLS = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    NUMERICAL_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
        self.critical_features = ADULT_CRITICAL_FEATURES
        self.non_functional_features = ADULT_NON_FUNCTIONAL_FEATURES
        self.feature_names = ADULT_ALL_FEATURES

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, np.ndarray]:
        train_path = os.path.join(self.data_dir, "adult.data")
        test_path = os.path.join(self.data_dir, "adult.test")

        if os.path.exists(train_path):
            df_train = pd.read_csv(
                train_path, names=self.COLUMN_NAMES, skipinitialspace=True, na_values="?"
            )
            df_test = pd.read_csv(
                test_path, names=self.COLUMN_NAMES, skipinitialspace=True,
                skiprows=1, na_values="?"
            )
            df = pd.concat([df_train, df_test], ignore_index=True)
        else:
            raise FileNotFoundError(f"Adult Census dataset not found at {self.data_dir}")

        df = df.dropna()

        df[ADULT_LABEL_COL] = df[ADULT_LABEL_COL].str.strip().str.rstrip(".")
        labels = (df[ADULT_LABEL_COL] == ">50K").astype(int).values

        df = df.drop(columns=[ADULT_LABEL_COL])

        df[self.NUMERICAL_COLS] = self.scaler.fit_transform(df[self.NUMERICAL_COLS])

        return df, labels

    def serialize_dataframe(self, df: pd.DataFrame) -> List[str]:
        serialized = []
        for _, row in df.iterrows():
            record = {col: row[col] for col in self.feature_names if col in df.columns}
            text = serialize_record(record, self.feature_names)
            serialized.append(text)
        return serialized

    def get_feature_categorization(self) -> Tuple[Set[str], Set[str]]:
        return self.critical_features, self.non_functional_features

    def get_cooccurrence_distribution(self, df: pd.DataFrame) -> Dict:
        cooccurrence = {}
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                value_counts = df[col].value_counts(normalize=True)
                cooccurrence[col] = value_counts.to_dict()
        return cooccurrence

    def get_feature_domains(self, df: pd.DataFrame) -> Dict[str, List]:
        domains = {}
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                domains[col] = df[col].unique().tolist()
        for col in self.NUMERICAL_COLS:
            if col in df.columns:
                domains[col] = [df[col].min(), df[col].max()]
        return domains


class MIMICIVDataProcessor:
    CATEGORICAL_COLS = ["ethnicity", "insurance", "religion", "marital_status", "language", "admission_type", "gender"]
    NUMERICAL_COLS = [
        "age", "lab_glucose", "lab_creatinine", "lab_bun",
        "lab_sodium", "lab_hematocrit"
    ]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler()
        self.critical_features = MIMIC_CRITICAL_FEATURES
        self.non_functional_features = MIMIC_NON_FUNCTIONAL_FEATURES
        self.feature_names = list(MIMIC_CRITICAL_FEATURES | MIMIC_NON_FUNCTIONAL_FEATURES) + \
                             ["age", "gender", "admission_type"]

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, np.ndarray]:
        mimic_path = os.path.join(self.data_dir, "mimic_iv_processed.csv")

        if not os.path.exists(mimic_path):
            raise FileNotFoundError(
                f"MIMIC-IV processed data not found at {mimic_path}. "
                "Please process MIMIC-IV following the PhysioNet data use agreement."
            )

        df = pd.read_csv(mimic_path)

        missing_threshold = 0.20
        df = df.dropna(thresh=int(len(df) * (1 - missing_threshold)), axis=1)
        df = df.dropna()

        labels = df[MIMIC_LABEL_COL].astype(int).values
        df = df.drop(columns=[MIMIC_LABEL_COL], errors="ignore")

        numerical_cols = [c for c in self.NUMERICAL_COLS if c in df.columns]
        if numerical_cols:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])

        return df, labels

    def serialize_dataframe(self, df: pd.DataFrame) -> List[str]:
        serialized = []
        feature_names = [f for f in self.feature_names if f in df.columns]
        for _, row in df.iterrows():
            record = {col: row[col] for col in feature_names}
            text = serialize_record(record, feature_names)
            serialized.append(text)
        return serialized

    def get_feature_categorization(self) -> Tuple[Set[str], Set[str]]:
        return self.critical_features, self.non_functional_features

    def get_cooccurrence_distribution(self, df: pd.DataFrame) -> Dict:
        cooccurrence = {}
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                value_counts = df[col].value_counts(normalize=True)
                cooccurrence[col] = value_counts.to_dict()
        return cooccurrence

    def get_feature_domains(self, df: pd.DataFrame) -> Dict[str, List]:
        domains = {}
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                domains[col] = df[col].unique().tolist()
        return domains


def create_stratified_splits(
    serialized_texts: List[str],
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    random_seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, stratify=labels, random_state=random_seed
    )
    val_size = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=labels[train_val_idx],
        random_state=random_seed,
    )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def create_cv_splits(
    labels: np.ndarray, n_folds: int = 5, random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    indices = np.arange(len(labels))
    return list(skf.split(indices, labels))


def build_dataloader(
    dataset: TabularTextDataset,
    batch_size: int,
    shuffle: bool = True,
    use_poisson_sampling: bool = False,
    poisson_rate: float = None,
) -> DataLoader:
    if use_poisson_sampling and poisson_rate is not None:
        from torch.utils.data import BatchSampler, RandomSampler
        sampler = PoissonSampler(len(dataset), poisson_rate)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


class PoissonSampler(torch.utils.data.Sampler):
    def __init__(self, n: int, rate: float):
        self.n = n
        self.rate = rate

    def __iter__(self):
        mask = torch.bernoulli(torch.full((self.n,), self.rate)).bool()
        indices = torch.where(mask)[0].tolist()
        if len(indices) == 0:
            indices = [0]
        yield indices

    def __len__(self):
        return 1
