import pandas as pd
import numpy as np
import os
import time
import wrapper
from rust_time_series.rust_time_series import (
    ForecastingDataSet,
    ClassificationDataSet,
    ImputeStrategy,
    SplittingStrategy,
)


from enum import Enum
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from rust_time_series.rust_time_series import (
    ForecastingDataSet,
    ClassificationDataSet,
    ImputeStrategy,
    SplittingStrategy,
)
import time
print("Rust Time Series Wrapper Loaded")


class DatasetType(Enum):
    Forecasting = "forecasting"
    Classification = "classification"


class BenchmarkingModule(wrapper.RustDataModule):
    def __init__(
        self,
        dataset: np.ndarray,
        dataset_type: DatasetType,
        past_window: int = 1,
        future_horizon: int = 1,
        stride: int = 1,
        labels: np.ndarray | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        downsampling_rate: int = 0,
        normalize: bool = False,
        standardize: bool = False,
        impute_strategy: ImputeStrategy = ImputeStrategy.LeaveNaN,
        splitting_strategy: SplittingStrategy = SplittingStrategy.Temporal,
        splitting_ratios: tuple = (0.7, 0.2, 0.1),  # Train, validation, test ratios
    ):
        super().__init__(
            dataset,
            dataset_type,
            past_window,
            future_horizon,
            stride,
            labels,
            batch_size,
            num_workers,
            downsampling_rate,
            normalize,
            standardize,
            impute_strategy,
            splitting_strategy,
            splitting_ratios
        )

        self.timings = {}

    def setup(self, stage: str):
        if self.dataset_type == DatasetType.Forecasting:
            # call the method that applies the preprocessing steps and returns the split datasets
            dataset = ForecastingDataSet(self.dataset)
        else:
            dataset = ClassificationDataSet(self.dataset, self.labels)

        # Apply imputation strategy if specified
        if self.impute_strategy != ImputeStrategy.LeaveNaN:
            dataset.impute(self.impute_strategy)

        # Apply downsampling if specified
        if self.downsampling_rate > 0:
            timer = time.time()
            dataset.downsample(self.downsampling_rate)
            print(f"Downsampling took {time.time() - timer:.2f} seconds")

        # Split the data
        split_args = (
            (self.splitting_strategy, *self.splitting_ratios)
            if self.dataset_type == DatasetType.Classification
            else self.splitting_ratios
        )
        timer = time.time()
        dataset.split(*split_args)
        print(f"Splitting took {time.time() - timer:.2f} seconds")

        # Apply normalization if specified
        if self.normalize:
            timer = time.time()
            dataset.normalize()
            print(f"Normalization took {time.time() - timer:.2f} seconds")

        # Apply standardization if specified
        if self.standardize:
            dataset.standardize()

        timer = time.time()
        # Collect the results
        collect_args = (
            (self.past_window, self.future_horizon, self.stride)
            if self.dataset_type == DatasetType.Forecasting
            else ()
        )
        (
            (X_train, y_train),
            (X_val, y_val),
            (X_test, y_test),
        ) = dataset.collect(*collect_args)
        print(f"Collecting data took {time.time() - timer:.2f} seconds")

        self.train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        self.val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        self.test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    def train_dataloader(self):
        # Return the training dataloader
        if self.train_data is None:
            raise ValueError("Training data is not set. Call setup() first.")
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        # Return the validation dataloader
        if self.val_data is None:
            raise ValueError("Validation data is not set. Call setup() first.")
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        # Return the test dataloader
        if self.test_data is None:
            raise ValueError("Test data is not set. Call setup() first.")
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )


# format assumed: (n_instances, n_channels, n_timepoints)

def load_electricity_data() -> np.ndarray:
    # Load the dataset (forecasting)
    file_path = "/home/kilian/github/ai_with_rust_final_project/source_code/python/data/LD/LD2011_2014.txt"
    df = pd.read_csv(file_path, sep=";", decimal=",")
    # drop first column (date)
    df = df.drop(columns=["date"])
    df = df.transpose()
    # turn pandas DataFrame into numPy array
    data = df.to_numpy(dtype=np.float64).reshape(df.shape[0], df.shape[1], 1)

    TIME_DIM = 1
    INSTANCES_DIM = 2
    FEATURE_DIM = 3


    assert data.ndim == 3
    return data

AEON_METADATA = {}
from aeon.datasets.tsc_datasets import multivariate, univariate
# get metadata for univariate and multivariate datasets
for dataset in univariate:
    AEON_METADATA[dataset] = {
        "Train": None,  # Placeholder, will be filled later
        "Test": None,   # Placeholder, will be filled later
        "Channel": 1,  # Default for univariate datasets
        "Class": None,    # Placeholder for classification datasets
    }
for dataset in multivariate:
    AEON_METADATA[dataset] = {
        "Train": None,  # Placeholder, will be filled later
        "Test": None,   # Placeholder, will be filled later
        "Channel": None,  # Placeholder for multivariate datasets
        "Class": None,    # Placeholder for classification datasets
    }

def load_aeon_data(dataset_title) -> tuple[np.ndarray, np.ndarray]:
    # Load the GunPoint dataset from aeon (classification)
    # Returns X (features) and y (labels) as numpy arrays
    if dataset_title not in AEON_METADATA:
        raise ValueError(f"Dataset {dataset_title} is not available in AEON metadata.")
    from aeon.datasets import load_classification
    X, y, metadata = load_classification(dataset_title,return_metadata=True)
    X = X.reshape(X.shape[0], -1)
    return X, y

ETT_METADATA = {
    "ETTh1": {"Train": 17420, "Channel": 7},
    "ETTh2": {"Train": 17420, "Channel": 7},
    "ETTm1": {"Train": 69680, "Channel": 7},
    "ETTm2": {"Train": 69680, "Channel": 7},
}

def load_ETT_data(dataset_title,base_path="data/ETT-small/") -> np.ndarray:
    # Load the ETT dataset (forecasting)
    file_path = os.path.join(base_path, dataset_title + ".csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    # read the csv files into a list of DataFrames
    df = pd.read_csv(file_path, sep=",", decimal=".")
    # drop first column (date)
    df = df.drop(columns=["date"])
    # turn pandas DataFrame into numPy array
    data = df.to_numpy(dtype=np.float64)
    return data

SERIES2VEC_METADATA = {
    "WISDM2":    {"Train": 134614, "Test": 14421, "Subject": 29,  "Length": 40,   "Channel": 3,   "Class": 6},
    "PAMAP2":    {"Train": 51192,  "Test": 11590, "Subject": 9,   "Length": 100,  "Channel": 52,  "Class": 18},
    "USC_HAD":   {"Train": 46899,  "Test": 9327,  "Subject": 14,  "Length": 100,  "Channel": 6,   "Class": 12},
    "Sleep":     {"Train": 25612,  "Test": 8910,  "Subject": 20,  "Length": 3000, "Channel": 1,   "Class": 5},
    "Skoda":     {"Train": 22587,  "Test": 5646,  "Subject": 1,   "Length": 50,   "Channel": 64,  "Class": 11},
    "Opportunity": {"Train": 15011, "Test": 2374, "Subject": 4,   "Length": 100,  "Channel": 113, "Class": 18},
    "WISDM":     {"Train": 11960,  "Test": 5207,  "Subject": 13,  "Length": 40,   "Channel": 3,   "Class": 6},
    "Epilepsy":  {"Train": 9200,   "Test": 2300,  "Subject": 500, "Length": 178,  "Channel": 1,   "Class": 2},
    "HAR":   {"Train": 7352,   "Test": 2947,  "Subject": 30,  "Length": 128,  "Channel": 9,   "Class": 6}
}

def load_series2vec_data(dataset_title,base_path="/home/kilian/github/ai_with_rust_final_project/source_code/python/data/series2vec/") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the Series2Vec datasets (classification)\n
    Returns train_data, train_label, test_data, test_label as numpy arrays\n
    source: https://drive.google.com/drive/folders/1YLdbzwslNkmi3No19C3aGdmfAUSoruzB (https://github.com/Navidfoumani/Series2Vec?tab=readme-ov-file)
    """

    file_path = os.path.join(base_path, dataset_title + ".npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    # load the dataset
    data = np.load(file_path, allow_pickle=True)
    train_data = data.item().get("train_data")
    train_label = data.item().get("train_label")
    test_data = data.item().get("test_data")
    test_label = data.item().get("test_label")
    total_data = np.concatenate((train_data, test_data), axis=0)
    total_label = np.concatenate((train_label, test_label), axis=0)

    return total_data, total_label

def test_torch_normalization(tensor):
    from pytorch_forecasting.data.encoders import TorchNormalizer

    normalizer = TorchNormalizer(method="standard")
    normalizer.fit(tensor)
    normalized = normalizer.transform(tensor)


if __name__ == "__main__":
    d = load_electricity_data()
    big_timer = time.time()
    m = wrapper.RustDataModule(d,wrapper.DatasetType.Forecasting,downsampling_rate=2,normalize=True)
    m.setup("stage")
    print(f"Total setup time: {time.time() - big_timer:.2f} seconds")
    print

