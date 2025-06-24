from enum import Enum
import pytorch_lightning as L
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from rust_time_series.rust_time_series import (
    ForecastingDataSet,
    ClassificationDataSet,
    ImputeStrategy,
    SplittingStrategy,
)

print("Rust Time Series Wrapper Loaded")


class DatasetType(Enum):
    Forecasting = "forecasting"
    Classification = "classification"


class RustDataModule(L.LightningDataModule):
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
        super().__init__()

        self.dataset = dataset
        self.labels = labels
        self.dataset_type = dataset_type

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.downsampling_rate = downsampling_rate
        self.normalize = normalize
        self.standardize = standardize

        self.impute_strategy = impute_strategy

        self.splitting_strategy = splitting_strategy
        self.splitting_ratios = splitting_ratios

        self.past_window = past_window
        self.future_horizon = future_horizon
        self.stride = stride

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
            dataset.downsample(self.downsampling_rate)

        # Split the data
        split_args = (
            (self.splitting_strategy, *self.splitting_ratios)
            if self.dataset_type == DatasetType.Classification
            else self.splitting_ratios
        )
        dataset.split(*split_args)

        # Apply normalization if specified
        if self.normalize:
            dataset.normalize()

        # Apply standardization if specified
        if self.standardize:
            dataset.standardize()

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

        self.train_data = TensorDataset(X_train, y_train)
        self.val_data = TensorDataset(X_val, y_val)
        self.test_data = TensorDataset(X_test, y_test)

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
