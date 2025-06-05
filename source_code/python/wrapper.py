import pytorch_lightning as L
from torch.utils.data import DataLoader
import numpy as np
from rust_time_series.rust_time_series import (
    ForecastingDataSet,
    DatasetType,
    ImputeStrategy,
    SplittingStrategy,
)

print ("Rust Time Series Wrapper Loaded")

class RustDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: np.ndarray,
        dataset_type: DatasetType,
        batch_size: int = 32,
        num_workers: int = 0,
        normalize: bool = False,
        standardize: bool = False,
        impute_strategy: ImputeStrategy = ImputeStrategy.LeaveNaN,
        splitting_strategy: SplittingStrategy = SplittingStrategy.Temporal,
        splitting_ratios: tuple = (0.7, 0.2, 0.1),  # Train, validation, test ratios
    ):
        super().__init__()

        self.dataset = dataset

        self.dataset_type = dataset_type

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.normalize = normalize
        self.standardize = standardize

        self.impute_strategy = impute_strategy

        self.splitting_strategy = splitting_strategy
        self.splitting_ratios = splitting_ratios

    def setup(self):
        # call the method that applies the preprocessing steps and returns the split datasets
        ts = ForecastingDataSet(self.dataset, self.dataset_type)

        # Apply normalization if specified
        if self.normalize:
            ts.normalize()

        # Apply standardization if specified
        if self.standardize:
            ts.standardize()

        # Apply imputation strategy if specified
        if self.impute_strategy != ImputeStrategy.LeaveNaN:
            ts.impute(self.impute_strategy)

        # Split the dataset into train, validation, and test sets
        (self.train_data, self.val_data, self.test_data) = ts.split(
            self.splitting_strategy, *self.splitting_ratios
        )

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
