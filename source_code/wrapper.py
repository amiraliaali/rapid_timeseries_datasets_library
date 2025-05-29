import pytorch_lightning as L
from torch.utils.data import DataLoader
from enum import Enum


class DataSetType(Enum):
    FORECASTING = "forecasting"
    CLASSIFICATION = "classification"


class SplittingStrategy(Enum):
    TEMPORAL = "temporal"  # Temporal split for time series data


class ImputeStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "forward_fill"  # Forward fill imputation
    NONE = "none"  # No imputation


class RustDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        data_set_type: DataSetType,
        batch_size: int = 32,
        num_workers: int = 0,
        normalize: bool = False,
        standardize: bool = False,
        impute_strategy: ImputeStrategy = ImputeStrategy.NONE,
        splitting_strategy: SplittingStrategy = SplittingStrategy.TEMPORAL,
        splitting_ratios: tuple = (0.7, 0.2, 0.1),  # Train, validation, test ratios
    ):
        super().__init__()

        self.dataset = dataset

        self.data_set_type = data_set_type

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.normalize = normalize
        self.standardize = standardize

        self.impute_strategy = impute_strategy

        self.splitting_strategy = splitting_strategy
        self.splitting_ratios = splitting_ratios

    # most likely not needed
    def prepare_data(self):
        # This method is used to download the dataset if needed
        pass

    def setup(self, stage: str):
        # call the method that applies the preprocessing steps and returns the split datasets

        self.train_data = None
        self.val_data = None
        self.test_data = None
        pass

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
