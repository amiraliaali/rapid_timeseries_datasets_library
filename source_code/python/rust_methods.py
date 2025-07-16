import time
import numpy as np
import wrapper
from rust_time_series.rust_time_series import (
    ForecastingDataSet,
    ClassificationDataSet,
    ImputeStrategy,
    SplittingStrategy,
)
import pytorch_lightning as L
import torch
from torch.utils.data import TensorDataset
import dataset_loaders

class BenchmarkingModule(wrapper.RustDataModule):
    def __init__(
        self,
        dataset: np.ndarray,
        dataset_type: wrapper.DatasetType,
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
        splitting_strategy: SplittingStrategy = SplittingStrategy.InOrder,
        splitting_ratios: tuple[float, float, float] = (
            0.7,
            0.2,
            0.1,
        ),  # Train, validation, test ratios
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
            splitting_ratios,
        )

        self.timings = {"rust": {}}

    def setup(self, stage: str):
        if self.dataset_type == wrapper.DatasetType.Forecasting:
            timer = time.time()
            dataset = ForecastingDataSet(self.dataset, *self.splitting_ratios)
            self.timings["rust"]["dataset_creation"] = time.time() - timer
        else:
            timer = time.time()
            dataset = ClassificationDataSet(
                self.dataset, self.labels, *self.splitting_ratios
            )
            self.timings["rust"]["dataset_creation"] = time.time() - timer

        # Apply imputation strategy if specified
        if self.impute_strategy != ImputeStrategy.LeaveNaN:
            timer = time.time()
            dataset.impute(self.impute_strategy)
            self.timings["rust"]["imputation"] = time.time() - timer

        # Apply downsampling if specified
        if self.downsampling_rate > 0:
            timer = time.time()
            dataset.downsample(self.downsampling_rate)
            self.timings["rust"]["downsampling"] = time.time() - timer

        # Split the data
        timer = time.time()
        if self.dataset_type == wrapper.DatasetType.Classification:
            dataset.split(self.splitting_strategy)
        else:
            dataset.split()
        self.timings["rust"]["data_splitting"] = time.time() - timer

        # Apply normalization if specified
        if self.normalize:
            timer = time.time()
            dataset.normalize()
            self.timings["rust"]["normalization"] = time.time() - timer

        # Apply standardization if specified
        if self.standardize:
            timer = time.time()
            dataset.standardize()
            self.timings["rust"]["standardization"] = time.time() - timer

        timer = time.time()
        # Collect the results
        collect_args = (
            (self.past_window, self.future_horizon, self.stride)
            if self.dataset_type == wrapper.DatasetType.Forecasting
            else ()
        )
        (
            (X_train, y_train),
            (X_val, y_val),
            (X_test, y_test),
        ) = dataset.collect(*collect_args)
        self.timings["rust"]["data_collection"] = time.time() - timer

        self.train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        self.val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        self.test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()

if __name__ == "__main__":
    d = dataset_loaders.load_electricity_data()
    big_timer = time.time()
    m = BenchmarkingModule(d, wrapper.DatasetType.Forecasting, downsampling_rate=2, normalize=True, impute_strategy=ImputeStrategy.Mean)
    m.setup("stage")
    print(f"Total setup time: {time.time() - big_timer:.2f} seconds")
    print(m.timings)

