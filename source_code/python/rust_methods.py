import time
import numpy as np
import wrapper
from rust_time_series.rust_time_series import (
    ForecastingDataSet,
    ClassificationDataSet,
    ImputeStrategy,
    SplittingStrategy,
)
import torch
from torch.utils.data import TensorDataset
import dataset_loaders
from memory_monitor import ProcessStepMemoryTracker


class RustBenchmarkingModule(wrapper.RustDataModule):
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
        self.memory_tracker = ProcessStepMemoryTracker("rust")
        self.memory_usage = self.memory_tracker.get_memory_usage()

    def setup(self, stage: str):
        if self.dataset_type == wrapper.DatasetType.Forecasting:
            with self.memory_tracker.track_step("dataset_creation"):
                timer = time.perf_counter()
                dataset = ForecastingDataSet(self.dataset, *self.splitting_ratios)
                self.timings["rust"]["dataset_creation"] = time.perf_counter() - timer
        else:
            with self.memory_tracker.track_step("dataset_creation"):
                timer = time.perf_counter()
                dataset = ClassificationDataSet(
                    self.dataset, self.labels, *self.splitting_ratios
                )
                self.timings["rust"]["dataset_creation"] = time.perf_counter() - timer

        if self.impute_strategy != ImputeStrategy.LeaveNaN:
            with self.memory_tracker.track_step("imputation"):
                timer = time.perf_counter()
                dataset.impute(self.impute_strategy)
                self.timings["rust"]["imputation"] = time.perf_counter() - timer

        if self.downsampling_rate > 0:
            with self.memory_tracker.track_step("downsampling"):
                timer = time.perf_counter()
                dataset.downsample(self.downsampling_rate)
                self.timings["rust"]["downsampling"] = time.perf_counter() - timer

        with self.memory_tracker.track_step("data_splitting"):
            timer = time.perf_counter()
            if self.dataset_type == wrapper.DatasetType.Classification:
                dataset.split(self.splitting_strategy)
            else:
                dataset.split()
            self.timings["rust"]["data_splitting"] = time.perf_counter() - timer

        if self.normalize:
            with self.memory_tracker.track_step("normalization"):
                timer = time.perf_counter()
                dataset.normalize()
                self.timings["rust"]["normalization"] = time.perf_counter() - timer
        if self.standardize:
            with self.memory_tracker.track_step("standardization"):
                timer = time.perf_counter()
                dataset.standardize()
                self.timings["rust"]["standardization"] = time.perf_counter() - timer

        with self.memory_tracker.track_step("data_collection"):
            timer = time.perf_counter()
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
            self.timings["rust"]["data_collection"] = time.perf_counter() - timer

        self.memory_usage = self.memory_tracker.get_memory_usage()

        self.train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        self.val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        self.test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
