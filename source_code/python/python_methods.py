import numpy as np
from rust_time_series.rust_time_series import (
    ImputeStrategy,
    SplittingStrategy,
)
import time
import numpy as np
import wrapper
import torch
import dataset_loaders
from torch.utils.data import TensorDataset
from memory_monitor import ProcessStepMemoryTracker
import math


def imputation(
    dataset: np.ndarray,
    impute_strategy: ImputeStrategy,
) -> np.ndarray:
    instance_count: int = dataset.shape[0]
    if impute_strategy != ImputeStrategy.LeaveNaN:
        # the dataset is a 3 dim numpy array
        for i in range(instance_count):
            feature_count: int = dataset[i].shape[1]
            for j in range(feature_count):
                if impute_strategy == ImputeStrategy.Mean:
                    # calculate mean value in pure python (no numpy)
                    values = [x for x in dataset[i, :, j] if not np.isnan(x)]
                    if values:
                        mean_value = sum(values) / len(values)
                        for k in range(len(dataset[i, :, j])):
                            if np.isnan(dataset[i, :, j][k]):
                                dataset[i, :, j][k] = mean_value
                elif impute_strategy == ImputeStrategy.Median:
                    values = [x for x in dataset[i, :, j] if not np.isnan(x)]
                    if values:
                        values.sort()
                        n = len(values)
                        median_value = (
                            values[n // 2]
                            if n % 2 == 1
                            else (values[n // 2 - 1] + values[n // 2]) / 2
                        )
                        for k in range(len(dataset[i, :, j])):
                            if np.isnan(dataset[i, :, j][k]):
                                dataset[i, :, j][k] = median_value
                elif impute_strategy == ImputeStrategy.ForwardFill:
                    fallback_value = None
                    for k in range(len(dataset[i, :, j])):
                        if np.isnan(dataset[i, :, j][k]):
                            if fallback_value is not None:
                                dataset[i, :, j][k] = fallback_value
                        else:
                            fallback_value = dataset[i, :, j][k]
                elif impute_strategy == ImputeStrategy.BackwardFill:
                    fallback_value = None
                    for k in range(len(dataset[i, :, j]) - 1, -1, -1):
                        if np.isnan(dataset[i, :, j][k]):
                            if fallback_value is not None:
                                dataset[i, :, j][k] = fallback_value
                        else:
                            fallback_value = dataset[i, :, j][k]
    return dataset


def downsample(
    dataset: np.ndarray,
    downsampling_rate: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if downsampling_rate <= 0:
        raise ValueError("Downsampling rate must be greater than 0.")
    downsampled = dataset[:, ::downsampling_rate, :]
    return downsampled, labels


def splitting(
    dataset: np.ndarray,
    splitting_strategy: SplittingStrategy,
    splitting_ratios: tuple[float, float, float],
    labels: np.ndarray | None = None,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]
):

    train_prop, val_prop, test_prop = splitting_ratios

    if train_prop < 0.0 or val_prop < 0.0 or test_prop < 0.0:
        raise ValueError("Sizes must be non-negative")

    if abs(train_prop + val_prop + test_prop - 1.0) > 1e-10:
        raise ValueError("Sizes must add up to 1")

    instances, timesteps, features = dataset.shape

    if labels is None:
        # split along time axis (timesteps)
        train_split_offset = int(math.floor(timesteps * train_prop + 0.5))
        val_split_offset = int(math.floor(timesteps * val_prop + 0.5))

        train_data = dataset[:, :train_split_offset, :]
        val_data = dataset[
            :, train_split_offset : train_split_offset + val_split_offset, :
        ]
        test_data = dataset[:, train_split_offset + val_split_offset :, :]

        return train_data, val_data, test_data

    else:
        # split along instance axis (samples)
        train_split_offset = int(math.floor(instances * train_prop + 0.5))
        val_split_offset = int(math.floor(instances * val_prop + 0.5))

        if splitting_strategy == SplittingStrategy.InOrder:
            train_data = dataset[:train_split_offset, :, :]
            val_data = dataset[
                train_split_offset : train_split_offset + val_split_offset, :, :
            ]
            test_data = dataset[train_split_offset + val_split_offset :, :, :]

            train_labels = labels[:train_split_offset]
            val_labels = labels[
                train_split_offset : train_split_offset + val_split_offset
            ]
            test_labels = labels[train_split_offset + val_split_offset :]

            return (
                (train_data, train_labels),
                (val_data, val_labels),
                (test_data, test_labels),
            )

        elif splitting_strategy == SplittingStrategy.Random:
            import random

            indices = list(range(instances))
            random.shuffle(indices)

            train_indices = indices[:train_split_offset]
            val_indices = indices[
                train_split_offset : train_split_offset + val_split_offset
            ]
            test_indices = indices[train_split_offset + val_split_offset :]

            train_data = dataset[train_indices, :, :]
            val_data = dataset[val_indices, :, :]
            test_data = dataset[test_indices, :, :]

            train_labels = labels[train_indices]
            val_labels = labels[val_indices]
            test_labels = labels[test_indices]

            return (
                (train_data, train_labels),
                (val_data, val_labels),
                (test_data, test_labels),
            )

        else:
            raise ValueError(f"Unknown splitting strategy: {splitting_strategy}")


def normalization(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = train_data.shape[2]

    for j in range(features):
        feature_data = train_data[:, :, j].flatten()
        min_value = min(feature_data)
        max_value = max(feature_data)

        range_value = max_value - min_value
        if range_value == 0.0:
            range_value = 1.0

        train_data[:, :, j] = (train_data[:, :, j] - min_value) / range_value
        val_data[:, :, j] = (val_data[:, :, j] - min_value) / range_value
        test_data[:, :, j] = (test_data[:, :, j] - min_value) / range_value

    return train_data, val_data, test_data


def standardization(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = train_data.shape[2]

    for j in range(features):
        feature_data = train_data[:, :, j].flatten()
        mean_value = sum(feature_data) / len(feature_data)
        variance = sum((x - mean_value) ** 2 for x in feature_data) / len(feature_data)
        std_value = variance**0.5

        if std_value == 0.0:
            std_value = 1.0

        train_data[:, :, j] = (train_data[:, :, j] - mean_value) / std_value
        val_data[:, :, j] = (val_data[:, :, j] - mean_value) / std_value
        test_data[:, :, j] = (test_data[:, :, j] - mean_value) / std_value

    return train_data, val_data, test_data


def collect_data(
    train_data,
    train_labels,
    val_data,
    val_labels,
    test_data,
    test_labels,
    dataset_type: wrapper.DatasetType,
    past_window=1,
    future_horizon=1,
    stride=1,
):
    if dataset_type == wrapper.DatasetType.Classification:
        return (
            (train_data, train_labels),
            (val_data, val_labels),
            (test_data, test_labels),
        )
    elif dataset_type == wrapper.DatasetType.Forecasting:

        def create_windows(data, past_window, future_horizon, stride):

            if past_window <= 0 or future_horizon <= 0 or stride <= 0:
                raise ValueError("past_window, future_horizon, and stride must be > 0")

            instances, timesteps, features = data.shape

            if past_window + future_horizon > timesteps:
                raise ValueError("past_window + future_horizon cannot exceed timesteps")

            windows_per_instance = (
                timesteps - past_window - future_horizon
            ) // stride + 1
            total_windows = int(instances * windows_per_instance)

            x_windows = np.zeros((total_windows, past_window, features))
            y_windows = np.zeros((total_windows, future_horizon, features))

            window_idx = 0

            for instance in range(instances):
                for i in range(0, timesteps - past_window - future_horizon + 1, stride):
                    x_windows[window_idx] = data[instance, i : i + past_window, :]

                    y_windows[window_idx] = data[
                        instance, i + past_window : i + past_window + future_horizon, :
                    ]

                    window_idx += 1

            return x_windows, y_windows

        train_x, train_y = create_windows(
            train_data, past_window, future_horizon, stride
        )
        val_x, val_y = create_windows(val_data, past_window, future_horizon, stride)
        test_x, test_y = create_windows(test_data, past_window, future_horizon, stride)

        return (train_x, train_y), (val_x, val_y), (test_x, test_y)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


class PythonBenchmarkingModule(wrapper.RustDataModule):
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

        self.timings = {"python": {}}
        self.memory_tracker = ProcessStepMemoryTracker("python")
        self.memory_usage = self.memory_tracker.get_memory_usage()

        self.working_datasets: dict[str, np.ndarray | None] = {"python": None}
        self.working_labels: dict[str, np.ndarray | None] = {"python": None}
        self.split_datasets: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ] = {
            "python": None,
        }
        self.split_labels: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ] = {
            "python": None,
        }

    def setup(self, stage: str):
        # make 2 copies of the dataset for benchmarking and assign them to the working datasets
        self.working_datasets["python"] = self.dataset.copy()
        self.working_labels["python"] = (
            self.labels.copy() if self.labels is not None else None
        )

        if self.impute_strategy != ImputeStrategy.LeaveNaN:
            with self.memory_tracker.track_step("imputation"):
                timer = time.perf_counter()
                self.working_datasets["python"] = imputation(
                    self.working_datasets["python"], self.impute_strategy
                )
                delta = time.perf_counter() - timer
                self.timings["python"]["imputation"] = delta

        if self.downsampling_rate > 0:
            with self.memory_tracker.track_step("downsampling"):
                timer = time.perf_counter()
                python_dataset, python_labels = downsample(
                    self.working_datasets["python"],
                    self.downsampling_rate,
                    self.working_labels["python"],
                )
                delta = time.perf_counter() - timer
                self.timings["python"]["downsampling"] = delta
                self.working_datasets["python"] = python_dataset
                if python_labels is not None:
                    self.working_labels["python"] = python_labels

        with self.memory_tracker.track_step("splitting"):
            timer = time.perf_counter()
            python_split_result = splitting(
                self.working_datasets["python"],
                self.splitting_strategy,
                self.splitting_ratios,
                self.working_labels["python"],
            )
            delta = time.perf_counter() - timer
            self.timings["python"]["splitting"] = delta

        if self.dataset_type == wrapper.DatasetType.Forecasting:
            train_data, val_data, test_data = python_split_result
            self.split_datasets["python"] = (train_data, val_data, test_data)
            self.split_labels["python"] = None
        else:
            (
                (train_data, train_labels),
                (val_data, val_labels),
                (test_data, test_labels),
            ) = python_split_result
            self.split_datasets["python"] = (train_data, val_data, test_data)
            self.split_labels["python"] = (train_labels, val_labels, test_labels)

        if self.normalize:
            with self.memory_tracker.track_step("normalization"):
                timer = time.perf_counter()
                python_dataset = normalization(
                    self.split_datasets["python"][0],
                    self.split_datasets["python"][1],
                    self.split_datasets["python"][2],
                )
                delta = time.perf_counter() - timer
                self.timings["python"]["normalization"] = delta
                self.split_datasets["python"] = python_dataset

        if self.standardize:
            with self.memory_tracker.track_step("standardization"):
                timer = time.perf_counter()
                python_dataset = standardization(
                    self.split_datasets["python"][0],
                    self.split_datasets["python"][1],
                    self.split_datasets["python"][2],
                )
                delta = time.perf_counter() - timer
                self.timings["python"]["standardization"] = delta
                self.split_datasets["python"] = python_dataset

        with self.memory_tracker.track_step("data_collection"):
            timer = time.perf_counter()
            if self.dataset_type == wrapper.DatasetType.Forecasting:
                # for forecasting, apply sliding window generation
                python_collected = collect_data(
                    self.split_datasets["python"][0],  # train_data
                    None,  # train_labels (None for forecasting)
                    self.split_datasets["python"][1],  # val_data
                    None,  # val_labels (None for forecasting)
                    self.split_datasets["python"][2],  # test_data
                    None,  # test_labels (None for forecasting)
                    self.dataset_type,
                    self.past_window,
                    self.future_horizon,
                    self.stride,
                )
            else:
                python_collected = collect_data(
                    self.split_datasets["python"][0],  # train_data
                    self.split_labels["python"][0],  # train_labels
                    self.split_datasets["python"][1],  # val_data
                    self.split_labels["python"][1],  # val_labels
                    self.split_datasets["python"][2],  # test_data
                    self.split_labels["python"][2],  # test_labels
                    self.dataset_type,
                    self.past_window,
                    self.future_horizon,
                    self.stride,
                )
            delta = time.perf_counter() - timer
            self.timings["python"]["data_collection"] = delta

        self.memory_usage = self.memory_tracker.get_memory_usage()

        (train_x, train_y), (val_x, val_y), (test_x, test_y) = python_collected

        self.train_data = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
        self.val_data = TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y))
        self.test_data = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
