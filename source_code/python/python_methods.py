import numpy as np
from rust_time_series.rust_time_series import (
    ImputeStrategy,
    SplittingStrategy,
)
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

# Helper function to extract data from tuples or arrays for type consistency
def extract_data(item):
    return item[0] if isinstance(item, tuple) else item

def imputation(
    dataset: np.ndarray,
    impute_strategy: ImputeStrategy,
) -> np.ndarray:
    """
    Apply the specified imputation strategy to the dataset.

    Args:
        dataset: The dataset to be processed.
        impute_strategy: The strategy to use for imputing missing values.

    Returns:
        The dataset with missing values imputed.
    """
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
    """
    Downsample the dataset by the specified rate.

    Args:
        dataset: The dataset to be processed.
        downsampling_rate: The rate at which to downsample the dataset.

    Returns:
        The downsampled dataset.
    """
    if downsampling_rate <= 0:
        raise ValueError("Downsampling rate must be greater than 0.")
    downsampled = dataset[:, ::downsampling_rate, :]
    if labels is not None:
        labels = labels[::downsampling_rate]
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
    """
    Split the dataset into training, validation, and testing sets.

    Args:
        dataset: The dataset to be processed (3D array: instances x timesteps x features).
        splitting_strategy: The strategy to use for splitting (Random or Temporal).
        splitting_ratios: Tuple of (train_prop, val_prop, test_prop) ratios.
        labels: Optional labels array for classification tasks.

    Returns:
        If labels is None: tuple of (train_data, val_data, test_data) - forecasting split
        If labels provided: tuple of ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels)) - classification split
    """
    train_prop, val_prop, test_prop = splitting_ratios

    # Validate proportions
    if train_prop < 0.0 or val_prop < 0.0 or test_prop < 0.0:
        raise ValueError("Sizes must be non-negative")

    if abs(train_prop + val_prop + test_prop - 1.0) > 1e-10:
        raise ValueError("Sizes must add up to 1")

    instances, timesteps, features = dataset.shape

    if labels is None:
        # split along time axis (timesteps)
        train_split_offset = int(round(timesteps * train_prop))
        val_split_offset = int(round(timesteps * val_prop))

        train_data = dataset[:, :train_split_offset, :]
        val_data = dataset[
            :, train_split_offset : train_split_offset + val_split_offset, :
        ]
        test_data = dataset[:, train_split_offset + val_split_offset :, :]

        return train_data, val_data, test_data

    else:
        # plit along instance axis (samples)
        train_split_offset = int(round(instances * train_prop))
        val_split_offset = int(round(instances * val_prop))

        if splitting_strategy == SplittingStrategy.Temporal:
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
    """
    Normalize the datasets using Min-Max normalization (global per-feature normalization)
    Args:
        train_data: Training dataset.
        val_data: Validation dataset.
        test_data: Testing dataset.
    Returns:
        Tuple of normalized datasets (train_data, val_data, test_data).
    """
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
    """
    Standardize the datasets using Z-score normalization (global per-feature standardization)
    Args:
        train_data: Training dataset.
        val_data: Validation dataset.
        test_data: Testing dataset.
    Returns:
        Tuple of standardized datasets (train_data, val_data, test_data).
    """
    features = train_data.shape[2]

    for j in range(features):
        feature_data = train_data[:, :, j].flatten()
        mean_value = sum(feature_data) / len(feature_data)
        std_value = sum((x - mean_value) ** 2 for x in feature_data) ** 0.5 / len(
            feature_data
        )

        if std_value == 0.0:
            std_value = 1.0

        train_data[:, :, j] = (train_data[:, :, j] - mean_value) / std_value
        val_data[:, :, j] = (val_data[:, :, j] - mean_value) / std_value
        test_data[:, :, j] = (test_data[:, :, j] - mean_value) / std_value

    return train_data, val_data, test_data


def collect_data(train_data, train_labels, val_data, val_labels, test_data, test_labels, dataset_type: wrapper.DatasetType, past_window=1, future_horizon=1, stride=1):
    """
    In case of classification, just return the split datasets and labels. In case of forecasting, perform sliding window generation on inputs and return those.
    Args:
        train_data, val_data, test_data: The input datasets.
        train_labels, val_labels, test_labels: The input labels.
        dataset_type: Classification or Forecasting.
        past_window: Number of past timesteps to include in each window.
        future_horizon: Number of future timesteps to predict.
        stride: Step size between consecutive windows.
    Returns:
        A tuple containing the collected dataset and labels.
    """
    if dataset_type == wrapper.DatasetType.Classification:
        return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
    elif dataset_type == wrapper.DatasetType.Forecasting:
        def create_windows(data, past_window, future_horizon, stride):
            """
            Create sliding windows for forecasting from the input data.
            
            Args:
                data: Input data of shape (instances, timesteps, features)
                past_window: Number of past timesteps to include
                future_horizon: Number of future timesteps to predict
                stride: Step size between windows
                
            Returns:
                Tuple of (x_windows, y_windows)
            """
            if past_window <= 0 or future_horizon <= 0 or stride <= 0:
                raise ValueError("past_window, future_horizon, and stride must be > 0")
            
            instances, timesteps, features = data.shape
            
            if past_window + future_horizon > timesteps:
                raise ValueError("past_window + future_horizon cannot exceed timesteps")
            
            windows_per_instance = (timesteps - past_window - future_horizon) / stride + 1
            total_windows = instances * windows_per_instance
            
            x_windows = np.zeros((total_windows, past_window, features))
            y_windows = np.zeros((total_windows, future_horizon, features))
            
            window_idx = 0
            
            for instance in range(instances):
                for i in range(0, timesteps - past_window - future_horizon + 1, stride):
                    x_windows[window_idx] = data[instance, i:i + past_window, :]
                    
                    y_windows[window_idx] = data[instance, i + past_window:i + past_window + future_horizon, :]
                    
                    window_idx += 1
            
            return x_windows, y_windows
        
        train_x, train_y = create_windows(train_data, past_window, future_horizon, stride)
        val_x, val_y = create_windows(val_data, past_window, future_horizon, stride)
        test_x, test_y = create_windows(test_data, past_window, future_horizon, stride)
        
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


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
            splitting_ratios,
        )

        self.timings = {"rust":{},
                        "python":{},
                        "pytorch":{}}

        self.working_datasets: dict[str, np.ndarray | None] = {"python": None,
                                 "pytorch": None}
        self.working_labels: dict[str, np.ndarray | None] = {"python": None,
                               "pytorch": None}
        self.split_datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {
            "python": None,
            "pytorch": None,
        }
        self.split_labels: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {
            "python": None,
            "pytorch": None,
        }

    def setup(self, stage: str):
        # make 2 copies of the dataset for benchmarking and assign them to the working datasets
        self.working_datasets["python"] = self.dataset.copy()
        self.working_datasets["pytorch"] = self.dataset.copy()
        self.working_labels["python"] = self.labels.copy() if self.labels is not None else None
        self.working_labels["pytorch"] = self.labels.copy() if self.labels is not None else None

        # Apply imputation strategy if specified
        if self.impute_strategy != ImputeStrategy.LeaveNaN:
            # Now benchmark the imputation in Python
            timer = time.time()
            self.working_datasets["python"] = imputation(
                self.working_datasets["python"], self.impute_strategy
            )
            delta = time.time() - timer
            self.timings["python"]["imputation"] = delta

        # Apply downsampling if specified
        if self.downsampling_rate > 0:
            # Now benchmark the downsampling in Python
            timer = time.time()
            python_dataset, python_labels = downsample(
                self.working_datasets["python"], self.downsampling_rate, self.working_labels["python"]
            )
            delta = time.time() - timer
            self.timings["python"]["downsampling"] = delta
            self.working_datasets["python"] = python_dataset
            if python_labels is not None:
                self.working_labels["python"] = python_labels

        # Now benchmark the splitting in Python
        timer = time.time()
        python_split_result = splitting(
            self.working_datasets["python"],
            self.splitting_strategy,
            self.splitting_ratios,
            self.working_labels["python"],
        )
        delta = time.time() - timer
        self.timings["python"]["splitting"] = delta
        
        # Handle the return values based on dataset type
        if self.dataset_type == wrapper.DatasetType.Forecasting:
            # For forecasting: returns (train_data, val_data, test_data)
            train_data, val_data, test_data = python_split_result
            self.split_datasets["python"] = (train_data, val_data, test_data)
            self.split_labels["python"] = None  # No labels for forecasting
        else:
            # For classification: returns ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels))
            (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = python_split_result
            self.split_datasets["python"] = (train_data, val_data, test_data)
            self.split_labels["python"] = (train_labels, val_labels, test_labels)

        # Apply normalization if specified
        if self.normalize:
            # Now benchmark the normalization in Python
            timer = time.time()
            # Ensure only data arrays are passed (not (data, label) tuples)
            
            python_dataset = normalization(
                extract_data(self.split_datasets["python"][0]),
                extract_data(self.split_datasets["python"][1]),
                extract_data(self.split_datasets["python"][2])
            )
            delta = time.time() - timer
            self.timings["python"]["normalization"] = delta

        # Apply standardization if specified
        if self.standardize:
            # Now benchmark the standardization in Python
            timer = time.time()
            python_dataset = standardization(
                extract_data(self.split_datasets["python"][0]),
                extract_data(self.split_datasets["python"][1]),
                extract_data(self.split_datasets["python"][2])
            )
            delta = time.time() - timer
            self.timings["python"]["standardization"] = delta

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
        delta = time.time() - timer
        self.timings["rust"]["data_collection"] = delta

        self.train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        self.val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
        self.test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()