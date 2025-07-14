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
from pytorch_forecasting.data.encoders import TorchNormalizer
import dataset_loaders
import python_methods

# Helper function to extract data from tuples or arrays for type consistency
def extract_data(item):
    return item[0] if isinstance(item, tuple) else item

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

        if self.dataset_type == wrapper.DatasetType.Forecasting:
            timer = time.time()
            dataset = ForecastingDataSet(self.dataset)
            delta = time.time() - timer
            self.timings["rust"]["dataset_creation (Forecasting)"] = delta
        else:
            timer = time.time()

            dataset = ClassificationDataSet(self.dataset, self.labels)
            delta = time.time() - timer
            self.timings["rust"]["dataset_creation (Classification)"] = delta

        # Apply imputation strategy if specified
        if self.impute_strategy != ImputeStrategy.LeaveNaN:
            # First benchmark the imputation in Rust
            timer = time.time()
            dataset.impute()
            delta = time.time() - timer
            self.timings["rust"]["imputation"] = delta

            # Now benchmark the imputation in Python
            timer = time.time()
            self.working_datasets["python"] = python_methods.imputation(
                self.working_datasets["python"], self.impute_strategy
            )
            delta = time.time() - timer
            self.timings["python"]["imputation"] = delta

        # Apply downsampling if specified
        if self.downsampling_rate > 0:
            timer = time.time()
            dataset.downsample(self.downsampling_rate)
            delta = time.time() - timer
            self.timings["rust"]["downsampling"] = delta

            # Now benchmark the downsampling in Python
            timer = time.time()
            python_dataset, python_labels = python_methods.downsample(
                self.working_datasets["python"], self.downsampling_rate, self.working_labels["python"]
            )
            delta = time.time() - timer
            self.timings["python"]["downsampling"] = delta
            self.working_datasets["python"] = python_dataset
            if python_labels is not None:
                self.working_labels["python"] = python_labels

        # Split the data
        split_args = (
            (self.splitting_strategy, *self.splitting_ratios)
            if self.dataset_type == wrapper.DatasetType.Classification
            else self.splitting_ratios
        )
        timer = time.time()
        dataset.split(*split_args)
        delta = time.time() - timer
        self.timings["rust"]["data_splitting"] = delta

        # Now benchmark the splitting in Python
        timer = time.time()
        python_split_result = python_methods.splitting(
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
            timer = time.time()
            dataset.normalize()
            delta = time.time() - timer
            self.timings["rust"]["normalization"] = delta

            # Now benchmark the normalization in Python
            timer = time.time()
            # Ensure only data arrays are passed (not (data, label) tuples)
            

            python_dataset = python_methods.normalization(
                extract_data(self.split_datasets["python"][0]),
                extract_data(self.split_datasets["python"][1]),
                extract_data(self.split_datasets["python"][2])
            )
            delta = time.time() - timer
            self.timings["python"]["normalization"] = delta

        # Apply standardization if specified
        if self.standardize:
            timer = time.time()
            dataset.standardize()
            delta = time.time() - timer
            self.timings["rust"]["standardization"] = delta

            # Now benchmark the standardization in Python
            timer = time.time()
            python_dataset = python_methods.standardization(
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

def test_torch_normalization(tensor):
    

    normalizer = TorchNormalizer(method="standard")
    normalizer.fit(tensor)
    normalized = normalizer.transform(tensor)


if __name__ == "__main__":
    d = dataset_loaders.load_electricity_data()
    big_timer = time.time()
    m = BenchmarkingModule(d,wrapper.DatasetType.Forecasting,downsampling_rate=2,normalize=True,impute_strategy=ImputeStrategy.Mean)
    m.setup("stage")
    print(f"Total setup time: {time.time() - big_timer:.2f} seconds")
    print(m.timings)

