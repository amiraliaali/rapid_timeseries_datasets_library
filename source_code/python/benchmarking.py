import numpy as np
from rust_time_series.rust_time_series import (
    ImputeStrategy,
    SplittingStrategy,
)
import time
import numpy as np
import wrapper
import psutil
import json
import os
from rust_time_series.rust_time_series import (
    ForecastingDataSet,
    ClassificationDataSet,
    ImputeStrategy,
    SplittingStrategy,
)
import pytorch_lightning as L
import torch
import dataset_loaders
from torch.utils.data import TensorDataset

from python_methods import PythonBenchmarkingModule
from numpy_methods import NumpyBenchmarkingModule
from rust_methods import RustBenchmarkingModule


def benchmark(
    original_data,
    dataset_type,
    past_window,
    future_horizon,
    stride,
    original_labels,
    batch_size,
    num_workers,
    downsampling_rate,
    normalize,
    standardize,
    impute_strategy,
    splitting_strategy,
    splitting_ratios,
):
    benchmark_results = []

    d = original_data.copy()
    l = original_labels.copy()
    python_timer = time.time()
    python_module = PythonBenchmarkingModule(
        d,
        dataset_type,
        past_window=past_window,
        future_horizon=future_horizon,
        stride=stride,
        labels=l,
        batch_size=batch_size,
        num_workers=num_workers,
        downsampling_rate=downsampling_rate,
        normalize=normalize,
        standardize=standardize,
        impute_strategy=impute_strategy,
        splitting_strategy=splitting_strategy,
        splitting_ratios=splitting_ratios,
    )
    python_module.setup("stage")
    python_train_loader = python_module.train_dataloader()
    python_test_loader = python_module.test_dataloader()
    python_data = next(iter(python_train_loader))
    python_timer = time.time() - python_timer
    print(f"Python Module Time: {python_timer:.4f} seconds")
    print(f"Python Data Shape: {python_data[0].shape}")
    print(
        f"Python Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"
    )
    benchmark_results.append(python_module.memory_usage)
    benchmark_results.append(python_module.timings)
    del python_module
    del python_train_loader
    del python_test_loader
    del python_data

    d = original_data.copy()
    l = original_labels.copy()
    numpy_timer = time.time()
    numpy_module = NumpyBenchmarkingModule(
        d,
        dataset_type,
        past_window=past_window,
        future_horizon=future_horizon,
        stride=stride,
        labels=l,
        batch_size=batch_size,
        num_workers=num_workers,
        downsampling_rate=downsampling_rate,
        normalize=normalize,
        standardize=standardize,
        impute_strategy=impute_strategy,
        splitting_strategy=splitting_strategy,
        splitting_ratios=splitting_ratios,
    )
    numpy_module.setup("stage")
    numpy_train_loader = numpy_module.train_dataloader()
    numpy_test_loader = numpy_module.test_dataloader()
    numpy_data = next(iter(numpy_train_loader))
    numpy_timer = time.time() - numpy_timer
    print(f"Numpy Module Time: {numpy_timer:.4f} seconds")
    print(f"Numpy Data Shape: {numpy_data[0].shape}")
    print(
        f"Numpy Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"
    )
    benchmark_results.append(numpy_module.memory_usage)
    benchmark_results.append(numpy_module.timings)
    del numpy_module
    del numpy_train_loader
    del numpy_test_loader
    del numpy_data

    d = original_data.copy()
    l = original_labels.copy()
    rust_timer = time.time()
    rust_module = RustBenchmarkingModule(
        d,
        dataset_type,
        past_window=past_window,
        future_horizon=future_horizon,
        stride=stride,
        labels=l,
        batch_size=batch_size,
        num_workers=num_workers,
        downsampling_rate=downsampling_rate,
        normalize=normalize,
        standardize=standardize,
        impute_strategy=impute_strategy,
        splitting_strategy=splitting_strategy,
        splitting_ratios=splitting_ratios,
    )
    rust_module.setup("stage")
    rust_train_loader = rust_module.train_dataloader()
    rust_test_loader = rust_module.test_dataloader()
    rust_data = next(iter(rust_train_loader))
    rust_timer = time.time() - rust_timer
    print(f"Rust Module Time: {rust_timer:.4f} seconds")
    print(f"Rust Data Shape: {rust_data[0].shape}")
    print(
        f"Rust Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"
    )
    benchmark_results.append(rust_module.memory_usage)
    benchmark_results.append(rust_module.timings)
    del rust_module
    del rust_train_loader
    del rust_test_loader
    del rust_data
    print("Benchmarking completed.")
    with open("benchmarking_results.json", "w") as f:
        json.dump(benchmark_results, f)


if __name__ == "__main__":
    original_data, original_labels = dataset_loaders.load_aeon_data(
        "ArticularyWordRecognition"
    )
    dataset_type = wrapper.DatasetType.Classification
    past_window = 12
    future_horizon = 6
    stride = 1
    batch_size = 32
    num_workers = 0
    downsampling_rate: int = 0
    normalize = False
    standardize = False
    impute_strategy = ImputeStrategy.LeaveNaN
    splitting_strategy = SplittingStrategy.InOrder
    splitting_ratios = (0.7, 0.2, 0.1)

    impute_strategy = ImputeStrategy.LeaveNaN
    splitting_strategy = SplittingStrategy.InOrder
    splitting_ratios = (0.7, 0.2, 0.1)

    benchmark(
        original_data,
        dataset_type,
        past_window,
        future_horizon,
        stride,
        original_labels,
        batch_size,
        num_workers,
        downsampling_rate,
        normalize,
        standardize,
        impute_strategy,
        splitting_strategy,
        splitting_ratios,
    )
