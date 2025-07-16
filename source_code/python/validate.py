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


def validate_data_consistency(
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
    print("Starting data consistency validation...")

    train_prop, val_prop, test_prop = splitting_ratios
    instances = original_data.shape[0]
    train_split_offset = int(round(instances * train_prop))
    val_split_offset = int(round(instances * val_prop))

    print(f"Debug: instances={instances}, train_prop={train_prop}, val_prop={val_prop}")
    print(
        f"Debug: train_split_offset={train_split_offset}, val_split_offset={val_split_offset}"
    )
    print(f"Debug: Expected train size: {train_split_offset}")
    print(f"Debug: Expected val size: {val_split_offset}")
    print(
        f"Debug: Expected test size: {instances - train_split_offset - val_split_offset}"
    )

    d = original_data.copy()
    l = original_labels.copy() if original_labels is not None else None
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
    python_validation_loader = python_module.val_dataloader()
    python_train_batch = next(iter(python_train_loader))
    python_test_batch = next(iter(python_test_loader))
    python_validation_batch = next(iter(python_validation_loader))
    print(
        f"Python dataset sizes: train={len(python_train_loader.dataset)}, val={len(python_validation_loader.dataset)}, test={len(python_test_loader.dataset)}"
    )

    d = original_data.copy()
    l = original_labels.copy() if original_labels is not None else None
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
    numpy_validation_loader = numpy_module.val_dataloader()
    numpy_train_batch = next(iter(numpy_train_loader))
    numpy_test_batch = next(iter(numpy_test_loader))
    numpy_validation_batch = next(iter(numpy_validation_loader))
    print(
        f"Numpy dataset sizes: train={len(numpy_train_loader.dataset)}, val={len(numpy_validation_loader.dataset)}, test={len(numpy_test_loader.dataset)}"
    )

    d = original_data.copy()
    l = original_labels.copy() if original_labels is not None else None
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
    rust_validation_loader = rust_module.val_dataloader()
    rust_validation_batch = next(iter(rust_validation_loader))
    rust_train_batch = next(iter(rust_train_loader))
    rust_test_batch = next(iter(rust_test_loader))
    print(
        f"Rust dataset sizes: train={len(rust_train_loader.dataset)}, val={len(rust_validation_loader.dataset)}, test={len(rust_test_loader.dataset)}"
    )
    all_consistent = True

    try:
        if not torch.allclose(
            python_train_batch[0], numpy_train_batch[0], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs NumPy training features differ")
            all_consistent = False
        else:
            print("PASS: Python vs NumPy training features match")

        if not torch.allclose(
            python_train_batch[0], rust_train_batch[0], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs Rust training features differ")
            all_consistent = False
        else:
            print("PASS: Python vs Rust training features match")

        if not torch.allclose(
            numpy_train_batch[0], rust_train_batch[0], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: NumPy vs Rust training features differ")
            all_consistent = False
        else:
            print("PASS: NumPy vs Rust training features match")

        if not torch.allclose(
            python_train_batch[1], numpy_train_batch[1], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs NumPy training labels differ")
            all_consistent = False
        else:
            print("PASS: Python vs NumPy training labels match")

        if not torch.allclose(
            python_train_batch[1], rust_train_batch[1], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs Rust training labels differ")
            all_consistent = False
        else:
            print("PASS: Python vs Rust training labels match")

        if not torch.allclose(
            numpy_train_batch[1], rust_train_batch[1], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: NumPy vs Rust training labels differ")
            all_consistent = False
        else:
            print("PASS: NumPy vs Rust training labels match")

    except Exception as e:
        print(f"ERROR: Error comparing training data: {e}")
        all_consistent = False

    try:
        if not torch.allclose(
            python_test_batch[0], numpy_test_batch[0], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs NumPy test features differ")
            all_consistent = False
        else:
            print("PASS: Python vs NumPy test features match")

        if not torch.allclose(
            python_test_batch[0], rust_test_batch[0], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs Rust test features differ")
            all_consistent = False
        else:
            print("PASS: Python vs Rust test features match")

        if not torch.allclose(
            numpy_test_batch[0], rust_test_batch[0], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: NumPy vs Rust test features differ")
            all_consistent = False
        else:
            print("PASS: NumPy vs Rust test features match")

        if not torch.allclose(
            python_test_batch[1], numpy_test_batch[1], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs NumPy test labels differ")
            all_consistent = False
        else:
            print("PASS: Python vs NumPy test labels match")

        if not torch.allclose(
            python_test_batch[1], rust_test_batch[1], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs Rust test labels differ")
            all_consistent = False
        else:
            print("PASS: Python vs Rust test labels match")

        if not torch.allclose(
            numpy_test_batch[1], rust_test_batch[1], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: NumPy vs Rust test labels differ")
            all_consistent = False
        else:
            print("PASS: NumPy vs Rust test labels match")

    except Exception as e:
        print(f"ERROR: Error comparing test data: {e}")
        all_consistent = False
    try:
        if not torch.allclose(
            python_validation_batch[0], numpy_validation_batch[0], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs NumPy validation features differ")
            all_consistent = False
        else:
            print("PASS: Python vs NumPy validation features match")
        if not torch.allclose(
            python_validation_batch[0], rust_validation_batch[0], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs Rust validation features differ")
            all_consistent = False
        else:
            print("PASS: Python vs Rust validation features match")
        if not torch.allclose(
            numpy_validation_batch[0], rust_validation_batch[0], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: NumPy vs Rust validation features differ")
            all_consistent = False
        else:
            print("PASS: NumPy vs Rust validation features match")
        if not torch.allclose(
            python_validation_batch[1], numpy_validation_batch[1], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs NumPy validation labels differ")
            all_consistent = False
        else:
            print("PASS: Python vs NumPy validation labels match")
        if not torch.allclose(
            python_validation_batch[1], rust_validation_batch[1], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: Python vs Rust validation labels differ")
            all_consistent = False
        else:
            print("PASS: Python vs Rust validation labels match")
        if not torch.allclose(
            numpy_validation_batch[1], rust_validation_batch[1], rtol=1e-5, atol=1e-8
        ):
            print("FAIL: NumPy vs Rust validation labels differ")
            all_consistent = False
        else:
            print("PASS: NumPy vs Rust validation labels match")

    except Exception as e:
        print(f"ERROR: Error comparing test data: {e}")
        all_consistent = False

    print(f"\nData shapes:")
    print(
        f"Python train: {python_train_batch[0].shape}, labels: {python_train_batch[1].shape}"
    )
    print(
        f"NumPy train: {numpy_train_batch[0].shape}, labels: {numpy_train_batch[1].shape}"
    )
    print(
        f"Rust train: {rust_train_batch[0].shape}, labels: {rust_train_batch[1].shape}"
    )
    print(
        f"Python test: {python_test_batch[0].shape}, labels: {python_test_batch[1].shape}"
    )
    print(
        f"NumPy test: {numpy_test_batch[0].shape}, labels: {numpy_test_batch[1].shape}"
    )
    print(f"Rust test: {rust_test_batch[0].shape}, labels: {rust_test_batch[1].shape}")

    del python_module, numpy_module, rust_module
    del python_train_loader, python_test_loader
    del numpy_train_loader, numpy_test_loader
    del rust_train_loader, rust_test_loader
    del python_train_batch, python_test_batch
    del numpy_train_batch, numpy_test_batch
    del rust_train_batch, rust_test_batch

    if all_consistent:
        print("\nSUCCESS: All implementations produce consistent data!")
    else:
        print("\nWARNING: Data inconsistencies detected between implementations!")

    return all_consistent


if __name__ == "__main__":
    original_data, original_labels = (
        dataset_loaders.load_aeon_data("ArticularyWordRecognition"),
    )
    dataset_type = wrapper.DatasetType.Classification
    past_window = 12
    future_horizon = 6
    stride = 1
    batch_size = 32
    num_workers = 0
    downsampling_rate: int = 2
    normalize = False
    standardize = True
    impute_strategy = ImputeStrategy.LeaveNaN
    splitting_strategy = SplittingStrategy.InOrder
    splitting_ratios = (0.7, 0.2, 0.1)

    validate_data_consistency(
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
