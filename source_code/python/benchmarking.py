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
from torch_methods import TorchBenchmarkingModule
from parameter_iterators import (
    ClassificationParameterIterator,
    ForecastingParameterIterator,
)


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
    config=None,
    num_runs=10,
    dataset_name=None,
):
    """Run benchmark for a specific configuration multiple times and average results."""

    all_python_setup_times = []
    all_python_iteration_times = []
    all_numpy_setup_times = []
    all_numpy_iteration_times = []
    all_rust_setup_times = []
    all_rust_iteration_times = []
    all_torch_setup_times = []
    all_torch_iteration_times = []

    python_batch_count = 0
    numpy_batch_count = 0
    rust_batch_count = 0
    torch_batch_count = 0
    sample_data_shape = None

    benchmark_results = []

    print(f"\nRunning {num_runs} iterations for current parameter configuration...")

    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        run_results = []

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
        python_setup_timer = time.time()
        python_module.setup("stage")
        python_setup_duration = time.time() - python_setup_timer
        python_train_loader = python_module.train_dataloader()
        python_test_loader = python_module.test_dataloader()
        python_data = next(iter(python_train_loader))

        if run == 0:
            sample_data_shape = python_data[0].shape

        python_iteration_timer = time.time()
        run_python_batch_count = 0
        for batch in python_train_loader:
            run_python_batch_count += 1
        python_iteration_duration = time.time() - python_iteration_timer
        if run == 0:
            python_batch_count = run_python_batch_count

        all_python_setup_times.append(python_setup_duration)
        all_python_iteration_times.append(python_iteration_duration)

        if run == num_runs - 1:
            run_results.append(python_module.memory_usage)
            run_results.append(python_module.timings)

        del python_module
        del python_train_loader
        del python_test_loader
        del python_data

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
        numpy_setup_timer = time.time()
        numpy_module.setup("stage")
        numpy_setup_duration = time.time() - numpy_setup_timer
        numpy_train_loader = numpy_module.train_dataloader()
        numpy_test_loader = numpy_module.test_dataloader()
        numpy_data = next(iter(numpy_train_loader))

        numpy_iteration_timer = time.time()
        run_numpy_batch_count = 0
        for batch in numpy_train_loader:
            run_numpy_batch_count += 1
        numpy_iteration_duration = time.time() - numpy_iteration_timer

        if run == 0:
            numpy_batch_count = run_numpy_batch_count

        all_numpy_setup_times.append(numpy_setup_duration)
        all_numpy_iteration_times.append(numpy_iteration_duration)

        if run == num_runs - 1:
            run_results.append(numpy_module.memory_usage)
            run_results.append(numpy_module.timings)

        del numpy_module
        del numpy_train_loader
        del numpy_test_loader
        del numpy_data

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
        rust_setup_timer = time.time()
        rust_module.setup("stage")
        rust_setup_duration = time.time() - rust_setup_timer
        rust_train_loader = rust_module.train_dataloader()
        rust_test_loader = rust_module.test_dataloader()
        rust_data = next(iter(rust_train_loader))

        rust_iteration_timer = time.time()
        run_rust_batch_count = 0
        for batch in rust_train_loader:
            run_rust_batch_count += 1
        rust_iteration_duration = time.time() - rust_iteration_timer

        if run == 0:
            rust_batch_count = run_rust_batch_count

        all_rust_setup_times.append(rust_setup_duration)
        all_rust_iteration_times.append(rust_iteration_duration)

        if run == num_runs - 1:
            run_results.append(rust_module.memory_usage)
            run_results.append(rust_module.timings)

        del rust_module
        del rust_train_loader
        del rust_test_loader
        del rust_data

        torch_setup_duration = None
        torch_iteration_duration = None
        run_torch_batch_count = 0
        torch_memory_usage = None
        torch_timings = None

        try:
            d = original_data.copy()
            l = original_labels.copy() if original_labels is not None else None
            torch_module = TorchBenchmarkingModule(
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
            torch_setup_timer = time.time()
            torch_module.setup("stage")
            torch_setup_duration = time.time() - torch_setup_timer
            torch_train_loader = torch_module.train_dataloader()
            torch_test_loader = torch_module.test_dataloader()
            torch_data = next(iter(torch_train_loader))

            torch_iteration_timer = time.time()
            run_torch_batch_count = 0
            for batch in torch_train_loader:
                run_torch_batch_count += 1
            torch_iteration_duration = time.time() - torch_iteration_timer
            if run == 0:
                torch_batch_count = run_torch_batch_count

            torch_memory_usage = torch_module.memory_usage
            torch_timings = torch_module.timings
            del torch_module
            del torch_train_loader
            del torch_test_loader
            del torch_data
        except Exception as e:
            if run == 0:
                print(f"Torch Module Error: {e}")
                print("Skipping torch benchmarking for this configuration.")
            torch_setup_duration = None
            torch_iteration_duration = None
            run_torch_batch_count = 0
            torch_memory_usage = {"error": str(e)}
            torch_timings = {"error": str(e)}

        if torch_setup_duration is not None:
            all_torch_setup_times.append(torch_setup_duration)
            all_torch_iteration_times.append(torch_iteration_duration)

        if run == num_runs - 1:
            run_results.append(torch_memory_usage)
            run_results.append(torch_timings)

    if "run_results" in locals():
        benchmark_results = run_results
    else:
        benchmark_results = []

    python_setup_duration = sum(all_python_setup_times) / len(all_python_setup_times)
    python_iteration_duration = sum(all_python_iteration_times) / len(
        all_python_iteration_times
    )
    numpy_setup_duration = sum(all_numpy_setup_times) / len(all_numpy_setup_times)
    numpy_iteration_duration = sum(all_numpy_iteration_times) / len(
        all_numpy_iteration_times
    )
    rust_setup_duration = sum(all_rust_setup_times) / len(all_rust_setup_times)
    rust_iteration_duration = sum(all_rust_iteration_times) / len(
        all_rust_iteration_times
    )

    if all_torch_setup_times:
        torch_setup_duration = sum(all_torch_setup_times) / len(all_torch_setup_times)
        torch_iteration_duration = sum(all_torch_iteration_times) / len(
            all_torch_iteration_times
        )
    else:
        torch_setup_duration = None
        torch_iteration_duration = None

    print("\n" + "=" * 50)
    print("SETUP DURATION COMPARISON")
    print("=" * 50)
    print(f"\nAveraged results over {num_runs} runs:")
    print(f"Python Module Setup Time: {python_setup_duration:.4f} seconds (avg)")
    print(f"Python Data Shape: {sample_data_shape}")
    print(
        f"Python Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB"
    )
    print(f"Numpy Module Setup Time: {numpy_setup_duration:.4f} seconds (avg)")
    print(f"Rust Module Setup Time: {rust_setup_duration:.4f} seconds (avg)")
    if torch_setup_duration is not None:
        print(f"Torch Module Setup Time: {torch_setup_duration:.4f} seconds (avg)")
    else:
        print("Torch Module Setup Time: N/A (error occurred)")

    print("\n" + "=" * 50)
    print("SETUP DURATION COMPARISON (AVERAGED)")
    print("=" * 50)
    print(f"Python Module Setup Time:  {python_setup_duration:.4f} seconds")
    print(f"Numpy Module Setup Time:   {numpy_setup_duration:.4f} seconds")
    print(f"Rust Module Setup Time:    {rust_setup_duration:.4f} seconds")
    if torch_setup_duration is not None:
        print(f"Torch Module Setup Time:   {torch_setup_duration:.4f} seconds")
    else:
        print("Torch Module Setup Time:   N/A (error occurred)")
    print("=" * 50)

    print("\n" + "=" * 50)
    print("TRAINING ITERATION DURATION COMPARISON (AVERAGED)")
    print("=" * 50)
    print(
        f"Python Training Iteration Time:  {python_iteration_duration:.4f} seconds ({python_batch_count} batches)"
    )
    print(
        f"Numpy Training Iteration Time:   {numpy_iteration_duration:.4f} seconds ({numpy_batch_count} batches)"
    )
    print(
        f"Rust Training Iteration Time:    {rust_iteration_duration:.4f} seconds ({rust_batch_count} batches)"
    )
    if torch_iteration_duration is not None:
        print(
            f"Torch Training Iteration Time:   {torch_iteration_duration:.4f} seconds ({torch_batch_count} batches)"
        )
    else:
        print("Torch Training Iteration Time:   N/A (error occurred)")
    print("=" * 50)

    print("Benchmarking completed.")

    setup_durations = {
        "python_setup_duration": python_setup_duration,
        "python_setup_times": all_python_setup_times,
        "python_setup_std": np.std(all_python_setup_times),
        "numpy_setup_duration": numpy_setup_duration,
        "numpy_setup_times": all_numpy_setup_times,
        "numpy_setup_std": np.std(all_numpy_setup_times),
        "rust_setup_duration": rust_setup_duration,
        "rust_setup_times": all_rust_setup_times,
        "rust_setup_std": np.std(all_rust_setup_times),
        "torch_setup_duration": torch_setup_duration,
        "torch_setup_times": all_torch_setup_times,
        "torch_setup_std": (
            np.std(all_torch_setup_times) if all_torch_setup_times else None
        ),
        "num_runs": num_runs,
    }

    iteration_durations = {
        "python_iteration_duration": python_iteration_duration,
        "python_iteration_times": all_python_iteration_times,
        "python_iteration_std": np.std(all_python_iteration_times),
        "python_batch_count": python_batch_count,
        "numpy_iteration_duration": numpy_iteration_duration,
        "numpy_iteration_times": all_numpy_iteration_times,
        "numpy_iteration_std": np.std(all_numpy_iteration_times),
        "numpy_batch_count": numpy_batch_count,
        "rust_iteration_duration": rust_iteration_duration,
        "rust_iteration_times": all_rust_iteration_times,
        "rust_iteration_std": np.std(all_rust_iteration_times),
        "rust_batch_count": rust_batch_count,
        "torch_iteration_duration": torch_iteration_duration,
        "torch_iteration_times": all_torch_iteration_times,
        "torch_iteration_std": (
            np.std(all_torch_iteration_times) if all_torch_iteration_times else None
        ),
        "torch_batch_count": torch_batch_count,
        "num_runs": num_runs,
    }

    config_for_json = config.copy() if config else {}
    if "original_data" in config_for_json:
        del config_for_json["original_data"]
    if "original_labels" in config_for_json:
        del config_for_json["original_labels"]

    if "dataset_type" in config_for_json:
        config_for_json["dataset_type"] = config_for_json["dataset_type"].value
    if "impute_strategy" in config_for_json:
        config_for_json["impute_strategy"] = str(config_for_json["impute_strategy"])
    if "splitting_strategy" in config_for_json:
        config_for_json["splitting_strategy"] = str(
            config_for_json["splitting_strategy"]
        )

    benchmark_entry = {
        "dataset_name": dataset_name,
        "config": config_for_json,
        "setup_durations": setup_durations,
        "iteration_durations": iteration_durations,
        "module_memory_usage": benchmark_results[:4],
        "module_timings": benchmark_results[4:8],
    }

    try:
        with open("benchmarking_results.json", "r") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = []

    all_results.append(benchmark_entry)

    with open("benchmarking_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    # initiate a logifle
    import logging

    logging.basicConfig(
        filename="benchmarking.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # first benchmark aeon (category classification) datasets
    for key in dataset_loaders.AEON_METADATA:
        logging.info(f"Benchmarking dataset: {key}")
        print(
            f"Key {list(dataset_loaders.AEON_METADATA.keys()).index(key) + 1} out of {len(dataset_loaders.AEON_METADATA)}: {key}"
        )
        try:
            original_data, original_labels = dataset_loaders.load_aeon_data(key)

        except Exception as e:
            logging.error(f"Error loading dataset {key}: {e}")
            continue

        try:
            i = ClassificationParameterIterator(
                original_data,
                original_labels,
                max_iterations=40,
            )
        except Exception as e:
            logging.error(f"Error creating parameter iterator for dataset {key}: {e}")
            continue
        iteration = 0
        for config in i:
            iteration += 1
            print(f"Iterations {iteration} out of {i.max_iterations}")
            try:
                pruned_config = {
                    k: v
                    for k, v in config.items()
                    if k != "original_data" and k != "original_labels"
                }
                logging.info(
                    f"Running benchmark with config: {pruned_config} on dataset {key}"
                )
                benchmark(
                    config["original_data"],
                    config["dataset_type"],
                    config["past_window"],
                    config["future_horizon"],
                    config["stride"],
                    config["original_labels"],
                    config["batch_size"],
                    config["num_workers"],
                    config["downsampling_rate"],
                    config["normalize"],
                    config["standardize"],
                    config["impute_strategy"],
                    config["splitting_strategy"],
                    config["splitting_ratios"],
                    config=config,
                    num_runs=10,
                    dataset_name=key,
                )
            except Exception as e:
                logging.error(
                    f"Error running benchmark for dataset {key} with config {pruned_config}: {e}"
                )
    # now benchmark ETT_METADATA
    for key in dataset_loaders.ETT_METADATA:
        logging.info(f"Benchmarking dataset: {key}")
        try:
            data = dataset_loaders.load_ETT_data(key)
        except Exception as e:
            logging.error(f"Error loading dataset {key}: {e}")
            continue

        try:
            i = ForecastingParameterIterator(data, max_iterations=40)
        except Exception as e:
            logging.error(f"Error creating parameter iterator for dataset {key}: {e}")
            continue
        iteration = 0
        for config in i:
            iteration += 1
            print(f"Iterations {iteration} out of {i.max_iterations}")
            try:
                pruned_config = {
                    k: v for k, v in config.items() if k != "original_data"
                }
                logging.info(
                    f"Running benchmark with config: {pruned_config} on dataset {key}"
                )
                benchmark(
                    config["original_data"],
                    config["dataset_type"],
                    config["past_window"],
                    config["future_horizon"],
                    config["stride"],
                    config["original_labels"],
                    config["batch_size"],
                    config["num_workers"],
                    config["downsampling_rate"],
                    config["normalize"],
                    config["standardize"],
                    config["impute_strategy"],
                    config["splitting_strategy"],
                    config["splitting_ratios"],
                    config=config,
                    num_runs=10,
                    dataset_name=key,
                )
            except Exception as e:
                logging.error(
                    f"Error running benchmark for dataset {key} with config {pruned_config}: {e}"
                )
    print("Benchmarking completed for all datasets.")
