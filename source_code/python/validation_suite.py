"""
Example script demonstrating how to use the parameter iterators with the validation function.
This script runs validation tests across different parameter configurations.
"""

from parameter_iterators import get_forecasting_iterator, get_classification_iterator
from validate import validate_data_consistency
import time
import json
import os
from datetime import datetime
from rust_time_series.rust_time_series import SplittingStrategy


def run_forecasting_validation_suite(
    max_iterations: int = 20, save_results: bool = True
):
    """Run validation suite for forecasting tasks with varying parameters."""

    print(f"Starting Forecasting Validation Suite ({max_iterations} iterations)")
    print("=" * 60)

    forecasting_iter = get_forecasting_iterator(max_iterations)
    results = []

    for config in forecasting_iter:
        print(f"\n--- Forecasting Iteration {config['iteration']}/{max_iterations} ---")
        print(
            f"Config: past_window={config['past_window']}, future_horizon={config['future_horizon']}"
        )
        print(f"        stride={config['stride']}, batch_size={config['batch_size']}")
        print(f"        downsampling_rate={config['downsampling_rate']}")
        print(
            f"        normalize={config['normalize']}, standardize={config['standardize']}"
        )
        print(f"        impute_strategy={config['impute_strategy']}")
        print(f"        splitting_ratios={config['splitting_ratios']}")

        start_time = time.time()

        try:
            # Run validation with current configuration
            is_consistent = validate_data_consistency(
                original_data=config["original_data"],
                dataset_type=config["dataset_type"],
                past_window=config["past_window"],
                future_horizon=config["future_horizon"],
                stride=config["stride"],
                original_labels=config["original_labels"],
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                downsampling_rate=config["downsampling_rate"],
                normalize=config["normalize"],
                standardize=config["standardize"],
                impute_strategy=config["impute_strategy"],
                splitting_strategy=SplittingStrategy.InOrder,  # deefault for forecasting
                splitting_ratios=config["splitting_ratios"],
            )

            execution_time = time.time() - start_time

            result = {
                "iteration": config["iteration"],
                "task_type": "forecasting",
                "parameters": {
                    "past_window": config["past_window"],
                    "future_horizon": config["future_horizon"],
                    "stride": config["stride"],
                    "batch_size": config["batch_size"],
                    "downsampling_rate": config["downsampling_rate"],
                    "normalize": config["normalize"],
                    "standardize": config["standardize"],
                    "impute_strategy": str(config["impute_strategy"]),
                    "splitting_ratios": config["splitting_ratios"],
                },
                "is_consistent": is_consistent,
                "execution_time": execution_time,
                "data_shape": config["original_data"].shape,
                "status": "success",
            }

            print(
                f"Result: {'PASS' if is_consistent else 'FAIL'} (took {execution_time:.2f}s)"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"ERROR: {str(e)} (took {execution_time:.2f}s)")

            result = {
                "iteration": config["iteration"],
                "task_type": "forecasting",
                "parameters": {
                    "past_window": config["past_window"],
                    "future_horizon": config["future_horizon"],
                    "stride": config["stride"],
                    "batch_size": config["batch_size"],
                    "downsampling_rate": config["downsampling_rate"],
                    "normalize": config["normalize"],
                    "standardize": config["standardize"],
                    "impute_strategy": str(config["impute_strategy"]),
                    "splitting_ratios": config["splitting_ratios"],
                },
                "is_consistent": False,
                "execution_time": execution_time,
                "data_shape": config["original_data"].shape,
                "status": "error",
                "error_message": str(e),
            }

        results.append(result)

    if save_results:
        save_validation_results(results, "forecasting")

    return results


def run_classification_validation_suite(
    max_iterations: int = 20, save_results: bool = True
):
    """Run validation suite for classification tasks with varying parameters."""

    print(f"\nStarting Classification Validation Suite ({max_iterations} iterations)")
    print("=" * 60)

    classification_iter = get_classification_iterator(max_iterations)
    results = []

    for config in classification_iter:
        print(
            f"\n--- Classification Iteration {config['iteration']}/{max_iterations} ---"
        )
        print(
            f"Config: past_window={config['past_window']}, future_horizon={config['future_horizon']}"
        )
        print(f"        stride={config['stride']}, batch_size={config['batch_size']}")
        print(f"        downsampling_rate={config['downsampling_rate']}")
        print(
            f"        normalize={config['normalize']}, standardize={config['standardize']}"
        )
        print(f"        impute_strategy={config['impute_strategy']}")
        print(f"        splitting_strategy={config['splitting_strategy']}")
        print(f"        splitting_ratios={config['splitting_ratios']}")

        start_time = time.time()

        try:
            is_consistent = validate_data_consistency(
                original_data=config["original_data"],
                dataset_type=config["dataset_type"],
                past_window=config["past_window"],
                future_horizon=config["future_horizon"],
                stride=config["stride"],
                original_labels=config["original_labels"],
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                downsampling_rate=config["downsampling_rate"],
                normalize=config["normalize"],
                standardize=config["standardize"],
                impute_strategy=config["impute_strategy"],
                splitting_strategy=config["splitting_strategy"],
                splitting_ratios=config["splitting_ratios"],
            )

            execution_time = time.time() - start_time

            result = {
                "iteration": config["iteration"],
                "task_type": "classification",
                "parameters": {
                    "past_window": config["past_window"],
                    "future_horizon": config["future_horizon"],
                    "stride": config["stride"],
                    "batch_size": config["batch_size"],
                    "downsampling_rate": config["downsampling_rate"],
                    "normalize": config["normalize"],
                    "standardize": config["standardize"],
                    "impute_strategy": str(config["impute_strategy"]),
                    "splitting_strategy": str(config["splitting_strategy"]),
                    "splitting_ratios": config["splitting_ratios"],
                },
                "is_consistent": is_consistent,
                "execution_time": execution_time,
                "data_shape": config["original_data"].shape,
                "labels_shape": (
                    config["original_labels"].shape
                    if config["original_labels"] is not None
                    else None
                ),
                "status": "success",
            }

            print(
                f"Result: {'PASS' if is_consistent else 'FAIL'} (took {execution_time:.2f}s)"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"ERROR: {str(e)} (took {execution_time:.2f}s)")

            result = {
                "iteration": config["iteration"],
                "task_type": "classification",
                "parameters": {
                    "past_window": config["past_window"],
                    "future_horizon": config["future_horizon"],
                    "stride": config["stride"],
                    "batch_size": config["batch_size"],
                    "downsampling_rate": config["downsampling_rate"],
                    "normalize": config["normalize"],
                    "standardize": config["standardize"],
                    "impute_strategy": str(config["impute_strategy"]),
                    "splitting_strategy": str(config["splitting_strategy"]),
                    "splitting_ratios": config["splitting_ratios"],
                },
                "is_consistent": False,
                "execution_time": execution_time,
                "data_shape": config["original_data"].shape,
                "labels_shape": (
                    config["original_labels"].shape
                    if config["original_labels"] is not None
                    else None
                ),
                "status": "error",
                "error_message": str(e),
            }

        results.append(result)

    if save_results:
        save_validation_results(results, "classification")

    return results


def save_validation_results(results, task_type):
    """Save validation results to a JSON file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"validation_results_{task_type}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {filename}")


def print_summary(results):
    """Print a summary of validation results."""
    total = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    consistent = sum(1 for r in results if r["is_consistent"])
    errors = sum(1 for r in results if r["status"] == "error")

    avg_time = sum(r["execution_time"] for r in results) / total if total > 0 else 0

    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Total iterations: {total}")
    print(f"Successful runs: {successful}")
    print(f"Consistent results: {consistent}")
    print(f"Errors: {errors}")
    print(f"Average execution time: {avg_time:.2f}s")
    print(f"Success rate: {successful/total*100:.1f}%")
    print(
        f"Consistency rate: {consistent/successful*100:.1f}%"
        if successful > 0
        else "Consistency rate: N/A"
    )


if __name__ == "__main__":
    print("Starting comprehensive validation testing...")

    forecasting_results = run_forecasting_validation_suite(max_iterations=5)
    print_summary(forecasting_results)

    classification_results = run_classification_validation_suite(max_iterations=5)
    print_summary(classification_results)

    all_results = forecasting_results + classification_results
    print(f"\n=== OVERALL SUMMARY ===")
    print_summary(all_results)
