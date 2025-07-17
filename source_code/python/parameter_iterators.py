"""
Parameter iterators for systematically varying settings across different configurations
for forecasting and classification tasks.
"""

import itertools
from typing import Iterator, Dict, Any, Tuple
import dataset_loaders
import wrapper
from rust_time_series.rust_time_series import ImputeStrategy, SplittingStrategy


class ForecastingParameterIterator:
    """Iterator for forecasting task parameters that systematically varies settings."""

    def __init__(self, data, max_iterations: int = 20):
        self.data = data
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.past_windows = [6, 12]
        self.future_horizons = [6, 12]
        self.strides = [1]
        self.batch_sizes = [16, 32]
        self.downsampling_rates = [1, 2]
        self.normalize_options = [False]
        self.standardize_options = [True, False]
        self.impute_strategies = [
            ImputeStrategy.LeaveNaN,
            # ImputeStrategy.Mean,
            # ImputeStrategy.Median,
            # ImputeStrategy.ForwardFill,
            # ImputeStrategy.BackwardFill,
        ]
        self.splitting_ratios_options = [
            (0.7, 0.2, 0.1),
            (0.6, 0.2, 0.2),
            (0.8, 0.1, 0.1),
        ]

        self.parameter_combinations = list(
            itertools.product(
                self.past_windows,
                self.future_horizons,
                self.strides,
                self.batch_sizes,
                self.downsampling_rates,
                self.normalize_options,
                self.standardize_options,
                self.impute_strategies,
                self.splitting_ratios_options,
            )
        )

        self.parameter_combinations = self.parameter_combinations[: self.max_iterations]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.iteration_count = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.iteration_count >= len(self.parameter_combinations):
            raise StopIteration

        (
            past_window,
            future_horizon,
            stride,
            batch_size,
            downsampling_rate,
            normalize,
            standardize,
            impute_strategy,
            splitting_ratios,
        ) = self.parameter_combinations[self.iteration_count]

        config = {
            "original_data": self.data,
            "original_labels": None,
            "dataset_type": wrapper.DatasetType.Forecasting,
            "past_window": past_window,
            "future_horizon": future_horizon,
            "stride": stride,
            "batch_size": batch_size,
            "num_workers": 0,
            "downsampling_rate": downsampling_rate,
            "normalize": normalize,
            "standardize": standardize,
            "impute_strategy": impute_strategy,
            "splitting_ratios": splitting_ratios,
            "splitting_strategy": SplittingStrategy.InOrder,  # not used but required
            "iteration": self.iteration_count + 1,
        }

        self.iteration_count += 1
        return config

    def __len__(self) -> int:
        return len(self.parameter_combinations)


class ClassificationParameterIterator:
    """Iterator for classification task parameters that systematically varies settings."""

    def __init__(self, data, labels, max_iterations: int = 20):
        self.data = data
        self.labels = labels
        self.max_iterations = max_iterations
        self.iteration_count = 0

        self.past_windows = [1]
        self.future_horizons = [
            1,
        ]
        self.strides = [1]
        self.batch_sizes = [16, 32]
        self.downsampling_rates = [1, 2]
        self.normalize_options = [False]
        self.standardize_options = [True, False]
        self.impute_strategies = [
            ImputeStrategy.LeaveNaN,
            # ImputeStrategy.Mean,
            # ImputeStrategy.Median,
            # ImputeStrategy.ForwardFill,
            # ImputeStrategy.BackwardFill,
        ]
        self.splitting_strategies = [
            SplittingStrategy.InOrder,
            SplittingStrategy.Random,
        ]
        self.splitting_ratios_options = [
            (0.7, 0.2, 0.1),
            (0.6, 0.2, 0.2),
            (0.8, 0.1, 0.1),
        ]

        self.parameter_combinations = list(
            itertools.product(
                self.past_windows,
                self.future_horizons,
                self.strides,
                self.batch_sizes,
                self.downsampling_rates,
                self.normalize_options,
                self.standardize_options,
                self.impute_strategies,
                self.splitting_strategies,
                self.splitting_ratios_options,
            )
        )
        self.parameter_combinations = self.parameter_combinations[: self.max_iterations]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.iteration_count = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.iteration_count >= len(self.parameter_combinations):
            raise StopIteration

        (
            past_window,
            future_horizon,
            stride,
            batch_size,
            downsampling_rate,
            normalize,
            standardize,
            impute_strategy,
            splitting_strategy,
            splitting_ratios,
        ) = self.parameter_combinations[self.iteration_count]

        config = {
            "original_data": self.data,
            "original_labels": self.labels,
            "dataset_type": wrapper.DatasetType.Classification,
            "past_window": past_window,
            "future_horizon": future_horizon,
            "stride": stride,
            "batch_size": batch_size,
            "num_workers": 0,
            "downsampling_rate": downsampling_rate,
            "normalize": normalize,
            "standardize": standardize,
            "impute_strategy": impute_strategy,
            "splitting_strategy": splitting_strategy,
            "splitting_ratios": splitting_ratios,
            "iteration": self.iteration_count + 1,
        }

        self.iteration_count += 1
        return config

    def __len__(self) -> int:
        return len(self.parameter_combinations)


def get_forecasting_iterator(max_iterations: int = 20) -> ForecastingParameterIterator:
    """Convenience function to get a forecasting parameter iterator."""
    data = dataset_loaders.load_electricity_data()
    return ForecastingParameterIterator(data, max_iterations)


def get_classification_iterator(
    max_iterations: int = 20,
) -> ClassificationParameterIterator:
    """Convenience function to get a classification parameter iterator."""
    data, labels = dataset_loaders.load_aeon_data("ArticularyWordRecognition")
    return ClassificationParameterIterator(data, labels, max_iterations)


if __name__ == "__main__":
    print("=== Forecasting Parameter Iterator Demo ===")
    forecasting_iter = get_forecasting_iterator(5)

    for i, config in enumerate(forecasting_iter):
        print(f"\nIteration {config['iteration']}:")
        print(f"  Past window: {config['past_window']}")
        print(f"  Future horizon: {config['future_horizon']}")
        print(f"  Stride: {config['stride']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Downsampling rate: {config['downsampling_rate']}")
        print(f"  Normalize: {config['normalize']}")
        print(f"  Standardize: {config['standardize']}")
        print(f"  Impute strategy: {config['impute_strategy']}")
        print(f"  Splitting ratios: {config['splitting_ratios']}")
        print(f"  Data shape: {config['original_data'].shape}")
        print(f"  Labels: {config['original_labels']}")

    print("\n" + "=" * 50)
    print("=== Classification Parameter Iterator Demo ===")
    classification_iter = get_classification_iterator(5)
    for i, config in enumerate(classification_iter):
        print(f"\nIteration {config['iteration']}:")
        print(f"  Past window: {config['past_window']}")
        print(f"  Future horizon: {config['future_horizon']}")
        print(f"  Stride: {config['stride']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Downsampling rate: {config['downsampling_rate']}")
        print(f"  Normalize: {config['normalize']}")
        print(f"  Standardize: {config['standardize']}")
        print(f"  Impute strategy: {config['impute_strategy']}")
        print(f"  Splitting strategy: {config['splitting_strategy']}")
        print(f"  Splitting ratios: {config['splitting_ratios']}")
        print(f"  Data shape: {config['original_data'].shape}")
        print(
            f"  Labels shape: {config['original_labels'].shape if config['original_labels'] is not None else 'None'}"
        )
