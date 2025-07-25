import itertools
from typing import Iterator, Dict, Any
import wrapper
from rust_time_series.rust_time_series import ImputeStrategy, SplittingStrategy


class ClassificationParameterIterator:
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
        self.batch_sizes = [32]
        self.downsampling_rates = [1, 2]
        self.normalize_options = [False, True]
        self.standardize_options = [True, False]
        self.impute_strategies = [
            ImputeStrategy.LeaveNaN,
            ImputeStrategy.Mean,
            ImputeStrategy.Median,
            ImputeStrategy.ForwardFill,
            ImputeStrategy.BackwardFill,
        ]
        self.splitting_strategies = [
            SplittingStrategy.InOrder,
            SplittingStrategy.Random,
        ]
        self.splitting_ratios_options = [
            (0.7, 0.2, 0.1),
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
