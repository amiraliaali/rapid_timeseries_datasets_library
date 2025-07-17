import numpy as np
from rust_time_series.rust_time_series import (
    ImputeStrategy,
    SplittingStrategy,
)
import time
import wrapper
import psutil
import os
import pytorch_lightning as L
import torch
import dataset_loaders
from torch.utils.data import TensorDataset, DataLoader
from pytorch_forecasting import TimeSeriesDataSet
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from typing import Dict, Any
import math
import numpy as np
import pandas as pd
from dataset_loaders import load_electricity_data, load_ETT_data
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import MultiNormalizer, TorchNormalizer
from rust_time_series.rust_time_series import (
    ImputeStrategy,
    SplittingStrategy,
)
import torch.utils.data


class TorchBenchmarkingModule(wrapper.RustDataModule):
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

        self.timings = {"torch": {}}
        self.memory_usage = {"torch": {}}
        self.process = psutil.Process(os.getpid())

        self.working_datasets: dict[str, np.ndarray | None] = {"torch": None}
        self.working_labels: dict[str, np.ndarray | None] = {"torch": None}
        self.split_datasets: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ] = {
            "torch": None,
        }
        self.split_labels: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ] = {
            "torch": None,
        }

    def _get_memory_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024

    def setup(self, stage: str):
        """
        Setup the TorchBenchmarkingModule using PyTorch Forecasting's TimeSeriesDataSet.
        This method creates train, validation, and test datasets for both forecasting and classification tasks.
        """
        start_memory = self._get_memory_mb()
        start_time = time.time()

    def forecasting_dataset(
        data: np.ndarray,
        past_window: int = 1,
        future_horizon: int = 1,
        stride: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
        downsampling_rate: int = 0,
        normalize: bool = False,
        standardize: bool = False,
        impute_strategy: ImputeStrategy = ImputeStrategy.LeaveNaN,
        splitting_ratios: tuple[float, float, float] = (
            0.7,
            0.2,
            0.1,
        ),  # Train, validation, test ratios
    ) -> list[torch.utils.data.DataLoader]:
        """
        Convert a 3D NumPy array (sources, timesteps, features) into a pandas DataFrame
        with one column per feature, plus group and time_idx.

        Parameters
        ----------
        arr : np.ndarray
            Input array of shape (sources, timesteps, features).
        feature_prefix : str
            Prefix for generated feature column names. Default: "feature".

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each feature, group, and time_idx.
        """
        if stride != 1:
            raise ValueError("Stride is not supported in this function due to torch.")
        if downsampling_rate > 1:
            data = data[:, ::downsampling_rate, :]

        # Ensure the input is a 3D array
        if data.ndim != 3:
            raise ValueError(
                "Input data must be a 3D NumPy array (sources, timesteps, features)"
            )

        if impute_strategy != ImputeStrategy.LeaveNaN:
            raise ValueError(
                "Imputation strategies other than LeaveNaN are not supported in torch natively."
            )

        sources, timesteps, features = data.shape

        if not standardize and not normalize:
            normalizer_method = "identity"
        elif standardize:
            normalizer_method = "standard"
        elif normalize:
            raise ValueError(
                "Normalization is not supported in this function due to torch. Use standardization instead."
            )

        else:
            raise ValueError(
                "Either standardize or normalize must be True or both must be False"
            )

        # splitting_ratios should be a tuple of three floats summing to 1.0, e.g. (0.6, 0.2, 0.2)
        assert np.isclose(
            sum(splitting_ratios), 1.0
        ), "Splitting ratios must sum to 1.0"
        train_end = int(timesteps * splitting_ratios[0])
        val_end = train_end + int(timesteps * splitting_ratios[1])
        train, val, test = np.split(data, [train_end, val_end], axis=1)

        datasets = []
        for d in (train, val, test):
            # Reshape: stack all sources * timesteps into one axis
            flat = d.reshape(-1, features)
            # Build the column names
            feature_cols = [f"feature_{i+1}" for i in range(features)]
            # Repeat group and tile time indices
            group = np.repeat(np.arange(sources), d.shape[1])
            time_idx = np.tile(np.arange(d.shape[1]), sources)
            # Combine into DataFrame
            df = pd.DataFrame(flat, columns=feature_cols)
            df["group"] = group
            df["time_idx"] = time_idx

            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import MultiNormalizer, TorchNormalizer

            # create the dataset from the pandas dataframe
            dataset = TimeSeriesDataSet(
                df,
                group_ids=["group"],
                target=feature_cols,
                time_idx="time_idx",
                min_encoder_length=past_window,
                max_encoder_length=past_window,
                min_prediction_length=future_horizon,
                max_prediction_length=future_horizon,
                time_varying_unknown_reals=feature_cols,
                target_normalizer=MultiNormalizer(
                    [TorchNormalizer(method=normalizer_method) for _ in feature_cols]
                ),
            )
            datasets.append(dataset)
        dataloaders = [
            dataset.to_dataloader(
                batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            for dataset in datasets
        ]
        return dataloaders
