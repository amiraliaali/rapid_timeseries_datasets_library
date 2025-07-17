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


# Create custom wrapper class to reshape targets
class ReshapedDataLoader:
    def __init__(self, original_dataloader, num_features):
        self.original_dataloader = original_dataloader
        self.num_features = num_features

    def __iter__(self):
        for batch_x, batch_y in self.original_dataloader:
            # Reshape targets from list of [batch_size, time] to [batch_size, time, features]
            reshaped_y = torch.stack(batch_y[0], dim=2)  # Stack along feature dimension
            yield batch_x["encoder_cont"], reshaped_y

    def __len__(self):
        return len(self.original_dataloader)

    @property
    def dataset(self):
        return self.original_dataloader.dataset


class ClassificationDataLoader:
    def __init__(self, original_dataloader, num_features):
        self.original_dataloader = original_dataloader
        self.num_features = num_features

    def __iter__(self):
        for batch_x, batch_y in self.original_dataloader:
            # For classification, concatenate encoder and decoder to get full sequence
            encoder_data = batch_x[
                "encoder_cont"
            ]  # Shape: [batch, encoder_length, features]
            decoder_data = batch_x[
                "decoder_cont"
            ]  # Shape: [batch, decoder_length, features]

            # Concatenate along time dimension to get full sequence
            full_sequence = torch.cat([encoder_data, decoder_data], dim=1)

            # Get the label (should be same for all timesteps in classification)
            reshaped_y = batch_y[0].squeeze(1)
            yield full_sequence, reshaped_y

    def __len__(self):
        return len(self.original_dataloader)

    @property
    def dataset(self):
        return self.original_dataloader.dataset


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

        # Make copies of the dataset for benchmarking
        self.working_datasets["torch"] = self.dataset.copy()
        self.working_labels["torch"] = (
            self.labels.copy() if self.labels is not None else None
        )

        # Apply preprocessing and create dataloaders based on dataset type
        memory_before = self._get_memory_mb()
        timer = time.time()

        if self.dataset_type == wrapper.DatasetType.Forecasting:
            (
                self.train_dataloader_torch,
                self.val_dataloader_torch,
                self.test_dataloader_torch,
            ) = self.forecasting_dataset(
                self.working_datasets["torch"],
                self.past_window,
                self.future_horizon,
                self.stride,
                self.batch_size,
                self.num_workers,
                self.downsampling_rate,
                self.normalize,
                self.standardize,
                self.impute_strategy,
                self.splitting_ratios,
            )
        else:  # Classification
            (
                self.train_dataloader_torch,
                self.val_dataloader_torch,
                self.test_dataloader_torch,
            ) = self.classification_dataset(
                self.working_datasets["torch"],
                self.working_labels["torch"],
                self.batch_size,
                self.num_workers,
                self.downsampling_rate,
                self.normalize,
                self.standardize,
                self.impute_strategy,
                self.splitting_strategy,
                self.splitting_ratios,
            )

        delta = time.time() - timer
        self.timings["torch"]["data_processing"] = delta
        memory_after = self._get_memory_mb()
        self.memory_usage["torch"]["data_processing"] = memory_after - memory_before

    def train_dataloader(self):
        return self.train_dataloader_torch

    def val_dataloader(self):
        return self.val_dataloader_torch

    def test_dataloader(self):
        return self.test_dataloader_torch

    def forecasting_dataset(
        self,
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
    ) -> tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
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

        # Wrap the dataloaders to reshape targets
        wrapped_dataloaders = []
        for dataloader in dataloaders:
            wrapped_dataloaders.append(ReshapedDataLoader(dataloader, features))

        return wrapped_dataloaders[0], wrapped_dataloaders[1], wrapped_dataloaders[2]

    def classification_dataset(
        self,
        data: np.ndarray,
        labels: np.ndarray,
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
    ) -> tuple[
        ClassificationDataLoader,
        ClassificationDataLoader,
        ClassificationDataLoader,
    ]:
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
        train_end = int(sources * splitting_ratios[0])
        val_end = train_end + int(sources * splitting_ratios[1])
        train, val, test = np.split(data, [train_end, val_end], axis=0)
        train_labels, val_labels, test_labels = np.split(
            labels, [train_end, val_end], axis=0
        )
        datasets = []
        for d, split_labels in zip(
            (train, val, test), (train_labels, val_labels, test_labels)
        ):
            # Reshape: stack all sources * timesteps into one axis
            flat = d.reshape(-1, features)
            # Build the column names
            feature_cols = [f"feature_{i+1}" for i in range(features)]
            # Repeat group and tile time indices
            split_sources = d.shape[0]  # Use the actual number of sources in this split
            group = np.repeat(np.arange(split_sources), d.shape[1])

            labels_repeated = np.repeat(split_labels, d.shape[1])

            time_idx = np.tile(np.arange(d.shape[1]), split_sources)
            # Combine into DataFrame
            df = pd.DataFrame(flat, columns=feature_cols)
            df["group"] = group
            df["time_idx"] = time_idx
            df["label"] = labels_repeated

            from pytorch_forecasting import TimeSeriesDataSet
            from pytorch_forecasting.data import MultiNormalizer, TorchNormalizer

            # create the dataset from the pandas dataframe
            # For classification, we want to use the entire time series as input
            max_time = int(df["time_idx"].max())
            # For classification, use full sequence as encoder, minimal prediction window
            dataset = TimeSeriesDataSet(
                df,
                group_ids=["group"],
                target="label",
                time_idx="time_idx",
                min_encoder_length=max_time,  # Use almost entire sequence as encoder
                max_encoder_length=max_time,  # Use almost entire sequence as encoder
                min_prediction_length=1,  # Predict single label using last timestep
                max_prediction_length=1,  # Predict single label using last timestep
                time_varying_unknown_reals=feature_cols,
                target_normalizer=TorchNormalizer(method=normalizer_method),
                allow_missing_timesteps=True,
            )
            datasets.append(dataset)
        dataloaders = [
            dataset.to_dataloader(
                batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            for dataset in datasets
        ]
        wrapper_dataloaders = [
            ClassificationDataLoader(dataloader, features) for dataloader in dataloaders
        ]
        return wrapper_dataloaders[0], wrapper_dataloaders[1], wrapper_dataloaders[2]
