import numpy as np
from rust_time_series.rust_time_series import (
    ImputeStrategy,
    SplittingStrategy,
)
import time
import wrapper
import pytorch_lightning as L
import torch
import pandas as pd
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import MultiNormalizer, TorchNormalizer
import torch.utils.data
from memory_monitor import ProcessStepMemoryTracker
from memory_monitor import ProcessStepMemoryTracker


class ReshapedDataLoader:
    def __init__(self, original_dataloader, num_features):
        self.original_dataloader = original_dataloader
        self.num_features = num_features

    def __iter__(self):
        for batch_x, batch_y in self.original_dataloader:
            reshaped_y = torch.stack(batch_y[0], dim=2)
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
            encoder_data = batch_x[
                "encoder_cont"
            ]  # Shape: [batch, encoder_length, features]
            decoder_data = batch_x[
                "decoder_cont"
            ]  # Shape: [batch, decoder_length, features]

            full_sequence = torch.cat([encoder_data, decoder_data], dim=1)

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
        ),
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
        self.memory_tracker = ProcessStepMemoryTracker("torch")
        self.memory_usage = self.memory_tracker.get_memory_usage()

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

    def setup(self, stage: str):
        self.working_datasets["torch"] = self.dataset.copy()
        self.working_labels["torch"] = (
            self.labels.copy() if self.labels is not None else None
        )

        if self.impute_strategy != ImputeStrategy.LeaveNaN:
            with self.memory_tracker.track_step("imputation"):
                timer = time.perf_counter()
                try:
                    raise ValueError(
                        "Imputation strategies other than LeaveNaN are not supported in torch natively."
                    )
                except ValueError:
                    delta = time.perf_counter() - timer
                    self.timings["torch"]["imputation"] = delta
                    raise

        if self.dataset_type == wrapper.DatasetType.Classification:  # Classification
            if self.working_labels["torch"] is None:
                raise ValueError("Labels are required for classification tasks")
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
        else:
            raise ValueError(
                f"Unsupported dataset type: {self.dataset_type}. Only Classification is supported in Torch."
            )

        self.memory_usage = self.memory_tracker.get_memory_usage()

    def train_dataloader(self):
        return self.train_dataloader_torch

    def val_dataloader(self):
        return self.val_dataloader_torch

    def test_dataloader(self):
        return self.test_dataloader_torch

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
        ),
    ) -> tuple[
        ClassificationDataLoader,
        ClassificationDataLoader,
        ClassificationDataLoader,
    ]:
        if downsampling_rate > 1:
            with self.memory_tracker.track_step("downsampling"):
                timer = time.perf_counter()
                data = data[:, ::downsampling_rate, :]
                delta = time.perf_counter() - timer
                self.timings["torch"]["downsampling"] = delta

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

        with self.memory_tracker.track_step("data_transformation"):
            timer = time.perf_counter()
            flat = data.reshape(-1, features)
            feature_cols = [f"feature_{i+1}" for i in range(features)]
            group = np.repeat(np.arange(sources), data.shape[1])
            time_idx = np.tile(np.arange(data.shape[1]), sources)

            labels_repeated = np.repeat(labels, data.shape[1])

            df = pd.DataFrame(flat, columns=feature_cols)
            df["group"] = group
            df["time_idx"] = time_idx
            df["label"] = labels_repeated
            delta = time.perf_counter() - timer
            self.timings["torch"]["data_transformation"] = delta

        with self.memory_tracker.track_step("splitting"):
            timer = time.perf_counter()
            assert np.isclose(
                sum(splitting_ratios), 1.0
            ), "Splitting ratios must sum to 1.0"

            max_group_id = int(df["group"].max())
            train_cutoff = int(max_group_id * splitting_ratios[0])
            val_cutoff = train_cutoff + int(max_group_id * splitting_ratios[1])

            max_time = int(df["time_idx"].max())

            training = TimeSeriesDataSet(
                df[df.group <= train_cutoff],
                group_ids=["group"],
                target="label",
                time_idx="time_idx",
                min_encoder_length=max_time,
                max_encoder_length=max_time,
                min_prediction_length=1,
                max_prediction_length=1,
                time_varying_unknown_reals=feature_cols,
                target_normalizer=TorchNormalizer(method=normalizer_method),
                allow_missing_timesteps=True,
            )

            validation = TimeSeriesDataSet.from_dataset(
                training,
                df[(df.group > train_cutoff) & (df.group <= val_cutoff)],
                stop_randomization=True,
            )

            test = TimeSeriesDataSet.from_dataset(
                training, df[df.group > val_cutoff], stop_randomization=True
            )

            delta = time.perf_counter() - timer
            self.timings["torch"]["splitting"] = delta

        with self.memory_tracker.track_step("data_collection"):
            timer = time.perf_counter()
            train_dataloader = training.to_dataloader(
                train=True, batch_size=batch_size, num_workers=num_workers
            )
            val_dataloader = validation.to_dataloader(
                train=False, batch_size=batch_size, num_workers=num_workers
            )
            test_dataloader = test.to_dataloader(
                train=False, batch_size=batch_size, num_workers=num_workers
            )

            wrapper_dataloaders = [
                ClassificationDataLoader(train_dataloader, features),
                ClassificationDataLoader(val_dataloader, features),
                ClassificationDataLoader(test_dataloader, features),
            ]

            delta = time.perf_counter() - timer
            self.timings["torch"]["data_collection"] = delta

        return wrapper_dataloaders[0], wrapper_dataloaders[1], wrapper_dataloaders[2]
