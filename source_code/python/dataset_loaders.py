import pandas as pd
import numpy as np
import os

# format assumed: (n_instances, n_timepoints, n_channels)

BASE_PATH = "/Users/kilian/github/ai_with_rust_final_project/data"


def load_electricity_data() -> np.ndarray:
    # Load the dataset (forecasting)
    file_path = os.path.join(BASE_PATH, "LD/LD2011_2014.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File {file_path} does not exist. Please check the path."
        )
    df = pd.read_csv(file_path, sep=";", decimal=",")
    # drop first column (date)
    df = df.drop(columns=["date"])
    df = df.transpose()
    # turn pandas DataFrame into numPy array
    data = df.to_numpy(dtype=np.float64).reshape(df.shape[0], df.shape[1], 1)

    assert data.ndim == 3
    return data


AEON_METADATA = {}
from aeon.datasets.tsc_datasets import multivariate, univariate

# get metadata for univariate and multivariate datasets
for dataset in univariate:
    AEON_METADATA[dataset] = {
        "Train": None,  # Placeholder, will be filled later
        "Test": None,  # Placeholder, will be filled later
        "Channel": 1,  # Default for univariate datasets
        "Class": None,  # Placeholder for classification datasets
    }
for dataset in multivariate:
    AEON_METADATA[dataset] = {
        "Train": None,  # Placeholder, will be filled later
        "Test": None,  # Placeholder, will be filled later
        "Channel": None,  # Placeholder for multivariate datasets
        "Class": None,  # Placeholder for classification datasets
    }


def load_aeon_data(dataset_title) -> tuple[np.ndarray, np.ndarray]:
    # Load the GunPoint dataset from aeon (classification)
    # Returns X (features) and y (labels) as numpy arrays
    if dataset_title not in AEON_METADATA:
        raise ValueError(f"Dataset {dataset_title} is not available in AEON metadata.")
    from aeon.datasets import load_classification

    x, y, metadata = load_classification(dataset_title, return_metadata=True)

    if len(x.shape) == 3:
        # aeon uses a different layout
        x = np.transpose(x, (0, 2, 1))

    # label encode y (always 1d categorical labels)
    if y.ndim > 1:
        raise ValueError(
            f"Expected 1D labels, but got {y.ndim}D labels for dataset {dataset_title}."
        )

    y = pd.Categorical(y).codes
    # turn to npt.NDArray[np.float64]
    y = y.astype(np.float64)

    return x, y


ETT_METADATA = {
    "ETTh1": {"Train": 17420, "Channel": 7},
    "ETTh2": {"Train": 17420, "Channel": 7},
    "ETTm1": {"Train": 69680, "Channel": 7},
    "ETTm2": {"Train": 69680, "Channel": 7},
}


def load_ETT_data(dataset_title) -> np.ndarray:
    # Load the ETT dataset (forecasting)
    file_path = os.path.join(BASE_PATH, "ETT-small/", dataset_title + ".csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File {file_path} does not exist. Please check the path."
        )
    df = pd.read_csv(file_path, sep=",", decimal=".")
    df = df.drop(columns=["date"])
    data = df.to_numpy(dtype=np.float64)
    timesteps = data.shape[0]
    channels = data.shape[1]
    data = data.reshape(1, timesteps, channels)

    return data


SERIES2VEC_METADATA = {
    "WISDM2": {
        "Timesteps": 40,
        "Channel": 3,
    },
    "PAMAP2": {
        "Timesteps": 100,
        "Channel": 52,
    },
    "USC_HAD": {
        "Timesteps": 100,
        "Channel": 6,
    },
    "Sleep": {
        "Timesteps": 3000,
        "Channel": 1,
    },
    "Skoda": {
        "Timesteps": 50,
        "Channel": 60,
    },
    "Opportunity": {
        "Timesteps": 100,
        "Channel": 113,
    },
    "WISDM": {
        "Timesteps": 40,
        "Channel": 3,
    },
    "Epilepsy": {
        "Timesteps": 178,
        "Channel": 1,
    },
    "HAR": {
        "Timesteps": 128,
        "Channel": 9,
    },
}


def load_series2vec_data(
    dataset_title,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Series2Vec datasets (classification)\n
    Returns train_data, train_label, test_data, test_label as numpy arrays\n
    source: https://drive.google.com/drive/folders/1YLdbzwslNkmi3No19C3aGdmfAUSoruzB (https://github.com/Navidfoumani/Series2Vec?tab=readme-ov-file)
    """

    file_path = os.path.join(BASE_PATH, "series2vec/", dataset_title + ".npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File {file_path} does not exist. Please check the path."
        )
    # load the dataset
    data = np.load(file_path, allow_pickle=True)
    train_data = data.item().get("train_data")
    train_label = data.item().get("train_label")
    test_data = data.item().get("test_data")
    test_label = data.item().get("test_label")

    # print shape
    print(f"Train data shape: {train_data.shape}")
    print(f"Train label shape: {train_label.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test label shape: {test_label.shape}")

    merged_data = np.concatenate((train_data, test_data), axis=0)
    merged_labels = np.concatenate((train_label, test_label), axis=0)

    expected_channels = SERIES2VEC_METADATA[dataset_title]["Channel"]
    channel_axis = [
        x for x in range(merged_data.ndim) if merged_data.shape[x] == expected_channels
    ]
    if len(channel_axis) != 1:
        raise ValueError(
            f"Invalid channel axis found in merged data for dataset {dataset_title}."
        )
    channel_axis = channel_axis[0]
    expected_timesteps = SERIES2VEC_METADATA[dataset_title]["Timesteps"]
    timesteps_axis = [
        x for x in range(merged_data.ndim) if merged_data.shape[x] == expected_timesteps
    ]
    if len(timesteps_axis) != 1:
        raise ValueError(
            f"Invalid timesteps axis found in merged data for dataset {dataset_title}."
        )
    timesteps_axis = timesteps_axis[0]
    instances_axis = [
        x for x in range(merged_data.ndim) if x != channel_axis and x != timesteps_axis
    ]
    if len(instances_axis) != 1:
        raise ValueError(
            f"Invalid instances axis found in merged data for dataset {dataset_title}."
        )
    instances_axis = instances_axis[0]

    # reshape the data to (n_instances, n_timesteps, n_channels)
    merged_data = np.transpose(
        merged_data, (instances_axis, timesteps_axis, channel_axis)
    )
    assert (
        merged_labels.ndim == 1
    ), f"Expected 1D labels, but got {merged_labels.ndim}D labels for dataset {dataset_title}."
    merged_labels = merged_labels.astype(np.float64)
    if merged_data.ndim != 3:
        raise ValueError(
            f"Expected 3D data, but got {merged_data.ndim}D data for dataset {dataset_title}."
        )
    merged_data = merged_data.astype(np.float64)
    return merged_data, merged_labels
