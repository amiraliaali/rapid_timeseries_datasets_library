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

    TIME_DIM = 1
    INSTANCES_DIM = 2
    FEATURE_DIM = 3

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

    X, y, metadata = load_classification(dataset_title, return_metadata=True)

    if len(X.shape) == 3:
        # aeon uses a different layout
        X = np.transpose(X, (0, 2, 1))

    # label encode y (always 1d categorical labels)
    if y.ndim > 1:
        raise ValueError(
            f"Expected 1D labels, but got {y.ndim}D labels for dataset {dataset_title}."
        )

    y = pd.Categorical(y).codes
    # turn to npt.NDArray[np.float64]
    y = y.astype(np.float64)

    return X, y


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
    # read the csv files into a list of DataFrames
    df = pd.read_csv(file_path, sep=",", decimal=".")
    # drop first column (date)
    df = df.drop(columns=["date"])
    # turn pandas DataFrame into numPy array
    data = df.to_numpy(dtype=np.float64)
    timesteps = data.shape[0]
    channels = data.shape[1]
    data = data.reshape(1, timesteps, channels)

    return data


SERIES2VEC_METADATA = {
    "WISDM2": {
        "Train": 134614,
        "Test": 14421,
        "Subject": 29,
        "Length": 40,
        "Channel": 3,
        "Class": 6,
    },
    "PAMAP2": {
        "Train": 51192,
        "Test": 11590,
        "Subject": 9,
        "Length": 100,
        "Channel": 52,
        "Class": 18,
    },
    "USC_HAD": {
        "Train": 46899,
        "Test": 9327,
        "Subject": 14,
        "Length": 100,
        "Channel": 6,
        "Class": 12,
    },
    "Sleep": {
        "Train": 25612,
        "Test": 8910,
        "Subject": 20,
        "Length": 3000,
        "Channel": 1,
        "Class": 5,
    },
    "Skoda": {
        "Train": 22587,
        "Test": 5646,
        "Subject": 1,
        "Length": 50,
        "Channel": 64,
        "Class": 11,
    },
    "Opportunity": {
        "Train": 15011,
        "Test": 2374,
        "Subject": 4,
        "Length": 100,
        "Channel": 113,
        "Class": 18,
    },
    "WISDM": {
        "Train": 11960,
        "Test": 5207,
        "Subject": 13,
        "Length": 40,
        "Channel": 3,
        "Class": 6,
    },
    "Epilepsy": {
        "Train": 9200,
        "Test": 2300,
        "Subject": 500,
        "Length": 178,
        "Channel": 1,
        "Class": 2,
    },
    "HAR": {
        "Train": 7352,
        "Test": 2947,
        "Subject": 30,
        "Length": 128,
        "Channel": 9,
        "Class": 6,
    },
}


def load_series2vec_data(
    dataset_title,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return train_data, train_label, test_data, test_label
