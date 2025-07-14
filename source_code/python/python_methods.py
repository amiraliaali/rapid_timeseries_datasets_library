import numpy as np
from rust_time_series.rust_time_series import (
    ImputeStrategy,
    SplittingStrategy,
)
def imputation(
    dataset: np.ndarray,
    impute_strategy: ImputeStrategy,
) -> np.ndarray:
    """
    Apply the specified imputation strategy to the dataset.
    
    Args:
        dataset: The dataset to be processed.
        impute_strategy: The strategy to use for imputing missing values.
    
    Returns:
        The dataset with missing values imputed.
    """
    instance_count:int = dataset.shape[0]
    if impute_strategy != ImputeStrategy.LeaveNaN:
        # the dataset is a 3 dim numpy array
        for i in range(instance_count):
            feature_count:int = dataset[i].shape[1]
            for j in range(feature_count):
                if impute_strategy == ImputeStrategy.Mean:
                    # calculate mean value in pure python (no numpy)
                    mean_value = sum(dataset[i, :, j]) / len(dataset[i, :, j])
                    for k in range(len(dataset[i, :, j])):
                        if np.isnan(dataset[i, :, j][k]):
                            dataset[i, :, j][k] = mean_value
                elif impute_strategy == ImputeStrategy.Median:
                    values = [x for x in dataset[i, :, j] if not np.isnan(x)]
                    values.sort()
                    n = len(values)
                    median_value = values[n//2] if n % 2 == 1 else (values[n//2-1] + values[n//2]) / 2
                    for k in range(len(dataset[i, :, j])):
                        if np.isnan(dataset[i, :, j][k]):
                            dataset[i, :, j][k] = median_value
                elif impute_strategy == ImputeStrategy.ForwardFill:
                    fallback_value = None
                    for k in range(len(dataset[i, :, j])):
                        if np.isnan(dataset[i, :, j][k]):
                            dataset[i, :, j][k] = fallback_value
                        else:
                            fallback_value = dataset[i, :, j][k]
                elif impute_strategy == ImputeStrategy.BackwardFill:
                    fallback_value = None
                    for k in range(len(dataset[i, :, j]) - 1, -1, -1):
                        if np.isnan(dataset[i, :, j][k]):
                            dataset[i, :, j][k] = fallback_value
                        else:
                            fallback_value = dataset[i, :, j][k]
    return dataset

def downsample(
    dataset: np.ndarray,
    downsampling_rate: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Downsample the dataset by the specified rate.
    
    Args:
        dataset: The dataset to be processed.
        downsampling_rate: The rate at which to downsample the dataset.
    
    Returns:
        The downsampled dataset.
    """
    if downsampling_rate <= 0:
        raise ValueError("Downsampling rate must be greater than 0.")
    downsampled = dataset[:, ::downsampling_rate, :]
    if labels is not None:
        labels = labels[::downsampling_rate]
    return downsampled, labels

def splitting(
    dataset: np.ndarray,
    splitting_strategy: SplittingStrategy,
    splitting_ratios: tuple[float, float, float],
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Split the dataset into training, validation, and testing sets.
    
    Args:
        dataset: The dataset to be processed (3D array: instances x timesteps x features).
        splitting_strategy: The strategy to use for splitting (Random or Temporal).
        splitting_ratios: Tuple of (train_prop, val_prop, test_prop) ratios.
        labels: Optional labels array for classification tasks.
    
    Returns:
        If labels is None: tuple of (train_data, val_data, test_data) - forecasting split
        If labels provided: tuple of ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels)) - classification split
    """
    train_prop, val_prop, test_prop = splitting_ratios
    
    # Validate proportions
    if train_prop < 0.0 or val_prop < 0.0 or test_prop < 0.0:
        raise ValueError("Sizes must be non-negative")
    
    if abs(train_prop + val_prop + test_prop - 1.0) > 1e-10:
        raise ValueError("Sizes must add up to 1")
    
    instances, timesteps, features = dataset.shape
    
    if labels is None:
        # FORECASTING: Split along time axis (timesteps)
        train_split_offset = int(round(timesteps * train_prop))
        val_split_offset = int(round(timesteps * val_prop))
        
        # Only temporal splitting makes sense for forecasting
        train_data = dataset[:, :train_split_offset, :]
        val_data = dataset[:, train_split_offset:train_split_offset + val_split_offset, :]
        test_data = dataset[:, train_split_offset + val_split_offset:, :]
        
        return train_data, val_data, test_data
    
    else:
        # CLASSIFICATION: Split along instance axis (samples)
        train_split_offset = int(round(instances * train_prop))
        val_split_offset = int(round(instances * val_prop))
        
        if splitting_strategy == SplittingStrategy.Temporal:
            # Temporal split - split instances sequentially
            train_data = dataset[:train_split_offset, :, :]
            val_data = dataset[train_split_offset:train_split_offset + val_split_offset, :, :]
            test_data = dataset[train_split_offset + val_split_offset:, :, :]
            
            train_labels = labels[:train_split_offset]
            val_labels = labels[train_split_offset:train_split_offset + val_split_offset]
            test_labels = labels[train_split_offset + val_split_offset:]
            
            return (
                (train_data, train_labels),
                (val_data, val_labels),
                (test_data, test_labels)
            )
        
        elif splitting_strategy == SplittingStrategy.Random:
            # Random split - shuffle instance indices and then split
            import random
            
            # Create shuffled indices for instances
            indices = list(range(instances))
            random.shuffle(indices)
            
            # Split indices
            train_indices = indices[:train_split_offset]
            val_indices = indices[train_split_offset:train_split_offset + val_split_offset]
            test_indices = indices[train_split_offset + val_split_offset:]
            
            # Select data using shuffled indices
            train_data = dataset[train_indices, :, :]
            val_data = dataset[val_indices, :, :]
            test_data = dataset[test_indices, :, :]
            
            train_labels = labels[train_indices]
            val_labels = labels[val_indices]
            test_labels = labels[test_indices]
            
            return (
                (train_data, train_labels),
                (val_data, val_labels),
                (test_data, test_labels)
            )
        
        else:
            raise ValueError(f"Unknown splitting strategy: {splitting_strategy}")

def normalization(
        train_data: np.ndarray,
        val_data: np.ndarray,
        test_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Apply Min max normalization to the datasets.
    """
    Normalize the datasets using Min-Max normalization
    Args:
        train_data: Training dataset.
        val_data: Validation dataset.
        test_data: Testing dataset.
    Returns:
        Tuple of normalized datasets (train_data, val_data, test_data).
    """
    # To avoid data leakage, we fit the scaler only on the training data
    instances = train_data.shape[0]
    for i in range(instances):
        feature_count = train_data[i].shape[1]
        for j in range(feature_count):
            min_value = min(train_data[i, :, j])
            max_value = max(train_data[i, :, j])
            for k in range(train_data[i, :, j].shape[0]):
                train_data[i, :, j][k] = (train_data[i, :, j][k] - min_value) / (max_value - min_value)

                val_data[i, :, j][k] = (val_data[i, :, j][k] - min_value) / (max_value - min_value)

                test_data[i, :, j][k] = (test_data[i, :, j][k] - min_value) / (max_value - min_value)
    return train_data, val_data, test_data

def standardization(
        train_data: np.ndarray,
        val_data: np.ndarray,
        test_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Apply Standardization to the datasets.
    """
    Standardize the datasets using Z-score normalization
    Args:
        train_data: Training dataset.
        val_data: Validation dataset.
        test_data: Testing dataset.
    Returns:
        Tuple of standardized datasets (train_data, val_data, test_data).
    """
    instances = train_data.shape[0]
    for i in range(instances):
        feature_count = train_data[i].shape[1]
        for j in range(feature_count):
            mean_value = sum(train_data[i, :, j]) / train_data[i, :, j].shape[0]
            std_value = (
                sum((x - mean_value) ** 2 for x in train_data[i, :, j]) / train_data[i, :, j].shape[0]
            ) ** 0.5
            for k in range(train_data[i, :, j].shape[0]):
                train_data[i, :, j][k] = (train_data[i, :, j][k] - mean_value) / std_value

                val_data[i, :, j][k] = (val_data[i, :, j][k] - mean_value) / std_value

                test_data[i, :, j][k] = (test_data[i, :, j][k] - mean_value) / std_value
    return train_data, val_data, test_data