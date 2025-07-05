#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use ndarray::{Array3 };
    use numpy::{PyArray3, IntoPyArray, PyArrayMethods};
    use rust_time_series::utils::{
        validate_split_indices
    };

    #[test]
    fn test_validate_split_indices() {
        Python::with_gil(|py| {
            // Valid indices
            assert!(validate_split_indices(Some(5), Some(8), 10).is_ok());

            // Invalid indices: train_split_index is None
            assert!(validate_split_indices(None, Some(8), 10).is_err());

            // Invalid indices: val_split_index is None
            assert!(validate_split_indices(Some(5), None, 10).is_err());

            // Invalid indices: train_split_index <= 0
            assert!(validate_split_indices(Some(0), Some(8), 10).is_err());

            // Invalid indices: train_split_index >= total_timesteps
            assert!(validate_split_indices(Some(10), Some(8), 10).is_err());

            // Invalid indices: val_split_index <= 0
            assert!(validate_split_indices(Some(5), Some(0), 10).is_err());

            // Invalid indices: val_split_index >= total_timesteps
            assert!(validate_split_indices(Some(5), Some(10), 10).is_err());
        });
    }

}
