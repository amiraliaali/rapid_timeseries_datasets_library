#[cfg(test)]
mod tests {
    use std::vec;

    use pyo3::prelude::*;
    use ndarray::{Array3 };
    use rust_time_series::splitting::{
        validate_props,
        get_n_timesteps,
        compute_split_offset,
        compute_split_offsets,
        get_split_offsets
    };

    #[test]
    fn test_validate_props() {
        Python::with_gil(|py| {
            // Valid proportions 1
            assert!(validate_props(0.7, 0.2, 0.1).is_ok());

            // Valid proportions 2
            assert!(validate_props(0.5, 0.5, 0.0).is_ok());

            // Invalid proportions 1
            assert!(validate_props(-0.1, 0.5, 0.6).is_err());

            // Invalid proportions 2
            assert!(validate_props(0.5, 0.6, -0.1).is_err());

            // Invalid proportions 3
            assert!(validate_props(0.5, 0.6, 0.1).is_err())
        });
    }

    #[test]
    fn test_get_n_timesteps() {
        Python::with_gil(|py| {
            let data = Array3::<f64>::ones((2, 10, 3));
            let data_view = data.view();

            let timesteps = get_n_timesteps(&data_view);
            assert_eq!(timesteps, 10);
        });
    }

    #[test]
    fn test_compute_split_offset() {
        Python::with_gil(|py| {
            let data = Array3::<f64>::ones((2, 10, 3));
            let data_view = data.view();

            let timesteps = get_n_timesteps(&data_view);
            let offset = compute_split_offset(timesteps, 0.7);
            assert_eq!(offset, 7);
        });
    }

    #[test]
    fn test_compute_split_offsets() {
        Python::with_gil(|py| {
            let data = Array3::<f64>::ones((2, 10, 3));
            let data_view = data.view();

            let timesteps = get_n_timesteps(&data_view);
            let (train_offset, val_offset) = compute_split_offsets(timesteps, 0.7, 0.2);
            assert_eq!(train_offset, 7);
            assert_eq!(val_offset, 2);
        });
    }

    #[test]
    fn test_get_split_offsets() {
        Python::with_gil(|py| {
            let data = Array3::<f64>::ones((2, 10, 3));
            let data_view = data.view();
            let train_prop = 0.7;
            let val_prop = 0.2;
            let (train_offset, val_offset) = get_split_offsets(&data_view, train_prop, val_prop);
            assert_eq!(train_offset, 7);
            assert_eq!(val_offset, 2);
        });
    }
}
