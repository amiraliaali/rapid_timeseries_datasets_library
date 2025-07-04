#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use numpy::{ ToPyArray, PyArray3, IntoPyArray, PyArrayMethods };
    use ndarray::{ Array1, Array3, Array };
    use pyo3::types::{PyAny, IntoPyDict};

    use rust_time_series::collecting::{ validate_window_params, create_windows };
    use pyo3::Python;

    #[test]
    fn test_validate_window_params() {
        Python::with_gil(|py| {
            // Valid parameters
            assert!(validate_window_params(5, 3, 1, 10).is_ok());
            assert!(validate_window_params(2, 2, 1, 5).is_ok());

            // Invalid parameters
            assert!(validate_window_params(0, 3, 1, 10).is_err());
            assert!(validate_window_params(5, 0, 1, 10).is_err());
            assert!(validate_window_params(5, 3, 0, 10).is_err());
            assert!(validate_window_params(5, 3, 1, 4).is_err());

        });
    }

    #[test]
    fn test_create_windows() {
        Python::with_gil(|py| {            
            let data = Array3::<f64>::ones((2, 10, 3));
            
            let data_view = data.view();
            let past_window = 5;
            let future_horizon = 3;
            let stride = 1;

            let (x_windows_py, y_windows_py) = create_windows(py, &data_view, past_window, future_horizon, stride).unwrap();

            let x_windows_bound = x_windows_py.bind(py);
            let y_windows_bound = y_windows_py.bind(py);

            let x_windows: Array3<f64> = x_windows_bound.to_owned_array();
            let y_windows: Array3<f64> = y_windows_bound.to_owned_array();

            let instances = 2;
            let windows_per_instance = (10 - past_window - future_horizon) / stride + 1;
            let window_count = windows_per_instance * instances;

            assert_eq!(x_windows.shape(), &[window_count, past_window, 3]);
            assert_eq!(y_windows.shape(), &[window_count, future_horizon, 3]);

        });
    }


}
