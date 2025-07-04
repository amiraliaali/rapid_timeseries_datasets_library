#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use numpy::{ ToPyArray, PyArray3, IntoPyArray, PyArrayMethods };
    use ndarray::{ Array1, Array3, Array };
    use pyo3::types::{ PyAny, IntoPyDict };

    use rust_time_series::collecting::{
        validate_window_params,
        create_windows,
        collect_forecasting,
        collect_classification,
    };
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

            let (x_windows_py, y_windows_py) = create_windows(
                py,
                &data_view,
                past_window,
                future_horizon,
                stride
            ).unwrap();

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

    #[test]
    fn test_collect_forecasting() {
        Python::with_gil(|py| {
            let instances = 2;
            let timesteps = 10;
            let features = 3;
            let past_window = 5;
            let future_horizon = 3;
            let stride = 1;

            let train_data = Array3::<f64>::ones((instances, timesteps, features));
            let val_data = Array3::<f64>::ones((instances, timesteps, features));
            let test_data = Array3::<f64>::ones((instances, timesteps, features));

            let train_view = train_data.view();
            let val_view = val_data.view();
            let test_view = test_data.view();

            let result = collect_forecasting(
                py,
                &train_view,
                &val_view,
                &test_view,
                past_window,
                future_horizon,
                stride
            ).expect("collect_forecasting failed");

            let ((train_x_py, train_y_py), (val_x_py, val_y_py), (test_x_py, test_y_py)) = result;

            let train_x = train_x_py.bind(py).to_owned_array();
            let train_y = train_y_py.bind(py).to_owned_array();
            let val_x = val_x_py.bind(py).to_owned_array();
            let val_y = val_y_py.bind(py).to_owned_array();
            let test_x = test_x_py.bind(py).to_owned_array();
            let test_y = test_y_py.bind(py).to_owned_array();

            let windows_per_instance = (timesteps - past_window - future_horizon) / stride + 1;
            let expected_window_count = windows_per_instance * instances;

            assert_eq!(train_x.shape(), &[expected_window_count, past_window, features]);
            assert_eq!(train_y.shape(), &[expected_window_count, future_horizon, features]);

            assert_eq!(val_x.shape(), &[expected_window_count, past_window, features]);
            assert_eq!(val_y.shape(), &[expected_window_count, future_horizon, features]);

            assert_eq!(test_x.shape(), &[expected_window_count, past_window, features]);
            assert_eq!(test_y.shape(), &[expected_window_count, future_horizon, features]);
        });
    }

    #[test]
    fn test_collect_classification() {
        Python::with_gil(|py| {
            let instances = 5;
            let timesteps = 10;
            let features = 4;

            let train_data = Array3::<f64>::ones((instances, timesteps, features));
            let val_data = Array3::<f64>::ones((instances, timesteps, features));
            let test_data = Array3::<f64>::ones((instances, timesteps, features));

            let train_labels = Array1::<f64>::from(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
            let val_labels = Array1::<f64>::from(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
            let test_labels = Array1::<f64>::from(vec![1.0, 1.0, 0.0, 0.0, 1.0]);

            let result = collect_classification(
                py,
                train_data,
                train_labels,
                val_data,
                val_labels,
                test_data,
                test_labels
            ).expect("collect_classification failed");

            let (
                (train_data_py, train_labels_py),
                (val_data_py, val_labels_py),
                (test_data_py, test_labels_py),
            ) = result;

            let train_data_arr = train_data_py.bind(py).to_owned_array();
            let train_labels_arr = train_labels_py.bind(py).to_owned_array();
            let val_data_arr = val_data_py.bind(py).to_owned_array();
            let val_labels_arr = val_labels_py.bind(py).to_owned_array();
            let test_data_arr = test_data_py.bind(py).to_owned_array();
            let test_labels_arr = test_labels_py.bind(py).to_owned_array();

            assert_eq!(train_data_arr.shape(), &[instances, timesteps, features]);
            assert_eq!(train_labels_arr.shape(), &[instances]);

            assert_eq!(val_data_arr.shape(), &[instances, timesteps, features]);
            assert_eq!(val_labels_arr.shape(), &[instances]);

            assert_eq!(test_data_arr.shape(), &[instances, timesteps, features]);
            assert_eq!(test_labels_arr.shape(), &[instances]);
        });
    }
}
