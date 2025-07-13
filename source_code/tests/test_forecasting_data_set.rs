mod common;

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use numpy::{ PyArray3, IntoPyArray };
    use ndarray::{ Array3, s, Axis };

    // Import common test utilities
    use crate::common::*;

    fn call_constructor<'py>(
        rust_time_series: &Bound<'py, PyModule>,
        data: &Bound<'py, PyArray3<f64>>
    ) -> PyResult<Bound<'py, PyAny>> {
        call_constructor_props(
            rust_time_series,
            data,
            DEFAULT_TRAIN_PROP,
            DEFAULT_VAL_PROP,
            DEFAULT_TEST_PROP
        )
    }

    fn call_constructor_props<'py>(
        rust_time_series: &Bound<'py, PyModule>,
        data: &Bound<'py, PyArray3<f64>>,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<Bound<'py, PyAny>> {
        call_forecasting_constructor(rust_time_series, data, train_prop, val_prop, test_prop)
    }

    fn call_constructor_unwrap<'py>(
        rust_time_series: &Bound<'py, PyModule>,
        data: &Bound<'py, PyArray3<f64>>
    ) -> Bound<'py, PyAny> {
        call_constructor(rust_time_series, data).unwrap()
    }

    // Function to initialize a ForecastingDataSet instance
    fn init_dataset(_py: Python) -> Bound<PyAny> {
        let (rust_time_series, data) = import_and_create_arrs(_py);

        call_constructor_unwrap(&rust_time_series, &data)
    }

    fn init_dataset_props(
        _py: Python,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> Bound<PyAny> {
        let (rust_time_series, data) = import_and_create_arrs(_py);

        call_constructor_props(&rust_time_series, &data, train_prop, val_prop, test_prop).unwrap()
    }

    fn import_and_create_arrs(_py: Python) -> (Bound<PyModule>, Bound<PyArray3<f64>>) {
        let rust_time_series = import_rust_time_series_unwrap(_py);
        let data = create_forecasting_test_data(_py);
        (rust_time_series, data)
    }

    // Testing if module can be imported successfully
    #[test]
    fn test_importing_module() {
        setup_python_test();
        Python::with_gil(|_py| {
            assert!(test_module_import(_py));
        });
    }

    // Testing if ForecastingDataSet can be initialized
    #[test]
    fn test_initialization() {
        setup_python_test();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);
            assert_is_pyany_instance(_py, &dataset);
        });
    }

    // Test to ensure that when we pass invalid proportions, it raises an error
    #[test]
    fn test_initialization_failure() {
        setup_python_test();
        Python::with_gil(|_py| {
            let rust_time_series = import_rust_time_series_unwrap(_py);
            let data = create_forecasting_test_data(_py);

            let result = call_constructor_props(&rust_time_series, &data, 0.8, 0.5, 0.2);

            // Should fail because proportions don't add up to 1.0
            assert!(result.is_err());
        });
    }

    // Test to check if the downsample method works correctly with factor 2
    #[test]
    fn test_downsample_factor_2() {
        setup_python_test();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);

            dataset.call_method1("downsample", (2,)).expect("Downsampling failed");

            let new_data = extract_array_data::<f64>(&dataset, "data");
            assert_eq!(new_data.dim(), (2, 30, 3));
        });
    }

    // More detailed test that checks downsample, this time with factor 3
    #[test]
    fn test_downsample_factor_3() {
        setup_python_test();
        Python::with_gil(|_py| {
            let rust_time_series = import_rust_time_series_unwrap(_py);

            let data = Vec::from_iter((0..9).map(|x| x as f64));
            let data_array = Array3::from_shape_vec((1, 9, 1), data)
                .unwrap()
                .into_pyarray(_py)
                .to_owned();

            let dataset = call_constructor_unwrap(&rust_time_series, &data_array);

            dataset.call_method1("downsample", (3,)).expect("Downsampling failed");

            let new_data = extract_array_data::<f64>(&dataset, "data");
            assert_eq!(new_data.dim(), (1, 3, 1));

            for i in 0..3 {
                assert_eq!(new_data[[0, i, 0]], (i * 3) as f64);
            }
        });
    }

    // Testing splitting strategy
    #[test]
    fn test_split() {
        setup_python_test();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);

            dataset.call_method0("split").expect("Splitting failed");

            let data = extract_array_data::<f64>(&dataset, "data");

            let train_split_index = extract_split_index(&dataset, "train_split_index");
            let val_split_index = extract_split_index(&dataset, "val_split_index");

            let (train_data, rest) = data.view().split_at(Axis(1), train_split_index);
            let (val_data, test_data) = rest.split_at(Axis(1), val_split_index);

            assert_eq!(train_data.dim(), (2, 42, 3));
            assert_eq!(test_data.dim(), (2, 6, 3));
            assert_eq!(val_data.dim(), (2, 12, 3));
        });
    }

    // Testing standardization of the dataset
    #[test]
    fn test_standardization() {
        setup_python_test();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);
            // we need to first split the dataset
            dataset.call_method0("split").expect("Splitting failed");

            // now we can normalize the train_data
            dataset.call_method0("normalize").expect("Normalization failed");
            let data = extract_array_data::<f64>(&dataset, "data");

            let train_split_index = extract_split_index(&dataset, "train_split_index");
            let val_split_index = extract_split_index(&dataset, "val_split_index");

            let (train_data, rest) = data
                .slice(s![.., .., ..])
                .split_at(Axis(1), train_split_index);
            let (val_data, test_data) = rest.split_at(Axis(1), val_split_index);

            // check if all values in train data are between -1 and 1
            for i in 0..train_data.dim().0 {
                for j in 0..train_data.dim().1 {
                    for k in 0..train_data.dim().2 {
                        assert!(train_data[[i, j, k]] >= -1.0 && train_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            // check if all values in test data are between -1 and 1
            for i in 0..test_data.dim().0 {
                for j in 0..test_data.dim().1 {
                    for k in 0..test_data.dim().2 {
                        assert!(test_data[[i, j, k]] >= -1.0 && test_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            // check if all values in val data are between -1 and 1
            for i in 0..val_data.dim().0 {
                for j in 0..val_data.dim().1 {
                    for k in 0..val_data.dim().2 {
                        assert!(val_data[[i, j, k]] >= -1.0 && val_data[[i, j, k]] <= 1.0);
                    }
                }
            }
        });
    }

    // Testing min max normalization of the dataset
    #[test]
    fn test_min_max_normalization() {
        setup_python_test();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);
            // we need to first split the dataset
            dataset.call_method0("split").expect("Splitting failed");

            // now we can normalize the train_data
            dataset.call_method0("standardize").expect("Standardization failed");

            let data = extract_array_data::<f64>(&dataset, "data");

            let train_split_index = extract_split_index(&dataset, "train_split_index");
            let val_split_index = extract_split_index(&dataset, "val_split_index");

            let (train_data, rest) = data
                .slice(s![.., .., ..])
                .split_at(Axis(1), train_split_index);
            let (val_data, test_data) = rest.split_at(Axis(1), val_split_index);

            // check if all values in train data are between 0 and 1
            for i in 0..train_data.dim().0 {
                for j in 0..train_data.dim().1 {
                    for k in 0..train_data.dim().2 {
                        assert!(train_data[[i, j, k]] >= 0.0 && train_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            // check if all values in test data are between 0 and 1
            for i in 0..test_data.dim().0 {
                for j in 0..test_data.dim().1 {
                    for k in 0..test_data.dim().2 {
                        assert!(test_data[[i, j, k]] >= 0.0 && test_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            // check if all values in val data are between 0 and 1
            for i in 0..val_data.dim().0 {
                for j in 0..val_data.dim().1 {
                    for k in 0..val_data.dim().2 {
                        assert!(val_data[[i, j, k]] >= 0.0 && val_data[[i, j, k]] <= 1.0);
                    }
                }
            }
        });
    }

    // Test splitting with different proportions
    #[test]
    fn test_split_different_proportions() {
        setup_python_test();
        Python::with_gil(|_py| {
            let dataset = init_dataset_props(_py, 0.6, 0.3, 0.1);

            dataset.call_method0("split").expect("Splitting failed");

            let data = extract_array_data::<f64>(&dataset, "data");

            let train_split_index = extract_split_index(&dataset, "train_split_index");
            let val_split_index = extract_split_index(&dataset, "val_split_index");

            let (train_data, rest) = data.view().split_at(Axis(1), train_split_index);
            let (val_data, test_data) = rest.split_at(Axis(1), val_split_index);

            assert_eq!(train_data.dim(), (2, 36, 3)); // 0.6 * 60 = 36
            assert_eq!(val_data.dim(), (2, 18, 3)); // 0.3 * 60 = 18
            assert_eq!(test_data.dim(), (2, 6, 3)); // 0.1 * 60 = 6
        });
    }
}
