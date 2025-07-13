mod common;

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use numpy::{ PyArray1, PyArray3, IntoPyArray };
    use ndarray::{ Array1 };

    // Import common test utilities
    use crate::common::*;

    fn call_constructor<'py>(
        rust_time_series: &Bound<'py, PyModule>,
        data: &Bound<'py, PyArray3<f64>>,
        labels: &Bound<'py, PyArray1<f64>>
    ) -> PyResult<Bound<'py, PyAny>> {
        call_constructor_props(
            rust_time_series,
            data,
            labels,
            DEFAULT_TRAIN_PROP,
            DEFAULT_VAL_PROP,
            DEFAULT_TEST_PROP
        )
    }

    fn call_constructor_props<'py>(
        rust_time_series: &Bound<'py, PyModule>,
        data: &Bound<'py, PyArray3<f64>>,
        labels: &Bound<'py, PyArray1<f64>>,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<Bound<'py, PyAny>> {
        call_classification_constructor(
            rust_time_series,
            data,
            labels,
            train_prop,
            val_prop,
            test_prop
        )
    }

    fn call_constructor_unwrap<'py>(
        rust_time_series: &Bound<'py, PyModule>,
        data: &Bound<'py, PyArray3<f64>>,
        labels: &Bound<'py, PyArray1<f64>>
    ) -> Bound<'py, PyAny> {
        call_constructor(rust_time_series, data, labels).unwrap()
    }

    // Function to initialize a ClassificationDataSet instance
    fn init_dataset(_py: Python) -> Bound<PyAny> {
        let (rust_time_series, data, labels) = import_and_create_arrs(_py);

        call_constructor_unwrap(&rust_time_series, &data, &labels)
    }

    fn init_dataset_props(
        _py: Python,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> Bound<PyAny> {
        let (rust_time_series, data, labels) = import_and_create_arrs(_py);

        call_constructor_props(
            &rust_time_series,
            &data,
            &labels,
            train_prop,
            val_prop,
            test_prop
        ).unwrap()
    }

    fn import_and_create_arrs(
        _py: Python
    ) -> (Bound<PyModule>, Bound<PyArray3<f64>>, Bound<PyArray1<f64>>) {
        let rust_time_series = import_rust_time_series_unwrap(_py);
        let (data, labels) = create_classification_test_data(_py);
        (rust_time_series, data, labels)
    }

    // Testing if module can be imported successfully
    #[test]
    fn test_importing_module() {
        setup_python_test();
        Python::with_gil(|_py| {
            assert!(test_module_import(_py));
        });
    }

    // Test to check if the ClassificationDataSet can be initialized successfully
    #[test]
    fn test_initialization_success() {
        setup_python_test();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);
            assert_is_pyany_instance(_py, &dataset);
        });
    }

    // Test to ensure that when we pass data and labels of mismatched lengths, it raises a ValueError
    #[test]
    fn test_initialization_failure() {
        setup_python_test();
        Python::with_gil(|_py| {
            let rust_time_series = import_rust_time_series_unwrap(_py);

            let data = create_custom_test_data(_py, (
                INSTANCES_CLASSIFICATION,
                TIMESTEPS,
                FEATURES,
            ));
            let labels = Array1::<f64>
                ::ones(INSTANCES_CLASSIFICATION + 1)
                .into_pyarray(_py)
                .to_owned();

            let result = call_constructor(&rust_time_series, &data, &labels);

            assert!(result.is_err());

            let err = result.unwrap_err();
            let err_str = err.to_string();

            assert_eq!(
                err_str,
                "ValueError: Labels length must match the number of instances in data"
            );
        });
    }

    // Test to check if the downsample method works correctly with factor 2
    #[test]
    fn test_downsample_factor_2() {
        setup_python_test();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);

            let downsampling_factor = 2;
            dataset
                .call_method1("downsample", (downsampling_factor,))
                .expect("Downsampling failed");

            let new_data = extract_array_data::<f64>(&dataset, "data");
            let new_labels = extract_1d_array_data::<f64>(&dataset, "labels");

            assert_eq!(new_data.dim(), (
                INSTANCES_CLASSIFICATION,
                TIMESTEPS / downsampling_factor,
                FEATURES,
            ));
            assert_eq!(new_labels.len(), INSTANCES_CLASSIFICATION);
        });
    }

    // More detailed test that checks downsample, this time with factor 3
    #[test]
    fn test_downsample_factor_3() {
        setup_python_test();
        Python::with_gil(|_py| {
            let rust_time_series = import_rust_time_series_unwrap(_py);

            let length = 9; // length of the sequence
            let (data_array, labels_array) = create_sequential_test_data(_py, length);

            let dataset = call_constructor_unwrap(&rust_time_series, &data_array, &labels_array);

            let downsampling_factor = 3;
            dataset
                .call_method1("downsample", (downsampling_factor,))
                .expect("Downsampling failed");

            let new_data = extract_array_data::<f64>(&dataset, "data");
            let new_labels = extract_1d_array_data::<f64>(&dataset, "labels");

            assert_eq!(new_data.dim(), (INSTANCES_CLASSIFICATION, length / downsampling_factor, 1));
            assert_eq!(new_labels.len(), INSTANCES_CLASSIFICATION);

            for i in 0..3 {
                assert_eq!(new_data[[0, i, 0]], (i * 3) as f64);
            }

            for i in 0..INSTANCES_CLASSIFICATION {
                assert_eq!(new_labels[i], i as f64);
            }
        });
    }

    // Testing in-order splitting strategy
    #[test]
    fn test_in_order_split() {
        setup_python_test();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);
            let strategy = get_splitting_strategy_unwrap(_py, "InOrder");

            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            let train_data = extract_array_data::<f64>(&dataset, "train_data");
            let train_labels = extract_1d_array_data::<f64>(&dataset, "train_labels");

            // making sure they have 0.7*10 = 7 instances
            let expected_train_instances = (
                (INSTANCES_CLASSIFICATION as f64) * DEFAULT_TRAIN_PROP
            ).round() as usize;
            assert_eq!(train_data.dim(), (expected_train_instances, TIMESTEPS, FEATURES));
            assert_eq!(train_labels.len(), expected_train_instances);

            let val_data = extract_array_data::<f64>(&dataset, "val_data");
            let val_labels = extract_1d_array_data::<f64>(&dataset, "val_labels");

            // making sure they have 0.2*10 = 2 instances
            let expected_val_instances = (
                (INSTANCES_CLASSIFICATION as f64) * DEFAULT_VAL_PROP
            ).round() as usize;
            assert_eq!(val_data.dim(), (expected_val_instances, TIMESTEPS, FEATURES));
            assert_eq!(val_labels.len(), expected_val_instances);

            let test_data = extract_array_data::<f64>(&dataset, "test_data");
            let test_labels = extract_1d_array_data::<f64>(&dataset, "test_labels");

            // making sure they have 0.1*10 = 1 instances
            let expected_test_instances = (
                (INSTANCES_CLASSIFICATION as f64) * DEFAULT_TEST_PROP
            ).round() as usize;
            assert_eq!(test_data.dim(), (expected_test_instances, TIMESTEPS, FEATURES));
            assert_eq!(test_labels.len(), expected_test_instances);
        });
    }

    // Testing in-order split and ensuring if order is preserved
    #[test]
    fn test_in_order_split_order() {
        setup_python_test();
        Python::with_gil(|py| {
            let (data_array, labels_array) = create_sequential_test_data(py, 10);
            let rust_time_series = import_rust_time_series_unwrap(py);
            let dataset = call_classification_constructor(
                &rust_time_series,
                &data_array,
                &labels_array,
                DEFAULT_TRAIN_PROP,
                DEFAULT_VAL_PROP,
                DEFAULT_TEST_PROP
            ).unwrap();
            let strategy = get_splitting_strategy_unwrap(py, "InOrder");

            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            let train_data: ndarray::Array3<f64> = extract_array_data(&dataset, "train_data");
            let train_labels: ndarray::Array1<f64> = extract_1d_array_data(
                &dataset,
                "train_labels"
            );

            // checking if the first element of train_data is 0.0 and train_labels is 0.0
            assert_eq!(train_data[[0, 0, 0]], 0.0);
            assert_eq!(train_labels[0], 0.0);
            // checking if the last element of train_data is 6.0 and train_labels is 6.0
            assert_eq!(train_data[[0, 6, 0]], 6.0);
            assert_eq!(train_labels[6], 6.0);
        });
    }

    // Testing random splitting strategy
    #[test]
    fn test_random_split() {
        setup_python_test();
        Python::with_gil(|py| {
            let dataset = init_dataset(py);
            let strategy = get_splitting_strategy_unwrap(py, "Random");

            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            let train_data: ndarray::Array3<f64> = extract_array_data(&dataset, "train_data");
            let train_labels: ndarray::Array1<f64> = extract_1d_array_data(
                &dataset,
                "train_labels"
            );

            // making sure they have 0.7*10 = 7 instances
            assert_eq!(train_data.dim(), (7, 60, 3));
            assert_eq!(train_labels.len(), 7);

            let val_data: ndarray::Array3<f64> = extract_array_data(&dataset, "val_data");
            let val_labels: ndarray::Array1<f64> = extract_1d_array_data(&dataset, "val_labels");

            // making sure they have 0.2*10 = 2 instances
            assert_eq!(val_data.dim(), (2, 60, 3));
            assert_eq!(val_labels.len(), 2);

            let test_data: ndarray::Array3<f64> = extract_array_data(&dataset, "test_data");
            let test_labels: ndarray::Array1<f64> = extract_1d_array_data(&dataset, "test_labels");

            // making sure they have 0.1*10 = 1 instances
            assert_eq!(test_data.dim(), (1, 60, 3));
            assert_eq!(test_labels.len(), 1);
        });
    }

    // Testing random split with strange proportions
    #[test]
    fn test_random_split_strange_proportions() {
        setup_python_test();
        Python::with_gil(|py| {
            let (train_prop, val_prop, test_prop) = (0.66, 0.33, 0.01);
            let dataset = init_dataset_props(py, train_prop, val_prop, test_prop);
            let strategy = get_splitting_strategy_unwrap(py, "Random");

            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            let train_data: ndarray::Array3<f64> = extract_array_data(&dataset, "train_data");
            let train_labels: ndarray::Array1<f64> = extract_1d_array_data(
                &dataset,
                "train_labels"
            );

            // making sure they have 0.66*10 = 6.6 instances, rounded to 7
            let expected_train_instances = (
                train_prop * (INSTANCES_CLASSIFICATION as f64)
            ).round() as usize;
            assert_eq!(train_data.dim(), (expected_train_instances, TIMESTEPS, FEATURES));
            assert_eq!(train_labels.len(), expected_train_instances);

            let val_data: ndarray::Array3<f64> = extract_array_data(&dataset, "val_data");
            let val_labels: ndarray::Array1<f64> = extract_1d_array_data(&dataset, "val_labels");
            // making sure they have 0.33*10 = 3.3 instances, rounded to 3
            let expected_val_instances = (
                val_prop * (INSTANCES_CLASSIFICATION as f64)
            ).round() as usize;
            assert_eq!(val_data.dim(), (expected_val_instances, TIMESTEPS, FEATURES));
            assert_eq!(val_labels.len(), expected_val_instances);

            let test_data: ndarray::Array3<f64> = extract_array_data(&dataset, "test_data");
            let test_labels: ndarray::Array1<f64> = extract_1d_array_data(&dataset, "test_labels");
            // making sure they have 0.01*10 = 0.1 instances, rounded to 0
            let expected_test_instances = (
                test_prop * (INSTANCES_CLASSIFICATION as f64)
            ).round() as usize;
            assert_eq!(test_data.dim(), (expected_test_instances, TIMESTEPS, FEATURES));
            assert_eq!(test_labels.len(), expected_test_instances);
        });
    }

    // Testing standardization of the dataset
    #[test]
    fn test_standardization() {
        setup_python_test();
        Python::with_gil(|py| {
            let dataset = init_dataset(py);
            // we need to first split the dataset
            let strategy = get_splitting_strategy_unwrap(py, "Random");
            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            // now we can normalize the train_data
            dataset.call_method0("normalize").expect("Normalization failed");
            let train_data: ndarray::Array3<f64> = extract_array_data(&dataset, "train_data");

            // check if all values are between -1 and 1
            for i in 0..train_data.dim().0 {
                for j in 0..train_data.dim().1 {
                    for k in 0..train_data.dim().2 {
                        assert!(train_data[[i, j, k]] >= -1.0 && train_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            let test_data: ndarray::Array3<f64> = extract_array_data(&dataset, "test_data");
            // check if all values are between -1 and 1
            for i in 0..test_data.dim().0 {
                for j in 0..test_data.dim().1 {
                    for k in 0..test_data.dim().2 {
                        assert!(test_data[[i, j, k]] >= -1.0 && test_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            let val_data: ndarray::Array3<f64> = extract_array_data(&dataset, "val_data");
            // check if all values are between -1 and 1
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
        Python::with_gil(|py| {
            let dataset = init_dataset(py);

            // we need to first split the dataset
            let strategy = get_splitting_strategy_unwrap(py, "Random");
            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            // now we can normalize the train_data
            dataset.call_method0("standardize").expect("Standardization failed");
            let train_data: ndarray::Array3<f64> = extract_array_data(&dataset, "train_data");

            // check if all values are between 0 and 1
            for i in 0..train_data.dim().0 {
                for j in 0..train_data.dim().1 {
                    for k in 0..train_data.dim().2 {
                        assert!(train_data[[i, j, k]] >= 0.0 && train_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            let test_data: ndarray::Array3<f64> = extract_array_data(&dataset, "test_data");
            // check if all values are between 0 and 1
            for i in 0..test_data.dim().0 {
                for j in 0..test_data.dim().1 {
                    for k in 0..test_data.dim().2 {
                        assert!(test_data[[i, j, k]] >= 0.0 && test_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            let val_data: ndarray::Array3<f64> = extract_array_data(&dataset, "val_data");
            // check if all values are between 0 and 1
            for i in 0..val_data.dim().0 {
                for j in 0..val_data.dim().1 {
                    for k in 0..val_data.dim().2 {
                        assert!(val_data[[i, j, k]] >= 0.0 && val_data[[i, j, k]] <= 1.0);
                    }
                }
            }
        });
    }
}
