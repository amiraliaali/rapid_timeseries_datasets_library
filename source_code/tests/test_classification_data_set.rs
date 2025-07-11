#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use numpy::{ PyArray1, PyArray3, IntoPyArray, PyArrayMethods };
    use ndarray::{ Array1, Array3 };

    fn call_constructor<'py>(
        rust_time_series: &Bound<'py, PyModule>,
        data: &Bound<'py, PyArray3<f64>>,
        labels: &Bound<'py, PyArray1<f64>>
    ) -> PyResult<Bound<'py, PyAny>> {
        call_constructor_props(rust_time_series, data, labels, 0.7, 0.2, 0.1)
    }

    fn call_constructor_props<'py>(
        rust_time_series: &Bound<'py, PyModule>,
        data: &Bound<'py, PyArray3<f64>>,
        labels: &Bound<'py, PyArray1<f64>>,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<Bound<'py, PyAny>> {
        rust_time_series
            .getattr("ClassificationDataSet")
            .and_then(|cls| cls.call1((data, labels, train_prop, val_prop, test_prop)))
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
        let rust_time_series = _py.import("rust_time_series").unwrap();

        let data = Array3::<f64>::ones((2, 60, 3)).into_pyarray(_py).to_owned();
        let labels = Array1::<f64>::ones(60).into_pyarray(_py).to_owned();

        (rust_time_series, data, labels)
    }

    // Testing if module can be imported successfully
    #[test]
    fn test_importing_module() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let module = _py.import("rust_time_series");
            assert!(module.is_ok());
        });
    }

    // Test to check if the ClassificationDataSet can be initialized successfully
    #[test]
    fn test_initialization_success() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);
            let pyany_type = _py.get_type::<PyAny>();
            assert!(dataset.is_instance(&pyany_type).unwrap());
        });
    }

    // Test to ensure that when we pass data and labels of mismatched lengths, it raises a ValueError
    #[test]
    fn test_initialization_failure() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let rust_time_series = _py.import("rust_time_series").unwrap();

            let data = Array3::<f64>::ones((2, 50, 3)).into_pyarray(_py).to_owned();
            let labels = Array1::<f64>::ones(60).into_pyarray(_py).to_owned();

            let result = call_constructor(&rust_time_series, &data, &labels);

            assert!(result.is_err());

            let err = result.unwrap_err();
            let err_str = err.to_string();

            assert_eq!(
                err_str,
                "ValueError: Labels length must match the number of timesteps in data"
            );
        });
    }

    // Test to check if the downsample method works correctly with factor 2
    #[test]
    fn test_downsample_factor_2() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);

            dataset.call_method1("downsample", (2,)).expect("Downsampling failed");

            let new_data = dataset
                .getattr("data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            let new_labels = dataset
                .getattr("labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            assert_eq!(new_data.dim(), (2, 30, 3));
            assert_eq!(new_labels.len(), 30);
        });
    }

    // More detailed test that checks downsample, this time with factor 3
    #[test]
    fn test_downsample_factor_3() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let rust_time_series = _py.import("rust_time_series").unwrap();

            let data = Vec::from_iter((0..9).map(|x| x as f64));
            let labels = data.clone();

            let data_array = Array3::from_shape_vec((1, 9, 1), data)
                .unwrap()
                .into_pyarray(_py)
                .to_owned();
            let labels_array = Array1::from_vec(labels).into_pyarray(_py).to_owned();

            let dataset = call_constructor_unwrap(&rust_time_series, &data_array, &labels_array);
            dataset.call_method1("downsample", (3,)).expect("Downsampling failed");

            let new_data = dataset
                .getattr("data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            let new_labels = dataset
                .getattr("labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            assert_eq!(new_data.dim(), (1, 3, 1));
            assert_eq!(new_labels.len(), 3);

            for i in 0..3 {
                assert_eq!(new_data[[0, i, 0]], (i * 3) as f64);
            }

            for i in 0..3 {
                assert_eq!(new_labels[i], (i * 3) as f64);
            }
        });
    }

    // Testing temporal splitting strategy
    #[test]
    fn test_temporal_split() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);

            let module = _py.import("rust_time_series").expect("Could not import module");
            let strategy = module
                .getattr("SplittingStrategy")
                .unwrap()
                .getattr("Temporal")
                .unwrap();

            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            let train_data = dataset
                .getattr("train_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            let train_labels = dataset
                .getattr("train_labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            // making sure they have 0.7*60 = 42 timesteps
            assert_eq!(train_data.dim(), (2, 42, 3));
            assert_eq!(train_labels.len(), 42);

            let val_data = dataset
                .getattr("val_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            let val_labels = dataset
                .getattr("val_labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            // making sure they have 0.2*60 = 12 timesteps
            assert_eq!(val_data.dim(), (2, 12, 3));
            assert_eq!(val_labels.len(), 12);

            let test_data = dataset
                .getattr("test_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            let test_labels = dataset
                .getattr("test_labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            // making sure they have 0.1*60 = 6 timesteps
            assert_eq!(test_data.dim(), (2, 6, 3));
            assert_eq!(test_labels.len(), 6);
        });
    }

    // Testing temporal split and ensuring if order is preserved
    #[test]
    fn test_temporal_split_order() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let data_vector = Vec::from_iter((0..10).map(|x| x as f64));
            let labels_vector = data_vector.clone();
            let data_array = Array3::from_shape_vec((1, 10, 1), data_vector)
                .unwrap()
                .into_pyarray(_py)
                .to_owned();
            let labels_array = Array1::from_vec(labels_vector).into_pyarray(_py).to_owned();
            let rust_time_series = _py.import("rust_time_series").unwrap();
            let dataset = call_constructor_unwrap(&rust_time_series, &data_array, &labels_array);
            let strategy = rust_time_series
                .getattr("SplittingStrategy")
                .unwrap()
                .getattr("Temporal")
                .unwrap();

            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            let train_data = dataset
                .getattr("train_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            let train_labels = dataset
                .getattr("train_labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

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
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);
            let module = _py.import("rust_time_series").expect("Could not import module");
            let strategy = module.getattr("SplittingStrategy").unwrap().getattr("Random").unwrap();

            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            let train_data = dataset
                .getattr("train_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            let train_labels = dataset
                .getattr("train_labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            // making sure they have 0.7*60 = 42 timesteps
            assert_eq!(train_data.dim(), (2, 42, 3));
            assert_eq!(train_labels.len(), 42);

            let val_data = dataset
                .getattr("val_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            let val_labels = dataset
                .getattr("val_labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            // making sure they have 0.2*60 = 12 timesteps
            assert_eq!(val_data.dim(), (2, 12, 3));
            assert_eq!(val_labels.len(), 12);

            let test_data = dataset
                .getattr("test_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            let test_labels = dataset
                .getattr("test_labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            // making sure they have 0.1*60 = 6 timesteps
            assert_eq!(test_data.dim(), (2, 6, 3));
            assert_eq!(test_labels.len(), 6);
        });
    }

    // Testing random split with strange proportions
    #[test]
    fn test_random_split_strange_proportions() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let dataset = init_dataset_props(_py, 0.66, 0.33, 0.01);
            let module = _py.import("rust_time_series").expect("Could not import module");
            let strategy = module.getattr("SplittingStrategy").unwrap().getattr("Random").unwrap();

            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            let train_data = dataset
                .getattr("train_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            let train_labels = dataset
                .getattr("train_labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            // making sure they have 0.66*60 = 39.6 timesteps, rounded to 40
            assert_eq!(train_data.dim(), (2, 40, 3));
            assert_eq!(train_labels.len(), 40);

            let val_data = dataset
                .getattr("val_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            let val_labels = dataset
                .getattr("val_labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            // making sure they have 0.33*60 = 19.8 timesteps, rounded to 20
            assert_eq!(val_data.dim(), (2, 20, 3));
            assert_eq!(val_labels.len(), 20);

            let test_data = dataset
                .getattr("test_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            let test_labels = dataset
                .getattr("test_labels")
                .unwrap()
                .downcast::<PyArray1<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            // making sure they have 0.01*60 = 0.6 timesteps, rounded to 0
            assert_eq!(test_data.dim(), (2, 0, 3));
            assert_eq!(test_labels.len(), 0);
        });
    }

    // Testing standardization of the dataset
    #[test]
    fn test_standardization() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);
            // we need to first split the dataset
            let module = _py.import("rust_time_series").expect("Could not import module");
            let strategy = module.getattr("SplittingStrategy").unwrap().getattr("Random").unwrap();
            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            // now we can normalize the train_data
            dataset.call_method0("normalize").expect("Normalization failed");
            let train_data = dataset
                .getattr("train_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            // check if all values are between -1 and 1
            for i in 0..train_data.dim().0 {
                for j in 0..train_data.dim().1 {
                    for k in 0..train_data.dim().2 {
                        assert!(train_data[[i, j, k]] >= -1.0 && train_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            let test_data = dataset
                .getattr("test_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            // check if all values are between -1 and 1
            for i in 0..test_data.dim().0 {
                for j in 0..test_data.dim().1 {
                    for k in 0..test_data.dim().2 {
                        assert!(test_data[[i, j, k]] >= -1.0 && test_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            let val_data = dataset
                .getattr("val_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
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
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|_py| {
            let dataset = init_dataset(_py);

            // we need to first split the dataset
            let module = _py.import("rust_time_series").expect("Could not import module");
            let strategy = module.getattr("SplittingStrategy").unwrap().getattr("Random").unwrap();
            dataset.call_method1("split", (strategy,)).expect("Splitting failed");

            // now we can normalize the train_data
            dataset.call_method0("standardize").expect("Standardization failed");
            let train_data = dataset
                .getattr("train_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            // check if all values are between 0 and 1
            for i in 0..train_data.dim().0 {
                for j in 0..train_data.dim().1 {
                    for k in 0..train_data.dim().2 {
                        assert!(train_data[[i, j, k]] >= 0.0 && train_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            let test_data = dataset
                .getattr("test_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
            // check if all values are between 0 and 1
            for i in 0..test_data.dim().0 {
                for j in 0..test_data.dim().1 {
                    for k in 0..test_data.dim().2 {
                        assert!(test_data[[i, j, k]] >= 0.0 && test_data[[i, j, k]] <= 1.0);
                    }
                }
            }

            let val_data = dataset
                .getattr("val_data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();
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
