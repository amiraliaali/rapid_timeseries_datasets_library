#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use numpy::{ PyArray1, PyArray3, IntoPyArray, PyArrayMethods };
    use ndarray::{ Array1, Array3, Array, s };
    use pyo3::types::IntoPyDict;
    use rust_time_series::data_abstract::SplittingStrategy;

    // Function to initialize a ClassificationDataSet instance
    fn init_dataset<'py>(py: Python<'py>) -> Bound<'py, PyAny> {
        let rust_time_series = py.import("rust_time_series").unwrap();

        let data = Array3::<f64>::ones((2, 60, 3)).into_pyarray(py).to_owned();

        rust_time_series.getattr("ForecastingDataSet").unwrap().call1((data, )).unwrap()
    }

    // Testing if module can be imported successfully
    #[test]
    fn test_importing_module() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = py.import("rust_time_series");
            assert!(module.is_ok());
        });
    }

    // Testing if ForecastingDataSet can be initialized
    #[test]
    fn test_initialization() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dataset = init_dataset(py);
            let pyany_type = py.get_type::<PyAny>();
            assert!(dataset.is_instance(&pyany_type).unwrap());
        });
    }

    // Test to check if the downsample method works correctly with factor 2
    #[test]
    fn test_downsample_factor_2() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dataset = init_dataset(py);

            dataset.call_method1("downsample", (2,)).expect("Downsampling failed");

            let new_data = dataset
                .getattr("data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            assert_eq!(new_data.dim(), (2, 30, 3));
        });
    }

    // More detailed test that checks downsample, this time with factor 3
    #[test]
    fn test_downsample_factor_3() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let rust_time_series = py.import("rust_time_series").unwrap();

            let data = Vec::from_iter((0..9).map(|x| x as f64));

            let data_array = Array3::from_shape_vec((1, 9, 1), data)
                .unwrap()
                .into_pyarray(py)
                .to_owned();

            let dataset = rust_time_series
                .getattr("ForecastingDataSet")
                .unwrap()
                .call1((data_array,))
                .unwrap();

            dataset.call_method1("downsample", (3,)).expect("Downsampling failed");

            let new_data = dataset
                .getattr("data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            assert_eq!(new_data.dim(), (1, 3, 1));

            for i in 0..3 {
                assert_eq!(new_data[[0, i, 0]], (i * 3) as f64);
            }
        });
    }

    // Testing splitting strategy
    #[test]
    fn test_split() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dataset = init_dataset(py);

            dataset.call_method1("split", (0.7, 0.2, 0.1)).expect("Splitting failed");

            let data = dataset
                .getattr("data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();


            let train_split_index = dataset.getattr("train_split_index").unwrap().extract::<usize>().unwrap();
            let val_split_index = dataset.getattr("val_split_index").unwrap().extract::<usize>().unwrap();

            let train_data = data.slice(s![.., 0..train_split_index, ..]);
            let val_data = data.slice(s![.., train_split_index..val_split_index, ..]);
            let test_data = data.slice(s![.., val_split_index.., ..]);

            assert_eq!(train_data.dim(), (2, 42, 3));
            assert_eq!(test_data.dim(), (2, 48, 3));
            assert_eq!(val_data.dim(), (2, 0, 3));
        });
    }

    // Testing standardization of the dataset
    #[test]
    fn test_standardization() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dataset = init_dataset(py);
            // we need to first split the dataset
            let module = py.import("rust_time_series").expect("Could not import module");
            dataset.call_method1("split", (0.7, 0.2, 0.1)).expect("Splitting failed");

            // now we can normalize the train_data
            dataset.call_method0("normalize").expect("Normalization failed");
            let data = dataset
                .getattr("data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            let train_split_index = dataset.getattr("train_split_index").unwrap().extract::<usize>().unwrap();
            let val_split_index = dataset.getattr("val_split_index").unwrap().extract::<usize>().unwrap();

            let train_data = data.slice(s![.., 0..train_split_index, ..]);
            let val_data = data.slice(s![.., train_split_index..val_split_index, ..]);
            let test_data = data.slice(s![.., val_split_index.., ..]);

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
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dataset = init_dataset(py);
            // we need to first split the dataset
            let module = py.import("rust_time_series").expect("Could not import module");
            dataset.call_method1("split", (0.7, 0.2, 0.1)).expect("Splitting failed");

            // now we can normalize the train_data
            dataset.call_method0("standardize").expect("Standardization failed");

            let data = dataset
                .getattr("data")
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .readonly()
                .as_array()
                .to_owned();

            let train_split_index = dataset.getattr("train_split_index").unwrap().extract::<usize>().unwrap();
            let val_split_index = dataset.getattr("val_split_index").unwrap().extract::<usize>().unwrap();
            
            let train_data = data.slice(s![.., 0..train_split_index, ..]);
            let val_data = data.slice(s![.., train_split_index..val_split_index, ..]);
            let test_data = data.slice(s![.., val_split_index.., ..]);

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

}