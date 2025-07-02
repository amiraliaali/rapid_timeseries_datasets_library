#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use numpy::{ PyArray1, PyArray3, IntoPyArray, PyArrayMethods };
    use ndarray::{ Array1, Array3, Array };
    use pyo3::types::IntoPyDict;

    // Function to initialize a ClassificationDataSet instance
    fn init_dataset<'py>(py: Python<'py>) -> Bound<'py, PyAny> {
        let rust_time_series = py.import("rust_time_series").unwrap();

        let data = Array3::<f64>::ones((2, 60, 3)).into_pyarray(py).to_owned();
        let labels = Array1::<f64>::ones(60).into_pyarray(py).to_owned();

        rust_time_series.getattr("ClassificationDataSet").unwrap().call1((data, labels)).unwrap()
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

    // Test to check if the ClassificationDataSet can be initialized successfully
    #[test]
    fn test_initialization_success() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dataset = init_dataset(py);
            let pyany_type = py.get_type::<PyAny>();
            assert!(dataset.is_instance(&pyany_type).unwrap());
        });
    }

    // Test to ensure that when we pass data and labels of mismatched lengths, it raises a ValueError
    #[test]
    fn test_initialization_failure() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let rust_time_series = py.import("rust_time_series").unwrap();

            let data = Array3::<f64>::ones((2, 50, 3)).into_pyarray(py).to_owned();
            let labels = Array1::<f64>::ones(60).into_pyarray(py).to_owned();

            let result = rust_time_series
                .getattr("ClassificationDataSet")
                .unwrap()
                .call1((data, labels));

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
        Python::with_gil(|py| {
            let dataset = init_dataset(py);

            dataset
                .call_method1("downsample", (2,))
                .expect("Downsampling failed");

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
        Python::with_gil(|py| {
            let rust_time_series = py.import("rust_time_series").unwrap();

            let data = Vec::from_iter((0..9).map(|x| x as f64));
            let labels = data.clone();

            let data_array = Array3::from_shape_vec((1, 9, 1), data).unwrap().into_pyarray(py).to_owned();
            let labels_array = Array1::from_vec(labels).into_pyarray(py).to_owned();

            let dataset = rust_time_series
                .getattr("ClassificationDataSet")
                .unwrap()
                .call1((data_array, labels_array))
                .unwrap();

            dataset
                .call_method1("downsample", (3,))
                .expect("Downsampling failed");

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
}
