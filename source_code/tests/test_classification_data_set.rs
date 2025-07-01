#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use numpy::{PyArray1, PyArray3, IntoPyArray};
    use ndarray::{Array1, Array3, Array};
    use pyo3::types::IntoPyDict;

    // Testing if module can be imported successfully
    #[test]
    fn test_importing_module() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = py.import("rust_time_series");
            assert!(module.is_ok());
        });
    }

    // Function to initialize a ClassificationDataSet instance
    fn init_dataset<'py>(py: Python<'py>) -> Bound<'py, PyAny> {
        let rust_time_series = py.import("rust_time_series").unwrap();

        let data = Array3::<f64>::ones((2, 60, 5)).into_pyarray(py).to_owned();
        let labels = Array1::<f64>::ones(60).into_pyarray(py).to_owned();

        rust_time_series
            .getattr("ClassificationDataSet")
            .unwrap()
            .call1((data, labels))
            .unwrap()
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

            let data = Array3::<f64>::ones((2, 50, 5)).into_pyarray(py).to_owned();
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

}
