// ignore unused code warning, since this is externally imported
#![allow(unused)]

// Common functionality for tests
use pyo3::prelude::*;
use numpy::{ PyArray1, PyArray3, IntoPyArray };
use ndarray::{ Array1, Array3 };

/// Common test setup function
pub fn setup_python_test() {
    pyo3::prepare_freethreaded_python();
}

/// Import the rust_time_series module
pub fn import_rust_time_series_unwrap<'py>(py: Python<'py>) -> Bound<'py, PyModule> {
    py.import("rust_time_series").unwrap()
}

/// Test if module can be imported successfully
pub fn test_module_import<'py>(py: Python<'py>) -> bool {
    py.import("rust_time_series").is_ok()
}

/// Create standard test data for classification datasets
pub fn create_classification_test_data<'py>(
    py: Python<'py>
) -> (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray1<f64>>) {
    let data = Array3::<f64>::ones((2, 60, 3)).into_pyarray(py).to_owned();
    let labels = Array1::<f64>::ones(60).into_pyarray(py).to_owned();
    (data, labels)
}

/// Create standard test data for forecasting datasets
pub fn create_forecasting_test_data<'py>(py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
    Array3::<f64>::ones((2, 60, 3)).into_pyarray(py).to_owned()
}

/// Create custom test data array
pub fn create_custom_test_data<'py>(
    py: Python<'py>,
    shape: (usize, usize, usize)
) -> Bound<'py, PyArray3<f64>> {
    Array3::<f64>::ones(shape).into_pyarray(py).to_owned()
}

/// Create sequential test data for detailed testing
pub fn create_sequential_test_data<'py>(
    py: Python<'py>,
    length: usize
) -> (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray1<f64>>) {
    let data = Vec::from_iter((0..length).map(|x| x as f64));
    let labels = data.clone();

    let data_array = Array3::from_shape_vec((1, length, 1), data)
        .unwrap()
        .into_pyarray(py)
        .to_owned();
    let labels_array = Array1::from_vec(labels).into_pyarray(py).to_owned();

    (data_array, labels_array)
}

/// Get splitting strategy from module
pub fn get_splitting_strategy_unwrap<'py>(
    py: Python<'py>,
    strategy_name: &str
) -> Bound<'py, PyAny> {
    let module = import_rust_time_series_unwrap(py);
    module.getattr("SplittingStrategy").unwrap().getattr(strategy_name).unwrap()
}

/// Common assertion for PyAny instance
pub fn assert_is_pyany_instance<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) {
    let pyany_type = py.get_type::<PyAny>();
    assert!(obj.is_instance(&pyany_type).unwrap());
}

/// Helper for extracting array data from Python dataset
pub fn extract_array_data<T>(dataset: &Bound<PyAny>, attr_name: &str) -> ndarray::Array3<T>
    where T: numpy::Element + Clone
{
    use numpy::PyArrayMethods;
    dataset
        .getattr(attr_name)
        .unwrap()
        .downcast::<PyArray3<T>>()
        .unwrap()
        .readonly()
        .as_array()
        .to_owned()
}

/// Helper for extracting 1D array data from Python dataset
pub fn extract_1d_array_data<T>(dataset: &Bound<PyAny>, attr_name: &str) -> ndarray::Array1<T>
    where T: numpy::Element + Clone
{
    use numpy::PyArrayMethods;
    dataset
        .getattr(attr_name)
        .unwrap()
        .downcast::<PyArray1<T>>()
        .unwrap()
        .readonly()
        .as_array()
        .to_owned()
}

/// Helper for extracting split indices
pub fn extract_split_index(dataset: &Bound<PyAny>, attr_name: &str) -> usize {
    dataset.getattr(attr_name).unwrap().extract::<usize>().unwrap()
}

/// Common function for classification dataset constructor calls
pub fn call_classification_constructor<'py>(
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

/// Common function for forecasting dataset constructor calls
pub fn call_forecasting_constructor<'py>(
    rust_time_series: &Bound<'py, PyModule>,
    data: &Bound<'py, PyArray3<f64>>,
    train_prop: f64,
    val_prop: f64,
    test_prop: f64
) -> PyResult<Bound<'py, PyAny>> {
    rust_time_series
        .getattr("ForecastingDataSet")
        .and_then(|cls| cls.call1((data, train_prop, val_prop, test_prop)))
}

/// Default proportions for testing
pub const DEFAULT_TRAIN_PROP: f64 = 0.7;
pub const DEFAULT_VAL_PROP: f64 = 0.2;
pub const DEFAULT_TEST_PROP: f64 = 0.1;
