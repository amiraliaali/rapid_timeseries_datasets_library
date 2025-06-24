use ndarray::{ Array1, Array3 };
use numpy::{ PyArray1, PyArray3 };
use pyo3::prelude::*;

pub type ClassificationSplitResult = PyResult<
    ((Array3<f64>, Array1<f64>), (Array3<f64>, Array1<f64>), (Array3<f64>, Array1<f64>))
>;

pub type ClassificationCollectResult = PyResult<
    (
        (Py<PyArray3<f64>>, Py<PyArray1<f64>>),
        (Py<PyArray3<f64>>, Py<PyArray1<f64>>),
        (Py<PyArray3<f64>>, Py<PyArray1<f64>>),
    )
>;

pub type ForecastingCollectResult = PyResult<
    (
        (Py<PyArray3<f64>>, Py<PyArray3<f64>>),
        (Py<PyArray3<f64>>, Py<PyArray3<f64>>),
        (Py<PyArray3<f64>>, Py<PyArray3<f64>>),
    )
>;
