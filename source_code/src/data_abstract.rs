use crate::py_definitions::{
    DatasetType,
};
use numpy::{ PyArray2 };
use pyo3::prelude::*;


#[pyclass]
#[derive(Debug)]
pub struct ForecastingSample {
    #[pyo3(get)]
    pub id: String,

    #[pyo3(get)]
    pub past: Py<PyArray2<f64>>,

    #[pyo3(get)]
    pub future: Py<PyArray2<f64>>,
}

#[pymethods]
impl ForecastingSample {
    fn id(&self) -> &str {
        &self.id
    }

    fn sequence(&self) -> &Py<PyArray2<f64>> {
        &self.past
    }
}

#[pyclass]
#[derive(Debug)]
pub struct ClassificationSample {
    #[pyo3(get)]
    pub id: String,

    #[pyo3(get)]
    pub features: Py<PyArray2<f64>>,

    #[pyo3(get)]
    pub label: String,
}

impl Clone for ClassificationSample {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            id: self.id.clone(),
            features: self.features.clone_ref(py),
            label: self.label.clone(),
        })
    }
}

#[pymethods]
impl ClassificationSample {
    fn id(&self) -> &str {
        &self.id
    }

    fn sequence(&self) -> &Py<PyArray2<f64>> {
        &self.features
    }
}

#[pyclass]
pub struct BaseDataSet {
    pub data: Py<PyArray2<f64>>,
    pub labels: Vec<String>,
    pub dataset_type: DatasetType,
    pub past_window: usize,
    pub future_horizon: usize,
    pub stride: usize,
}
