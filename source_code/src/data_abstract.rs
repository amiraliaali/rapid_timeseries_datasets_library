use ndarray::{ArrayBase, Dim, OwnedRepr};
use numpy::{ PyArray2,PyArray3 };
use pyo3::prelude::*;

macro_rules! py_enum {
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident { $($variant:ident),* $(,)? }
    ) => {
        #[pyclass]
        #[derive(PartialEq, Clone, Debug)]
        $(#[$meta])*
        $vis enum $name {
            $($variant),*
        }
    };
}

// Replace your existing enum definitions with:
py_enum! {
    pub DatasetType {
        Classification,
        Forecasting,
    }
}

py_enum! {
    pub SplittingStrategy {
        Temporal,
        Random,
    }
}

py_enum! {
    pub ImputeStrategy {
        LeaveNaN,
        Mean,
        Median,
        ForwardFill,
        BackwardFill,
    }
}


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

    fn past(&self) -> &Py<PyArray2<f64>> {
        &self.past
    }

    fn future(&self) -> &Py<PyArray2<f64>> {
        &self.future
    }
}
impl Clone for ForecastingSample {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            id: self.id.clone(),
            past: self.past.clone_ref(py),
            future: self.future.clone_ref(py),
        })
    }
}

#[pyclass]
#[derive(Debug)]
pub struct ClassificationSample {
    #[pyo3(get)]
    pub id: String,

    #[pyo3(get)]
    pub past: Py<PyArray2<f64>>,

    #[pyo3(get)]
    pub label: f64,
}

impl Clone for ClassificationSample {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            id: self.id.clone(),
            past: self.past.clone_ref(py),
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
        &self.past
    }
    fn label(&self) -> f64 {
        self.label
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum SampleType {
    Classification(ClassificationSample),
    Forecasting(ForecastingSample),
}

#[pyclass]
pub struct BaseDataSet {
    pub data: Py<PyArray3<f64>>,
    pub labels: Option<ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>>,
    pub dataset_type: DatasetType,
    pub past_window: usize,
    pub future_horizon: usize,
    pub stride: usize,
    pub x_windows: ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>,
    pub y_windows: Option<ArrayBase<OwnedRepr<f64>, Dim<[usize; 3]>>>,
}
