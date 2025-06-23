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
pub struct Sample {
    #[pyo3(get)]
    pub id: String,

    #[pyo3(get)]
    pub past: Option<Py<PyArray2<f64>>>,

    #[pyo3(get)]
    pub future: Option<Py<PyArray2<f64>>>,

    #[pyo3(get)]
    pub label: Option<f64>,
}

#[pymethods]
impl Sample {
    fn id(&self) -> &str {
        &self.id
    }

    fn past(&self) -> Option<&Py<PyArray2<f64>>> {
        self.past.as_ref()
    }

    fn future(&self) -> Option<&Py<PyArray2<f64>>> {
        self.future.as_ref()
    }

    fn label(&self) -> Option<f64> {
        self.label
    }
}
impl Clone for Sample {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            id: self.id.clone(),
            past: self.past.as_ref().map(|x| x.clone_ref(py)),
            future: self.future.as_ref().map(|x| x.clone_ref(py)),
            label: self.label,
        })
    }
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
