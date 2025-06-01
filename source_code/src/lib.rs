// mod data_abstract;
// use data_abstract::{ClassificationDataset, ForecastingDataset};

use pyo3::prelude::*;
use log::{ info, debug };
use numpy::{ ndarray::Axis, IntoPyArray, PyArray2, PyArrayMethods };

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
        Mode,
        ForwardFill,
    }
}

#[pyclass]
pub struct RustTimeSeries {
    dataset_type: DatasetType,
    data: Py<PyArray2<f64>>,
}

#[pymethods]
impl RustTimeSeries {
    #[new]
    pub fn new(data: Py<PyArray2<f64>>, dataset_type: DatasetType) -> Self {
        debug!("Creating RustTimeSeries instance with dataset type: {:?}", dataset_type);
        RustTimeSeries { data, dataset_type }
    }

    fn set_to_100(&mut self, py: Python) {
        let array = self.data.bind(py);
        let mut array_mut = unsafe { array.as_array_mut() };

        // set the first element to 100.0
        if let Some(first_elem) = array_mut.get_mut([0, 0]) {
            *first_elem = 100.0;
        } else {
            panic!("Array is empty, cannot modify first element");
        }
    }

    fn normalize(&mut self, py: Python) -> PyResult<()> {
        debug!("Normalizing array");
        Ok(())
    }

    fn standardize(&mut self, py: Python) -> PyResult<()> {
        debug!("Standardizing array");
        Ok(())
    }

    fn impute(&mut self, py: Python, strategy: ImputeStrategy) -> PyResult<()> {
        debug!("Imputing array with strategy: {:?}", strategy);
        Ok(())
    }

    fn split(
        &mut self,
        py: Python,
        split_strategy: SplittingStrategy,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
        debug!(
            "Splitting array with sizes: train={}, val={}, test={}, adding up to {}\n",
            train_prop,
            val_prop,
            test_prop,
            train_prop + val_prop + test_prop
        );

        // Validate the sizes
        if train_prop < 0.0 || val_prop < 0.0 || test_prop < 0.0 {
            return Err(
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Sizes must be non-negative")
            );
        }
        const EPSILON: f64 = 1e-10;
        if (train_prop + val_prop + test_prop - 1.0).abs() > EPSILON {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Sizes must sum to 1.0"));
        }

        if split_strategy == SplittingStrategy::Random {
            return Err(
                PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Random splitting is not implemented yet"
                )
            );
        }

        let bound_array = self.data.bind(py);
        let array = unsafe { bound_array.as_array() };
        let (rows, _) = array.dim();

        let train_split = (train_prop * (rows as f64)).round() as usize;
        let val_split = ((val_prop * (rows as f64)).round() as usize) + train_split;

        let (train_data, remainder) = array.split_at(Axis(0), train_split);
        let (val_data, test_data) = remainder.split_at(Axis(0), val_split - train_split);

        let train_data_py = train_data.to_owned().into_pyarray(py);
        let val_data_py = val_data.to_owned().into_pyarray(py);
        let test_data_py = test_data.to_owned().into_pyarray(py);

        Ok((train_data_py.into(), val_data_py.into(), test_data_py.into()))
    }
}

#[pymodule]
fn rust_time_series(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<RustTimeSeries>()?;
    m.add_class::<DatasetType>()?;
    m.add_class::<ImputeStrategy>()?;
    m.add_class::<SplittingStrategy>()?;
    Ok(())
}
