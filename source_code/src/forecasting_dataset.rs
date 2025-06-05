use crate::py_definitions::{
    DatasetType,
    SplittingStrategy,
    ImputeStrategy,
};
use numpy::{ ndarray::{s, Axis}, IntoPyArray, PyArray2, PyArrayMethods };
use log::{ info, debug };
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
pub struct ForecastingDataSet {
    data: Py<PyArray2<f64>>,
    dataset_type: DatasetType,
    past_window: usize,
    future_horizon: usize,
    stride: usize,
}

#[pymethods]
impl ForecastingDataSet {
    #[new]
    pub fn new(data: Py<PyArray2<f64>>, dataset_type: DatasetType, past_window: usize, future_horizon: usize, stride: usize) -> Self {
        debug!("Creating RustTimeSeries instance with dataset type: {:?}", dataset_type);
        ForecastingDataSet { data, dataset_type, past_window, future_horizon, stride }
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

    fn len(&self, py: Python) -> PyResult<usize> {
        let total_window = self.past_window + self.future_horizon;

        let bound_array = self.data.bind(py);
        let array = unsafe { bound_array.as_array() };
        let (rows, _) = array.dim();

        if rows < total_window {
            Ok(0)
        } else {
            Ok((rows - total_window) / self.stride + 1)
        }
    }

    fn get(&self, py: Python, index: usize) -> PyResult<Option<ForecastingSample>>{
        let bound_array = self.data.bind(py);
        let array = unsafe { bound_array.as_array() };
        let (rows, _) = array.dim();

        if index >= rows {
            return Ok(None);
        }

        let start_pos = index * self.stride;
        let total_window_size = self.past_window + self.future_horizon;

        if start_pos + total_window_size > rows {
            return Ok(None);
        }

        let past = array.slice(s![start_pos..start_pos + self.past_window, ..]).to_owned();
        let future = array.slice(s![start_pos + self.past_window..start_pos + total_window_size, ..]).to_owned();
        let past_py = past.into_pyarray(py);
        let future_py = future.into_pyarray(py);

        let sample = ForecastingSample {
            id: index.to_string(),
            past: past_py.into(),
            future: future_py.into(),
        };
        Ok(Some(sample))
    }
}