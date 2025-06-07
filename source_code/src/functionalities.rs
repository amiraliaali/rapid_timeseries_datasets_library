use crate::data_abstract::{
    DatasetType,
    SplittingStrategy,
    ImputeStrategy,
    BaseDataSet, 
    ForecastingSample
};
use numpy::{ ndarray::{s}, IntoPyArray, PyArray2, PyArrayMethods };
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use log::{ debug };


#[pymethods]
impl BaseDataSet {
    #[new]
    pub fn new_forecasting(data: Py<PyArray2<f64>>, past_window: usize, future_horizon: usize, stride: usize) -> PyResult<Self> {
        if past_window == 0 || future_horizon == 0 || stride == 0 {
            return Err(PyValueError::new_err("past_window, future_horizon, and stride must be greater than 0"));
        }

        debug!("Creating RustTimeSeries instance with dataset type: {:?}", DatasetType::Forecasting);
        Ok(BaseDataSet {
            data,
            labels: Vec::new(),
            dataset_type: DatasetType::Forecasting,
            past_window,
            future_horizon,
            stride
        })
    }

    #[staticmethod]
    pub fn new_classification(features: Py<PyArray2<f64>>, labels: Vec<String>, py: Python) -> PyResult<Self> {
        debug!("Creating ClassificationDataset instance with dataset type: {:?}", DatasetType::Classification);
        
        let bound_array = features.bind(py);
        let array = unsafe { bound_array.as_array() };
        let (rows, _) = array.dim();

        if rows != labels.len() {
            return Err(PyValueError::new_err("Number of rows in features does not match number of labels"));
        }

        Ok(BaseDataSet {
            data: features,
            labels,
            dataset_type: DatasetType::Classification,
            past_window: 0, // Not used for classification datasets
            future_horizon: 0, // Not used for classification datasets
            stride: 1, // Default stride for classification datasets
        })
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

    fn len(&self, py: Python) -> PyResult<usize> {
        if self.dataset_type == DatasetType::Classification {
            return Ok(self.labels.len());
        }

        let bound_array = self.data.bind(py);
        let array = unsafe { bound_array.as_array() };
        let (rows, _) = array.dim();

        let total_window = self.past_window + self.future_horizon;

        if rows < total_window {
            Ok(0)
        } else {
            Ok((rows - total_window) / self.stride + 1)
        }
    }

    fn get(&self, py: Python, index: usize) -> PyResult<Option<ForecastingSample>>{
        if self.dataset_type == DatasetType::Classification {
            return Err(PyValueError::new_err("get method is not applicable for classification datasets"));
        }

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

    #[pyo3(name = "split")]
    fn split_binding(
        mut slf: PyRefMut<Self>,
        py: Python,
        split_strategy: SplittingStrategy,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)>{
        slf.split(py, split_strategy, train_prop, val_prop, test_prop)
    }
}
