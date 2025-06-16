use crate::data_abstract::{
    DatasetType,
    SplittingStrategy,
    ImputeStrategy,
    BaseDataSet, 
    ForecastingSample
};
use crate::splitting::split;
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
    
    /// Normalizes the dataset by min-max scaling each column, transforming the data to a range between 0 and 1.
    #[pyo3(signature = (indices=None))]
    fn normalize(&mut self, py: Python, indices: Option<Vec<usize>>) -> PyResult<()> {
        debug!("Normalizing array");
        let array = self.data.bind(py);
        let mut array_mut = unsafe { array.as_array_mut() };
        let (_rows, cols) = array_mut.dim();
        if let Some(ref indices) = indices {
            for &index in indices {
                if index >= cols {
                    return Err(PyValueError::new_err(format!("Index {} is out of bounds for columns ({})", index, cols)));
                }
            }
        }

        for col in 0..cols {
            if indices.is_none() || indices.as_ref().unwrap().contains(&col) {
            let column_slice = array_mut.slice(s![.., col]);
            let min = column_slice.iter().copied().fold(f64::INFINITY, f64::min);
            let max = column_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if max - min != 0.0 {
                let mut column_mut = array_mut.slice_mut(s![.., col]);
                column_mut -= min;
                column_mut /= max - min;
            } else {
                return Err(PyValueError::new_err("Column has zero range, cannot normalize"));
            }
        }
        }
        Ok(())

        // let min = array_mut.iter().copied().fold(f64::INFINITY, f64::min);
        // let max = array_mut.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // array_mut -= min;
        // array_mut /= max - min;
        // Ok(())
    }

    /// Standardizes the dataset by subtracting the mean and dividing by the standard deviation for each column.
    #[pyo3(signature = (indices=None))]
    fn standardize(&mut self, py: Python, indices: Option<Vec<usize>>) -> PyResult<()> {
        debug!("Standardizing array");
        let array = self.data.bind(py);
        let mut array_mut = unsafe { array.as_array_mut() };
        let (_rows, cols) = array_mut.dim();
        // Checks if indices if provided are within bounds
        if let Some(ref indices) = indices {
            for &index in indices {
                if index >= cols {
                    return Err(PyValueError::new_err(format!("Index {} is out of bounds for columns ({})", index, cols)));
                }
            }
        }

        for col in 0..cols {
            if indices.is_none() || indices.as_ref().unwrap().contains(&col) {
            let column_slice = array_mut.slice(s![.., col]);
            let mean = column_slice.mean().unwrap_or(0.0);
            let std = column_slice.std(0.0);
            if std != 0.0 {
                let mut column_mut = array_mut.slice_mut(s![.., col]);
                column_mut -= mean;
                column_mut /= std;
            } else {
                return Err(PyValueError::new_err("Standard deviation is zero, cannot standardize"));
            }
        }
        }
        Ok(())
    }

    #[pyo3(signature = (strategy=ImputeStrategy::LeaveNaN,indices=None))]
    fn impute(&mut self, py: Python, strategy: ImputeStrategy, indices: Option<Vec<usize>>) -> PyResult<()> {
        debug!("Imputing array with strategy: {:?}", strategy);
        let array = self.data.bind(py);
        let mut array_mut = unsafe { array.as_array_mut() };
        let (_rows, cols) = array_mut.dim();
        // Checks if indices if provided are within bounds
        if let Some(ref indices) = indices {
            for &index in indices {
                if index >= cols {
                    return Err(PyValueError::new_err(format!("Index {} is out of bounds for columns ({})", index, cols)));
                }
            }
        }
        for col in 0..cols {
            if indices.is_none() || indices.as_ref().unwrap().contains(&col) {
                let mut column_slice = array_mut.slice_mut(s![.., col]);
                match strategy {
                    ImputeStrategy::LeaveNaN => continue,
                    ImputeStrategy::Mean => {
                        // Calculate the mean of the filtered values (dropping NaNs) (using numpy api would return NaN)
                        let mean = column_slice.iter()
                            .filter(|&&x| !x.is_nan())
                            .cloned()
                            .sum::<f64>() / column_slice.iter().filter(|&&x| !x.is_nan()).count() as f64;
                        column_slice.iter_mut().for_each(|x| {
                            if x.is_nan() {
                                *x = mean;
                            }
                        });
                    }
                    ImputeStrategy::Median => {
                        let vals = column_slice.iter().filter(|&&x| !x.is_nan()).cloned().collect::<Vec<_>>();
                        let mut sorted_vals = vals.clone();
                        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median = if sorted_vals.is_empty() {
                            0.0
                        } else if sorted_vals.len() % 2 == 1 {
                            sorted_vals[sorted_vals.len() / 2]
                        } else {
                            (sorted_vals[sorted_vals.len() / 2 - 1] + sorted_vals[sorted_vals.len() / 2]) / 2.0
                        };
                        column_slice.iter_mut().for_each(|x| {
                            if x.is_nan() {
                                *x = median;
                            }
                        });
                    }
                    ImputeStrategy::ForwardFill => {
                        let mut last_valid = None;
                        for x in column_slice.iter_mut() {
                            if x.is_nan() {
                                if let Some(last) = last_valid {
                                    *x = last;
                                }
                            } else {
                                last_valid = Some(*x);
                            }
                        }
                    }
                    ImputeStrategy::BackwardFill => {
                        let mut next_valid = None;
                        for x in column_slice.iter_mut().rev() {
                            if x.is_nan() {
                                if let Some(next) = next_valid {
                                    *x = next;
                                }
                            } else {
                                next_valid = Some(*x);
                            }
                        }
                    }
                    
                }
            }
        }
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

    fn split(
        &self,
        py: Python,
        split_strategy: SplittingStrategy,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)>{
        split(&self.dataset_type, &self.data, py, split_strategy, train_prop, val_prop, test_prop)
    }
}
