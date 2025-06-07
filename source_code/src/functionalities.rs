use crate::py_definitions::{
    DatasetType,
    SplittingStrategy,
    ImputeStrategy,
};
use crate::data_abstract::{BaseDataSet, ForecastingSample};
use numpy::{ ndarray::{s, Axis, Array2}, IntoPyArray, PyArray2, PyArrayMethods };
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use log::{ debug };
use rand::seq::SliceRandom;
use rand::thread_rng;


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

    fn split(
        &mut self,
        py: Python,
        split_strategy: SplittingStrategy,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
        if split_strategy == SplittingStrategy::Temporal {
            if self.dataset_type != DatasetType::Forecasting {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Temporal splitting is only applicable for forecasting datasets"
                ));
            }
            else{
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
        } else {
            if self.dataset_type != DatasetType::Classification {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Random splitting is only applicable for classification datasets"
                ));
            }
            else {
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

                let bound_array = self.data.bind(py);
                let array = unsafe { bound_array.as_array() };
                let (rows, cols) = array.dim();
                
                let mut rows_vec: Vec<_> = array.outer_iter().map(|row| row.to_owned()).collect();
                
                // Shuffle rows
                let mut rng = thread_rng();
                rows_vec.shuffle(&mut rng);

                // Compute split indices
                let train_split = (train_prop * (rows as f64)).round() as usize;
                let val_split = (val_prop * (rows as f64)).round() as usize;
                let test_split = rows - train_split - val_split;

                let train_data = Array2::from_shape_vec(
                    (train_split, cols),
                    rows_vec[..train_split].iter().flat_map(|r| r.iter().cloned()).collect()
                ).unwrap();

                let val_data = Array2::from_shape_vec(
                    (val_split, cols),
                    rows_vec[train_split..train_split + val_split].iter().flat_map(|r| r.iter().cloned()).collect()
                ).unwrap();

                let test_data = Array2::from_shape_vec(
                    (test_split, cols),
                    rows_vec[train_split + val_split..].iter().flat_map(|r| r.iter().cloned()).collect()
                ).unwrap();

                Ok((
                    train_data.into_pyarray(py).into(),
                    val_data.into_pyarray(py).into(),
                    test_data.into_pyarray(py).into()
                ))
            }
        }
        
    }
}
