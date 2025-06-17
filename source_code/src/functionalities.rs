use crate::data_abstract::{
    BaseDataSet, ClassificationSample, DatasetType, ForecastingSample, ImputeStrategy, SampleType, SplittingStrategy
};
use crate::splitting::split;
use numpy::PyArray1;
use numpy::{ ndarray::{s}, IntoPyArray, PyArray2,PyArray3, PyArrayMethods };
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use log::{ debug };


#[pymethods]
impl BaseDataSet {
    #[new]
    pub fn new_forecasting(py: Python,data: Py<PyArray3<f64>>, past_window: usize, future_horizon: usize, stride: usize) -> PyResult<Self> {
        if past_window == 0 || future_horizon == 0 || stride == 0 {
            return Err(PyValueError::new_err("past_window, future_horizon, and stride must be greater than 0"));
        }
        let bound_array = data.bind(py);
        let mut data_array = unsafe { bound_array.as_array_mut() };
        debug!("Creating RustTimeSeries instance with dataset type: {:?}", DatasetType::Forecasting);
        let (instances, timesteps, features) = data_array.dim();
        
        // print to consol (not debug)
        println!("Data dimensions: instances={}, timesteps={}, features={}", instances, timesteps, features);
        
        // Apply sliding window logic
        println!("Applying sliding window with past_window={}, future_horizon={}, stride={}", past_window, future_horizon, stride);
        // make a new 3d array
        let windows_per_instance = ((timesteps - past_window - future_horizon) / stride) + 1;
        let window_count = windows_per_instance * instances;
        let mut x_windows = ndarray::Array3::<f64>::zeros((window_count, past_window, features));
        let mut y_windows = ndarray::Array3::<f64>::zeros((window_count, future_horizon, features));
        println!("Total windows to process: {}", window_count);
        let mut window_index = 0 as usize;
        for instance in 0..instances {
            println!("Processing instance {}/{}", instance + 1, instances);
            let start = 0;
            let end = timesteps - future_horizon;
            for i in (start..end).step_by(stride) {
                println!("Processing window starting at index {}", i);
                let x_start = i;
                let x_end = i + past_window;
                let y_start = i + past_window;
                let y_end = y_start + future_horizon;

                if x_end > timesteps || y_end > timesteps {
                    continue; // Skip if the window exceeds the bounds
                }
                println!("Processing instance {}, window start: {}, x_start: {}, x_end: {}, y_start: {}, y_end: {}", instance, i, x_start, x_end, y_start, y_end);
                let x_slice = data_array.slice(s![instance, x_start..x_end, ..]).to_owned();
                let y_slice = data_array.slice(s![instance, y_start..y_end, ..]).to_owned();

                println!("x_slice shape: {:?}", x_slice.shape());
                x_windows.slice_mut(s![window_index, .., ..]).assign(&x_slice);
                y_windows.slice_mut(s![window_index, .., ..]).assign(&y_slice);
                window_index += 1;
                println!("Window index incremented to {}", window_index);
            }
            
        }

        println!("Creating RustTimeSeries instance with dataset type: {:?}", DatasetType::Forecasting);
        Ok(BaseDataSet {
            data,
            labels: None,
            dataset_type: DatasetType::Forecasting,
            past_window,
            future_horizon,
            stride,
            x_windows,
            y_windows: Some(y_windows),
        })
    }

    #[staticmethod]
    pub fn new_classification(py: Python,data: Py<PyArray3<f64>>, past_window: usize, future_horizon: usize, stride: usize, labels: Py<PyArray1<f64>>) -> PyResult<Self> {
        debug!("Creating ClassificationDataset instance with dataset type: {:?}", DatasetType::Classification);
        
        let bound_array = data.bind(py);
        let array = unsafe { bound_array.as_array() };
        let (rows, _,_) = array.dim();

        let bound_labels = labels.bind(py);
        let labels_array = unsafe { bound_labels.as_array() };

        if rows != labels_array.len() {
            return Err(PyValueError::new_err("Number of rows in features does not match number of labels"));
        }

        let (instances, timesteps, features) = array.dim();
        
        
        // Apply sliding window logic
        
        // make a new 3d array
        let window_count = (((timesteps - past_window - future_horizon) / stride) + 1) * instances;
        let mut x_windows = ndarray::Array3::<f64>::zeros((window_count, past_window, features));
        let mut y_windows = ndarray::Array1::<f64>::zeros(window_count);
        
        for instance in 0..instances {
            let start = 0;
            let end = timesteps;
            for i in (start..end).step_by(stride) {
                let x_start = i;
                let x_end = i + past_window;
                
                // ignore future_horizon for classification since it is zero
                if x_end > timesteps {
                    continue; // Skip if the window exceeds the bounds
                }
                let x_slice = array.slice(s![instance, x_start..x_end, ..]).to_owned();
                
                let window_index = instance * ((end - start) / stride) + (i - start) / stride;
                x_windows.slice_mut(s![window_index, .., ..]).assign(&x_slice);
                
                // Assign the label for the current instance
                if instance < labels_array.len() {
                    y_windows[window_index] = labels_array[instance];
                } else {
                    return Err(PyValueError::new_err("Labels array length does not match data instances"));
                }
            }
        }

        Ok(BaseDataSet {
            data,
            labels: Some(y_windows),
            dataset_type: DatasetType::Forecasting,
            past_window,
            future_horizon,
            stride,
            x_windows,
            y_windows: None,
        })
    }

    fn len(&self) -> PyResult<usize> {
        if self.dataset_type == DatasetType::Classification {
            if let Some(labels) = &self.labels {
                return Ok(labels.len());
            } else {
                return Err(PyValueError::new_err("Labels are missing for classification dataset"));
            }
        } else{
            if let Some(y_windows) = &self.y_windows {
                return Ok(y_windows.len());
            } else {
                return Err(PyValueError::new_err("y_windows are missing for forecasting dataset"));
            }
        }
    }

    fn get(&self, index: usize, py: Python) -> PyResult<Option<SampleType>>{
        if self.dataset_type == DatasetType::Classification {
            let x_piece = self.x_windows.slice(s![index, .., ..]).to_owned();
            let x_py_array: Py<PyArray2<f64>> = x_piece.into_pyarray(py).into();
            if let Some(labels) = &self.labels {
                let y_value = labels[index];
                return Ok(Some(SampleType::Classification(ClassificationSample {
                id: format!("sample_{}", index),
                past: x_py_array,
                label: y_value,
            })));
            } else {
                return Err(PyValueError::new_err("Labels are missing for classification dataset"));
            }
        } else {
            if let Some(y_windows) = &self.y_windows {
                if index >= y_windows.len() {
                    return Ok(None); // Index out of bounds
                }
                let x_piece = self.x_windows.slice(s![index, .., ..]).to_owned();
                let y_piece = y_windows.slice(s![index, .., ..]).to_owned();
                
                // Convert x_piece and y_piece to Py<PyArray2<f64>>
                let x_pyarray: Py<PyArray2<f64>> = x_piece.into_pyarray(py).into();
                let y_pyarray: Py<PyArray2<f64>> = y_piece.into_pyarray(py).into();
                
                return Ok(Some(SampleType::Forecasting(ForecastingSample {
                id: format!("sample_{}", index),
                past: x_pyarray,
                future: y_pyarray,
            })));
            } else {
                return Err(PyValueError::new_err("y_windows are missing for forecasting dataset"));
            }
        }
    }
    fn split(
        &self,
        py: Python,
        split_strategy: SplittingStrategy,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray3<f64>>, Py<PyArray3<f64>>)>{
        split(&self.dataset_type, &self.data, py, split_strategy, train_prop, val_prop, test_prop)
    }
}
