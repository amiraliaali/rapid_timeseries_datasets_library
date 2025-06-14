use crate::data_abstract::{DatasetType, SplittingStrategy};
use numpy::{PyArray2, IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use numpy::ndarray::{Array2, Axis};
use log::debug;
use rand::seq::SliceRandom;
use rand::thread_rng;


pub fn split(
    self_dataset_type: &DatasetType,
    self_data: &Py<PyArray2<f64>>,
    py: Python,
    split_strategy: SplittingStrategy,
    train_prop: f64,
    val_prop: f64,
    test_prop: f64
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    if split_strategy == SplittingStrategy::Temporal {
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

        let bound_array = self_data.bind(py);
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
        
    } else {
        if *self_dataset_type != DatasetType::Classification {
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

            let bound_array = self_data.bind(py);
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
