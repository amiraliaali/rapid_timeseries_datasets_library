use crate::py_definitions::{
    DatasetType,
    SplittingStrategy,
    ImputeStrategy,
};
use numpy::{ ndarray::{s, Axis}, IntoPyArray, PyArray2, PyArrayMethods };
use log::{ info, debug };
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;



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
pub struct ClassificationDataSet {
    features: Py<PyArray2<f64>>,
    labels: Vec<String>,
    dataset_type: DatasetType
}

#[pymethods]
impl ClassificationDataSet {
    #[new]
    pub fn new(features: Py<PyArray2<f64>>, labels: Vec<String>, dataset_type: DatasetType, py: Python) -> PyResult<Self> {
        debug!("Creating ClassificationDataset instance with dataset type: {:?}", dataset_type);
        
        let bound_array = features.bind(py);
        let array = unsafe { bound_array.as_array() };
        let (rows, _) = array.dim();

        if rows != labels.len() {
            return Err(PyValueError::new_err("Number of rows in features does not match number of labels"));
        }

        Ok(ClassificationDataSet {
            features: features,
            labels,
            dataset_type
        })
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
    ) -> PyResult<()> {
        debug!("Splitting dataset with strategy: {:?}, train_prop: {}, val_prop: {}, test_prop: {}", 
            split_strategy, train_prop, val_prop, test_prop);

        Ok(())
    }

    fn len(&self, py: Python) -> PyResult<usize> {
        Ok(self.labels.len())
    }

    fn get(&self, index: usize, py: Python) -> PyResult<Option<ClassificationSample>> {
        return Err(PyValueError::new_err("get method not implemented yet"));
    }

}