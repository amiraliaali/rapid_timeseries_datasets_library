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
pub struct ClassificationSample {
    #[pyo3(get)]
    pub id: String,

    #[pyo3(get)]
    pub features: Py<PyArray2<f64>>,

    #[pyo3(get)]
    pub label: String,
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
pub struct ClassificationDataset {
    samples: Vec<ClassificationSample>,
    dataset_type: DatasetType
}

#[pymethods]
impl ClassificationDataset {
    #[new]
    pub fn new(samples: Vec<ClassificationSample>, dataset_type: DatasetType) -> Self {
        debug!("Creating ClassificationDataset instance with dataset type: {:?}", dataset_type);
        ClassificationDataset { samples, dataset_type }
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
        debug!("Splitting dataset with strategy: {:?}, train_prop: {}, val_prop: {}, test_prop: {}", 
            split_strategy, train_prop, val_prop, test_prop);

        Ok(())
    }

    fn len(&self, py: Python) -> PyResult<usize> {
        Ok(self.samples.len())
    }

    fn get(&self, index: usize, py: Python) -> PyResult<Option<ClassificationSample>> {
        if index < self.samples.len() {
            Ok(Some(self.samples[index].clone()))
        } else {
            Ok(None)
        }
    }

}

    