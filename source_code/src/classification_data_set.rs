use crate::abrev_types::ClassificationCollectResult;
use crate::collecting::collect_classification;
use crate::data_abstract::SplittingStrategy;
use crate::preprocessing::{ normalize, standardize, 
    downsampling_classification };
use crate::splitting::split_classification;
use crate::utils::{ bind_array_1d, bind_array_3d, check_arrays_set };
use numpy::{ ndarray::*, PyArray1, PyArray3 };
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass]
pub struct ClassificationDataSet {
    data: Py<PyArray3<f64>>,
    labels: Py<PyArray1<f64>>,
    train_data: Option<Array3<f64>>,
    train_labels: Option<Array1<f64>>,
    val_data: Option<Array3<f64>>,
    val_labels: Option<Array1<f64>>,
    test_data: Option<Array3<f64>>,
    test_labels: Option<Array1<f64>>,
}

#[pymethods]
impl ClassificationDataSet {
    #[new]
    pub fn new(_py: Python, data: Py<PyArray3<f64>>, labels: Py<PyArray1<f64>>) -> PyResult<Self> {
        let data_view = bind_array_3d(_py, &data);
        let labels_view = bind_array_1d(_py, &labels);

        let (_instances, timesteps, _features) = data_view.dim();
        if labels_view.len() != timesteps {
            return Err(
                PyValueError::new_err("Labels length must match the number of timesteps in data")
            );
        }

        Ok(Self {
            data,
            labels,
            train_data: None,
            train_labels: None,
            val_data: None,
            val_labels: None,
            test_data: None,
            test_labels: None,
        })
    }

    fn impute(&mut self, _py: Python) -> PyResult<()> {
        // TODO: Pass the self.data (/self.labels) to an imputation function!
        Ok(())
    }

    fn downsample(&mut self, _py: Python, factor: usize) -> PyResult<()> {
        let (new_data, new_labels) = downsampling_classification(
            _py,
            &self.data,
            &self.labels,
            factor
        )?;

        self.data = new_data;
        self.labels = new_labels;

        Ok(())
    }

    fn split(
        &mut self,
        _py: Python,
        splitting_strategy: SplittingStrategy,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<()> {
        let ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels)) =
            split_classification(
                _py,
                &self.data,
                &self.labels,
                splitting_strategy,
                train_prop,
                val_prop,
                test_prop
            )?;
        self.train_data = Some(train_data);
        self.train_labels = Some(train_labels);
        self.val_data = Some(val_data);
        self.val_labels = Some(val_labels);
        self.test_data = Some(test_data);
        self.test_labels = Some(test_labels);

        Ok(())
    }

    fn normalize(&mut self, _py: Python) -> PyResult<()> {
        check_arrays_set(&self.train_data, &self.val_data, &self.test_data)?;

        normalize(
            &mut self.train_data.as_mut().unwrap(),
            &mut self.val_data.as_mut().unwrap(),
            &mut self.test_data.as_mut().unwrap()
        )?;

        Ok(())
    }

    fn standardize(&mut self, _py: Python) -> PyResult<()> {
        check_arrays_set(&self.train_data, &self.val_data, &self.test_data)?;

        standardize(
            &mut self.train_data.as_mut().unwrap(),
            &mut self.val_data.as_mut().unwrap(),
            &mut self.test_data.as_mut().unwrap()
        )?;

        Ok(())
    }

    fn collect(&mut self, _py: Python) -> ClassificationCollectResult {
        check_arrays_set(&self.train_data, &self.val_data, &self.test_data)?;

        collect_classification(
            _py,
            self.train_data.take().unwrap(),
            self.train_labels.take().unwrap(),
            self.val_data.take().unwrap(),
            self.val_labels.take().unwrap(),
            self.test_data.take().unwrap(),
            self.test_labels.take().unwrap()
        )
    }
}
