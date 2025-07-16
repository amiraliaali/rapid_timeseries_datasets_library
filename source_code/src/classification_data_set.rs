use crate::abrev_types::ClassificationCollectResult;
use crate::collecting::collect_classification;
use crate::data_abstract::{ ImputeStrategy, SplittingStrategy };
use crate::preprocessing::{ downsample, impute, normalize, standardize };
use crate::splitting::{ split_classification, validate_props };
use crate::utils::{ bind_array_1d, bind_array_3d, check_arrays_set, get_split_views_by_prop_mut };
use numpy::{ ndarray::*, PyArray1, PyArray3 };
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[cfg(feature = "test_expose")]
use numpy::IntoPyArray;

#[pyclass]
pub struct ClassificationDataSet {
    data: Py<PyArray3<f64>>,
    labels: Py<PyArray1<f64>>,
    train_prop: f64,
    val_prop: f64,
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
    pub fn new(
        _py: Python,
        data: Py<PyArray3<f64>>,
        labels: Py<PyArray1<f64>>,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<Self> {
        validate_props(train_prop, val_prop, test_prop)?;

        let data_view = bind_array_3d(_py, &data);
        let labels_view = bind_array_1d(_py, &labels);

        let instances = data_view.dim().0;
        if labels_view.len() != instances {
            return Err(
                PyValueError::new_err("Labels length must match the number of instances in data")
            );
        }

        Ok(Self {
            data,
            labels,
            train_prop,
            val_prop,
            train_data: None,
            train_labels: None,
            val_data: None,
            val_labels: None,
            test_data: None,
            test_labels: None,
        })
    }

    fn impute(&mut self, _py: Python, impute_strategy: ImputeStrategy) -> PyResult<()> {
        let (mut train_view, mut val_view, mut test_view) = get_split_views_by_prop_mut(
            _py,
            &self.data,
            self.train_prop,
            self.val_prop
        )?;

        impute(_py, &mut train_view, &mut val_view, &mut test_view, impute_strategy)?;
        Ok(())
    }

    fn downsample(&mut self, _py: Python, factor: usize) -> PyResult<()> {
        self.data = downsample(_py, &self.data, factor)?;
        Ok(())
    }

    fn split(&mut self, _py: Python, splitting_strategy: SplittingStrategy) -> PyResult<()> {
        let ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels)) =
            split_classification(
                _py,
                &self.data,
                &self.labels,
                splitting_strategy,
                self.train_prop,
                self.val_prop
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

    #[cfg(feature = "test_expose")]
    #[getter(data)]
    fn get_data<'py>(&self, _py: Python<'py>) -> Py<PyArray3<f64>> {
        self.data.clone_ref(_py)
    }

    #[cfg(feature = "test_expose")]
    #[getter(labels)]
    fn get_labels<'py>(&self, _py: Python<'py>) -> Py<PyArray1<f64>> {
        self.labels.clone_ref(_py)
    }

    #[cfg(feature = "test_expose")]
    #[getter(train_data)]
    fn get_train_data<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyArray3<f64>>>> {
        Ok(self.train_data.as_ref().map(|arr| arr.to_owned().into_pyarray(py).into()))
    }

    #[cfg(feature = "test_expose")]
    #[getter(train_labels)]
    fn get_train_labels<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyArray1<f64>>>> {
        Ok(self.train_labels.as_ref().map(|arr| arr.to_owned().into_pyarray(py).into()))
    }

    #[cfg(feature = "test_expose")]
    #[getter(val_data)]
    fn get_val_data<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyArray3<f64>>>> {
        Ok(self.val_data.as_ref().map(|arr| arr.to_owned().into_pyarray(py).into()))
    }

    #[cfg(feature = "test_expose")]
    #[getter(val_labels)]
    fn get_val_labels<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyArray1<f64>>>> {
        Ok(self.val_labels.as_ref().map(|arr| arr.to_owned().into_pyarray(py).into()))
    }

    #[cfg(feature = "test_expose")]
    #[getter(test_data)]
    fn get_test_data<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyArray3<f64>>>> {
        Ok(self.test_data.as_ref().map(|arr| arr.to_owned().into_pyarray(py).into()))
    }

    #[cfg(feature = "test_expose")]
    #[getter(test_labels)]
    fn get_test_labels<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyArray1<f64>>>> {
        Ok(self.test_labels.as_ref().map(|arr| arr.to_owned().into_pyarray(py).into()))
    }
}
