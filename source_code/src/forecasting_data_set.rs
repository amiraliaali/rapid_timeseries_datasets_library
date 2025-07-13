use crate::collecting::collect_forecasting;
use crate::abrev_types::ForecastingCollectResult;
use crate::data_abstract::ImputeStrategy;
use crate::preprocessing::{ downsample, impute, normalize, standardize };
use crate::splitting::{ split_forecasting, validate_props };
use crate::utils::{ get_split_views, get_split_views_by_prop_mut, get_split_views_mut };
use numpy::PyArray3;
use pyo3::prelude::*;

#[pyclass]
pub struct ForecastingDataSet {
    data: Py<PyArray3<f64>>,
    train_prop: f64,
    val_prop: f64,
    train_split_index: Option<usize>,
    val_split_index: Option<usize>,
}

#[pymethods]
impl ForecastingDataSet {
    #[new]
    fn new(
        _py: Python,
        data: Py<PyArray3<f64>>,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<Self> {
        validate_props(train_prop, val_prop, test_prop)?;

        Ok(Self {
            data,
            train_prop,
            val_prop,
            train_split_index: None,
            val_split_index: None,
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
        let (new_data, _) = downsample(_py, &self.data, None, factor)?;

        self.data = new_data;
        Ok(())
    }

    fn split(&mut self, _py: Python) -> PyResult<()> {
        let (train_split_index, val_split_index) = split_forecasting(
            _py,
            &self.data,
            self.train_prop,
            self.val_prop
        )?;

        self.train_split_index = Some(train_split_index);
        self.val_split_index = Some(val_split_index);
        Ok(())
    }

    fn normalize(&mut self, _py: Python) -> PyResult<()> {
        let (mut train_view, mut val_view, mut test_view) = get_split_views_mut(
            _py,
            &self.data,
            self.train_split_index,
            self.val_split_index
        )?;

        normalize(&mut train_view, &mut val_view, &mut test_view)?;
        Ok(())
    }

    fn standardize(&mut self, _py: Python) -> PyResult<()> {
        let (mut train_view, mut val_view, mut test_view) = get_split_views_mut(
            _py,
            &self.data,
            self.train_split_index,
            self.val_split_index
        )?;

        standardize(&mut train_view, &mut val_view, &mut test_view)?;
        Ok(())
    }

    fn collect(
        &self,
        _py: Python,
        past_window: usize,
        future_horizon: usize,
        stride: usize
    ) -> ForecastingCollectResult {
        let (train_view, val_view, test_view) = get_split_views(
            _py,
            &self.data,
            self.train_split_index,
            self.val_split_index
        )?;

        collect_forecasting(
            _py,
            &train_view,
            &val_view,
            &test_view,
            past_window,
            future_horizon,
            stride
        )
    }

    #[cfg(feature = "test_expose")]
    #[getter(data)]
    fn get_data<'py>(&self, py: Python<'py>) -> Py<PyArray3<f64>> {
        self.data.clone_ref(py)
    }

    #[cfg(feature = "test_expose")]
    #[getter(train_split_index)]
    fn get_train_split_index(&self) -> Option<usize> {
        self.train_split_index
    }

    #[cfg(feature = "test_expose")]
    #[getter(val_split_index)]
    fn get_val_split_index(&self) -> Option<usize> {
        self.val_split_index
    }
}
