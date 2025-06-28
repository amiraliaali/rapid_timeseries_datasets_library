use crate::collecting::collect_forecasting;
use crate::abrev_types::ForecastingCollectResult;
use crate::preprocessing::{ normalize, standardize };
use crate::splitting::split_forecasting;
use crate::utils::{ get_split_views, get_split_views_mut };
use numpy::PyArray3;
use pyo3::prelude::*;

#[pyclass]
pub struct ForecastingDataSet {
    data: Py<PyArray3<f64>>,
    train_split_index: Option<usize>,
    val_split_index: Option<usize>,
}

#[pymethods]
impl ForecastingDataSet {
    #[new]
    fn new(_py: Python, data: Py<PyArray3<f64>>) -> PyResult<Self> {
        Ok(Self {
            data,
            train_split_index: None,
            val_split_index: None,
        })
    }

    fn impute(&mut self, _py: Python) -> PyResult<()> {
        // TODO: Pass the self.data to an imputation function!
        Ok(())
    }

    fn downsample(&mut self, _py: Python, factor: usize) -> PyResult<()> {
        // TODO: Pass the self.data to a downsampling function!
        Ok(())
    }

    fn split(
        &mut self,
        _py: Python,
        train_prop: f64,
        val_prop: f64,
        test_prop: f64
    ) -> PyResult<()> {
        let indices = split_forecasting(_py, &self.data, train_prop, val_prop, test_prop)?;
        self.train_split_index = Some(indices.0);
        self.val_split_index = Some(indices.1);
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
}
