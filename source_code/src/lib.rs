pub mod data_abstract;
pub mod collecting;
pub mod splitting;
pub mod abrev_types;
pub mod forecasting_data_set;
pub mod classification_data_set;
pub mod preprocessing;
pub mod utils;

use crate::{
    classification_data_set::ClassificationDataSet,
    forecasting_data_set::ForecastingDataSet,
};
use data_abstract::{ ImputeStrategy, SplittingStrategy };
use pyo3::prelude::*;

#[pymodule]
fn rust_time_series(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<ForecastingDataSet>()?;
    m.add_class::<ClassificationDataSet>()?;
    m.add_class::<ImputeStrategy>()?;
    m.add_class::<SplittingStrategy>()?;
    Ok(())
}
