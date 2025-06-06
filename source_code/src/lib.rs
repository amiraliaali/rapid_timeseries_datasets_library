pub mod py_definitions;
pub mod forecasting_dataset;
pub mod classification_dataset;
pub mod data_abstract;

use data_abstract::BaseDataSet;
use py_definitions::{DatasetType, ImputeStrategy, SplittingStrategy};
use pyo3::prelude::*;

#[pymodule]
fn rust_time_series(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    pyo3_log::init();

    m.add_class::<BaseDataSet>()?;
    m.add_class::<DatasetType>()?;
    m.add_class::<ImputeStrategy>()?;
    m.add_class::<SplittingStrategy>()?;
    Ok(())
}