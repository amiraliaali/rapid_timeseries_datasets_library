pub mod data_abstract;
pub mod functionalities;
pub mod splitting;

use data_abstract::{DatasetType, ImputeStrategy, SplittingStrategy, BaseDataSet};
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