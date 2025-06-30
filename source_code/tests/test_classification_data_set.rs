#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use numpy::{PyArray1, PyArray3, IntoPyArray};
    use ndarray::Array;
    use pyo3::types::PyTuple;

    #[test]
    fn test_importing_module() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = py.import("rust_time_series");
            assert!(module.is_ok());
        });
    }
}
