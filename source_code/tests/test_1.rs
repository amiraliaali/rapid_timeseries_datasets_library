#[cfg(test)]
mod tests {
    use pyo3::prelude::*;

    #[test]
    fn test_py_function() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let module = py.import("rust_time_series").unwrap();
            assert_eq!(2, 3);
        });
    }
}
