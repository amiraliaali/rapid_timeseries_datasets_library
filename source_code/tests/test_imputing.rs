#[cfg(test)]
mod tests {
    use std::vec;

    use pyo3::prelude::*;
    use ndarray::{Array3 };
    use rust_time_series::preprocessing::{
        impute,impute_view,
    };
    use rust_time_series::data_abstract::{ImputeStrategy};

    #[test]
    fn test_impute_view_basic() {
        Python::with_gil(|py| {
            let row_one_data = vec![1.0, 4.0, 7.0];
            let row_two_data = vec![2.0, 5.0, 8.0];
            let row_three_data = vec![3.0, 6.0, 9.0];

            let mut data = Array3::<f64>::from_shape_vec(
                (1, 3, 3),
                row_one_data.into_iter().chain(row_two_data).chain(row_three_data).collect()
            ).unwrap();

            // let data_view = data.view_mut();
            let strategy = ImputeStrategy::Mean;
            impute_view(py, &strategy, &mut data.view_mut());
        });
    }
}