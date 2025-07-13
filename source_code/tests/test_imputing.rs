#[cfg(test)]
mod tests {
    use std::vec;

    use pyo3::prelude::*;
    use ndarray::{ Array3, array };
    use rust_time_series::preprocessing::{
        impute,
        impute_view,
        impute_mean,
        impute_median,
        impute_forward_fill,
        impute_backward_fill,
    };
    use rust_time_series::data_abstract::{ ImputeStrategy };

    #[test]
    fn test_impute_view_basic() {
        Python::with_gil(|py| {
            let row_one_data = vec![1.0, 4.0, 7.0];
            let row_two_data = vec![2.0, 5.0, 8.0];
            let row_three_data = vec![3.0, 6.0, 9.0];

            let mut data = Array3::<f64>
                ::from_shape_vec(
                    (1, 3, 3),
                    row_one_data.into_iter().chain(row_two_data).chain(row_three_data).collect()
                )
                .unwrap();

            // let data_view = data.view_mut();
            let strategy = ImputeStrategy::Mean;
            impute_view(py, &strategy, &mut data.view_mut());
        });
    }

    // Here we test the impute function directly. Taking the mean of 2, 4, and 6 should yield 4.0.
    // The NaN values should be replaced with this mean.
    #[test]
    fn test_impute_mean() {
        let mut data = array![f64::NAN, 2.0, f64::NAN, 4.0, 6.0]
            .into_dyn()
            .into_shape((5,))
            .unwrap();
        let mut view = data.view_mut();
        impute_mean(&mut view);

        let expected = array![4.0, 2.0, 4.0, 4.0, 6.0];
        assert_eq!(view.to_vec(), expected.to_vec());
    }

    // Here we test the impute function directly. Taking the median of 2, 20, and 6 should yield 6.0.
    // The NaN values should be replaced with this median.
    #[test]
    fn test_impute_median() {
        let mut data = array![f64::NAN, 2.0, f64::NAN, 20.0, 6.0]
            .into_dyn()
            .into_shape((5,))
            .unwrap();
        let mut view = data.view_mut();
        impute_median(&mut view);

        let expected = array![6.0, 2.0, 6.0, 20.0, 6.0];
        assert_eq!(view.to_vec(), expected.to_vec());
    }

    // This time we test the impute function with an even number of elements.
    // The median of 2, 4, 6, and 8 should be 5.0.
    // The NaN values should be replaced with this median.
    #[test]
    fn test_impute_median_even_length() {
        let mut data = array![f64::NAN, 2.0, f64::NAN, 4.0, 6.0, 8.0]
            .into_dyn()
            .into_shape((6,))
            .unwrap();
        let mut view = data.view_mut();
        impute_median(&mut view);

        let expected = array![5.0, 2.0, 5.0, 4.0, 6.0, 8.0];
        assert_eq!(view.to_vec(), expected.to_vec());
    }

    #[test]
    fn test_impute_forward_fill() {
        let mut data = array![1.0, 2.0, f64::NAN, f64::NAN, 6.0, f64::NAN]
            .into_dyn()
            .into_shape((6,))
            .unwrap();
        let mut view = data.view_mut();
        impute_forward_fill(&mut view);

        let expected = array![1.0, 2.0, 2.0, 2.0, 6.0, 6.0];
        assert_eq!(view.to_vec(), expected.to_vec());
    }

    #[test]
    fn test_impute_backward_fill() {
        let mut data = array![1.0, 2.0, f64::NAN, f64::NAN, 6.0, f64::NAN, 7.0]
            .into_dyn()
            .into_shape((7,))
            .unwrap();
        let mut view = data.view_mut();
        impute_backward_fill(&mut view);

        let expected = array![1.0, 2.0, 6.0, 6.0, 6.0, 7.0, 7.0];
        assert_eq!(view.to_vec(), expected.to_vec());
    }
}
