#[cfg(test)]
mod tests {
    use std::vec;

    use pyo3::prelude::*;
    use ndarray::{ Array3 };
    use rust_time_series::preprocessing::{
        compute_feature_statistics,
        compute_standardization_per_column,
        compute_min_max,
        compute_min_max_normalization,
        standardize,
    };

    #[test]
    fn test_compute_feature_statistics() {
        Python::with_gil(|_py| {
            let row_one_data = vec![1.0, 4.0, 7.0];
            let row_two_data = vec![2.0, 5.0, 8.0];
            let row_three_data = vec![3.0, 6.0, 9.0];

            let data = Array3::<f64>
                ::from_shape_vec(
                    (1, 3, 3),
                    row_one_data.into_iter().chain(row_two_data).chain(row_three_data).collect()
                )
                .unwrap();

            let data_view = data.view();

            let (means, stds) = compute_feature_statistics(&data_view);

            assert_eq!(means, vec![2.0, 5.0, 8.0]);

            for std in stds {
                assert!(std < 0.82 && std > 0.81); // std should be around 0.8165
            }
        });
    }

    #[test]
    fn test_compute_feature_statistics_empty() {
        Python::with_gil(|_py| {
            let data = Array3::<f64>::zeros((0, 0, 0));
            let data_view = data.view();
            let (means, stds) = compute_feature_statistics(&data_view);
            assert!(means.is_empty());
            assert!(stds.is_empty());
        });
    }

    #[test]
    fn test_compute_standardization_per_column() {
        Python::with_gil(|_py| {
            let row_one_data = vec![1.0, 4.0, 7.0];
            let row_two_data = vec![2.0, 5.0, 8.0];
            let row_three_data = vec![3.0, 6.0, 9.0];

            let mut data = Array3::<f64>
                ::from_shape_vec(
                    (1, 3, 3),
                    row_one_data.into_iter().chain(row_two_data).chain(row_three_data).collect()
                )
                .unwrap();

            let (means, stds) = compute_feature_statistics(&data.view());
            compute_standardization_per_column(&mut data.view_mut(), &means, &stds);

            let data_view = data.view();
            let num_features = data_view.shape()[2];

            for feature_idx in 0..num_features {
                let feature_data = data_view.slice(ndarray::s![.., .., feature_idx]);
                let mean = feature_data.mean().unwrap();
                let std = feature_data.std(0.0);

                assert!(mean == 0.0);
                assert!((std - 1.0).abs() < 1e-8);
            }
        });
    }

    #[test]
    fn test_compute_min_max() {
        Python::with_gil(|_py| {
            let row_one_data = vec![1.0, 4.0, 7.0];
            let row_two_data = vec![2.0, 5.0, 8.0];
            let row_three_data = vec![3.0, 6.0, 9.0];

            let data = Array3::<f64>
                ::from_shape_vec(
                    (1, 3, 3),
                    row_one_data.into_iter().chain(row_two_data).chain(row_three_data).collect()
                )
                .unwrap();

            let data_view = data.view();

            let (mins, maxs) = compute_min_max(&data_view);

            assert_eq!(mins, vec![1.0, 4.0, 7.0]);
            assert_eq!(maxs, vec![3.0, 6.0, 9.0]);
        });
    }

    #[test]
    fn test_compute_min_max_normalization() {
        Python::with_gil(|_py| {
            let row_one_data = vec![1.0, 4.0, 7.0];
            let row_two_data = vec![2.0, 5.0, 8.0];
            let row_three_data = vec![3.0, 6.0, 9.0];

            let mut data = Array3::<f64>
                ::from_shape_vec(
                    (1, 3, 3),
                    row_one_data.into_iter().chain(row_two_data).chain(row_three_data).collect()
                )
                .unwrap();

            let (mins, maxs) = compute_min_max(&data.view());
            compute_min_max_normalization(&mut data.view_mut(), &mins, &maxs);

            let data_view = data.view();
            for val in data_view.iter() {
                assert!(*val >= 0.0 && *val <= 1.0, "Value not in [0,1]: {}", val);
            }
        });
    }

    #[test]
    fn test_standardize() {
        Python::with_gil(|_py| {
            let train_data_raw = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
            let val_data_raw = vec![4.0, 7.0, 5.0, 8.0, 6.0, 9.0];
            let test_data_raw = vec![7.0, 10.0, 8.0, 11.0, 9.0, 12.0];

            let mut train_data = Array3::from_shape_vec((1, 3, 2), train_data_raw).unwrap();
            let mut val_data = Array3::from_shape_vec((1, 3, 2), val_data_raw).unwrap();
            let mut test_data = Array3::from_shape_vec((1, 3, 2), test_data_raw).unwrap();

            let result = standardize(
                &mut train_data.view_mut(),
                &mut val_data.view_mut(),
                &mut test_data.view_mut()
            );

            assert!(result.is_ok());

            let (means, stds) = compute_feature_statistics(&train_data.view());

            for (_i, &mean) in means.iter().enumerate() {
                assert!(mean == 0.0);
            }

            for (_i, &std) in stds.iter().enumerate() {
                assert!((std - 1.0).abs() < 1e-8);
            }
        });
    }
}
