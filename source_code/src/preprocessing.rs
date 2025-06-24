use ndarray::{ ArrayBase, ArrayView3, DataMut, Dim };
use pyo3::prelude::*;

fn compute_feature_statistics(data_view: &ArrayView3<f64>) -> (Vec<f64>, Vec<f64>) {
    let num_features = data_view.shape()[2];
    let mut means = Vec::with_capacity(num_features);
    let mut stds = Vec::with_capacity(num_features);

    for feature_idx in 0..num_features {
        // Get a view of the current feature across all instances and timestamps
        let feature_data = data_view.slice(ndarray::s![.., .., feature_idx]);

        // Compute statistics for this feature
        let mean = feature_data.mean().unwrap();
        let std = feature_data.std(0.0);

        // Handle case where std is 0 (constant feature)
        let std = if std == 0.0 { 1.0 } else { std };

        means.push(mean);
        stds.push(std);
    }

    (means, stds)
}

fn compute_normalization_per_column<S>(
    data_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    means: &[f64],
    stds: &[f64]
)
    where S: DataMut<Elem = f64>
{
    let num_features = data_view.shape()[2];

    // Normalize each feature column separately
    for feature_idx in 0..num_features {
        let mut feature_column = data_view.slice_mut(ndarray::s![.., .., feature_idx]);
        feature_column -= means[feature_idx];
        feature_column /= stds[feature_idx];
    }
}

pub fn normalize<S>(
    train_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    val_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    test_view: &mut ArrayBase<S, Dim<[usize; 3]>>
) -> PyResult<()>
    where S: DataMut<Elem = f64>
{
    // Compute mean and std for each feature column from training data
    let train_view_immutable = train_view.view();
    let (means, stds) = compute_feature_statistics(&train_view_immutable);

    // Apply normalization to all splits using the training statistics
    compute_normalization_per_column(train_view, &means, &stds);
    compute_normalization_per_column(val_view, &means, &stds);
    compute_normalization_per_column(test_view, &means, &stds);

    Ok(())
}

pub fn standardize<S>(
    train_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    val_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    test_view: &mut ArrayBase<S, Dim<[usize; 3]>>
) -> PyResult<()>
    where S: DataMut<Elem = f64>
{
    // TODO: Standardize the data in-place

    Ok(())
}
