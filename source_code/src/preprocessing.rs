use ndarray::{ s, ArrayBase, ArrayView3, DataMut, Dim };
use numpy::{ PyArray1, PyArray3, IntoPyArray };
use crate::utils::{ bind_array_1d, bind_array_3d, bind_array_mut_3d };
use pyo3::prelude::*;
use ndarray::{ Array3, Array1 };
use pyo3::{ Python, PyResult, PyErr };
use pyo3::exceptions::PyValueError;
use crate::data_abstract::{ ImputeStrategy };

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
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

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn compute_standardization_per_column<S>(
    data_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    means: &[f64],
    stds: &[f64]
)
    where S: DataMut<Elem = f64>
{
    let num_features = data_view.shape()[2];

    // Standardize each feature column separately
    for feature_idx in 0..num_features {
        let mut feature_column = data_view.slice_mut(ndarray::s![.., .., feature_idx]);
        feature_column -= means[feature_idx];
        feature_column /= stds[feature_idx];
    }
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn compute_min_max(data_view: &ArrayView3<f64>) -> (Vec<f64>, Vec<f64>) {
    let num_features = data_view.shape()[2];
    let mut mins = Vec::with_capacity(num_features);
    let mut maxs = Vec::with_capacity(num_features);

    for feature_idx in 0..num_features {
        let feature_data = data_view.slice(ndarray::s![.., .., feature_idx]);

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for &value in feature_data.iter() {
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }

        mins.push(min);
        maxs.push(max);
    }

    (mins, maxs)
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn compute_min_max_normalization<S>(
    data_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    mins: &[f64],
    maxs: &[f64]
)
    where S: DataMut<Elem = f64>
{
    let num_features = data_view.shape()[2];

    for feature_idx in 0..num_features {
        let mut feature_column = data_view.slice_mut(ndarray::s![.., .., feature_idx]);
        let range = maxs[feature_idx] - mins[feature_idx];
        // to avoid division by zero, we set range to 1.0 if it is 0.0
        let range = if range == 0.0 { 1.0 } else { range };
        feature_column -= mins[feature_idx];
        feature_column /= range;
    }
}

pub fn standardize<S>(
    train_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    val_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    test_view: &mut ArrayBase<S, Dim<[usize; 3]>>
) -> PyResult<()>
    where S: DataMut<Elem = f64>
{
    // Compute mean and std for each feature column from training data
    let train_view_immutable = train_view.view();
    let (means, stds) = compute_feature_statistics(&train_view_immutable);

    // Apply standardization to all splits using the training statistics
    compute_standardization_per_column(train_view, &means, &stds);
    compute_standardization_per_column(val_view, &means, &stds);
    compute_standardization_per_column(test_view, &means, &stds);

    Ok(())
}

pub fn normalize<S>(
    train_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    val_view: &mut ArrayBase<S, Dim<[usize; 3]>>,
    test_view: &mut ArrayBase<S, Dim<[usize; 3]>>
) -> PyResult<()>
    where S: DataMut<Elem = f64>
{
    let train_view_immutable = train_view.view();
    let (mins, maxs) = compute_min_max(&train_view_immutable);

    compute_min_max_normalization(train_view, &mins, &maxs);
    compute_min_max_normalization(val_view, &mins, &maxs);
    compute_min_max_normalization(test_view, &mins, &maxs);

    Ok(())
}

fn downsample_data(
    _py: Python,
    instances: usize,
    old_timesteps: usize,
    new_timesteps: usize,
    features: usize,
    factor: usize,
    data_view: &ArrayView3<f64>
) -> Py<PyArray3<f64>> {
    let mut new_data = Array3::<f64>::zeros((instances, new_timesteps, features));

    for instance in 0..instances {
        for new_timestep in 0..new_timesteps {
            let old_timestep = new_timestep * factor;
            if old_timestep < old_timesteps {
                new_data
                    .slice_mut(s![instance, new_timestep, ..])
                    .assign(&data_view.slice(s![instance, old_timestep, ..]));
            }
        }
    }

    new_data.into_pyarray(_py).into()
}

fn downsample_labels(
    _py: Python,
    old_timesteps: usize,
    new_timesteps: usize,
    factor: usize,
    labels_py: &Py<PyArray1<f64>>
) -> Py<PyArray1<f64>> {
    let labels_view = bind_array_1d(_py, labels_py);
    let mut new_labels = Array1::<f64>::zeros(new_timesteps);

    for new_timestep in 0..new_timesteps {
        let old_timestep = new_timestep * factor;
        if old_timestep < old_timesteps {
            new_labels[new_timestep] = labels_view[old_timestep];
        }
    }

    new_labels.into_pyarray(_py).into()
}

pub fn downsample(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    labels: Option<&Py<PyArray1<f64>>>,
    factor: usize
) -> PyResult<(Py<PyArray3<f64>>, Option<Py<PyArray1<f64>>>)> {
    if factor <= 0 {
        return Err(PyErr::new::<PyValueError, _>("Downsampling factor must be greater than 0"));
    }

    let data_view = bind_array_3d(_py, data);
    let (instances, timesteps, features) = data_view.dim();

    let new_timesteps = (timesteps + factor - 1) / factor;
    let new_data = downsample_data(
        _py,
        instances,
        timesteps,
        new_timesteps,
        features,
        factor,
        &data_view
    );

    let new_labels = if let Some(labels) = labels {
        Some(downsample_labels(_py, timesteps, new_timesteps, factor, labels))
    } else {
        None
    };

    Ok((new_data, new_labels))
}

pub fn impute(
    _py: Python,
    train_view: &mut ArrayBase<ndarray::ViewRepr<&mut f64>, Dim<[usize; 3]>>,
    val_view: &mut ArrayBase<ndarray::ViewRepr<&mut f64>, Dim<[usize; 3]>>,
    test_view: &mut ArrayBase<ndarray::ViewRepr<&mut f64>, Dim<[usize; 3]>>,
    strategy: ImputeStrategy
) -> PyResult<()> {

    if strategy == ImputeStrategy::LeaveNaN{
        return Ok(());
    }

    sub_impute(_py, &strategy, train_view);
    sub_impute(_py, &strategy, val_view);
    sub_impute(_py, &strategy, test_view);
    Ok(())
}

fn sub_impute(_py: Python, strategy: &ImputeStrategy, mut_view: &mut ArrayBase<ndarray::ViewRepr<&mut f64>, Dim<[usize; 3]>>) {
    let (instances, _, features) = mut_view.dim();
    for instance in 0..instances {
        for feature in 0..features {
            let mut column_slice = mut_view.slice_mut(s![instance, .., feature]);
            match strategy {
                ImputeStrategy::LeaveNaN => continue,
                ImputeStrategy::Mean => {
                    // Calculate the mean of the filtered values (dropping NaNs) (using numpy api would return NaN)
                    let mean = column_slice.iter()
                        .filter(|&&x| !x.is_nan())
                        .cloned()
                        .sum::<f64>() / column_slice.iter().filter(|&&x| !x.is_nan()).count() as f64;
                    column_slice.iter_mut().for_each(|x| {
                        if x.is_nan() {
                            *x = mean;
                        }
                    });
                }
                ImputeStrategy::Median => {
                    let vals = column_slice.iter().filter(|&&x| !x.is_nan()).collect::<Vec<_>>();
                    let mut sorted_vals = vals.clone();
                    sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = if sorted_vals.is_empty() {
                        0.0
                    } else if sorted_vals.len() % 2 == 1 {
                        *sorted_vals[sorted_vals.len() / 2]
                    } else {
                        (*sorted_vals[sorted_vals.len() / 2 - 1] + *sorted_vals[sorted_vals.len() / 2]) / 2.0
                    };
                    column_slice.iter_mut().for_each(|x| {
                        if x.is_nan() {
                            *x = median;
                        }
                    });
                }
                ImputeStrategy::ForwardFill => {
                    let mut last_valid = None;
                    for x in column_slice.iter_mut() {
                        if x.is_nan() {
                            if let Some(last) = last_valid {
                                *x = last;
                            }
                        } else {
                            last_valid = Some(*x);
                        }
                    }
                }
                ImputeStrategy::BackwardFill => {
                    let mut next_valid = None;
                    for x in column_slice.iter_mut().rev() {
                        if x.is_nan() {
                            if let Some(next) = next_valid {
                                *x = next;
                            }
                        } else {
                            next_valid = Some(*x);
                        }
                    }
                }
                
            }
        }
    }
    }