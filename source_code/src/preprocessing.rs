use ndarray::{ s, ArrayBase, ArrayView3, ArrayViewMut1, ArrayViewMut3, DataMut, Dim };
use numpy::{ PyArray3, IntoPyArray };
use crate::utils::bind_array_3d;
use pyo3::prelude::*;
use ndarray::Array3;
use pyo3::{ Python, PyResult, PyErr };
use pyo3::exceptions::PyValueError;
use crate::data_abstract::ImputeStrategy;

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

fn downsample_data(_py: Python, factor: usize, data_view: &ArrayView3<f64>) -> Py<PyArray3<f64>> {
    let (instances, old_timesteps, features) = data_view.dim();
    let new_timesteps = (old_timesteps + factor - 1) / factor;

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

pub fn downsample(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    factor: usize
) -> PyResult<Py<PyArray3<f64>>> {
    if factor <= 0 {
        return Err(PyErr::new::<PyValueError, _>("Downsampling factor must be greater than 0"));
    }

    let data_view = bind_array_3d(_py, data);

    Ok(downsample_data(_py, factor, &data_view))
}

pub fn impute(
    _py: Python,
    train_view: &mut ArrayViewMut3<f64>,
    val_view: &mut ArrayViewMut3<f64>,
    test_view: &mut ArrayViewMut3<f64>,
    strategy: ImputeStrategy
) -> PyResult<()> {
    if strategy == ImputeStrategy::LeaveNaN {
        return Ok(());
    }

    impute_view(_py, &strategy, train_view);
    impute_view(_py, &strategy, val_view);
    impute_view(_py, &strategy, test_view);
    Ok(())
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn impute_view(_py: Python, strategy: &ImputeStrategy, mut_view: &mut ArrayViewMut3<f64>) {
    let (instances, _, features) = mut_view.dim();
    for instance in 0..instances {
        for feature in 0..features {
            let mut column_slice = mut_view.slice_mut(s![instance, .., feature]);
            match strategy {
                ImputeStrategy::LeaveNaN => {
                    break;
                }
                ImputeStrategy::Mean => {
                    impute_mean(&mut column_slice);
                }
                ImputeStrategy::Median => {
                    impute_median(&mut column_slice);
                }
                ImputeStrategy::ForwardFill => {
                    impute_forward_fill(&mut column_slice);
                }
                ImputeStrategy::BackwardFill => {
                    impute_backward_fill(&mut column_slice);
                }
            }
        }
    }
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn impute_mean(column_slice: &mut ArrayViewMut1<f64>) {
    let mean =
        column_slice
            .iter()
            .filter(|&&x| !x.is_nan())
            .sum::<f64>() /
        (
            column_slice
                .iter()
                .filter(|&&x| !x.is_nan())
                .count() as f64
        );
    column_slice.iter_mut().for_each(|x| {
        if x.is_nan() {
            *x = mean;
        }
    });
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn impute_median(column_slice: &mut ArrayViewMut1<f64>) {
    let mut vals = column_slice
        .iter()
        .filter(|&&x| !x.is_nan())
        .collect::<Vec<_>>();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if vals.is_empty() {
        0.0
    } else if vals.len() % 2 == 1 {
        *vals[vals.len() / 2]
    } else {
        (*vals[vals.len() / 2 - 1] + *vals[vals.len() / 2]) / 2.0
    };
    column_slice.iter_mut().for_each(|x| {
        if x.is_nan() {
            *x = median;
        }
    });
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn impute_forward_fill(column_slice: &mut ArrayViewMut1<f64>) {
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

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn impute_backward_fill(column_slice: &mut ArrayViewMut1<f64>) {
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
