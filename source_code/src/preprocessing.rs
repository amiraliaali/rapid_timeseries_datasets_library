use ndarray::{s,  ArrayBase, ArrayView3, DataMut, Dim };
use numpy::{ PyArray1, PyArray3, IntoPyArray };
use crate::{
    utils::{ bind_array_1d, bind_array_3d },
};
use pyo3::prelude::*;
use ndarray::{Array3, Array1};
use pyo3::{Python, PyResult, PyErr};
use pyo3::exceptions::PyValueError;

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

pub fn downsampling_forecasting(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    factor: usize,
) -> PyResult<Py<PyArray3<f64>>> {
    let data_view = bind_array_3d(_py, data);
    let (instances, timesteps, features) = data_view.dim();

    // creating an empty array for downsampled data
    let new_timesteps = (timesteps + factor - 1) / factor;
    let mut new_data = Array3::<f64>::zeros((instances, new_timesteps, features));

    // downsampling the data
    for instance in 0..instances {
        for new_timestep in 0..new_timesteps {
            let old_timestep = new_timestep * factor;
            if old_timestep < timesteps {
                new_data.slice_mut(s![instance, new_timestep, ..])
                    .assign(&data_view.slice(s![instance, old_timestep, ..]));
            }
        }
    }

    let new_data_py = new_data.into_pyarray(_py);

    Ok(new_data_py.into())
}

pub fn downsampling_classification(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    labels: &Py<PyArray1<f64>>,
    factor: usize,
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray1<f64>>)> {
    let data_view = bind_array_3d(_py, data);
    let labels_view = bind_array_1d(_py, labels);

    let (instances, timesteps, features) = data_view.dim();
    assert!(timesteps == labels_view.len(), "Labels length must match the number of timesteps in data");

    // creating two empty arrays for downsampled data and labels
    let new_timesteps = (timesteps + factor - 1) / factor;
    let mut new_data = Array3::<f64>::zeros((instances, new_timesteps, features));
    let mut new_labels = Array1::<f64>::zeros(new_timesteps);

    // downsampling the data and labels
    for instance in 0..instances {
        for new_timestep in 0..new_timesteps {
            let old_timestep = new_timestep * factor;
            if old_timestep < timesteps {
                new_data.slice_mut(s![instance, new_timestep, ..])
                    .assign(&data_view.slice(s![instance, old_timestep, ..]));
            }
        }
    }
    for new_timestep in 0..new_timesteps {
        let old_timestep = new_timestep * factor;
        if old_timestep < labels_view.len() {
            new_labels[new_timestep] = labels_view[old_timestep];
        }
    }

    let new_data_py = new_data.into_pyarray(_py);
    let new_labels_py = new_labels.into_pyarray(_py);

    Ok((new_data_py.into(), new_labels_py.into()))
}

pub fn downsampling(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    labels: Option<&Py<PyArray1<f64>>>,
    factor: usize,
) -> PyResult<(Py<PyArray3<f64>>, Option<Py<PyArray1<f64>>>)> {
    if factor == 0 {
        return Err(PyErr::new::<PyValueError, _>("Downsampling factor must be greater than 0"));
    }

    if let Some(labels) = labels {
        let (new_data, new_labels) = downsampling_classification(_py, data, labels, factor)?;
        Ok((new_data, Some(new_labels)))
    } else {
        let new_data = downsampling_forecasting(_py, data, factor)?;
        Ok((new_data, None))
    }
}