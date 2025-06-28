use crate::{
    abrev_types::ClassificationSplitResult,
    data_abstract::SplittingStrategy,
    utils::{ bind_array_1d, bind_array_3d },
};
use log::debug;
use ndarray::{ ArrayView3, Axis };
use numpy::{ PyArray1, PyArray3 };
use pyo3::prelude::*;
use rand::seq::SliceRandom;

fn log_split_sizes(train_prop: f64, val_prop: f64, test_prop: f64) {
    debug!(
        "Splitting array with sizes: train={}, val={}, test={}, adding up to {}\n",
        train_prop,
        val_prop,
        test_prop,
        train_prop + val_prop + test_prop
    );
}

fn validate_props(train_prop: f64, val_prop: f64, test_prop: f64) -> PyResult<()> {
    log_split_sizes(train_prop, val_prop, test_prop);

    if train_prop < 0.0 || val_prop < 0.0 || test_prop < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Sizes must be non-negative"));
    }

    const EPSILON: f64 = 1e-10;
    if (train_prop + val_prop + test_prop - 1.0).abs() > EPSILON {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Sizes must add up to 1"));
    }

    Ok(())
}

fn get_n_timesteps(data_view: &ArrayView3<f64>) -> usize {
    let (_instances, timesteps, _features) = data_view.dim();
    timesteps
}

fn compute_split_offset(timesteps: usize, prop: f64) -> usize {
    ((timesteps as f64) * prop).round() as usize
}

fn compute_split_offsets(timesteps: usize, train_prop: f64, val_prop: f64) -> (usize, usize) {
    let train_split_offset = compute_split_offset(timesteps, train_prop);
    let val_split_offset = compute_split_offset(timesteps, val_prop);
    (train_split_offset, val_split_offset)
}

fn get_split_offsets(
    data_view: &ArrayView3<f64>,
    train_prop: f64,
    val_prop: f64
) -> (usize, usize) {
    let timesteps = get_n_timesteps(data_view);
    compute_split_offsets(timesteps, train_prop, val_prop)
}

pub fn split_forecasting(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    train_prop: f64,
    val_prop: f64,
    test_prop: f64
) -> PyResult<(usize, usize)> {
    // split_strategy is necessarily Temporal for forecasting datasets

    // Validate the sizes
    validate_props(train_prop, val_prop, test_prop)?;

    // bind the array
    let data_view = bind_array_3d(_py, data);

    // Calculate the split indices
    Ok(get_split_offsets(&data_view, train_prop, val_prop))
}

fn split_temporal(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    labels: &Py<PyArray1<f64>>,
    train_prop: f64,
    val_prop: f64,
    test_prop: f64
) -> ClassificationSplitResult {
    // Validate the sizes
    validate_props(train_prop, val_prop, test_prop)?;

    // bind the arrays
    let data_view = bind_array_3d(_py, data);
    let labels_view = bind_array_1d(_py, labels);

    let (train_split_offset, val_split_offset) = get_split_offsets(
        &data_view,
        train_prop,
        val_prop
    );

    // Split the data and labels
    let timesteps_axis = Axis(1);
    let (train_data, remainder_data) = data_view.split_at(timesteps_axis, train_split_offset);
    let (val_data, test_data) = remainder_data.split_at(timesteps_axis, val_split_offset);

    let label_axis = Axis(0);
    let (train_labels, remainder_labels) = labels_view.split_at(label_axis, train_split_offset);
    let (val_labels, test_labels) = remainder_labels.split_at(label_axis, val_split_offset);

    Ok((
        (train_data.to_owned(), train_labels.to_owned()),
        (val_data.to_owned(), val_labels.to_owned()),
        (test_data.to_owned(), test_labels.to_owned()),
    ))
}

fn split_random(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    labels: &Py<PyArray1<f64>>,
    train_prop: f64,
    val_prop: f64,
    test_prop: f64
) -> ClassificationSplitResult {
    // Validate the sizes
    validate_props(train_prop, val_prop, test_prop)?;

    // bind the arrays
    let data_view = bind_array_3d(_py, data);
    let labels_view = bind_array_1d(_py, labels);

    // Calculate the number of samples for each set
    let timesteps = get_n_timesteps(&data_view);

    // Compute split offsets
    let (train_split_offset, val_split_offset) = compute_split_offsets(
        timesteps,
        train_prop,
        val_prop
    );

    // Create random indices for shuffling
    let mut indices: Vec<usize> = (0..timesteps).collect();
    indices.shuffle(&mut rand::thread_rng());

    // Split the data and labels using the shuffled indices
    let train_indices: Vec<usize> = indices[0..train_split_offset].to_vec();
    let val_indices: Vec<usize> =
        indices[train_split_offset..train_split_offset + val_split_offset].to_vec();
    let test_indices: Vec<usize> = indices[train_split_offset + val_split_offset..].to_vec();

    let timesteps_axis = Axis(1);
    let train_data = data_view.select(timesteps_axis, &train_indices);
    let val_data = data_view.select(timesteps_axis, &val_indices);
    let test_data = data_view.select(timesteps_axis, &test_indices);

    let label_axis = Axis(0);
    let train_labels = labels_view.select(label_axis, &train_indices);
    let val_labels = labels_view.select(label_axis, &val_indices);
    let test_labels = labels_view.select(label_axis, &test_indices);

    Ok((
        (train_data.to_owned(), train_labels.to_owned()),
        (val_data.to_owned(), val_labels.to_owned()),
        (test_data.to_owned(), test_labels.to_owned()),
    ))
}

pub fn split_classification(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    labels: &Py<PyArray1<f64>>,
    splitting_strategy: SplittingStrategy,
    train_prop: f64,
    val_prop: f64,
    test_prop: f64
) -> ClassificationSplitResult {
    validate_props(train_prop, val_prop, test_prop)?;

    match splitting_strategy {
        SplittingStrategy::Random => {
            split_random(_py, &data, &labels, train_prop, val_prop, test_prop)
        }
        SplittingStrategy::Temporal => {
            split_temporal(_py, &data, &labels, train_prop, val_prop, test_prop)
        }
    }
}
