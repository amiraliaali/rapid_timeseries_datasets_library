use crate::{
    abrev_types::ClassificationSplitResult,
    data_abstract::SplittingStrategy,
    utils::{ bind_array_1d, bind_array_3d },
};
use log::debug;
use ndarray::{ ArrayView1, ArrayView3, Axis };
use numpy::{ PyArray1, PyArray3 };
use pyo3::prelude::*;
use rand::seq::SliceRandom;

pub fn split_forecasting(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    train_prop: f64,
    val_prop: f64
) -> PyResult<(usize, usize)> {
    // split_strategy is necessarily Temporal for forecasting datasets

    // bind the array
    let data_view = bind_array_3d(_py, data);

    // Calculate the split indices
    Ok(get_split_offsets(&data_view, train_prop, val_prop))
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn get_split_offsets(
    data_view: &ArrayView3<f64>,
    train_prop: f64,
    val_prop: f64
) -> (usize, usize) {
    let timesteps = get_n_timesteps(data_view);
    compute_split_offsets(timesteps, train_prop, val_prop)
}

pub fn compute_split_offsets(units: usize, train_prop: f64, val_prop: f64) -> (usize, usize) {
    let train_split_offset = compute_split_offset(units, train_prop);
    let val_split_offset = compute_split_offset(units, val_prop);
    (train_split_offset, val_split_offset)
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn compute_split_offset(units: usize, prop: f64) -> usize {
    ((units as f64) * prop).round() as usize
}

pub fn validate_props(train_prop: f64, val_prop: f64, test_prop: f64) -> PyResult<()> {
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

fn log_split_sizes(train_prop: f64, val_prop: f64, test_prop: f64) {
    debug!(
        "Splitting array with sizes: train={}, val={}, test={}, adding up to {}\n",
        train_prop,
        val_prop,
        test_prop,
        train_prop + val_prop + test_prop
    );
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn get_n_timesteps(data_view: &ArrayView3<f64>) -> usize {
    let (_instances, timesteps, _features) = data_view.dim();
    timesteps
}

pub fn split_classification(
    _py: Python,
    data: &Py<PyArray3<f64>>,
    labels: &Py<PyArray1<f64>>,
    splitting_strategy: SplittingStrategy,
    train_prop: f64,
    val_prop: f64
) -> ClassificationSplitResult {
    // bind the arrays
    let data_view = bind_array_3d(_py, data);
    let labels_view = bind_array_1d(_py, labels);

    // Calculate the number of instances for each set
    let instances = get_n_instances(&data_view);

    // Compute split offsets
    let (train_split_offset, val_split_offset) = compute_split_offsets(
        instances,
        train_prop,
        val_prop
    );

    match splitting_strategy {
        SplittingStrategy::InOrder => {
            split_by_instance_in_order(
                _py,
                &data_view,
                &labels_view,
                train_split_offset,
                val_split_offset
            )
        }
        SplittingStrategy::Random => {
            split_by_instance_random(
                _py,
                &data_view,
                &labels_view,
                instances,
                train_split_offset,
                val_split_offset
            )
        }
    }
}

fn split_by_instance_random(
    _py: Python,
    data_view: &ArrayView3<f64>,
    labels_view: &ArrayView1<f64>,
    instances: usize,
    train_split_offset: usize,
    val_split_offset: usize
) -> ClassificationSplitResult {
    // Create random indices for shuffling
    let mut indices: Vec<usize> = (0..instances).collect();
    indices.shuffle(&mut rand::thread_rng());

    // Split the data and labels using the shuffled indices
    let train_indices: Vec<usize> = indices[0..train_split_offset].to_vec();
    let val_indices: Vec<usize> =
        indices[train_split_offset..train_split_offset + val_split_offset].to_vec();
    let test_indices: Vec<usize> = indices[train_split_offset + val_split_offset..].to_vec();

    let top_axis = Axis(0);
    let train_data = data_view.select(top_axis, &train_indices);
    let val_data = data_view.select(top_axis, &val_indices);
    let test_data = data_view.select(top_axis, &test_indices);

    let train_labels = labels_view.select(top_axis, &train_indices);
    let val_labels = labels_view.select(top_axis, &val_indices);
    let test_labels = labels_view.select(top_axis, &test_indices);

    Ok(((train_data, train_labels), (val_data, val_labels), (test_data, test_labels)))
}

fn split_by_instance_in_order(
    _py: Python,
    data_view: &ArrayView3<f64>,
    labels_view: &ArrayView1<f64>,
    train_split_offset: usize,
    val_split_offset: usize
) -> ClassificationSplitResult {
    let top_axis = Axis(0);
    let (train_data, remainder) = data_view.split_at(top_axis, train_split_offset);
    let (val_data, test_data) = remainder.split_at(top_axis, val_split_offset);

    let (train_labels, remainder_labels) = labels_view.split_at(top_axis, train_split_offset);
    let (val_labels, test_labels) = remainder_labels.split_at(top_axis, val_split_offset);

    Ok((
        (train_data.to_owned(), train_labels.to_owned()),
        (val_data.to_owned(), val_labels.to_owned()),
        (test_data.to_owned(), test_labels.to_owned()),
    ))
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn get_n_instances(data_view: &ArrayView3<f64>) -> usize {
    let (instances, _timesteps, _features) = data_view.dim();
    instances
}
