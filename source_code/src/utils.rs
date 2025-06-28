use ndarray::{ Array3, ArrayView1, ArrayView3, ArrayViewMut1, ArrayViewMut3, Axis };
use numpy::{ PyArray1, PyArray3, PyArrayMethods };
use pyo3::prelude::*;

pub fn bind_array_1d<'py>(py: Python<'py>, data: &'py Py<PyArray1<f64>>) -> ArrayView1<'py, f64> {
    let bound_array = data.bind(py);
    unsafe { bound_array.as_array() }
}

pub fn bind_array_3d<'py>(py: Python<'py>, data: &'py Py<PyArray3<f64>>) -> ArrayView3<'py, f64> {
    let bound_array = data.bind(py);
    unsafe { bound_array.as_array() }
}

pub fn bind_array_mut_1d<'py>(
    py: Python<'py>,
    data: &'py Py<PyArray1<f64>>
) -> ArrayViewMut1<'py, f64> {
    let bound_array = data.bind(py);
    unsafe { bound_array.as_array_mut() }
}

pub fn bind_array_mut_3d<'py>(
    py: Python<'py>,
    data: &'py Py<PyArray3<f64>>
) -> ArrayViewMut3<'py, f64> {
    let bound_array = data.bind(py);
    unsafe { bound_array.as_array_mut() }
}

fn validate_split_indices(
    train_split_index: Option<usize>,
    val_split_index: Option<usize>,
    total_timesteps: usize
) -> PyResult<(usize, usize)> {
    if train_split_index.is_none() || val_split_index.is_none() {
        return Err(
            PyErr::new::<pyo3::exceptions::PyValueError, _>("View-split before split() operation")
        );
    }
    let train_split_index = train_split_index.unwrap();
    let val_split_index = val_split_index.unwrap();

    if train_split_index <= 0 || train_split_index >= total_timesteps {
        return Err(
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "train_split_index must be greater than 0 and less than the number of timesteps"
            )
        );
    }
    if val_split_index <= 0 || val_split_index >= total_timesteps {
        return Err(
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "val_split_index must be greater than 0 and less than the number of timesteps"
            )
        );
    }
    Ok((train_split_index, val_split_index))
}

pub fn get_split_views<'py>(
    py: Python<'py>,
    data: &'py Py<PyArray3<f64>>,
    train_split_index: Option<usize>,
    val_split_index: Option<usize>
) -> PyResult<(ArrayView3<'py, f64>, ArrayView3<'py, f64>, ArrayView3<'py, f64>)> {
    let data_view = bind_array_3d(py, data);

    let (train_split_index, val_split_index) = validate_split_indices(
        train_split_index,
        val_split_index,
        data_view.shape()[1]
    )?;

    let (train_view, remainder) = data_view.split_at(Axis(1), train_split_index);
    let (val_view, test_view) = remainder.split_at(Axis(1), val_split_index);

    Ok((train_view, val_view, test_view))
}

pub fn get_split_views_mut<'py>(
    py: Python<'py>,
    data: &'py Py<PyArray3<f64>>,
    train_split_index: Option<usize>,
    val_split_index: Option<usize>
) -> PyResult<(ArrayViewMut3<'py, f64>, ArrayViewMut3<'py, f64>, ArrayViewMut3<'py, f64>)> {
    let data_view = bind_array_mut_3d(py, data);

    let (train_split_index, val_split_index) = validate_split_indices(
        train_split_index,
        val_split_index,
        data_view.shape()[1]
    )?;

    let (train_view, remainder) = data_view.split_at(Axis(1), train_split_index);
    let (val_view, test_view) = remainder.split_at(Axis(1), val_split_index);

    Ok((train_view, val_view, test_view))
}

pub fn check_arrays_set(
    train_data: &Option<Array3<f64>>,
    val_data: &Option<Array3<f64>>,
    test_data: &Option<Array3<f64>>
) -> PyResult<()> {
    if train_data.is_none() || val_data.is_none() || test_data.is_none() {
        return Err(
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "function requiring split called before calling split()"
            )
        );
    }

    Ok(())
}
