use crate::abrev_types::{ ClassificationCollectResult, ForecastingCollectResult };
use ndarray::{ s, Array1, Array3, ArrayView3 };
use numpy::{ IntoPyArray, PyArray3 };
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn validate_window_params(
    past_window: usize,
    future_horizon: usize,
    stride: usize,
    timesteps: usize
) -> PyResult<()> {
    if past_window == 0 || future_horizon == 0 || stride == 0 {
        return Err(
            PyErr::new::<PyValueError, _>(
                "past_window, future_horizon, and stride must be greater than 0"
            )
        );
    }

    if past_window + future_horizon > timesteps {
        return Err(
            PyErr::new::<PyValueError, _>(
                "past_window + future_horizon must be less than or equal to the number of timesteps"
            )
        );
    }

    Ok(())
}

#[cfg_attr(feature = "test_expose", visibility::make(pub))]
fn create_windows(
    _py: Python,
    data_view: &ArrayView3<f64>,
    past_window: usize,
    future_horizon: usize,
    stride: usize
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray3<f64>>)> {
    let (instances, timesteps, features) = data_view.dim();

    validate_window_params(past_window, future_horizon, stride, timesteps)?;

    // Apply sliding window logic
    // make a new 3d array
    let windows_per_instance = (timesteps - past_window - future_horizon) / stride + 1;
    let window_count = windows_per_instance * instances;
    let mut x_windows = ndarray::Array3::<f64>::zeros((window_count, past_window, features));
    let mut y_windows = ndarray::Array3::<f64>::zeros((window_count, future_horizon, features));
    let mut window_index = 0 as usize;
    for instance in 0..instances {
        let start = 0;
        let end = timesteps - future_horizon;
        for i in (start..end).step_by(stride) {
            let x_start = i;
            let x_end = i + past_window;
            let y_start = i + past_window;
            let y_end = y_start + future_horizon;

            if x_end > timesteps || y_end > timesteps {
                continue; // Skip if the window exceeds the bounds
            }
            let x_slice = data_view.slice(s![instance, x_start..x_end, ..]).to_owned();
            let y_slice = data_view.slice(s![instance, y_start..y_end, ..]).to_owned();

            x_windows.slice_mut(s![window_index, .., ..]).assign(&x_slice);
            y_windows.slice_mut(s![window_index, .., ..]).assign(&y_slice);
            window_index += 1;
        }
    }

    Ok((x_windows.into_pyarray(_py).into(), y_windows.into_pyarray(_py).into()))
}

pub fn collect_forecasting(
    _py: Python,
    train_view: &ArrayView3<f64>,
    val_view: &ArrayView3<f64>,
    test_view: &ArrayView3<f64>,
    past_window: usize,
    future_horizon: usize,
    stride: usize
) -> ForecastingCollectResult {
    let train_windows = create_windows(_py, train_view, past_window, future_horizon, stride)?;
    let val_windows = create_windows(_py, val_view, past_window, future_horizon, stride)?;
    let test_windows = create_windows(_py, test_view, past_window, future_horizon, stride)?;

    Ok((train_windows, val_windows, test_windows))
}

pub fn collect_classification(
    _py: Python,
    train_data: Array3<f64>,
    train_labels: Array1<f64>,
    val_data: Array3<f64>,
    val_labels: Array1<f64>,
    test_data: Array3<f64>,
    test_labels: Array1<f64>
) -> ClassificationCollectResult {
    Ok((
        (train_data.into_pyarray(_py).into(), train_labels.into_pyarray(_py).into()),
        (val_data.into_pyarray(_py).into(), val_labels.into_pyarray(_py).into()),
        (test_data.into_pyarray(_py).into(), test_labels.into_pyarray(_py).into()),
    ))
}
