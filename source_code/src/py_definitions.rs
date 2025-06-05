use pyo3::prelude::*;

macro_rules! py_enum {
    (
        $(#[$meta:meta])*
        $vis:vis $name:ident { $($variant:ident),* $(,)? }
    ) => {
        #[pyclass]
        #[derive(PartialEq, Clone, Debug)]
        $(#[$meta])*
        $vis enum $name {
            $($variant),*
        }
    };
}

// Replace your existing enum definitions with:
py_enum! {
    pub DatasetType {
        Classification,
        Forecasting,
    }
}

py_enum! {
    pub SplittingStrategy {
        Temporal,
        Random,
    }
}

py_enum! {
    pub ImputeStrategy {
        LeaveNaN,
        Mean,
        Median,
        Mode,
        ForwardFill,
    }
}
