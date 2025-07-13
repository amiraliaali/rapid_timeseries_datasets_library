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

py_enum! {
    pub SplittingStrategy {
        InOrder,
        Random,
    }
}

py_enum! {
    pub ImputeStrategy {
        LeaveNaN,
        Mean,
        Median,
        ForwardFill,
        BackwardFill,
    }
}
