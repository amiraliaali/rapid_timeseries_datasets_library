// We know that we are going to deal with two types of data:
// Forecasting data and Classification (UCR/UEA) data
// Therefore we need to be able to handle both types of data

#![allow(dead_code)]

// we first implement a trait for a single data point in our dataset
pub trait TimeSeriesSample{
    fn id(&self) -> &str;

    fn sequence(&self) -> &Vec<Vec<f32>>;
}

// Example for forecasting data:
//  id  f1      f2
//  0   12.0    6.5 <--
//  1   15.0    3.7 <--
//  2   19.0    7.1 <--
// ---past to future---
//  3   24.0    1.4
//  4   13.0    6.9
// Note that for this kind of dataset we woudl need a sliding window parameter.
// In the example I assumed the sliding window to be 3
#[derive(Clone, Debug)]
pub struct ForecastingSample{
    pub id: String,
    pub past: Vec<Vec<f32>>,
    pub future: Vec<Vec<f32>>
}

impl TimeSeriesSample for ForecastingSample {
    fn id(&self) -> &str{
        &self.id
    }

    fn sequence(&self) -> &Vec<Vec<f32>> {
        &self.past
    }
}

// Example for classification data:
//  id  f1      f2      f3      c
//  0   12.0    5.5     2.5     1
//  1   15.0    7.3     6.3     0     <--
//  2   19.0    2.2     8.9     3
// However we usually only access one row as a sample, but since the trait
// also covers forecating dataset and in forecasting dataset we coudl have 
// vector of vectors, we would also define here vec<vec>>
#[derive(Clone, Debug)]
pub struct ClassificationSample {
    pub id: String,
    pub series: Vec<Vec<f32>>,
    pub label: usize,
}

impl TimeSeriesSample for ClassificationSample {
    fn id(&self) -> &str {
        &self.id
    }

    fn sequence(&self) -> &Vec<Vec<f32>> {
        &self.series
    }
}

// And now we create a trait for the whole dataset, which is basically
// a vector of samples
pub trait TimeSeriesDataset{
    // we create a place holder for Sample and later give it either the classification
    // or forecasting data type. This is used later as the return type of get()
    type Sample;

    fn len(&self) -> usize;

    fn get(&self, index: usize) -> Option<Self::Sample>;
}

pub struct ClassificationDataset {
    pub samples: Vec<ClassificationSample>
}

impl TimeSeriesDataset for ClassificationDataset {
    type Sample = ClassificationSample;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Self::Sample> {
        self.samples.get(index).cloned()
    }
}

pub struct ForecastingDataset {
    pub data: Vec<Vec<f32>>,
    pub past_window: usize,
    pub future_horizon: usize,
    pub stride: usize,
}

impl TimeSeriesDataset for ForecastingDataset {
    type Sample = ForecastingSample;

    // We want to see how many sliding window can we have in the whole data, given
    // the past and future window sizes as well as stride
    fn len(&self) -> usize {
        let total_window = self.past_window + self.future_horizon;
        if self.data.len() < total_window {
            0
        } else {
            (self.data.len() - total_window) / self.stride + 1
        }
    }

    fn get(&self, index: usize) -> Option<Self::Sample> {
        let total_window_size = self.past_window + self.future_horizon;
        let start_pos = index * self.stride;
        if start_pos + total_window_size > self.data.len() {
            return None;
        }

        let past = self.data[start_pos..start_pos + self.past_window].to_vec();
        let future = self.data[start_pos + self.past_window..start_pos + total_window_size].to_vec();

        Some(ForecastingSample {
            id: index.to_string(),
            past,
            future,
        })
    }
}