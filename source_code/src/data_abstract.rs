// We know that we are oging to deal with two types of data:
// Forecasting data and Classification (UCR/UEA) data
// Therefore we need to be able to handle both types of data

#![allow(dead_code)]

// we first implement a trait for a single data point in our dataset
pub trait TimeSeriesSample{
    fn id(&self) -> &str;

    fn sequence(&self) -> &Vec<f32>;
}

// Example for forecasting data:
//  id  f1
//  0   12.0    <--
//  1   15.0    <--
//  2   19.0    <--
// ---past to future---
//  3   24.0
//  4   13.0
#[derive(Clone, Debug)]
pub struct ForecastingSample{
    pub id: String,
    pub past: Vec<f32>,
    pub future: Vec<f32>
}

impl TimeSeriesSample for ForecastingSample {
    fn id(&self) -> &str{
        &self.id
    }

    fn sequence(&self) -> &Vec<f32> {
        &self.past
    }
}

// Example for classification data:
//  id  f1      f2      f3      c
//  0   12.0    5.5     2.5     1
//  1   15.0    7.3     6.3     0     <--
//  2   19.0    2.2     8.9     3
#[derive(Clone, Debug)]
pub struct ClassificationSample {
    pub id: String,
    pub series: Vec<f32>,
    pub label: usize,
}

impl TimeSeriesSample for ClassificationSample {
    fn id(&self) -> &str {
        &self.id
    }

    fn sequence(&self) -> &Vec<f32> {
        &self.series
    }
}

// And now we create a trait for the whole dataset, which is basically
// a vector of samples
pub trait TimeSeriesDataset{
    // we don't define the exact type of the sample here because
    // in each of the cases it could differ (categorical vs. forecasting)
    type Sample;

    fn len(&self) -> usize;

    fn get(&self, index: usize) -> Option<Self::Sample>;
}

pub struct ForecastingDataset {
    pub samples: Vec<ForecastingSample>
}

impl TimeSeriesDataset for ForecastingDataset {
    type Sample = ForecastingSample;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Self::Sample> {
        self.samples.get(index).cloned()
    }
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