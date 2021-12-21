use smartcore::dataset::iris::load_dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::accuracy;

pub fn regression() -> f32 {

    let iris_data = load_dataset();

    let x = DenseMatrix::from_array(
        iris_data.num_samples,
        iris_data.num_features,
        &iris_data.data,
    );
    let y = iris_data.target;

    let lr = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
    let y_hat = lr.predict(&x).unwrap();

    accuracy(&y, &y_hat)
}
