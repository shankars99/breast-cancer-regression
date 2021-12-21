//importing breast cancer dataset
use smartcore::dataset::breast_cancer;

//importing dense matrix and logistic regression
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;

//importing accuracy metric and Kfold cross validation
use smartcore::metrics::accuracy;
use smartcore::model_selection::{cross_validate, KFold};

//function returns an array of 2 elements: train accuracy and test accuracy
pub fn linear_regression() -> [f32; 2] {
    //importing breast cancer dataset
    let breast_cancer_data = breast_cancer::load_dataset();

    //creating a dense matrix from the imported dataset
    let x = DenseMatrix::from_array(
        breast_cancer_data.num_samples,
        breast_cancer_data.num_features,
        &breast_cancer_data.data,
    );

    //setting target values for the inputs
    let y = breast_cancer_data.target;

    //creating a KFold cross validation object
    let results = cross_validate(
        LogisticRegression::fit,
        &x,
        &y,
        Default::default(),
        KFold::default().with_n_splits(2),
        accuracy,
    )
    .unwrap();

    //returning train and test accuracy
    [results.mean_test_score(), results.mean_train_score()]
}
