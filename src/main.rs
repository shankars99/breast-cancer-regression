//importing the linear regression file
mod linear_regression;

fn main() {
    //getting the output from the linear regression function
    let regression_eval: [f32; 2] = linear_regression::linear_regression();
    //printing out the values
    println!(
        "Breast cancer with train {} and test accuracy {}",
        regression_eval[0], regression_eval[1]
    );
}
