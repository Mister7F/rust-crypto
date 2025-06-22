use pyo3::prelude::*;

pub mod aes {
    pub mod aes;
    pub mod aes_5_rounds;
    pub mod aes_constants;
    pub mod aes_ni;
    pub mod aes_r8faults;
}
pub mod expression {
    pub mod expression;
    pub mod expression_bin;
    pub mod expression_bin_config;
    pub mod expression_generic;
    pub mod expression_generic_config;
}
pub mod matrix {
    pub mod matrix;
    pub mod matrix_bin;
    pub mod matrix_gen;
}
pub mod rings {
    pub mod fraction;
    pub mod zmod;
}

pub mod utils;

pub mod feal;

#[pyfunction]
fn example_py_callback(py: Python, a: usize, b: usize, callback: PyObject) -> PyResult<u64> {
    let result = callback
        .call(py, (a, b, (a + b).to_string()), None)?
        .extract::<[u64; 2]>(py)?;
    Ok(result[0] + 1)
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_crypto(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(example_py_callback, m)?)?;
    m.add_function(wrap_pyfunction!(aes::aes_5_rounds::decrypt_round_5_4, m)?)?;
    m.add_function(wrap_pyfunction!(aes::aes_5_rounds::aes_5_rounds, m)?)?;
    m.add_function(wrap_pyfunction!(aes::aes_r8faults::aes_r8faults_filter, m)?)?;
    m.add_class::<expression::expression_bin::ExpressionBin>()?;
    m.add_class::<expression::expression_bin_config::ExpressionBinConfig>()?;
    m.add_function(wrap_pyfunction!(feal::feal_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(feal::py_feal_break_6_rounds, m)?)?;
    m.add_class::<matrix::matrix_bin::MatrixBin>()?;
    // m.add_class::<matrix::matrix::Matrix<BigInt>>()?;
    Ok(())
}
