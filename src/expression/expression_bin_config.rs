use crate::expression::expression::ExpressionConfig;
use crate::expression::expression_bin::ExpressionBin;
use crate::impl_expression_config_pymethods;
use crate::matrix::matrix_bin::MatrixBin;
use pyo3::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

#[pyclass(subclass)]
#[derive(Debug, Clone)]
pub struct ExpressionBinConfig {
    pub variables: Arc<Mutex<Vec<String>>>,
}

impl ExpressionConfig<ExpressionBin, MatrixBin, bool> for ExpressionBinConfig {
    fn new() -> Self {
        ExpressionBinConfig {
            variables: Arc::new(Mutex::new(vec![])),
        }
    }

    fn gen(&mut self, name: String) -> ExpressionBin {
        let mut variables = self.variables.lock().unwrap();
        let index = variables
            .iter()
            .position(|x| x == &name)
            .unwrap_or_else(|| {
                variables.push(name);
                variables.len() - 1
            });

        let mut coeffs = vec![0u64; index / 64];
        coeffs.push(1u64 << (index % 64));
        ExpressionBin::new(coeffs, false, &self.clone())
    }

    fn from_matrix(&self, matrix: MatrixBin, constants: Vec<bool>) -> Vec<ExpressionBin> {
        let n_variables = self.variables.lock().unwrap().len();
        assert!(matrix.ncols <= n_variables);
        assert!(constants.len() == matrix.nrows());

        let stride = matrix.ncols.div_ceil(64);
        matrix
            .cells
            .chunks(stride)
            .zip(constants.into_iter())
            .map(|(coeffs, constant)| ExpressionBin::new(coeffs.into(), constant, &self.clone()))
            .collect()
    }
}

impl_expression_config_pymethods!(ExpressionBinConfig, ExpressionBin, MatrixBin, bool);
