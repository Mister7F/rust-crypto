use crate::expression::expression::ExpressionConfig;
use crate::expression::expression_bin::ExpressionBin;
use crate::impl_expression_config_pymethods;
use pyo3::prelude::*;

use std::sync::Arc;
use std::sync::Mutex;

#[pyclass(subclass)]
#[derive(Debug, Clone)]
pub struct ExpressionBinConfig {
    pub variables: Arc<Mutex<Vec<String>>>,
}

impl ExpressionConfig<ExpressionBin> for ExpressionBinConfig {
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
}

impl_expression_config_pymethods!(ExpressionBinConfig, ExpressionBin);
