use crate::expression::expression::ExpressionConfig;
use crate::expression::expression_bin::ExpressionBin;
use crate::impl_expression_config_pymethods;
use pyo3::prelude::*;

use std::sync::Arc;
use std::sync::Mutex;

#[pyclass]
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

        ExpressionBin::new(
            (0u64..(index / 64) as u64)
                .chain(std::iter::once(1u64 << (index % 64)))
                .collect(),
            false,
            &self.clone(),
        )
    }
}

impl_expression_config_pymethods!(ExpressionBinConfig);
