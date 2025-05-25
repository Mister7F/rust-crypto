use core::ops;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::{Arc, Mutex};

use crate::expression::expression::{Expression, ExpressionConfig};
use crate::impl_expression_config_pymethods;
use crate::matrix::matrix_bin::MatrixBin;
use pyo3::exceptions::PyValueError;

// Represent a multi-variate polynomial in GF(2)

// --------------------------------------------------
//                      PYTHON
// --------------------------------------------------

#[pyclass]
#[derive(Debug, Clone)]
pub struct ExpressionBinConfig {
    variables: Arc<Mutex<Vec<String>>>,
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

        ExpressionBin {
            coeffs: (0u64..(index / 64) as u64)
                .chain(std::iter::once(1u64 << (index % 64)))
                .collect(),
            constant: false,
            config: self.clone(),
        }
    }
}

impl_expression_config_pymethods!(ExpressionBinConfig);

#[derive(Debug, FromPyObject)]
#[pyclass(frozen)]
pub struct ExpressionBin {
    // Each element contains 64 variables coefficients, for performance
    coeffs: Vec<u64>,
    constant: bool,
    config: ExpressionBinConfig,
}

#[pymethods]
impl ExpressionBin {
    #[new]
    fn new(coeffs: Vec<u64>, config: &ExpressionBinConfig) -> Self {
        ExpressionBin {
            coeffs: coeffs,
            constant: false,
            config: config.clone(),
        }
    }

    pub fn __xor__(&self, other: ExpressionBinOrInt) -> ExpressionBin {
        let borrowed;
        match other {
            ExpressionBinOrInt::ExpressionBin(other) => {
                borrowed = other.borrow();
                assert!(
                    Arc::ptr_eq(&self.config.variables, &borrowed.config.variables),
                    "Expression config is not shared"
                );
                return self._add(&borrowed.coeffs, borrowed.constant);
            }
            ExpressionBinOrInt::Int(other) => {
                if other % 2 == 1 {
                    return self._add(&vec![], true);
                } else {
                    ExpressionBin {
                        coeffs: self.coeffs.clone(),
                        constant: self.constant,
                        config: self.config.clone(),
                    }
                }
            }
        }
    }

    pub fn __add__(&self, other: ExpressionBinOrInt) -> ExpressionBin {
        self.__xor__(other)
    }

    pub fn __radd__(&self, other: ExpressionBinOrInt) -> ExpressionBin {
        self.__xor__(other)
    }

    pub fn __sub__(&self, other: ExpressionBinOrInt) -> ExpressionBin {
        self.__xor__(other)
    }

    pub fn __rsub__(&self, other: ExpressionBinOrInt) -> ExpressionBin {
        self.__xor__(other)
    }

    pub fn __rxor__(&self, other: ExpressionBinOrInt) -> ExpressionBin {
        self.__xor__(other)
    }

    pub fn __mod__(&self, other: u64) -> PyResult<ExpressionBin> {
        if other != 2 {
            return Err(PyValueError::new_err("Only 2 is a valid modulus"));
        }
        Ok(ExpressionBin {
            coeffs: self.coeffs.clone(),
            constant: self.constant,
            config: self.config.clone(),
        })
    }

    pub fn __mul__(&self, other: u64) -> ExpressionBin {
        self._mul(other)
    }

    pub fn __bool__(&self) -> bool {
        Expression::bool(self)
    }

    #[getter]
    pub fn constant(&self) -> bool {
        Expression::constant(self)
    }

    #[getter]
    pub fn degree(&self) -> u32 {
        Expression::degree(self)
    }

    pub fn var_name(&self) -> Option<String> {
        Expression::var_name(self)
    }

    pub fn lin_coeffs(&self) -> Vec<(bool, String)> {
        let mut bits: Vec<bool> = Vec::with_capacity(64 * self.coeffs.len() + 1);
        bits.push(self.constant);

        for c in &self.coeffs {
            for j in 0..64 {
                bits.push((c >> j) & 1 != 0)
            }
        }

        let variables = self.config.variables.lock().unwrap();
        if bits.len() < variables.len() {
            bits.extend(vec![false; variables.len() - bits.len()]);
        }

        bits.iter()
            .zip(std::iter::once(&String::new()).chain(variables.iter()))
            .map(|(coeff, name)| (*coeff, name.clone()))
            .collect()
    }

    pub fn __str__(&self) -> String {
        if !self.__bool__() {
            return "0".into();
        }
        self.lin_coeffs()
            .iter()
            .filter(|(coeff, _name)| *coeff)
            .map(|(_coeff, name)| if name.len() != 0 { name } else { "1" })
            .join(" + ")
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }

    #[classmethod]
    pub fn to_matrix(
        _cls: &Bound<PyType>,
        equations: Vec<Bound<ExpressionBin>>,
    ) -> (MatrixBin, Vec<bool>) {
        let equations: Vec<PyRef<ExpressionBin>> = equations
            .iter()
            .map(|e| e.extract::<PyRef<ExpressionBin>>().unwrap())
            .collect();

        Expression::to_matrix(equations.iter().map(|e| &**e).collect())
    }
}

impl Expression<bool, MatrixBin, ExpressionBinConfig> for ExpressionBin {
    fn var_name(&self) -> Option<String> {
        if self.constant {
            return None;
        }
        let res: Vec<(usize, &u64)> = self
            .coeffs
            .iter()
            .enumerate()
            .filter(|(_i, c)| **c != 0)
            .collect();
        if res.len() != 1 || res[0].1.count_ones() != 1 {
            return None;
        }

        let i = res[0].0;
        let c = res[0].1;
        Some(self.config.variables.lock().unwrap()[i * 64 + (c.ilog2() as usize)].clone())
    }

    fn degree(&self) -> u32 {
        self.coeffs.iter().any(|c| *c != 0) as u32
    }

    fn constant(&self) -> bool {
        self.constant
    }

    fn to_matrix(equations: Vec<&ExpressionBin>) -> (MatrixBin, Vec<bool>) {
        let cols = equations[0].config.variables.lock().unwrap().len();
        let stride = (cols + 63) / 64;
        (
            MatrixBin {
                cols: cols,
                rows: equations.len(),
                cells: equations
                    .iter()
                    .map(|e| {
                        if e.coeffs.len() == stride {
                            return e.coeffs.clone();
                        }
                        e.coeffs
                            .iter()
                            .copied()
                            .chain(vec![0u64; stride - e.coeffs.len()].iter().copied())
                            .collect()
                    })
                    .flatten()
                    .collect(),
            },
            equations.iter().map(|e| e.constant).collect(),
        )
    }

    fn bool(&self) -> bool {
        return self.constant || self.coeffs.iter().any(|c| *c != 0u64);
    }
}

// --------------------------------------------------
//                      MATH
// --------------------------------------------------

impl ExpressionBin {
    #[inline(always)]
    fn _add(&self, coeffs: &Vec<u64>, constant: bool) -> ExpressionBin {
        let self_len = self.coeffs.len();
        let other_len = coeffs.len();

        let mut new_coeffs: Vec<u64> = Vec::with_capacity(self_len.max(other_len));

        let min = self_len.min(other_len);
        for i in 0..min {
            new_coeffs.push(self.coeffs[i] ^ coeffs[i])
        }

        if self_len > other_len {
            new_coeffs.extend(&self.coeffs[other_len..])
        } else if other_len > self_len {
            new_coeffs.extend(&coeffs[self_len..])
        }

        ExpressionBin {
            coeffs: new_coeffs,
            constant: self.constant ^ constant,
            config: self.config.clone(),
        }
    }

    #[inline(always)]
    fn _mul(&self, other: u64) -> ExpressionBin {
        if other % 2 == 0 {
            ExpressionBin {
                coeffs: vec![],
                constant: self.constant,
                config: self.config.clone(),
            }
        } else {
            ExpressionBin {
                coeffs: self.coeffs.clone(),
                constant: self.constant,
                config: self.config.clone(),
            }
        }
    }
}

// --------------------------------------------------
//                      RUST
// --------------------------------------------------

impl ops::Add<ExpressionBin> for ExpressionBin {
    type Output = ExpressionBin;

    fn add(self, rhs: ExpressionBin) -> ExpressionBin {
        self._add(&rhs.coeffs, rhs.constant)
    }
}

impl ops::Add<bool> for ExpressionBin {
    type Output = ExpressionBin;

    fn add(self, rhs: bool) -> ExpressionBin {
        self._add(&vec![], rhs)
    }
}

impl ops::Mul<u64> for ExpressionBin {
    type Output = ExpressionBin;

    fn mul(self, rhs: u64) -> ExpressionBin {
        self._mul(rhs)
    }
}

#[derive(FromPyObject)]
pub enum ExpressionBinOrInt<'a> {
    Int(u64),
    ExpressionBin(Bound<'a, ExpressionBin>),
}

// --------------------------------------------------
//                      TESTS
// --------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::expression::expression_bin::ExpressionBinConfig;

    #[test]
    fn test_expression_bin() {
        let mut config = ExpressionBinConfig::new();

        let mut c = |name: &str| config.gen(name.into());

        assert_eq!((c("a") + c("b") + true).__str__(), "1 + a + b");
        assert_eq!((c("a") + c("b") + true + c("a")).__str__(), "1 + b");
        assert_eq!((c("a") + c("b") + true + c("b") + false).__str__(), "1 + a");
        assert_eq!(
            (c("a") + c("b") + true + c("b") + false + c("d")).__str__(),
            "1 + a + d"
        );
        assert_eq!((c("a") * 1).__str__(), "a");
        assert_eq!((c("a") * 0).__str__(), "0");
    }

    #[test]
    fn test_var_name() {
        let mut config = ExpressionBinConfig::new();

        let mut c = |name: &str| config.gen(name.into());

        assert_eq!(c("a").var_name(), Some("a".into()));
        assert_eq!(c("b").var_name(), Some("b".into()));
        assert_eq!((c("a") + c("b")).var_name(), None);
        assert_eq!((c("a") + true).var_name(), None);
    }
}
