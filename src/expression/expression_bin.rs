use core::ops;
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::Arc;

use crate::expression::expression::Expression;
use crate::expression::expression_bin_config::ExpressionBinConfig;
use crate::matrix::matrix_bin::MatrixBin;
use pyo3::exceptions::PyValueError;

// Represent a multi-variate polynomial in GF(2)

// --------------------------------------------------
//                      PYTHON
// --------------------------------------------------

#[derive(Debug, FromPyObject)]
#[pyclass(frozen, subclass)]
pub struct ExpressionBin {
    // Each element contains 64 variables coefficients, for performance
    coeffs: Vec<u64>,
    constant: bool,
    config: ExpressionBinConfig,
    var_name: Option<String>,
}

#[pymethods]
impl ExpressionBin {
    #[new]
    pub fn new(coeffs: Vec<u64>, constant: bool, config: &ExpressionBinConfig) -> Self {
        ExpressionBin {
            coeffs,
            constant,
            config: config.clone(),
            var_name: None,
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
                self._add(&borrowed.coeffs, borrowed.constant)
            }
            ExpressionBinOrInt::Int(other) => {
                if other % 2 == 1 {
                    self._add(&[], true)
                } else {
                    ExpressionBin {
                        coeffs: self.coeffs.clone(),
                        constant: self.constant,
                        config: self.config.clone(),
                        var_name: None,
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
            var_name: None,
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

    #[getter]
    pub fn config(&self) -> ExpressionBinConfig {
        self.config.clone()
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
            .map(|(_coeff, name)| if !name.is_empty() { name } else { "1" })
            .join(" + ")
    }

    pub fn __int__(&self) -> PyResult<u8> {
        if self.degree() != 0 {
            return Err(PyValueError::new_err("Not a constant"));
        }
        Ok(self.constant as u8)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }

    pub fn get_coeff(&self, name: String) -> bool {
        let variables = self.config.variables.lock().unwrap();
        let idx = variables.iter().position(|n| n == &name);
        if let Some(idx) = idx {
            return self.coeffs.get(idx / 64).unwrap_or(&0) >> (idx % 64) & 1 != 0;
        }
        false
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
        if let Some(name) = &self.var_name {
            return Some(name.clone());
        }
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
        let ncols = equations[0].config.variables.lock().unwrap().len();
        let stride = ncols.div_ceil(64);
        (
            MatrixBin {
                ncols,
                nrows: equations.len(),
                cells: equations
                    .iter()
                    .flat_map(|e| {
                        if e.coeffs.len() == stride {
                            return e.coeffs.clone();
                        }
                        e.coeffs
                            .iter()
                            .copied()
                            .chain(vec![0u64; stride - e.coeffs.len()].iter().copied())
                            .collect()
                    })
                    .collect(),
            },
            equations.iter().map(|e| e.constant).collect(),
        )
    }

    fn bool(&self) -> bool {
        self.constant || self.coeffs.iter().any(|c| *c != 0u64)
    }
}

// --------------------------------------------------
//                      MATH
// --------------------------------------------------

impl ExpressionBin {
    #[inline(always)]
    fn _add(&self, coeffs: &[u64], constant: bool) -> ExpressionBin {
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
            var_name: None,
        }
    }

    #[inline(always)]
    fn _mul(&self, other: u64) -> ExpressionBin {
        if other % 2 == 0 {
            ExpressionBin {
                coeffs: vec![],
                constant: self.constant,
                config: self.config.clone(),
                var_name: None,
            }
        } else {
            ExpressionBin {
                coeffs: self.coeffs.clone(),
                constant: self.constant,
                config: self.config.clone(),
                var_name: None,
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
        self._add(&[], rhs)
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

        let e = c("a") + c("d") + true;
        assert_eq!(e.get_coeff("a".into()), true);
        assert_eq!(e.get_coeff("b".into()), false);
        assert_eq!(e.get_coeff("c".into()), false);
        assert_eq!(e.get_coeff("d".into()), true);
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
