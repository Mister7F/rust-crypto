use crate::matrix::matrix_gen::GenElement;
use itertools::Itertools;
use std::fmt;
use std::ops;

#[derive(Debug, Clone)]
pub struct Monomial<T> {
    pub coefficient: T,
    pub exponents: Vec<(String, usize)>,
}

#[derive(Debug, Clone)]
pub struct ExpressionGeneric<T: GenElement> {
    pub constant: T,
    pub monomials: Vec<Monomial<T>>,
}

struct ZipMonomials<'a, T> {
    self_monomials: std::slice::Iter<'a, Monomial<T>>,
    rhs_monomials: std::slice::Iter<'a, Monomial<T>>,
    current_s: Option<&'a Monomial<T>>,
    current_r: Option<&'a Monomial<T>>,
    started: bool,
}

impl<T: GenElement> Iterator for ZipMonomials<'_, T> {
    type Item = Monomial<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started {
            self.current_s = self.self_monomials.next();
            self.current_r = self.rhs_monomials.next();
            self.started = true;
        }
        if self.current_s.is_none() && self.current_r.is_none() {
            return None;
        }
        let current_sx = match self.current_s {
            Some(current_sx) => {
                if self.current_r.is_none() {
                    self.current_s = self.self_monomials.next();
                    return Some(current_sx.clone());
                }
                current_sx
            }
            None => {
                return self.current_r.cloned();
            }
        };
        let current_rx = self.current_r?;

        // Both not none
        if current_sx.exponents == current_rx.exponents {
            let result = Monomial {
                exponents: current_rx.exponents.clone(),
                coefficient: current_sx.coefficient.clone() + current_sx.coefficient.clone(),
            };
            self.current_s = self.self_monomials.next();
            self.current_r = self.rhs_monomials.next();
            return Some(result);
        }
        if current_sx.exponents < current_rx.exponents {
            self.current_s = self.self_monomials.next();
            return Some(current_sx.clone());
        }
        self.current_r = self.rhs_monomials.next();
        Some(current_rx.clone())
    }
}

impl<T: GenElement> ExpressionGeneric<T> {
    fn zip_monomials<'a>(&'a self, rhs: &'a ExpressionGeneric<T>) -> ZipMonomials<'a, T> {
        ZipMonomials {
            self_monomials: self.monomials.iter(),
            rhs_monomials: rhs.monomials.iter(),
            current_s: None,
            current_r: None,
            started: false,
        }
    }
}

impl<T: GenElement> fmt::Display for ExpressionGeneric<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ret = self
            .monomials
            .iter()
            .map(|m| format!("{}", m))
            .chain(if self.constant != T::zero() {
                Some(format!("{}", self.constant))
            } else {
                None
            })
            .join(" + ");

        write!(f, "{}", ret)
    }
}

impl<T: GenElement> fmt::Display for Monomial<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ret = (if self.coefficient != T::one() {
            Some(format!("{}", self.coefficient))
        } else {
            None
        })
        .into_iter()
        .chain(self.exponents.iter().map(|(name, exp)| {
            if *exp == 1 {
                name.clone()
            } else {
                format!("{} ** {}", name, exp)
            }
        }))
        .join(" * ");

        write!(f, "{}", ret)
    }
}

impl<T: GenElement> ops::Add<ExpressionGeneric<T>> for ExpressionGeneric<T> {
    type Output = ExpressionGeneric<T>;

    fn add(self, rhs: ExpressionGeneric<T>) -> ExpressionGeneric<T> {
        ExpressionGeneric {
            monomials: self.zip_monomials(&rhs).collect(),
            constant: self.constant + rhs.constant,
        }
    }
}

impl<T: GenElement> ops::Add<T> for ExpressionGeneric<T> {
    type Output = ExpressionGeneric<T>;

    fn add(self, rhs: T) -> ExpressionGeneric<T> {
        ExpressionGeneric {
            monomials: self.monomials,
            constant: self.constant + rhs,
        }
    }
}

impl<T: GenElement> ops::Sub<T> for ExpressionGeneric<T> {
    type Output = ExpressionGeneric<T>;

    fn sub(self, rhs: T) -> ExpressionGeneric<T> {
        ExpressionGeneric {
            monomials: self.monomials,
            constant: self.constant - rhs,
        }
    }
}

impl<T: GenElement> ops::Mul<T> for ExpressionGeneric<T> {
    type Output = ExpressionGeneric<T>;

    fn mul(self, rhs: T) -> ExpressionGeneric<T> {
        ExpressionGeneric {
            monomials: self
                .monomials
                .into_iter()
                .map(|m| Monomial {
                    coefficient: m.coefficient * rhs.clone(),
                    exponents: m.exponents,
                })
                .collect(),
            constant: self.constant * rhs,
        }
    }
}

// --------------------------------------------------
//                      TESTS
// --------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::expression::expression::ExpressionConfig;
    use crate::expression::expression_generic_config::ExpressionGenericConfig;

    #[test]
    fn test_expression_generic() {
        let mut config: ExpressionGenericConfig<u32> = ExpressionGenericConfig::new();
        let p1 = config.gen("x".into());
        assert_eq!("x", format!("{}", p1));

        let p2 = config.gen("y".into());
        assert_eq!("y", format!("{}", p2));
        assert_eq!("y + 1", format!("{}", p2.clone() + 1));
        assert_eq!("y", format!("{}", (p2.clone() + 1) - 1));
        assert_eq!("5 * y + 5", format!("{}", (p2.clone() + 1) * 5));
        assert_eq!(
            "5 * x + 10 * y + 15",
            format!("{}", (p2.clone() * 2 + 3 + p1.clone()) * 5)
        );
    }
}
