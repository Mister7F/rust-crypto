use crate::matrix::matrix_gen::GenElement;
use itertools::Itertools;
use std::fmt;
use std::ops;

#[derive(Debug, Clone)]
pub struct Monomial<T> {
    pub coefficient: T,
    // Empty vector for the constant, to ease the algorithms bellow
    pub exponents: Vec<(String, usize)>,
}

#[derive(Debug, Clone)]
pub struct ExpressionGeneric<T: GenElement> {
    pub monomials: Vec<Monomial<T>>,
}

// Zip iterate on a list of iterator <Coefficient, T>
// Sort by `T`, if 2 elements have the same `T`, then
// apply `coefficient_operator` on them.
struct ZipTuple<C, T> {
    iterators: Vec<std::vec::IntoIter<(C, T)>>,
    currents: Vec<Option<(C, T)>>, // Current element for each iterator
    started: bool,
    coefficient_operator: fn(a: C, b: C) -> C,
}

impl<C: GenElement, T: Ord> Iterator for ZipTuple<C, T>
where
    T: std::cmp::PartialEq + Clone + std::cmp::Ord,
{
    type Item = (C, T);

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started {
            self.currents = self.iterators.iter_mut().map(|i| i.next()).collect();
            self.started = true;
        }

        let min = &self
            .currents
            .iter()
            .flatten()
            .min_by_key(|c| c.1.clone())?
            .1
            .clone();

        let mut ret_coeffs: Vec<C> = Vec::new();
        for (i, iterator) in self.iterators.iter_mut().enumerate() {
            if let Some((ref coeff, ref element)) = self.currents[i] {
                if element == min {
                    ret_coeffs.push(coeff.clone());
                    self.currents[i] = iterator.next();
                }
            }
        }
        let ret_coeff = ret_coeffs.into_iter().reduce(self.coefficient_operator)?;
        if ret_coeff == C::zero() {
            return self.next();
        }
        Some((ret_coeff, min.clone()))
    }
}

impl<C: GenElement> ExpressionGeneric<C> {
    fn _zip_monomials<'a, T>(
        &self,
        rhs: &ExpressionGeneric<C>,
        coefficient_operator: fn(a: C, b: C) -> C,
    ) -> ZipTuple<C, Vec<(String, usize)>> {
        let iter_1 = self
            .monomials
            .iter()
            .map(|m| (m.coefficient.clone(), m.exponents.clone()))
            .collect::<Vec<_>>();
        let iter_2 = rhs
            .monomials
            .iter()
            .map(|m| (m.coefficient.clone(), m.exponents.clone()))
            .collect::<Vec<_>>();
        ZipTuple {
            iterators: vec![iter_1.into_iter(), iter_2.into_iter()],
            currents: vec![None, None],
            started: false,
            coefficient_operator,
        }
    }
}

impl<T: GenElement> fmt::Display for ExpressionGeneric<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ret = self
            .monomials
            .iter()
            .map(|m| {
                if m.coefficient > T::zero() {
                    format!("+ {}", m)
                } else {
                    format!("{}", m)
                }
            })
            .join(" ");
        write!(
            f,
            "{}",
            if ret.starts_with("+ ") {
                ret[2..].into()
            } else {
                ret
            }
        )
    }
}

impl<T: GenElement> fmt::Display for Monomial<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ret = (if self.coefficient == T::one() && !self.exponents.is_empty() {
            None
        } else {
            let mut r = format!("{}", self.coefficient);
            if r.starts_with("-") {
                r = format!("- {}", &r[1..]);
            }
            Some(r)
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
            monomials: self
                ._zip_monomials::<T>(&rhs, |a, b| a + b)
                .map(|(coefficient, exponents)| Monomial {
                    coefficient,
                    exponents,
                })
                .collect(),
        }
    }
}

impl<T: GenElement> ops::Add<T> for ExpressionGeneric<T> {
    type Output = ExpressionGeneric<T>;

    fn add(self, rhs: T) -> ExpressionGeneric<T> {
        let mut ret_monomials = self.monomials.clone();
        match ret_monomials.iter_mut().find(|m| m.exponents.is_empty()) {
            Some(constant) => {
                constant.coefficient = constant.coefficient.clone() + rhs;
            }
            None => {
                ret_monomials.insert(
                    0,
                    Monomial {
                        coefficient: rhs,
                        exponents: vec![],
                    },
                );
            }
        };

        ExpressionGeneric {
            monomials: ret_monomials
                .into_iter()
                .filter(|m| m.coefficient != T::zero())
                .collect(),
        }
    }
}

impl<T: GenElement> ops::Sub<T> for ExpressionGeneric<T> {
    type Output = ExpressionGeneric<T>;

    fn sub(self, rhs: T) -> ExpressionGeneric<T> {
        self + (T::zero() - rhs)
    }
}

impl<T: GenElement> ops::Sub<ExpressionGeneric<T>> for ExpressionGeneric<T> {
    type Output = ExpressionGeneric<T>;

    fn sub(self, rhs: ExpressionGeneric<T>) -> ExpressionGeneric<T> {
        ExpressionGeneric {
            monomials: self
                ._zip_monomials::<T>(&rhs, |a, b| a - b)
                .map(|(coefficient, exponents)| Monomial {
                    coefficient,
                    exponents,
                })
                .collect(),
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
        }
    }
}

impl<T: GenElement> ops::Mul<ExpressionGeneric<T>> for ExpressionGeneric<T> {
    type Output = ExpressionGeneric<T>;

    fn mul(self, rhs: ExpressionGeneric<T>) -> ExpressionGeneric<T> {
        let mut monomials: Vec<_> = vec![];

        for rhs_monomial in &rhs.monomials {
            for lhs_monomial in &self.monomials {
                let iter_1: Vec<(usize, String)> = lhs_monomial
                    .exponents
                    .iter()
                    .map(|(n, e)| (*e, n.clone()))
                    .collect();

                let iter_2: Vec<(usize, String)> = rhs_monomial
                    .exponents
                    .iter()
                    .map(|(n, e)| (*e, n.clone()))
                    .collect();

                let zip_iter: ZipTuple<usize, String> = ZipTuple {
                    iterators: vec![iter_1.into_iter(), iter_2.into_iter()],
                    currents: vec![None, None],
                    started: false,
                    coefficient_operator: |exp_1, exp_2| exp_1 + exp_2,
                };

                monomials.push(Monomial {
                    coefficient: lhs_monomial.coefficient.clone()
                        * rhs_monomial.coefficient.clone(),
                    exponents: zip_iter.map(|(e, n)| (n, e)).collect(),
                });
            }
        }

        let zip_iter = ZipTuple {
            iterators: monomials
                .iter()
                .map(|m| vec![(m.coefficient.clone(), m.exponents.clone())].into_iter())
                .collect::<Vec<_>>(),
            currents: vec![None; monomials.len()],
            started: false,
            coefficient_operator: |exp_1, exp_2| exp_1 + exp_2,
        };

        ExpressionGeneric {
            monomials: zip_iter
                .map(|(coefficient, exponents)| Monomial {
                    coefficient,
                    exponents,
                })
                .collect(),
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
        let mut config: ExpressionGenericConfig<i32> = ExpressionGenericConfig::new();
        let p1 = config.gen("x".into());
        assert_eq!("x", format!("{}", p1));

        let p2 = config.gen("y".into());
        assert_eq!("y", format!("{}", p2));
        assert_eq!("1 + y", format!("{}", p2.clone() + 1));
        assert_eq!("y", format!("{}", (p2.clone() + 1) - 1));
        assert_eq!("5 + 5 * y", format!("{}", (p2.clone() + 1) * 5));
        let p = (p2.clone() * 2 + 3 + p1.clone()) * 5;
        assert_eq!("15 + 5 * x + 10 * y", format!("{}", p));
        let p = p + p2.clone();
        assert_eq!("15 + 5 * x + 11 * y", format!("{}", p));
        let pp = p.clone() + (p.clone() * 3);
        assert_eq!("60 + 20 * x + 44 * y", format!("{}", pp));
        let p = pp.clone() - (p * 3);
        assert_eq!("15 + 5 * x + 11 * y", format!("{}", p));
        let p = p.clone() - (p1.clone() * 5);
        assert_eq!("15 + 11 * y", format!("{}", p));
        let p = p.clone() - (p2.clone() * 9 + 1);
        assert_eq!("14 + 2 * y", format!("{}", p));

        assert_eq!(
            "840 + 280 * x + 40 * x * y + 736 * y + 88 * y ** 2",
            format!("{}", p.clone() * pp.clone())
        );

        let p = p.clone() - (p2.clone() * 2 - 2);
        assert_eq!("16", format!("{}", p));

        // Make `y` term vanish after multiplication
        let p1 = config.gen("y".into()) * 2 - 5;
        let p2 = config.gen("x".into()) * 33 + config.gen("y".into()) * 4 + 10;
        assert_eq!(
            "- 50 - 165 * x + 66 * x * y + 8 * y ** 2",
            format!("{}", p1 * p2)
        );
    }
}
