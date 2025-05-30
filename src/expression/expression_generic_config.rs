use crate::expression::expression::ExpressionConfig;
use crate::expression::expression_generic::{ExpressionGeneric, Monomial};
use crate::matrix::matrix_gen::GenElement;

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct ExpressionGenericConfig<T> {
    pub variables: Arc<Mutex<Vec<String>>>,
    _marker: PhantomData<T>,
}

impl<T: GenElement> ExpressionConfig<ExpressionGeneric<T>> for ExpressionGenericConfig<T> {
    fn new() -> Self {
        ExpressionGenericConfig::<T> {
            variables: Arc::new(Mutex::new(vec![])),
            _marker: PhantomData,
        }
    }

    fn gen(&mut self, name: String) -> ExpressionGeneric<T> {
        ExpressionGeneric::<T> {
            monomials: vec![{
                Monomial {
                    exponents: vec![(name, 1usize)],
                    coefficient: T::one(),
                }
            }],
            constant: T::zero(),
        }
    }
}

// Need to wrap all types...
// impl_expression_config_pymethods!(ExpressionGenericConfig, ExpressionGeneric<BigUint>);
