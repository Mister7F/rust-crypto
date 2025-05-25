use std::ops::{Add, Mul};

pub trait ExpressionConfig<Expression> {
    fn new() -> Self;

    fn gen(&mut self, name: String) -> Expression;

    fn gens(&mut self, name: String, n: usize) -> Vec<Expression> {
        (0..n)
            .map(|i| self.gen(name.as_str().to_owned() + "_" + &i.to_string()))
            .collect()
    }
}

pub trait Expression<T, Matrix, Config>: Add<Self> + Add<T> + Mul<u64>
where
    Self: Sized,
    Config: ExpressionConfig<Self>,
{
    fn constant(&self) -> T;
    fn degree(&self) -> u32;
    fn var_name(&self) -> Option<String>;
    fn to_matrix(equations: Vec<&Self>) -> (Matrix, Vec<T>);
    fn bool(&self) -> bool;
}

// Macro to create the python interface
#[macro_export]
macro_rules! impl_expression_config_pymethods {
    ($type:ty) => {
        #[pymethods]
        impl $type {
            #[new]
            fn new() -> Self {
                ExpressionConfig::new()
            }

            pub fn gen(&mut self, name: String) -> ExpressionBin {
                ExpressionConfig::gen(self, name)
            }

            pub fn gens(&mut self, name: String, n: usize) -> Vec<ExpressionBin> {
                ExpressionConfig::gens(self, name, n)
            }
        }
    };
}
