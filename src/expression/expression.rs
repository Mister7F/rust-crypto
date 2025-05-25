use std::ops::{Add, Mul};

pub trait Expression<T, Matrix>: Add<Self> + Add<T> + Mul<u64>
where
    Self: Sized,
{
    fn constant(&self) -> T;
    fn degree(&self) -> u32;
    fn var_name(&self) -> Option<String>;
    fn to_matrix(equations: Vec<&Self>) -> (Matrix, Vec<T>);
    fn bool(&self) -> bool;
}
