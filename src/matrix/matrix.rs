pub trait Matrix<T>
where
    Self: Sized,
{
    fn from_list(lines: Vec<Vec<T>>) -> Self;
    fn to_list(&self) -> Vec<Vec<T>>;

    fn is_rref(&self) -> bool;
    fn solve_right(&self, target: Vec<T>) -> Result<(Vec<T>, Self, usize), String>;
    fn echelon_form(
        &self,
        target: Vec<T>,
    ) -> Result<(Self, Vec<T>, Vec<Option<usize>>, usize), String>;
    fn right_kernel_matrix(&self) -> Result<Self, String>;
    fn identity(n: usize) -> Self;
    fn inverse(&self) -> Result<Self, String>;
    fn transpose(&self) -> Self;
    fn at(&self, row: usize, col: usize) -> T;
}
