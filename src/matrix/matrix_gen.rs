use num_traits::{One, Zero};

use std::ops;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct MatrixGen<T> {
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<T>,
}

impl<T> MatrixGen<T>
where
    T: Clone
        + Zero
        + One
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::iter::Sum,
{
    pub fn to_list(&self) -> Vec<Vec<T>> {
        self.cells
            .chunks(self.cols)
            .map(|line| line.into())
            .collect()
    }

    pub fn __add__(&self, rhs: &MatrixGen<T>) -> Result<MatrixGen<T>, String> {
        self.add(rhs)
    }

    pub fn __mul__(&self, rhs: &MatrixGen<T>) -> Result<MatrixGen<T>, String> {
        if self.cols != rhs.rows {
            return Err("Dimensions not compatible".into());
        }
        self.mul(rhs)
    }

    pub fn inverse(&self) -> Option<MatrixGen<T>> {
        todo!()
    }

    pub fn echelon_form(&self, mut target: Vec<T>) -> Result<(MatrixGen<T>, Vec<T>), &str> {
        if target.len() != self.rows {
            return Err("Target size does not match number of rows");
        }

        let mut mat = self.clone();

        for col in 0..mat.cols {
            let mut pivot_row = None;
            for row in col..mat.rows {
                if mat._at(row, col) != T::zero() {
                    pivot_row = Some(row);
                    break;
                }
            }

            let pivot_row = match pivot_row {
                Some(row) => row,
                None => continue,
            };

            if pivot_row != col {
                for k in 0..mat.cols {
                    mat.cells.swap(col * mat.cols + k, pivot_row * mat.cols + k);
                }
                target.swap(col, pivot_row);
            }

            for row in (col + 1)..mat.rows {
                let pivot = mat._at(col, col);
                let factor = mat._at(row, col);

                for k in 0..mat.cols {
                    let a = mat._at(row, k) * pivot.clone();
                    let b = mat._at(col, k) * factor.clone();
                    mat.cells[row * mat.cols + k] = a - b;
                }

                target[row] = target[row].clone() * pivot - target[col].clone() * factor;
            }
        }

        Ok((mat, target))
    }

    pub fn solve_right(&self, target: Vec<T>) -> Result<Vec<T>, &str> {
        let (a, target) = self.echelon_form(target)?;
        let n = a.rows;
        let mut x = vec![T::zero(); n];

        for i in (0..n).rev() {
            let mut acc = target[i].clone();
            for j in i + 1..n {
                acc = acc - a._at(i, j) * x[j].clone();
            }

            let diag = a._at(i, i);
            if diag == T::zero() {
                return Err("Singular matrix");
            }

            x[i] = acc / diag;
        }

        Ok(x)
    }

    pub fn identity(n: usize) -> MatrixGen<T> {
        MatrixGen {
            rows: n,
            cols: n,
            cells: (0..n)
                .flat_map(|i| (0..n).map(move |j| if i == j { T::one() } else { T::zero() }))
                .collect(),
        }
    }

    pub fn new(rows: usize, cols: usize) -> MatrixGen<T> {
        MatrixGen {
            rows,
            cols,
            cells: (0..(rows * cols)).map(|_| T::zero()).collect(),
        }
    }

    pub fn from_list(lines: Vec<Vec<T>>) -> Result<Self, String> {
        let cols = lines.iter().map(|l| l.len()).max().unwrap_or(0);
        let rows = lines.len();
        if lines.iter().any(|l| l.len() != cols) {
            return Err("Bad Dimensions".into());
        }

        Ok(MatrixGen {
            rows,
            cols,
            cells: lines.into_iter().flatten().collect(),
        })
    }

    #[inline(always)]
    fn _at(&self, row: usize, col: usize) -> T {
        self.cells[row * self.cols + col].clone()
    }
}

impl<T> ops::Add<&MatrixGen<T>> for &MatrixGen<T>
where
    T: Clone
        + Zero
        + One
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::iter::Sum,
{
    type Output = Result<MatrixGen<T>, String>;

    fn add(self, rhs: &MatrixGen<T>) -> Result<MatrixGen<T>, String> {
        if self.cols != rhs.cols || self.rows != rhs.rows {
            return Err("Dimensions not compatible".into());
        }

        Ok(MatrixGen {
            rows: self.rows,
            cols: self.cols,
            cells: self
                .cells
                .iter()
                .zip(rhs.cells.iter())
                .map(|(a, b)| a.to_owned() + b.to_owned())
                .collect(),
        })
    }
}

impl<T> ops::Mul<&MatrixGen<T>> for &MatrixGen<T>
where
    T: Clone
        + Zero
        + One
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::iter::Sum,
{
    type Output = Result<MatrixGen<T>, String>;

    fn mul(self, rhs: &MatrixGen<T>) -> Result<MatrixGen<T>, String> {
        if self.rows != rhs.cols {
            return Err("Dimensions not compatible".into());
        }

        Ok(MatrixGen {
            rows: self.rows,
            cols: rhs.cols,
            cells: (0..self.rows)
                .flat_map(|i| {
                    (0..rhs.cols).map(move |j| {
                        (0..self.cols)
                            .map(|k| (self._at(i, k) * rhs._at(k, j)))
                            .sum()
                    })
                })
                .collect(),
        })
    }
}

// --------------------------------------------------
//                      TESTS
// --------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    #[test]
    fn test_matrix_generic() {
        let bi = |s: &str| BigInt::parse_bytes(s.as_bytes(), 10).unwrap();

        let a = MatrixGen::<BigInt>::identity(2);
        let b =
            MatrixGen::<BigInt>::from_list(vec![vec![bi("2"), bi("3")], vec![bi("4"), bi("5")]])
                .unwrap();

        let c = (&a + &b).unwrap();
        assert_eq!(
            c.to_list(),
            vec![vec![bi("3"), bi("3")], vec![bi("4"), bi("6")]]
        );

        let a = MatrixGen::<BigInt>::from_list(vec![
            vec![
                bi("100000000000000000000000000000000000000000000000000000000000006"),
                bi("-101"),
            ],
            vec![bi("1"), bi("-1")],
        ])
        .unwrap();

        let c = (&a + &b).unwrap();
        assert_eq!(
            c.to_list(),
            vec![
                vec![
                    bi("100000000000000000000000000000000000000000000000000000000000008"),
                    bi("-98")
                ],
                vec![bi("5"), bi("4")]
            ]
        );

        let c = (&a * &b).unwrap();
        assert_eq!(
            c.to_list(),
            vec![
                vec![
                    bi("199999999999999999999999999999999999999999999999999999999999608"),
                    bi("299999999999999999999999999999999999999999999999999999999999513")
                ],
                vec![bi("-2"), bi("-2")]
            ]
        );

        let a =
            MatrixGen::<BigInt>::from_list(vec![vec![bi("16"), bi("4")], vec![bi("8"), bi("16")]])
                .unwrap();
        assert_eq!(
            a.solve_right(vec![bi("896"), bi("1792")]).unwrap(),
            vec![bi("32"), bi("96")],
        );
    }
}
