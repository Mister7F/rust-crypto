use num_traits::{One, Zero};

use crate::rings::fraction::Fraction;
use std::ops;
use std::ops::{Add, Div, Mul, Sub};

pub trait GenElement:  // Avoid repeating all the traits
    Clone
    + Zero
    + One
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + std::iter::Sum<Self>
    + std::fmt::Display
    + std::cmp::Ord
    + std::fmt::Debug
{
}

impl<T> GenElement for T where
    T: Clone
        + Zero
        + One
        + PartialEq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + std::iter::Sum<T>
        + std::fmt::Display
        + std::cmp::Ord
        + std::fmt::Debug
{
}

#[derive(Debug, Clone)]
pub struct MatrixGen<T> {
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<T>,
}

impl<T: GenElement> MatrixGen<T> {
    // TODO: implement `Matrix` trait
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

    pub fn echelon_form(&self, mut target: Vec<T>) -> Result<(MatrixGen<T>, Vec<T>, usize), &str> {
        if target.len() != self.rows {
            return Err("Target size does not match number of rows");
        }

        let mut mat = self.clone();
        let mut rank = 0;
        let mut row = 0;

        for col in 0..mat.cols {
            let mut pivot_row = None;
            for r in row..mat.rows {
                if mat._at(r, col) != T::zero() {
                    pivot_row = Some(r);
                    break;
                }
            }

            let pivot_row = match pivot_row {
                Some(r) => r,
                None => continue,
            };

            if pivot_row != row {
                for k in 0..mat.cols {
                    mat.cells.swap(row * mat.cols + k, pivot_row * mat.cols + k);
                }
                target.swap(row, pivot_row);
            }

            let pivot_val = mat._at(row, col);

            for r in 0..mat.rows {
                if r == row {
                    continue;
                }

                let factor = mat._at(r, col);
                for k in 0..mat.cols {
                    let a = mat._at(r, k) * pivot_val.clone();
                    let b = mat._at(row, k) * factor.clone();
                    mat.cells[r * mat.cols + k] = a - b;
                }

                let a = target[r].clone() * pivot_val.clone();
                let b = target[row].clone() * factor;
                target[r] = a - b;
            }

            rank += 1;
            row += 1;
            if row >= mat.rows {
                break;
            }
        }

        Ok((mat, target, rank))
    }

    pub fn solve_right(&self, target: Vec<T>) -> Result<(Vec<T>, MatrixGen<T>, usize), &str> {
        let (echelon, target, rank) = self.echelon_form(target)?;
        let n = echelon.rows;
        let mut x = vec![T::zero(); n];

        for i in (0..n).rev() {
            let mut acc = target[i].clone();
            for j in i + 1..n {
                acc = acc - echelon._at(i, j) * x[j].clone();
            }

            let diag = echelon._at(i, i);
            if diag == T::zero() {
                return Err("Singular matrix");
            }

            x[i] = acc / diag;
        }

        Ok((x, echelon, rank))
    }

    pub fn right_kernel_matrix(&self) -> MatrixGen<T> {
        let mut mat = self.clone();
        if !self.is_rref() {
            let zero_vec = vec![T::zero(); self.rows];
            mat = self.echelon_form(zero_vec).unwrap().0;
        }

        let mut pivot_col_for_row = vec![None; mat.rows];
        let mut pivot_row_for_col = vec![None; mat.cols];

        for r in 0..mat.rows {
            for c in 0..mat.cols {
                if mat._at(r, c) != T::zero() {
                    pivot_col_for_row[r] = Some(c);
                    pivot_row_for_col[c] = Some(r);
                    break;
                }
            }
        }

        let free_cols: Vec<usize> = (0..mat.cols)
            .filter(|&c| pivot_row_for_col[c].is_none())
            .collect();

        let mut basis = vec![];

        for &free_col in &free_cols {
            let mut vec = vec![T::zero(); mat.cols];
            vec[free_col] = T::one();

            for r in (0..mat.rows).rev() {
                if let Some(pivot_col) = pivot_col_for_row[r] {
                    let mut acc = T::zero();
                    for c in pivot_col + 1..mat.cols {
                        acc = acc + mat._at(r, c) * vec[c].clone();
                    }
                    let pivot_val = mat._at(r, pivot_col);
                    vec[pivot_col] = (T::zero() - acc) / pivot_val;
                }
            }

            basis.push(vec);
        }

        (MatrixGen {
            rows: mat.cols,
            cols: basis.len(),
            cells: basis.into_iter().flatten().collect(),
        })
        .transpose()
    }

    pub fn is_rref(&self) -> bool {
        let mut lead = None;

        for i in 0..self.rows {
            let row = &self.cells[i * self.cols..(i + 1) * self.cols];
            let pivot_col_opt = row.iter().position(|x| *x != T::zero());

            match pivot_col_opt {
                None => {
                    for r in i + 1..self.rows {
                        let next_row = &self.cells[r * self.cols..(r + 1) * self.cols];
                        if next_row.iter().any(|x| *x != T::zero()) {
                            return false;
                        }
                    }
                    break;
                }
                Some(pivot_col) => {
                    if let Some(prev_lead) = lead {
                        if pivot_col <= prev_lead {
                            return false;
                        }
                    }
                    lead = Some(pivot_col);

                    if row[pivot_col] != T::one() {
                        return false;
                    }

                    for r in 0..self.rows {
                        if r != i && self._at(r, pivot_col) != T::zero() {
                            return false;
                        }
                    }
                }
            }
        }
        true
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

    pub fn transpose(&self) -> MatrixGen<T> {
        MatrixGen {
            rows: self.cols,
            cols: self.rows,
            cells: (0..self.cols)
                .map(|c| (0..self.rows).map(move |r| self._at(r, c)))
                .flatten()
                .collect(),
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

impl<T: GenElement> ops::Add<&MatrixGen<T>> for &MatrixGen<T> {
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

impl<T: GenElement> ops::Mul<&MatrixGen<T>> for &MatrixGen<T> {
    type Output = Result<MatrixGen<T>, String>;

    fn mul(self, rhs: &MatrixGen<T>) -> Result<MatrixGen<T>, String> {
        if self.cols != rhs.rows {
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
            a.solve_right(vec![bi("896"), bi("1792")]).unwrap().0,
            vec![bi("32"), bi("96")],
        );
    }

    #[test]
    fn test_matrix_generic_kernel() {
        let m = MatrixGen::<i64>::from_list(vec![
            vec![1, 2, 5, 77],
            vec![3, 4, 7, 11],
            vec![4, 8, 9, 1],
        ])
        .unwrap();

        assert_eq!(
            m.transpose().to_list(),
            vec![vec![1, 3, 4], vec![2, 4, 8], vec![5, 7, 9], vec![77, 11, 1],]
        );

        assert_eq!(m.solve_right(vec![11, 11, 33]).unwrap().0, vec![-8, 7, 1]);

        let m = MatrixGen::<Fraction>::from_list(vec![
            vec![
                Fraction::from_str("1").unwrap(),
                Fraction::from_str("2").unwrap(),
                Fraction::from_str("5").unwrap(),
                Fraction::from_str("77").unwrap(),
            ],
            vec![
                Fraction::from_str("3").unwrap(),
                Fraction::from_str("4").unwrap(),
                Fraction::from_str("7").unwrap(),
                Fraction::from_str("11").unwrap(),
            ],
            vec![
                Fraction::from_str("4").unwrap(),
                Fraction::from_str("8").unwrap(),
                Fraction::from_str("9").unwrap(),
                Fraction::from_str("1").unwrap(),
            ],
        ])
        .unwrap();

        let res = m
            .solve_right(vec![
                Fraction::from_str("11").unwrap(),
                Fraction::from_str("11").unwrap(),
                Fraction::from_str("33").unwrap(),
            ])
            .unwrap()
            .0;
        assert_eq!(res, vec![-8, 7, 1]);
        assert_ne!(res, vec![8, 7, 1]);
        assert_ne!(res, vec![-8, 8, 1]);
        assert_ne!(res, vec![-8, 7, 2]);
        assert_ne!(res, vec![-16, 14, 2]);

        let expected = vec![
            // TODO: make same result as sage math
            // Sage math in python produce: [-652  -18  307  -11]
            vec![
                Fraction::from_str("652/11").unwrap(),
                Fraction::from_str("18/11").unwrap(),
                Fraction::from_str("-307/11").unwrap(),
                Fraction::from_str("11/11").unwrap(),
            ],
        ];
        assert_eq!(m.right_kernel_matrix().to_list(), expected);

        assert_eq!(
            (&m * &m.right_kernel_matrix().transpose())
                .unwrap()
                .to_list(),
            vec![vec![0], vec![0], vec![0]]
        );
        assert_ne!(
            (&m * &m.right_kernel_matrix().transpose())
                .unwrap()
                .to_list(),
            vec![vec![0], vec![1], vec![0]]
        );
    }
}
