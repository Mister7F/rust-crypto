use crate::utils::bits_to_u64;
use crate::utils::u64_to_bits;
use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::prelude::*;
use std::ops;
use std::ops::Add;
use std::ops::Mul;

#[derive(Debug, Clone)]
#[pyclass(frozen)]
pub struct MatrixBin {
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<u64>,
}

#[pymethods]
impl MatrixBin {
    #[classmethod]
    pub fn from_list(_cls: &Bound<PyType>, lines: Vec<Vec<u8>>) -> Self {
        let rows = lines.len();
        let cols = lines.iter().map(|line| line.len()).max().unwrap_or(0);
        assert!(lines.iter().all(|line| line.len() == cols));

        let cells = lines
            .iter()
            .map(|line| {
                line.chunks(64)
                    .map(|c| bits_to_u64(c.iter().map(|c| c & 1 != 0)))
            })
            .flatten()
            .collect();

        MatrixBin { cols, rows, cells }
    }

    pub fn to_list(&self) -> Vec<Vec<bool>> {
        let stride = (self.cols + 63) / 64;
        self.cells
            .chunks(stride)
            .map(|line| {
                line.iter()
                    .map(|x| u64_to_bits(*x))
                    .flatten()
                    .take(self.cols)
                    .collect()
            })
            .collect()
    }

    pub fn __add__(&self, rhs: &MatrixBin) -> MatrixBin {
        self.add(rhs)
    }

    pub fn __mul__(&self, rhs: &MatrixBin) -> MatrixBin {
        self.mul(rhs)
    }

    pub fn inverse(&self) -> Option<MatrixBin> {
        if self.rows != self.cols {
            return None;
        }

        let n = self.rows;
        let stride = (self.cols + 63) / 64;
        let mut aug = self.clone();
        let mut identity = MatrixBin::identity(n);

        for col in 0..n {
            let mut pivot_row = None;
            for row in col..n {
                let word = aug.cells[row * stride + col / 64];
                if ((word >> (col % 64)) & 1) == 1 {
                    pivot_row = Some(row);
                    break;
                }
            }

            let pivot = match pivot_row {
                Some(p) => p,
                None => return None,
            };

            if pivot != col {
                for k in 0..stride {
                    aug.cells.swap(col * stride + k, pivot * stride + k);
                    identity.cells.swap(col * stride + k, pivot * stride + k);
                }
            }

            for row in 0..n {
                if row != col {
                    let word = aug.cells[row * stride + col / 64];
                    if ((word >> (col % 64)) & 1) == 1 {
                        for k in 0..stride {
                            aug.cells[row * stride + k] ^= aug.cells[col * stride + k];
                            identity.cells[row * stride + k] ^= identity.cells[col * stride + k];
                        }
                    }
                }
            }
        }

        Some(identity)
    }

    pub fn echelon_form(&self, target: Vec<bool>) -> (MatrixBin, Vec<bool>) {
        assert_eq!(target.len(), self.rows);
        let mut aug = self.clone();
        let n = self.rows;
        let stride = (self.cols + 63) / 64;
        let mut target = target.clone();

        for col in 0..n {
            let mut pivot_row = col;
            for row in (col + 1)..n {
                let bit = (aug.cells[row * stride + col / 64] >> (col % 64)) & 1;
                if bit > (aug.cells[pivot_row * stride + col / 64] >> (col % 64)) & 1 {
                    pivot_row = row;
                }
            }

            if pivot_row != col {
                for k in 0..stride {
                    aug.cells.swap(col * stride + k, pivot_row * stride + k);
                }
                target.swap(col, pivot_row);
            }

            let (top, bottom) = aug.cells.split_at_mut((col + 1) * stride);
            let pivot = &top[col * stride..(col + 1) * stride];

            bottom
                .chunks_mut(stride)
                .enumerate()
                .filter(|(_, row)| ((row[col / 64] >> (col % 64)) & 1) == 1)
                .for_each(|(i, row)| {
                    for k in 0..stride {
                        row[k] ^= pivot[k];
                    }
                    target[(col + 1) + i] ^= target[col];
                });
        }

        (aug, target)
    }

    pub fn solve_right(&self, target: Vec<u8>) -> Option<Vec<bool>> {
        let target: Vec<bool> = target.iter().map(|t| t & 1 != 0).collect();

        if target.len() != self.rows {
            return None;
        }

        let (echelon, target) = self.echelon_form(target);
        let stride = (self.cols + 63) / 64;
        let n_unknowns = self.cols;
        let mut res = vec![false; n_unknowns];

        for i in (0..n_unknowns).rev() {
            let row_i = i.min(self.rows - 1);

            let row = &echelon.cells[row_i * stride..(row_i + 1) * stride];

            let mut sum = target[row_i];
            for j in (i + 1)..n_unknowns {
                let bit = (row[j / 64] >> (j % 64)) & 1;
                if bit == 1 {
                    sum ^= res[j];
                }
            }

            let pivot = (row[i / 64] >> (i % 64)) & 1;
            if pivot == 0 {
                if sum && i == row_i {
                    return None;
                }
            } else {
                res[i] = sum;
            }
        }

        Some(res)
    }
}

impl MatrixBin {
    pub fn identity(n: usize) -> MatrixBin {
        let stride = (n + 63) / 64;
        let mut cells = vec![0u64; n * stride];
        for i in 0..n {
            cells[i * stride + i / 64] |= 1u64 << (i % 64);
        }
        MatrixBin {
            rows: n,
            cols: n,
            cells,
        }
    }

    pub fn new(rows: usize, cols: usize) -> MatrixBin {
        let stride = (cols + 63) / 64;
        MatrixBin {
            rows,
            cols,
            cells: vec![0u64; rows * stride],
        }
    }
}

impl ops::Mul<&MatrixBin> for &MatrixBin {
    type Output = MatrixBin;

    fn mul(self, rhs: &MatrixBin) -> MatrixBin {
        assert_eq!(self.cols, rhs.rows);

        let lhs_stride = (self.cols + 63) / 64;
        let rhs_stride = (rhs.cols + 63) / 64;
        let mut result = MatrixBin::new(self.rows, rhs.cols);
        let res_stride = rhs_stride;

        let mut rot: Vec<Vec<u64>> = (0..rhs.cols).map(|_| vec![0u64; lhs_stride]).collect();

        for c in 0..rhs.cols {
            for r in 0..rhs.rows {
                let bit = (rhs.cells[r * rhs_stride + c / 64] >> (c % 64)) & 1;
                if bit == 1 {
                    rot[c][r / 64] |= 1u64 << (r % 64);
                }
            }
        }

        result
            .cells
            .par_chunks_mut(res_stride)
            .enumerate()
            .for_each(|(r, row)| {
                for c in 0..rhs.cols {
                    let mut dot = 0u64;
                    for k in 0..lhs_stride {
                        dot ^= self.cells[r * lhs_stride + k] & rot[c][k];
                    }
                    let count = dot.count_ones() as u64 & 1;
                    row[c / 64] |= count << (c % 64);
                }
            });

        result
    }
}

impl ops::Add<&MatrixBin> for &MatrixBin {
    type Output = MatrixBin;

    fn add(self, rhs: &MatrixBin) -> MatrixBin {
        assert_eq!(self.cols, rhs.cols);
        assert_eq!(self.rows, rhs.rows);

        MatrixBin {
            cols: self.cols,
            rows: self.rows,
            cells: self
                .cells
                .iter()
                .zip(rhs.cells.iter())
                .map(|(a, b)| a ^ b)
                .collect(),
        }
    }
}
