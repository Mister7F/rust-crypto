use crate::matrix::matrix::Matrix;
use crate::utils::bits_to_u64;
use crate::utils::u64_to_bits;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::prelude::*;
use std::ops;
use std::ops::Add;
use std::ops::Mul;

#[derive(Debug, Clone)]
#[pyclass(frozen)]
pub struct MatrixBin {
    pub ncols: usize,
    pub nrows: usize,
    pub cells: Vec<u64>,
}

#[pymethods]
impl MatrixBin {
    #[classmethod]
    pub fn from_list(_cls: &Bound<PyType>, lines: Vec<Vec<u8>>) -> Self {
        Matrix::from_list(
            lines
                .iter()
                .map(|l| l.iter().map(|x| x & 1 != 0).collect())
                .collect(),
        )
    }

    pub fn to_list(&self) -> Vec<Vec<bool>> {
        Matrix::to_list(self)
    }

    pub fn __add__(&self, rhs: &MatrixBin) -> PyResult<MatrixBin> {
        match self.add(rhs) {
            Ok(result) => Ok(result),
            Err(error) => Err(PyValueError::new_err(error)),
        }
    }

    pub fn __mul__(&self, rhs: &MatrixBin) -> PyResult<MatrixBin> {
        match self.mul(rhs) {
            Ok(result) => Ok(result),
            Err(error) => Err(PyValueError::new_err(error)),
        }
    }

    pub fn at(&self, row: usize, col: usize) -> bool {
        Matrix::at(self, row, col)
    }

    #[getter]
    pub fn T(&self) -> MatrixBin {
        self.transpose()
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn __getitem__(&self, pos: (usize, usize)) -> bool {
        self.at(pos.0, pos.1)
    }

    pub fn echelon_form(
        &self,
        target: Vec<bool>,
    ) -> PyResult<(MatrixBin, Vec<bool>, Vec<Option<usize>>, usize)> {
        match Matrix::echelon_form(self, target) {
            Ok(value) => Ok(value),
            Err(error) => Err(PyValueError::new_err(error)),
        }
    }

    pub fn solve_right(&self, target: Vec<u8>) -> PyResult<(Vec<bool>, MatrixBin, usize)> {
        match Matrix::solve_right(self, target.iter().map(|t| t & 1 != 0).collect()) {
            Ok(value) => Ok(value),
            Err(error) => Err(PyValueError::new_err(error)),
        }
    }

    pub fn right_kernel_matrix(&self) -> PyResult<MatrixBin> {
        match Matrix::right_kernel_matrix(self) {
            Ok(value) => Ok(value),
            Err(error) => Err(PyValueError::new_err(error)),
        }
    }

    pub fn inverse(&self) -> PyResult<MatrixBin> {
        match Matrix::inverse(self) {
            Ok(value) => Ok(value),
            Err(error) => Err(PyValueError::new_err(error)),
        }
    }
}

impl Matrix<bool> for MatrixBin {
    fn from_list(lines: Vec<Vec<bool>>) -> Self {
        let nrows = lines.len();
        let ncols = lines.iter().map(|line| line.len()).max().unwrap_or(0);
        assert!(lines.iter().all(|line| line.len() == ncols));

        let cells = lines
            .iter()
            .flat_map(|line| line.chunks(64).map(|c| bits_to_u64(c.iter().copied())))
            .collect();

        MatrixBin {
            ncols,
            nrows,
            cells,
        }
    }

    fn to_list(&self) -> Vec<Vec<bool>> {
        let stride = self.ncols.div_ceil(64);
        self.cells
            .chunks(stride)
            .map(|line| {
                line.iter()
                    .flat_map(|x| u64_to_bits(*x))
                    .take(self.ncols)
                    .collect()
            })
            .collect()
    }

    fn is_rref(&self) -> bool {
        let mut last_pivot_col = None;

        for row in 0..self.nrows {
            let pivot_col_opt = (0..self.ncols).find(|&col| self.at(row, col));

            if let Some(pivot_col) = pivot_col_opt {
                if let Some(last) = last_pivot_col {
                    if pivot_col <= last {
                        return false;
                    }
                }

                for r in 0..self.nrows {
                    if r != row && self.at(r, pivot_col) {
                        return false;
                    }
                }

                last_pivot_col = Some(pivot_col);
            } else {
                if (0..self.ncols).any(|col| self.at(row, col)) {
                    return false;
                }
            }
        }

        true
    }

    fn solve_right(&self, target: Vec<bool>) -> Result<(Vec<bool>, MatrixBin, usize), String> {
        let (echelon, target, pivot_map, rank) = self.echelon_form(target).unwrap(); // TODO `?`

        let n_vars = self.ncols;
        let stride = self.ncols.div_ceil(64);

        let mut solution = vec![false; n_vars];
        let mut pivot_positions = vec![None; n_vars];

        for (row_idx, pivot_col_opt) in pivot_map.iter().enumerate() {
            if let Some(pivot_col) = pivot_col_opt {
                pivot_positions[*pivot_col] = Some(row_idx);
            } else if target[row_idx] {
                return Err("Impossible system".into());
            }
        }

        for col in (0..n_vars).rev() {
            if let Some(row_idx) = pivot_positions[col] {
                let row = &echelon.cells[row_idx * stride..(row_idx + 1) * stride];
                let mut val = target[row_idx];

                for j in (col + 1)..n_vars {
                    if (row[j / 64] >> (j % 64)) & 1 == 1 {
                        val ^= solution[j];
                    }
                }

                solution[col] = val;
            }
        }

        Ok((solution, echelon, rank))
    }

    fn echelon_form(
        &self,
        target: Vec<bool>,
    ) -> Result<(MatrixBin, Vec<bool>, Vec<Option<usize>>, usize), String> {
        if target.len() != self.nrows {
            return Err("Target size does not match the number of rows".into());
        }

        let mut aug = self.clone();
        let mut target = target.clone();
        let n = self.nrows;
        let stride = self.ncols.div_ceil(64);
        let mut pivot_cols = vec![None; n];

        let mut rank = 0;

        for col in 0..self.ncols {
            let mut pivot_row = None;
            for row in rank..n {
                let bit = (aug.cells[row * stride + col / 64] >> (col % 64)) & 1;
                if bit == 1 {
                    pivot_row = Some(row);
                    break;
                }
            }

            if let Some(pivot) = pivot_row {
                if pivot != rank {
                    for k in 0..stride {
                        aug.cells.swap(rank * stride + k, pivot * stride + k);
                    }
                    target.swap(rank, pivot);
                }

                pivot_cols[rank] = Some(col);

                let pivot_slice = aug.cells[rank * stride..(rank + 1) * stride].to_vec();

                for row in (rank + 1)..n {
                    let bit = (aug.cells[row * stride + col / 64] >> (col % 64)) & 1;
                    if bit == 1 {
                        for k in 0..stride {
                            aug.cells[row * stride + k] ^= pivot_slice[k];
                        }
                        target[row] ^= target[rank];
                    }
                }

                rank += 1;
            }
        }

        // Row Echelon Form -> Reduced Row Echelon Form
        for i in (0..rank).rev() {
            if let Some(pivot_col) = pivot_cols[i] {
                for row_above in 0..i {
                    let bit =
                        (aug.cells[row_above * stride + pivot_col / 64] >> (pivot_col % 64)) & 1;
                    if bit == 1 {
                        for k in 0..stride {
                            aug.cells[row_above * stride + k] ^= aug.cells[i * stride + k];
                        }
                        target[row_above] ^= target[i];
                    }
                }
            }
        }

        Ok((aug, target, pivot_cols, rank))
    }

    fn right_kernel_matrix(&self) -> Result<MatrixBin, String> {
        let mut mat = self.clone();
        if !self.is_rref() {
            mat = Matrix::echelon_form(self, vec![false; self.nrows])?.0;
        }

        let stride = mat.ncols.div_ceil(64);

        let mut pivot_pos = vec![None; mat.nrows];
        for row in 0..mat.nrows {
            for col in 0..mat.ncols {
                let word = mat.cells[row * stride + col / 64];
                if ((word >> (col % 64)) & 1) == 1 {
                    pivot_pos[row] = Some(col);
                    break;
                }
            }
        }

        let mut is_pivot = vec![false; mat.ncols];
        for &p in &pivot_pos {
            if let Some(col) = p {
                is_pivot[col] = true;
            }
        }

        let free_cols: Vec<usize> = (0..mat.ncols).filter(|&c| !is_pivot[c]).collect();

        let nullity = free_cols.len();
        let null_stride = nullity.div_ceil(64);

        let mut nullspace_cells = vec![0u64; mat.ncols * null_stride];

        for (idx, &free_col) in free_cols.iter().enumerate() {
            let row = free_col;
            let cell_idx = row * null_stride + idx / 64;
            let bit_pos = idx % 64;
            nullspace_cells[cell_idx] |= 1u64 << bit_pos;

            for (row_idx, &pivot_col_opt) in pivot_pos.iter().enumerate() {
                if let Some(pivot_col) = pivot_col_opt {
                    let word = mat.cells[row_idx * stride + free_col / 64];
                    if ((word >> (free_col % 64)) & 1) == 1 {
                        let cell_idx = pivot_col * null_stride + idx / 64;
                        nullspace_cells[cell_idx] ^= 1u64 << bit_pos;
                    }
                }
            }
        }

        Ok(MatrixBin {
            nrows: mat.ncols,
            ncols: nullity,
            cells: nullspace_cells,
        })
    }

    fn identity(n: usize) -> Self {
        let stride = n.div_ceil(64);
        let mut cells = vec![0u64; n * stride];
        for i in 0..n {
            cells[i * stride + i / 64] |= 1u64 << (i % 64);
        }
        MatrixBin {
            nrows: n,
            ncols: n,
            cells,
        }
    }

    fn inverse(&self) -> Result<MatrixBin, String> {
        if self.nrows != self.ncols {
            return Err("Matrix is not square".into());
        }

        let n = self.nrows;
        let stride = self.ncols.div_ceil(64);
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

            let pivot = pivot_row.ok_or("Not invertible")?;
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

        Ok(identity)
    }

    fn transpose(&self) -> MatrixBin {
        let col_stride = self.ncols.div_ceil(64);
        let row_stride = self.nrows.div_ceil(64);
        let mut rot: Vec<Vec<u64>> = (0..self.ncols).map(|_| vec![0u64; row_stride]).collect();

        for c in 0..self.ncols {
            for r in 0..self.nrows {
                let bit = (self.cells[r * col_stride + c / 64] >> (c % 64)) & 1;
                if bit == 1 {
                    rot[c][r / 64] |= 1u64 << (r % 64);
                }
            }
        }

        MatrixBin {
            ncols: self.nrows,
            nrows: self.ncols,
            cells: rot.into_iter().flatten().collect(),
        }
    }

    fn at(&self, row: usize, col: usize) -> bool {
        let stride = self.ncols.div_ceil(64);
        let cell_idx = row * stride + col / 64;
        let bit_pos = col % 64;
        if cell_idx >= self.cells.len() {
            return false;
        }
        ((self.cells[cell_idx] >> bit_pos) & 1) == 1
    }
}

impl MatrixBin {
    // TODO: move in Matrix trait
    pub fn new(nrows: usize, ncols: usize) -> Self {
        let stride = ncols.div_ceil(64);
        MatrixBin {
            nrows,
            ncols,
            cells: vec![0u64; nrows * stride],
        }
    }
}

impl ops::Mul<&MatrixBin> for &MatrixBin {
    type Output = Result<MatrixBin, String>;

    fn mul(self, rhs: &MatrixBin) -> Result<MatrixBin, String> {
        if self.ncols != rhs.nrows {
            return Err("Dimensions not compatible".into());
        }

        let lhs_stride = self.ncols.div_ceil(64);
        let rhs_stride = rhs.ncols.div_ceil(64);
        let mut result = MatrixBin::new(self.nrows, rhs.ncols);
        let res_stride = rhs_stride;

        let rot = rhs.transpose();

        result
            .cells
            .par_chunks_mut(res_stride)
            .enumerate()
            .for_each(|(r, row)| {
                for c in 0..rhs.ncols {
                    let mut dot = 0u64;
                    for k in 0..lhs_stride {
                        dot ^= self.cells[r * lhs_stride + k] & rot.cells[c * lhs_stride + k];
                    }
                    let count = dot.count_ones() as u64 & 1;
                    row[c / 64] |= count << (c % 64);
                }
            });

        Ok(result)
    }
}

impl ops::Add<&MatrixBin> for &MatrixBin {
    type Output = Result<MatrixBin, String>;

    fn add(self, rhs: &MatrixBin) -> Result<MatrixBin, String> {
        if self.ncols != rhs.ncols || self.nrows != rhs.nrows {
            return Err("Dimensions not compatible".into());
        }

        Ok(MatrixBin {
            ncols: self.ncols,
            nrows: self.nrows,
            cells: self
                .cells
                .iter()
                .zip(rhs.cells.iter())
                .map(|(a, b)| a ^ b)
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

    #[test]
    fn test_kernel_1() {
        let row_echelon_form = <MatrixBin as Matrix<bool>>::from_list(vec![
            vec![true, true, false],
            vec![false, true, true],
        ]);
        let reduced_row_echelon_form = row_echelon_form.echelon_form(vec![false, false]).unwrap().0;
        assert_eq!(
            reduced_row_echelon_form.to_list(),
            vec![vec![true, false, true], vec![false, true, true]]
        );
        let kernel = reduced_row_echelon_form.right_kernel_matrix().unwrap();
        assert_eq!(kernel.to_list(), vec![vec![true], vec![true], vec![true]]);

        assert!(!row_echelon_form.is_rref());
        assert!(reduced_row_echelon_form.is_rref());
    }

    #[test]
    fn test_kernel_2() {
        let row_echelon_form = <MatrixBin as Matrix<bool>>::from_list(vec![
            vec![true, true, true],
            vec![false, false, true],
        ]);
        let reduced_row_echelon_form = row_echelon_form.echelon_form(vec![false, false]).unwrap().0;
        assert_eq!(
            reduced_row_echelon_form.to_list(),
            vec![vec![true, true, false], vec![false, false, true]]
        );
        let kernel = reduced_row_echelon_form.right_kernel_matrix().unwrap();
        assert_eq!(kernel.to_list(), vec![vec![true], vec![true], vec![false]]);

        assert!(!row_echelon_form.is_rref());
        assert!(reduced_row_echelon_form.is_rref());
    }

    #[test]
    fn test_kernel_3() {
        let m = <MatrixBin as Matrix<bool>>::from_list(vec![
            vec![true, true, false],
            vec![false, false, true],
            vec![true, true, true],
        ]);
        let reduced_row_echelon_form = m.echelon_form(vec![false, false, false]).unwrap().0;
        assert_eq!(
            reduced_row_echelon_form.to_list(),
            vec![
                vec![true, true, false],
                vec![false, false, true],
                vec![false, false, false]
            ]
        );
        let kernel = reduced_row_echelon_form.right_kernel_matrix().unwrap();
        assert_eq!(kernel.to_list(), vec![vec![true], vec![true], vec![false]]);

        assert!(!m.is_rref());
        assert!(reduced_row_echelon_form.is_rref());
    }

    #[test]
    fn test_kernel_4() {
        let m = <MatrixBin as Matrix<bool>>::from_list(vec![
            vec![true, true, false],
            vec![false, false, true],
            vec![true, true, true],
            vec![true, true, true],
        ]);
        let reduced_row_echelon_form = m.echelon_form(vec![false, false, false, false]).unwrap().0;
        assert_eq!(
            reduced_row_echelon_form.to_list(),
            vec![
                vec![true, true, false],
                vec![false, false, true],
                vec![false, false, false],
                vec![false, false, false]
            ]
        );
        let kernel = reduced_row_echelon_form.right_kernel_matrix().unwrap();
        assert_eq!(kernel.to_list(), vec![vec![true], vec![true], vec![false]]);

        assert!(!m.is_rref());
        assert!(reduced_row_echelon_form.is_rref());
    }
}
