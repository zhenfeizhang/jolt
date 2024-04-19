#![allow(clippy::too_many_arguments)]
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::utils::{self, compute_dotproduct, compute_dotproduct_low_optimized};

use crate::utils::math::Math;
use core::ops::Index;
use ff::PrimeField;
use rayon::prelude::*;
use std::ops::AddAssign;

#[derive(Debug, Clone, PartialEq)]
pub struct DensePolynomial<F> {
    num_vars: usize, // the number of variables in the multilinear polynomial
    len: usize,
    Z: Vec<F>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<F: PrimeField> DensePolynomial<F> {
    pub fn new(Z: Vec<F>) -> Self {
        assert!(
            utils::is_power_of_two(Z.len()),
            "Dense multi-linear polynomials must be made from a power of 2 (not {})",
            Z.len()
        );

        DensePolynomial {
            num_vars: Z.len().log_2(),
            len: Z.len(),
            Z,
        }
    }

    pub fn new_padded(evals: Vec<F>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_evals = evals;
        while !(utils::is_power_of_two(poly_evals.len())) {
            poly_evals.push(F::ZERO);
        }

        DensePolynomial {
            num_vars: poly_evals.len().log_2(),
            len: poly_evals.len(),
            Z: poly_evals,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn clone(&self) -> Self {
        Self::new(self.Z[0..self.len].to_vec())
    }

    pub fn split(&self, idx: usize) -> (Self, Self) {
        assert!(idx < self.len());
        (
            Self::new(self.Z[..idx].to_vec()),
            Self::new(self.Z[idx..2 * idx].to_vec()),
        )
    }

    pub fn split_evals(&self, idx: usize) -> (&[F], &[F]) {
        (&self.Z[..idx], &self.Z[idx..])
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            *a += *r * (*b - *a);
        });

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_top_par(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(a, b)| {
                *a += *r * (*b - *a);
            });

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_top_many_ones(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.iter_mut()
            .zip(right.iter())
            .filter(|(&mut a, &b)| a != b)
            .for_each(|(a, b)| {
                let m = *b - *a;
                if m == F::ONE {
                    *a += *r;
                } else {
                    *a += *r * m;
                }
            });

        self.num_vars -= 1;
        self.len = n;
    }

    /// Bounds the polynomial's most significant index bit to 'r' optimized for a
    /// high P(eval = 0).
    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_top_zero_optimized(&mut self, r: &F) {
        let n = self.len() / 2;

        let (left, right) = self.Z.split_at_mut(n);

        left.par_iter_mut()
            .zip(right.par_iter())
            .filter(|(&mut a, &b)| a != b)
            .for_each(|(a, b)| {
                *a += *r * (*b - *a);
            });

        self.Z.resize(n, F::ZERO);
        self.num_vars -= 1;
        self.len = n;
    }

    #[tracing::instrument(skip_all)]
    pub fn new_poly_from_bound_poly_var_top(&self, r: &F) -> Self {
        let n = self.len() / 2;
        let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(n);

        for i in 0..n {
            // let low' = low + r * (high - low)
            let low = self.Z[i];
            let high = self.Z[i + n];
            if !(low.is_zero_vartime() && high.is_zero_vartime()) {
                let m = high - low;
                new_evals[i] = low + *r * m;
            }
        }
        let num_vars = self.num_vars - 1;
        let len = n;

        Self {
            num_vars,
            len,
            Z: new_evals,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn new_poly_from_bound_poly_var_top_flags(&self, r: &F) -> Self {
        let n = self.len() / 2;
        let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(n);

        for i in 0..n {
            // let low' = low + r * (high - low)
            // Special truth table here
            //         high 0   high 1
            // low 0     0        r
            // low 1   (1-r)      1
            let low = self.Z[i];
            let high = self.Z[i + n];

            if low.is_zero_vartime() {
                if high == F::ONE {
                    new_evals[i] = *r;
                } else if !high.is_zero_vartime() {
                    panic!("Shouldn't happen for a flag poly");
                }
            } else if low == F::ONE {
                if high == F::ONE {
                    new_evals[i] = F::ONE;
                } else if high.is_zero_vartime() {
                    new_evals[i] = F::ONE - r;
                } else {
                    panic!("Shouldn't happen for a flag poly");
                }
            }
        }
        let num_vars = self.num_vars - 1;
        let len = n;

        Self {
            num_vars,
            len,
            Z: new_evals,
        }
    }

    pub fn bound_poly_var_bot(&mut self, r: &F) {
        let n = self.len() / 2;
        for i in 0..n {
            self.Z[i] = self.Z[2 * i] + *r * (self.Z[2 * i + 1] - self.Z[2 * i]);
        }
        self.num_vars -= 1;
        self.len = n;
    }

    // returns Z(r) in O(n) time
    pub fn evaluate(&self, r: &[F]) -> F {
        // r must have a value for each variable
        assert_eq!(r.len(), self.get_num_vars());
        let chis = EqPolynomial::new(r.to_vec()).evals();
        assert_eq!(chis.len(), self.Z.len());
        compute_dotproduct(&self.Z, &chis)
    }

    pub fn evaluate_at_chi(&self, chis: &Vec<F>) -> F {
        compute_dotproduct(&self.Z, &chis)
    }

    pub fn evaluate_at_chi_low_optimized(&self, chis: &Vec<F>) -> F {
        assert_eq!(self.Z.len(), chis.len());
        compute_dotproduct_low_optimized(&self.Z, &chis)
    }

    pub fn evals(&self) -> Vec<F> {
        self.Z.clone()
    }

    pub fn evals_ref(&self) -> &[F] {
        self.Z.as_ref()
    }

    #[tracing::instrument(skip_all, name = "DensePoly::flatten")]
    pub fn flatten(polys: &[DensePolynomial<F>]) -> Vec<F> {
        let poly_len = polys[0].len();
        polys
            .iter()
            .for_each(|poly| assert_eq!(poly_len, poly.len()));

        let num_polys = polys.len();
        let flat_len = num_polys * poly_len;
        let mut flat: Vec<F> = unsafe_allocate_zero_vec(flat_len);
        flat.par_chunks_mut(poly_len)
            .enumerate()
            .for_each(|(poly_index, result)| {
                let evals = polys[poly_index].evals_ref();
                for (eval_index, eval) in evals.iter().enumerate() {
                    result[eval_index] = *eval;
                }
            });
        flat
    }

    #[tracing::instrument(skip_all, name = "DensePolynomial::from")]
    pub fn from_usize(Z: &[usize]) -> Self {
        DensePolynomial::new(
            (0..Z.len())
                .map(|i| F::from(Z[i] as u64))
                .collect::<Vec<F>>(),
        )
    }

    #[tracing::instrument(skip_all, name = "DensePolynomial::from")]
    pub fn from_u64(Z: &[u64]) -> Self {
        DensePolynomial::new((0..Z.len()).map(|i| F::from(Z[i])).collect::<Vec<F>>())
    }
}

impl<F> Index<usize> for DensePolynomial<F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, _index: usize) -> &F {
        &(self.Z[_index])
    }
}

impl<F> AsRef<DensePolynomial<F>> for DensePolynomial<F> {
    fn as_ref(&self) -> &DensePolynomial<F> {
        self
    }
}

impl<F: PrimeField> AddAssign<&DensePolynomial<F>> for DensePolynomial<F> {
    fn add_assign(&mut self, rhs: &DensePolynomial<F>) {
        assert_eq!(self.num_vars, rhs.num_vars);
        assert_eq!(self.len, rhs.len);
        let summed_evaluations: Vec<F> = self.Z.iter().zip(&rhs.Z).map(|(a, b)| *a + *b).collect();

        *self = Self {
            num_vars: self.num_vars,
            len: self.len,
            Z: summed_evaluations,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::hyrax::matrix_dimensions;

    use super::*;
    use ark_std::test_rng;
    use ff::PrimeField;
    use goldilocks::Goldilocks;
    use goldilocks::GoldilocksExt2;
    use halo2curves::bn256::Fr;

    fn evaluate_with_LR<F: PrimeField>(Z: &[F], r: &[F]) -> F {
        let ell = r.len();
        let (L_size, _R_size) = matrix_dimensions(ell, 1);
        let eq = EqPolynomial::<F>::new(r.to_vec());
        let (L, R) = eq.compute_factored_evals(L_size);

        // ensure ell is even
        assert!(ell % 2 == 0);
        // compute n = 2^\ell
        let n = ell.pow2();
        // compute m = sqrt(n) = 2^{\ell/2}
        let m = n.square_root();

        // compute vector-matrix product between L and Z viewed as a matrix
        let LZ = (0..m)
            .map(|i| (0..m).map(|j| L[j] * Z[j * m + i]).sum())
            .collect::<Vec<F>>();

        // compute dot product between LZ and R
        compute_dotproduct(&LZ, &R)
    }

    #[test]
    fn check_polynomial_evaluation() {
        check_polynomial_evaluation_helper::<Fr>();
        check_polynomial_evaluation_helper::<Goldilocks>();
        check_polynomial_evaluation_helper::<GoldilocksExt2>();
    }

    fn check_polynomial_evaluation_helper<F: PrimeField>() {
        // Z = [1, 2, 1, 4]
        let Z = vec![F::ONE, F::from(2u64), F::ONE, F::from(4u64)];

        // r = [4,3]
        let r = vec![F::from(4u64), F::from(3u64)];

        let eval_with_LR = evaluate_with_LR::<F>(&Z, &r);
        let poly = DensePolynomial::new(Z);

        let eval = poly.evaluate(&r);
        assert_eq!(eval, F::from(28u64));
        assert_eq!(eval_with_LR, eval);
    }

    pub fn compute_factored_chis_at_r<F: PrimeField>(r: &[F]) -> (Vec<F>, Vec<F>) {
        let mut L: Vec<F> = Vec::new();
        let mut R: Vec<F> = Vec::new();

        let ell = r.len();
        assert!(ell % 2 == 0); // ensure ell is even
        let n = ell.pow2();
        let m = n.square_root();

        // compute row vector L
        for i in 0..m {
            let mut chi_i = F::ONE;
            for j in 0..ell / 2 {
                let bit_j = ((m * i) & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::ONE - r[j];
                }
            }
            L.push(chi_i);
        }

        // compute column vector R
        for i in 0..m {
            let mut chi_i = F::ONE;
            for j in ell / 2..ell {
                let bit_j = (i & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::ONE - r[j];
                }
            }
            R.push(chi_i);
        }
        (L, R)
    }

    pub fn compute_chis_at_r<F: PrimeField>(r: &[F]) -> Vec<F> {
        let ell = r.len();
        let n = ell.pow2();
        let mut chis: Vec<F> = Vec::new();
        for i in 0..n {
            let mut chi_i = F::ONE;
            for j in 0..r.len() {
                let bit_j = (i & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::ONE - r[j];
                }
            }
            chis.push(chi_i);
        }
        chis
    }

    pub fn compute_outerproduct<F: PrimeField>(L: &[F], R: &[F]) -> Vec<F> {
        assert_eq!(L.len(), R.len());
        (0..L.len())
            .map(|i| (0..R.len()).map(|j| L[i] * R[j]).collect::<Vec<F>>())
            .collect::<Vec<Vec<F>>>()
            .into_iter()
            .flatten()
            .collect::<Vec<F>>()
    }

    #[test]
    fn check_memoized_chis() {
        check_memoized_chis_helper::<Fr>();
        check_memoized_chis_helper::<Goldilocks>();
        check_memoized_chis_helper::<GoldilocksExt2>();
    }

    fn check_memoized_chis_helper<F: PrimeField>() {
        let mut prng = test_rng();

        let s = 10;
        let mut r: Vec<F> = Vec::new();
        for _i in 0..s {
            r.push(F::random(&mut prng));
        }
        let chis = compute_chis_at_r::<F>(&r);
        let chis_m = EqPolynomial::<F>::new(r).evals();
        assert_eq!(chis, chis_m);
    }

    #[test]
    fn check_factored_chis() {
        check_factored_chis_helper::<Fr>()
    }

    fn check_factored_chis_helper<F: PrimeField>() {
        let mut prng = test_rng();

        let s = 10;
        let mut r: Vec<F> = Vec::new();
        for _i in 0..s {
            r.push(F::random(&mut prng));
        }
        let chis = EqPolynomial::new(r.clone()).evals();
        let (L_size, _R_size) = matrix_dimensions(r.len(), 1);
        let (L, R) = EqPolynomial::new(r).compute_factored_evals(L_size);
        let O = compute_outerproduct(&L, &R);
        assert_eq!(chis, O);
    }

    #[test]
    fn check_memoized_factored_chis() {
        check_memoized_factored_chis_helper::<Fr>()
    }

    fn check_memoized_factored_chis_helper<F: PrimeField>() {
        let mut prng = test_rng();

        let s = 10;
        let mut r: Vec<F> = Vec::new();
        for _i in 0..s {
            r.push(F::random(&mut prng));
        }
        let (L_size, _R_size) = matrix_dimensions(r.len(), 1);
        let (L, R) = compute_factored_chis_at_r(&r);
        let eq = EqPolynomial::new(r);
        let (L2, R2) = eq.compute_factored_evals(L_size);
        assert_eq!(L, L2);
        assert_eq!(R, R2);
    }

    #[test]
    fn evaluation() {
        let num_evals = 4;
        let mut evals: Vec<Fr> = Vec::with_capacity(num_evals);
        for _ in 0..num_evals {
            evals.push(Fr::from(8));
        }
        let dense_poly: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

        // Evaluate at 3:
        // (0, 0) = 1
        // (0, 1) = 1
        // (1, 0) = 1
        // (1, 1) = 1
        // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
        // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
        // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
        assert_eq!(
            dense_poly.evaluate(vec![Fr::from(3), Fr::from(4)].as_slice()),
            Fr::from(8)
        );
    }
}
