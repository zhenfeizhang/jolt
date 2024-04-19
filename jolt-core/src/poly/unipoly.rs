#![allow(dead_code)]
use crate::utils::gaussian_elimination::gaussian_elimination;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use ff::PrimeField;
use serde::{Deserialize, Serialize};

// ax^2 + bx + c stored as vec![c,b,a]
// ax^3 + bx^2 + cx + d stored as vec![d,c,b,a]
#[derive(Debug, Clone)]
pub struct UniPoly<F> {
    coeffs: Vec<F>,
}

// ax^2 + bx + c stored as vec![c,a]
// ax^3 + bx^2 + cx + d stored as vec![d,b,a]
#[derive(Serialize, Deserialize, Debug)]
pub struct CompressedUniPoly<F: PrimeField> {
    coeffs_except_linear_term: Vec<F>,
}

impl<F: PrimeField> UniPoly<F> {
    #[allow(dead_code)]
    pub fn from_coeff(coeffs: Vec<F>) -> Self {
        UniPoly { coeffs }
    }

    pub fn from_evals(evals: &[F]) -> Self {
        UniPoly {
            coeffs: Self::vandermonde_interpolation(evals),
        }
    }

    fn vandermonde_interpolation(evals: &[F]) -> Vec<F> {
        let n = evals.len();
        let xs: Vec<F> = (0..n).map(|x| F::from(x as u64)).collect();

        let mut vandermonde: Vec<Vec<F>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            let x = xs[i];
            row.push(F::ONE);
            row.push(x);
            for j in 2..n {
                row.push(row[j - 1] * x);
            }
            row.push(evals[i]);
            vandermonde.push(row);
        }

        gaussian_elimination(&mut vandermonde)
    }

    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }

    pub fn as_vec(&self) -> Vec<F> {
        self.coeffs.clone()
    }

    pub fn eval_at_zero(&self) -> F {
        self.coeffs[0]
    }

    pub fn eval_at_one(&self) -> F {
        (0..self.coeffs.len()).map(|i| self.coeffs[i]).sum()
    }

    pub fn evaluate(&self, r: &F) -> F {
        let mut eval = self.coeffs[0];
        let mut power = *r;
        for i in 1..self.coeffs.len() {
            eval += power * self.coeffs[i];
            power *= r;
        }
        eval
    }

    pub fn compress(&self) -> CompressedUniPoly<F> {
        let coeffs_except_linear_term = [&self.coeffs[..1], &self.coeffs[2..]].concat();
        debug_assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
        CompressedUniPoly {
            coeffs_except_linear_term,
        }
    }
}

impl<F: PrimeField> CompressedUniPoly<F> {
    // we require eval(0) + eval(1) = hint, so we can solve for the linear term as:
    // linear_term = hint - 2 * constant_term - deg2 term - deg3 term
    pub fn decompress(&self, hint: &F) -> UniPoly<F> {
        let mut linear_term =
            *hint - self.coeffs_except_linear_term[0] - self.coeffs_except_linear_term[0];
        for i in 1..self.coeffs_except_linear_term.len() {
            linear_term -= self.coeffs_except_linear_term[i];
        }

        let mut coeffs = vec![self.coeffs_except_linear_term[0], linear_term];
        coeffs.extend(&self.coeffs_except_linear_term[1..]);
        assert_eq!(self.coeffs_except_linear_term.len() + 1, coeffs.len());
        UniPoly { coeffs }
    }
}

impl<F: PrimeField> AppendToTranscript for UniPoly<F> {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        transcript.append_message(label, b"UniPoly_begin");
        for i in 0..self.coeffs.len() {
            transcript.append_scalar(b"coeff", &self.coeffs[i]);
        }
        transcript.append_message(label, b"UniPoly_end");
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use ff::PrimeField;
    use goldilocks::Goldilocks;
    use goldilocks::GoldilocksExt2;
    use halo2curves::bn256::Fr;

    #[test]
    fn test_from_evals_quad() {
        test_from_evals_quad_helper::<Fr>();
        test_from_evals_quad_helper::<Goldilocks>();
        test_from_evals_quad_helper::<GoldilocksExt2>();
    }

    fn test_from_evals_quad_helper<F: PrimeField>() {
        // polynomial is 2x^2 + 3x + 1
        let e0 = F::ONE;
        let e1 = F::from(6u64);
        let e2 = F::from(15u64);
        let evals = vec![e0, e1, e2];
        let poly = UniPoly::from_evals(&evals);

        assert_eq!(poly.eval_at_zero(), e0);
        assert_eq!(poly.eval_at_one(), e1);
        assert_eq!(poly.coeffs.len(), 3);
        assert_eq!(poly.coeffs[0], F::ONE);
        assert_eq!(poly.coeffs[1], F::from(3u64));
        assert_eq!(poly.coeffs[2], F::from(2u64));

        let hint = e0 + e1;
        let compressed_poly = poly.compress();
        let decompressed_poly = compressed_poly.decompress(&hint);
        for i in 0..decompressed_poly.coeffs.len() {
            assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
        }

        let e3 = F::from(28u64);
        assert_eq!(poly.evaluate(&F::from(3u64)), e3);
    }

    #[test]
    fn test_from_evals_cubic() {
        test_from_evals_cubic_helper::<Fr>();
        test_from_evals_cubic_helper::<Goldilocks>();
        test_from_evals_cubic_helper::<GoldilocksExt2>();
    }
    fn test_from_evals_cubic_helper<F: PrimeField>() {
        // polynomial is x^3 + 2x^2 + 3x + 1
        let e0 = F::ONE;
        let e1 = F::from(7u64);
        let e2 = F::from(23u64);
        let e3 = F::from(55u64);
        let evals = vec![e0, e1, e2, e3];
        let poly = UniPoly::from_evals(&evals);

        assert_eq!(poly.eval_at_zero(), e0);
        assert_eq!(poly.eval_at_one(), e1);
        assert_eq!(poly.coeffs.len(), 4);
        assert_eq!(poly.coeffs[0], F::ONE);
        assert_eq!(poly.coeffs[1], F::from(3u64));
        assert_eq!(poly.coeffs[2], F::from(2u64));
        assert_eq!(poly.coeffs[3], F::ONE);

        let hint = e0 + e1;
        let compressed_poly = poly.compress();
        let decompressed_poly = compressed_poly.decompress(&hint);
        for i in 0..decompressed_poly.coeffs.len() {
            assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
        }

        let e4 = F::from(109u64);
        assert_eq!(poly.evaluate(&F::from(4u64)), e4);
    }
}
