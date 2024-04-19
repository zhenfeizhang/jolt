use ff::PrimeField;

use crate::utils::math::Math;

pub struct IdentityPolynomial {
    size_point: usize,
}

impl IdentityPolynomial {
    pub fn new(size_point: usize) -> Self {
        IdentityPolynomial { size_point }
    }

    pub fn evaluate<F: PrimeField>(&self, r: &[F]) -> F {
        let len = r.len();
        assert_eq!(len, self.size_point);
        (0..len)
            .map(|i| F::from((len - i - 1).pow2() as u64) * r[i])
            .sum()
    }
}
