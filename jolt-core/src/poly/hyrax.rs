use super::dense_mlpoly::DensePolynomial;
use super::pedersen::{PedersenCommitment, PedersenGenerators};
use crate::msm::best_multiexp;
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::errors::ProofVerifyError;
use crate::utils::math::Math;
use crate::utils::transcript::{AppendToTranscript, ProofTranscript};
use crate::utils::{compute_dotproduct, mul_0_1_optimized};
use ff::{Field, FromUniformBytes};
use halo2curves::group::Curve;
use halo2curves::group::Group;
use halo2curves::CurveAffine;
use num_integer::Roots;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::trace_span;

// use crate::msm::VariableBaseMSM;

pub fn matrix_dimensions(num_vars: usize, matrix_aspect_ratio: usize) -> (usize, usize) {
    let mut row_size = (num_vars / 2).pow2();
    row_size = (row_size * matrix_aspect_ratio.sqrt()).next_power_of_two();

    let right_num_vars = std::cmp::min(row_size.log_2(), num_vars - 1);
    row_size = right_num_vars.pow2();
    let left_num_vars = num_vars - right_num_vars;
    let col_size = left_num_vars.pow2();

    (col_size, row_size)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct HyraxGenerators<const RATIO: usize, G: CurveAffine> {
    pub gens: PedersenGenerators<G>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyraxCommitment<const RATIO: usize, G: CurveAffine> {
    row_commitments: Vec<G>,
}

impl<const RATIO: usize, G: CurveAffine> HyraxCommitment<RATIO, G> {
    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit")]
    pub fn commit(poly: &DensePolynomial<G::Scalar>, generators: &PedersenGenerators<G>) -> Self {
        let n = poly.len();
        let ell = poly.get_num_vars();
        assert_eq!(n, ell.pow2());

        Self::commit_slice(poly.evals_ref(), generators)
    }

    #[tracing::instrument(skip_all, name = "HyraxCommitment::commit_slice")]
    pub fn commit_slice(eval_slice: &[G::Scalar], generators: &PedersenGenerators<G>) -> Self {
        let n = eval_slice.len();
        let ell = n.log_2();

        let (L_size, R_size) = matrix_dimensions(ell, RATIO);
        assert_eq!(L_size * R_size, n);

        let gens = &generators.generators[..R_size];
        let row_commitments = eval_slice
            .par_chunks(R_size)
            .map(|row| PedersenCommitment::commit_vector(row, &gens).to_affine())
            .collect();
        Self { row_commitments }
    }

    #[tracing::instrument(skip_all, name = "HyraxCommitment::batch_commit")]
    pub fn batch_commit(
        batch: &Vec<Vec<G::Scalar>>,
        generators: &PedersenGenerators<G>,
    ) -> Vec<Self> {
        let n = batch[0].len();
        batch.iter().for_each(|poly| assert_eq!(poly.len(), n));
        let ell = n.log_2();

        let (L_size, R_size) = matrix_dimensions(ell, RATIO);
        assert_eq!(L_size * R_size, n);

        let gens = &generators.generators[..R_size];

        let rows = batch.par_iter().flat_map(|poly| poly.par_chunks(R_size));
        let row_commitments: Vec<G> = rows
            .map(|row| PedersenCommitment::commit_vector(row, &gens).to_affine())
            .collect();

        row_commitments
            .par_chunks(L_size)
            .map(|chunk| Self {
                row_commitments: chunk.to_vec(),
            })
            .collect()
    }

    #[tracing::instrument(skip_all, name = "HyraxCommitment::batch_commit_polys")]
    pub fn batch_commit_polys(
        polys: Vec<&DensePolynomial<G::Scalar>>,
        generators: &PedersenGenerators<G>,
    ) -> Vec<Self> {
        let num_vars = polys[0].get_num_vars();
        let n = num_vars.pow2();
        polys
            .iter()
            .for_each(|poly| assert_eq!(poly.as_ref().len(), n));

        let (L_size, R_size) = matrix_dimensions(num_vars, RATIO);
        assert_eq!(L_size * R_size, n);

        let gens = &generators.generators[..R_size];

        let rows = polys
            .par_iter()
            .flat_map(|poly| poly.evals_ref().par_chunks(R_size));
        let row_commitments: Vec<G> = rows
            .map(|row| PedersenCommitment::commit_vector(row, &gens).to_affine())
            .collect();

        row_commitments
            .par_chunks(L_size)
            .map(|chunk| Self {
                row_commitments: chunk.to_vec(),
            })
            .collect()
    }
}

impl<const RATIO: usize, G: CurveAffine> AppendToTranscript for HyraxCommitment<RATIO, G> {
    fn append_to_transcript(&self, label: &'static [u8], transcript: &mut ProofTranscript) {
        transcript.append_message(label, b"poly_commitment_begin");
        for i in 0..self.row_commitments.len() {
            transcript.append_point(b"poly_commitment_share", &self.row_commitments[i]);
        }
        transcript.append_message(label, b"poly_commitment_end");
    }
}

#[derive(Debug)]
pub struct HyraxOpeningProof<const RATIO: usize, G: CurveAffine> {
    vector_matrix_product: Vec<G::Scalar>,
}

/// See Section 14.3 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<const RATIO: usize, G: CurveAffine> HyraxOpeningProof<RATIO, G> {
    fn protocol_name() -> &'static [u8] {
        b"Hyrax opening proof"
    }

    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::prove")]
    pub fn prove(
        poly: &DensePolynomial<G::Scalar>,
        opening_point: &[G::Scalar], // point at which the polynomial is evaluated
        transcript: &mut ProofTranscript,
    ) -> HyraxOpeningProof<RATIO, G> {
        transcript.append_protocol_name(Self::protocol_name());

        // assert vectors are of the right size
        assert_eq!(poly.get_num_vars(), opening_point.len());

        // compute the L and R vectors
        let (L_size, _R_size) = matrix_dimensions(poly.get_num_vars(), RATIO);
        let eq = EqPolynomial::new(opening_point.to_vec());
        let (L, _R) = eq.compute_factored_evals(L_size);

        // compute vector-matrix product between L and Z viewed as a matrix
        let vector_matrix_product = Self::vector_matrix_product(poly, &L);

        HyraxOpeningProof {
            vector_matrix_product,
        }
    }

    pub fn verify(
        &self,
        pedersen_generators: &PedersenGenerators<G>,
        transcript: &mut ProofTranscript,
        opening_point: &[G::Scalar], // point at which the polynomial is evaluated
        opening: &G::Scalar,         // evaluation \widetilde{Z}(r)
        commitment: &HyraxCommitment<RATIO, G>,
    ) -> Result<(), ProofVerifyError> {
        transcript.append_protocol_name(Self::protocol_name());

        // compute L and R
        let (L_size, R_size) = matrix_dimensions(opening_point.len(), RATIO);
        let eq: EqPolynomial<_> = EqPolynomial::new(opening_point.to_vec());
        let (L, R) = eq.compute_factored_evals(L_size);

        // Verifier-derived commitment to u * a = \prod Com(u_j)^{a_j}
        let homomorphically_derived_commitment = best_multiexp(&L, &commitment.row_commitments);

        let product_commitment = best_multiexp(
            &self.vector_matrix_product,
            &pedersen_generators.generators[..R_size],
        );

        let dot_product = compute_dotproduct(&self.vector_matrix_product, &R);

        if (homomorphically_derived_commitment == product_commitment) && (dot_product == *opening) {
            Ok(())
        } else {
            assert!(false, "hyrax verify error");
            Err(ProofVerifyError::InternalError)
        }
    }

    #[tracing::instrument(skip_all, name = "HyraxOpeningProof::vector_matrix_product")]
    fn vector_matrix_product(poly: &DensePolynomial<G::Scalar>, L: &[G::Scalar]) -> Vec<G::Scalar> {
        let (_, R_size) = matrix_dimensions(poly.get_num_vars(), RATIO);

        poly.evals_ref()
            .par_chunks(R_size)
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .map(|x| mul_0_1_optimized(&L[i], x))
                    .collect::<Vec<G::Scalar>>()
            })
            .reduce(
                || vec![G::Scalar::ZERO; R_size],
                |mut acc: Vec<_>, row| {
                    acc.iter_mut().zip(row).for_each(|(x, y)| *x += y);
                    acc
                },
            )
    }
}

#[derive(Debug)]
pub struct BatchedHyraxOpeningProof<const RATIO: usize, G: CurveAffine> {
    joint_proof: HyraxOpeningProof<RATIO, G>,
}

/// See Section 16.1 of Thaler's Proofs, Arguments, and Zero-Knowledge
impl<const RATIO: usize, G: CurveAffine> BatchedHyraxOpeningProof<RATIO, G>
where
    G::Scalar: FromUniformBytes<64>,
{
    #[tracing::instrument(skip_all, name = "BatchedHyraxOpeningProof::prove")]
    pub fn prove(
        polynomials: &[&DensePolynomial<G::Scalar>],
        opening_point: &[G::Scalar],
        openings: &[G::Scalar],
        transcript: &mut ProofTranscript,
    ) -> Self {
        transcript.append_protocol_name(Self::protocol_name());

        // append the claimed evaluations to transcript
        transcript.append_scalars(b"evals_ops_val", &openings);

        let rlc_coefficients: Vec<_> =
            transcript.challenge_vector(b"challenge_combine_n_to_one", polynomials.len());

        let _span = trace_span!("Compute RLC of polynomials");
        let _enter = _span.enter();

        let poly_len = polynomials[0].len();

        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = poly_len / num_chunks;

        let rlc_poly = if chunk_size > 0 {
            (0..num_chunks)
                .into_par_iter()
                .flat_map_iter(|chunk_index| {
                    let mut chunk = vec![G::Scalar::ZERO; chunk_size];
                    for (coeff, poly) in rlc_coefficients.iter().zip(polynomials.iter()) {
                        for (rlc, poly_eval) in chunk
                            .iter_mut()
                            .zip(poly.evals_ref()[chunk_index * chunk_size..].iter())
                        {
                            *rlc += mul_0_1_optimized(poly_eval, coeff);
                        }
                    }
                    chunk
                })
                .collect::<Vec<_>>()
        } else {
            rlc_coefficients
                .par_iter()
                .zip(polynomials.par_iter())
                .map(|(coeff, poly)| poly.evals_ref().iter().map(|eval| *coeff * eval).collect())
                .reduce(
                    || vec![G::Scalar::ZERO; poly_len],
                    |running, new| {
                        debug_assert_eq!(running.len(), new.len());
                        running
                            .iter()
                            .zip(new.iter())
                            .map(|(r, n)| *r + n)
                            .collect()
                    },
                )
        };

        drop(_enter);
        drop(_span);

        let joint_proof =
            HyraxOpeningProof::prove(&DensePolynomial::new(rlc_poly), opening_point, transcript);
        Self { joint_proof }
    }

    pub fn verify(
        &self,
        pedersen_generators: &PedersenGenerators<G>,
        opening_point: &[G::Scalar],
        openings: &[G::Scalar],
        commitments: &[&HyraxCommitment<RATIO, G>],
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let (L_size, _R_size) = matrix_dimensions(opening_point.len(), RATIO);

        transcript.append_protocol_name(Self::protocol_name());

        // append the claimed evaluations to transcript
        transcript.append_scalars(b"evals_ops_val", &openings);

        let rlc_coefficients: Vec<_> =
            transcript.challenge_vector(b"challenge_combine_n_to_one", openings.len());

        let rlc_eval = compute_dotproduct(&rlc_coefficients, openings);

        let rlc_commitment = rlc_coefficients
            .par_iter()
            .zip(commitments.par_iter())
            .map(|(coeff, commitment)| {
                commitment
                    .row_commitments
                    .iter()
                    .map(|row_commitment| *row_commitment * coeff)
                    .collect()
            })
            .reduce(
                || vec![G::Curve::identity(); L_size],
                |running, new| {
                    debug_assert_eq!(running.len(), new.len());
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(r, n)| *r + n)
                        .collect()
                },
            );
        let mut rlc_commitment_affine = vec![G::default(); rlc_commitment.len()];
        G::CurveExt::batch_normalize(&rlc_commitment, &mut rlc_commitment_affine);

        self.joint_proof.verify(
            pedersen_generators,
            transcript,
            opening_point,
            &rlc_eval,
            &HyraxCommitment {
                row_commitments: rlc_commitment_affine,
            },
        )
    }

    fn protocol_name() -> &'static [u8] {
        b"Jolt BatchedHyraxOpeningProof"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2curves::bn256;
    use halo2curves::grumpkin;

    #[test]
    fn check_polynomial_commit() {
        check_polynomial_commit_helper::<bn256::G1Affine>();
        check_polynomial_commit_helper::<grumpkin::G1Affine>();
    }

    fn check_polynomial_commit_helper<G: CurveAffine>() {
        let Z = vec![
            G::Scalar::ONE,
            G::Scalar::from(2u64),
            G::Scalar::ONE,
            G::Scalar::from(4u64),
        ];
        let poly = DensePolynomial::new(Z);

        // r = [4,3]
        let r = vec![G::Scalar::from(4u64), G::Scalar::from(3u64)];
        let eval = poly.evaluate(&r);
        assert_eq!(eval, G::Scalar::from(28u64));

        let generators: PedersenGenerators<G> = PedersenGenerators::new(1 << 8, b"test-two");
        let poly_commitment: HyraxCommitment<1, G> = HyraxCommitment::commit(&poly, &generators);

        let mut prover_transcript = ProofTranscript::new(b"example");
        let proof = HyraxOpeningProof::prove(&poly, &r, &mut prover_transcript);

        let mut verifier_transcript = ProofTranscript::new(b"example");

        assert!(proof
            .verify(
                &generators,
                &mut verifier_transcript,
                &r,
                &eval,
                &poly_commitment
            )
            .is_ok());
    }
}
