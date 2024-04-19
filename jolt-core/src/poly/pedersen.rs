use ark_std::rand::SeedableRng;
use digest::{ExtendableOutput, Input};
use halo2curves::group::Group;
use halo2curves::{group::Curve, CurveAffine};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use sha3::Shake256;
use std::io::Read;

use crate::msm::best_multiexp;

#[derive(Clone, Serialize, Deserialize)]
pub struct PedersenGenerators<G: CurveAffine> {
    pub generators: Vec<G>,
}

impl<G: CurveAffine> PedersenGenerators<G> {
    #[tracing::instrument(skip_all, name = "PedersenGenerators::new")]
    pub fn new(len: usize, label: &[u8]) -> Self {
        let mut shake = Shake256::default();
        shake.input(label);

        let buf = G::generator().to_bytes();
        shake.input(buf.as_ref());

        let mut reader = shake.xof_result();
        let mut seed = [0u8; 32];
        reader.read_exact(&mut seed).unwrap();
        let mut rng = ChaCha20Rng::from_seed(seed);

        let mut generators: Vec<G> = Vec::new();
        for _ in 0..len {
            generators.push(G::Curve::random(&mut rng).to_affine());
        }

        Self { generators }
    }

    pub fn clone_n(&self, n: usize) -> PedersenGenerators<G> {
        assert!(
            self.generators.len() >= n,
            "Insufficient number of generators for clone_n: required {}, available {}",
            n,
            self.generators.len()
        );
        let slice = &self.generators[..n];
        PedersenGenerators {
            generators: slice.into(),
        }
    }
}

pub trait PedersenCommitment<G: CurveAffine>: Sized {
    fn commit(&self, gens: &PedersenGenerators<G>) -> G::Curve;
    fn commit_vector(inputs: &[Self], bases: &[G]) -> G::Curve;
}

impl<G: CurveAffine> PedersenCommitment<G> for G::Scalar {
    #[tracing::instrument(skip_all, name = "PedersenCommitment::commit")]
    fn commit(&self, gens: &PedersenGenerators<G>) -> G::Curve {
        assert_eq!(gens.generators.len(), 1);
        gens.generators[0].mul(self)
    }

    fn commit_vector(inputs: &[Self], bases: &[G]) -> G::Curve {
        assert_eq!(bases.len(), inputs.len());
        best_multiexp(&inputs, &bases)
    }
}
