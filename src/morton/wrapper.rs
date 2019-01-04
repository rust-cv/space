use crate::*;
use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};
use std::hash::{Hash, Hasher};

/// This wraps a morton to convey special external trait implementations to it that are specific to mortons.
///
/// This includes:
/// - `Hash`
/// - `From<Vector3<S>>`
/// - `Into<Vector3<S>>`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MortonWrapper<M>(pub M);

impl<M> Default for MortonWrapper<M>
where
    M: Morton,
{
    #[inline]
    fn default() -> Self {
        MortonWrapper(M::zero())
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl<M> Hash for MortonWrapper<M>
where
    M: Morton,
{
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        state.write_u64((self.0 & M::from_u64(!0).unwrap()).to_u64().unwrap())
    }
}

impl<S, M> From<Vector3<S>> for MortonWrapper<M>
where
    M: Morton + std::fmt::Debug + 'static,
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn from(point: Vector3<S>) -> Self {
        let point = point.map(|d| {
            M::from_u64(
                (d * (S::one() + S::one()).powi(M::dim_bits() as i32))
                    .to_u64()
                    .unwrap(),
            )
            .unwrap()
        });
        MortonWrapper(M::encode(point))
    }
}

impl<S, M> Into<Vector3<S>> for MortonWrapper<M>
where
    M: Morton,
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn into(self) -> Vector3<S> {
        let point = self.0.decode();
        let scale = (S::one() + S::one()).powi(-(M::dim_bits() as i32));

        point.map(|d| {
            (S::from_u64(d.to_u64().unwrap()).unwrap() + S::from_f32(0.5).unwrap()) * scale
        })
    }
}
