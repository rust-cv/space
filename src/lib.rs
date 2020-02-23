//! This library is intended to be incredibly lightweight (cost nothing), but provide
//! common traits that can be shared among spatial data structures.

#![no_std]

mod hamming_impls;
#[cfg(feature = "simd")]
mod simd_impls;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "simd")]
pub use simd_impls::*;

/// This trait is implemented by points inside of a metric space.
///
/// It is important that all points that implement this trait satisfy
/// the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality).
/// This requirement basically means that the sum of distances that start
/// at a point A and end at a point B can never be less than the distance
/// from A to B directly. All implementors must also take care to avoid
/// negative numbers, as valid distances are only positive numbers.
///
/// In practice, the `u32` distance returned by this trait should only be used
/// to compare distances between points in a metric space. If the distances are added,
/// one would need to take care of overflow. Regardless of the underlying representation
/// (float or integer), one can map the metric distance into the set of 32-bit integers.
/// This may cause some loss of precision, but the choice to use 32 bits of precision
/// is one that is done with practicality in mind. Specifically, there may be cases
/// where only a few bits of precision are needed (hamming distance), but there may also
/// be cases where a 32-bit floating point number may be different by only one bit of precision.
/// It is trivial to map a 32-bit float to the unsigned integers for comparison, as
/// [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) is designed such that a direct
/// bit-to-bit translation of a 32-bit float to a 32-bit signed integer will compare
/// as intended using a standard signed integer comparation operation. 64-bit floating point
/// numbers must be truncated to their upper 32-bits and loose some precision. This loss
/// is acceptable in most scenarios. 32-bit integers are widely supported on embedded and
/// desktop processors as native registers.
///
/// If you have a floating distance, use [`f32_metric`] or [`f64_metric`]. Keep in mind
/// that [`f64_metric`] will cause precision loss of 32 bits.
pub trait MetricPoint {
    fn distance(&self, rhs: &Self) -> u32;
}

/// Any data contained in this struct is treated such that all of the bits
/// of the data are each separate dimensions that can be of length `0` or `1`.
/// This is referred to as [hamming space](https://en.wikipedia.org/wiki/Hamming_space).
/// This leads to all [Lp spaces](https://en.wikipedia.org/wiki/Lp_space) being
/// equal and all distances being no larger than the number of bits of the data.
///
/// This is typically used to perform searches on data that was binarized using
/// hyperplane comparisons on high dimensional floating-point features or data built
/// on a series of binary comparisons. Hamming distance is incredibly fast
/// to compute. Hamming space can be difficult to search because the
/// amount of equidistant points is high (see "Thick Boundaries in Binary Space and
/// Their Influence on Nearest-Neighbor Search") and they can quickly
/// [grow in dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Hamming<T>(pub T);

/// L1 distance is applied to items wrapped in this type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct L1<T>(pub T);

/// L2 distance is applied to items wrapped in this type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct L2<T>(pub T);

/// L-infinity distance is applied to items wrapped in this type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LInfinity<T>(pub T);

/// Converts a `f32` metric to a `u32` metric with no loss in precision.
pub fn f32_metric(n: f32) -> u32 {
    n.to_bits()
}

/// Converts a `f64` metric into a `u32` metric by truncating 32-bits of precision.
pub fn f64_metric(n: f64) -> u32 {
    (n.to_bits() >> 32) as u32
}

/// This can be used to return an iterable set of indices over a search index
/// by wrapping any type that implements `AsRef<[usize]>`.
///
/// This is useful when you have an array of indicies that were deposited into
/// a mutable slice or stored in an array as the result of a kNN search and would
/// like to return them in such a way that they can be iterated over easily.
pub struct Indices<A>(pub A);

impl<A> IntoIterator for Indices<A>
where
    A: AsRef<[usize]>,
{
    type Item = usize;
    type IntoIter = IndexIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        let Self(indices) = self;
        IndexIter {
            indices,
            position: 0,
        }
    }
}

/// An iterator created by calling [`IndexIntoIter::into_iter`](IntoIterator::into_iter).
pub struct IndexIter<A> {
    // These members must remain hidden to avoid undefined behavior in the
    // iterator next implementation.
    indices: A,
    position: usize,
}

impl<A> Iterator for IndexIter<A>
where
    A: AsRef<[usize]>,
{
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let indices = self.indices.as_ref();
        if indices.len() == self.position {
            None
        } else {
            // This should be safe as the position cannot exceed
            // indices.len(). The members of IndexIter are hidden.
            let index = unsafe { *indices.get_unchecked(self.position) };
            self.position += 1;
            Some(index)
        }
    }
}
