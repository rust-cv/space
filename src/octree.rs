//! Octree types and algorithms.

mod linear;
mod pointer;

pub use self::linear::Linear;
pub use self::pointer::Pointer;

use crate::morton::*;
use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

/// Implement this trait to perform a tree fold across the octree.
///
/// This will convert leaf nodes into the internal `Sum` type and then propogate them up to parent regions by
/// calling `fold`.
pub trait Folder<Item, M> {
    /// This is the type that `gather` and `fold` will produce and acts as the accumulator.
    type Sum;

    /// `gather` converts a leaf node into the internal `Sum` type.
    fn gather<'a>(&self, morton: M, item: &'a Item) -> Self::Sum;

    /// `fold` is allowed to assume the `it` gives at least one item and no more than 8 items.
    fn fold<I>(&self, it: I) -> Self::Sum
    where
        I: Iterator<Item = Self::Sum>;
}

/// Null folder that only produces only tuples.
pub struct NullFolder;

impl<Item, M> Folder<Item, M> for NullFolder {
    type Sum = ();

    fn gather<'a>(&self, _: M, _: &'a Item) -> Self::Sum {}

    fn fold<I>(&self, _: I) -> Self::Sum
    where
        I: Iterator<Item = Self::Sum>,
    {
    }
}

/// This defines a region from [-2**n, 2**n).
#[derive(Copy, Clone, Debug)]
pub struct LeveledRegion(pub i32);

impl LeveledRegion {
    /// This allows the discretization of a `Vector3` `point` to a morton code using the region.
    ///
    /// ```
    /// let region = space::octree::LeveledRegion(0);
    /// // This is inside the bounds, so it gives back `Some(morton)`.
    /// let m = region.discretize::<f32, u64>(nalgebra::Vector3::new(0.5, 0.5, 0.5)).unwrap();
    /// // This is outside the bounds, so it gives back `None`.
    /// assert!(region.discretize::<f32, u64>(nalgebra::Vector3::new(1.5, 1.5, 1.5)).is_none());
    /// ```
    pub fn discretize<S, M>(self, point: Vector3<S>) -> Option<M>
    where
        S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
        M: Morton + std::fmt::Debug + 'static,
    {
        let bound = (S::one() + S::one()).powi(self.0);
        if point.iter().any(|n| n.abs() > bound) {
            None
        } else {
            // Convert the point into normalized space.
            let MortonWrapper(m) =
                (point.map(|n| (n + bound) / (S::one() + S::one()).powi(self.0 + 1))).into();
            Some(m)
        }
    }
}
