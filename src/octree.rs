//! Octree types and algorithms.

mod linear;
mod pointer;

pub use self::linear::Linear;
pub use self::pointer::Pointer;

use crate::morton::*;
use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

/// Implement this trait to gather the leaf nodes of an octree in a region.
pub trait Gatherer<Item, M> {
    type Sum;

    /// `gather` is allowed to assume the `it` gives at least one item.
    fn gather<'a, I>(&self, it: I) -> Self::Sum
    where
        Item: 'a,
        I: Iterator<Item = (M, &'a Item)>;
}

/// Implement this trait to gather subregion changes in the octree into the parent region.
pub trait Folder {
    type Sum;
    /// `sum` is allowed to assume the `it` gives at least one item.
    fn sum<I>(&self, it: I) -> Option<Self::Sum>
    where
        I: Iterator<Item = Self::Sum>;
}

/// This defines a region from [-2**n, 2**n).
#[derive(Copy, Clone, Debug)]
pub struct LeveledRegion(pub i32);

impl LeveledRegion {
    /// This allows the discretization of a `Vector3` `point` to a morton code using the region.
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
