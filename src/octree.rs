mod linear;
mod morton;
mod pointer;

pub use self::linear::Linear;
pub use self::morton::*;
pub use self::pointer::Pointer;

use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

pub trait Gatherer<Item, N> {
        type Sum;
        fn gather<'a, I>(&self, it: I) -> Self::Sum
        where
                Item: 'a,
                I: Iterator<Item = (Morton<N>, &'a Item)>;
}

pub trait Folder {
        type Sum;
        fn sum<I>(&self, it: I) -> Option<Self::Sum>
        where
                I: Iterator<Item = Self::Sum>;
}

pub struct LeveledRegion(pub i32);

impl LeveledRegion {
        pub fn discretize<S, T>(&self, point: &Vector3<S>) -> Option<T>
        where
                S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
                Vector3<S>: Into<T>,
        {
                let bound = (S::one() + S::one()).powi(self.0);
                if point.iter().any(|n| n.abs() > bound) {
                        None
                } else {
                        // Convert the point into normalized space.
                        Some(
                                (point.map(|n| {
                                        (n + bound) / (S::one() + S::one()).powi(self.0 + 1)
                                }))
                                .into(),
                        )
                }
        }
}
