//! Octree types and algorithms.

mod linear;
mod pointer;

pub use self::linear::LinearOctree;
pub use self::pointer::PointerOctree;

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

impl<Item, M, F> Folder<Item, M> for &F
where
    F: Folder<Item, M>,
{
    type Sum = F::Sum;

    fn gather<'a>(&self, morton: M, item: &'a Item) -> Self::Sum {
        (*self).gather(morton, item)
    }

    fn fold<I>(&self, it: I) -> Self::Sum
    where
        I: Iterator<Item = Self::Sum>,
    {
        (*self).fold(it)
    }
}

macro_rules! tuple_folder {
    ({$($id: ident),* $(,)?}, {$($sm: ident),* $(,)?}, {$($acc: ident),* $(,)?}, {$($item: ident),* $(,)?}) => {
        #[allow(non_snake_case)]
        impl <Item, M, $($id:),*> Folder<Item, M> for ($($id),*)
            where M: Morton, $($id: Folder<Item, M>,)*
        {
            type Sum = ($($id::Sum),*);

            fn gather<'a>(&self, morton: M, item: &'a Item) -> Self::Sum {
                let ($(ref $id),*) = *self;
                ($($id.gather(morton, item)),*)
            }

            fn fold<IT>(&self, it: IT) -> Self::Sum
            where
                IT: Iterator<Item = Self::Sum>,
            {
                let ($($sm),*): ($(smallvec::SmallVec<[$id::Sum; 8]>),*) =
                    it.fold(<($(smallvec::SmallVec<[$id::Sum; 8]>),*)>::default(),
                            |($(mut $acc),*), ($($item),*)| {
                                $($acc.push($item);)*
                                ($($acc),*)
                            });
                let ($(ref $id),*) = *self;
                ($($id.fold($sm.into_iter())),*)
            }
        }
    }
}

tuple_folder!({A, B},
              {A_sm, B_sm},
              {A_acc, B_acc},
              {A_item, B_item});
tuple_folder!({A, B, C},
              {A_sm, B_sm, C_sm},
              {A_acc, B_acc, C_acc},
              {A_item, B_item, C_item});
tuple_folder!({A, B, C, D},
              {A_sm, B_sm, C_sm, D_sm},
              {A_acc, B_acc, C_acc, D_acc},
              {A_item, B_item, C_item, D_item});
tuple_folder!({A, B, C, D, E},
              {A_sm, B_sm, C_sm, D_sm, E_sm},
              {A_acc, B_acc, C_acc, D_acc, E_acc},
              {A_item, B_item, C_item, D_item, E_item});
tuple_folder!({A, B, C, D, E, F},
              {A_sm, B_sm, C_sm, D_sm, E_sm, F_sm},
              {A_acc, B_acc, C_acc, D_acc, E_acc, F_acc},
              {A_item, B_item, C_item, D_item, E_item, F_item});
tuple_folder!({A, B, C, D, E, F, G},
              {A_sm, B_sm, C_sm, D_sm, E_sm, F_sm, G_sm},
              {A_acc, B_acc, C_acc, D_acc, E_acc, F_acc, G_acc},
              {A_item, B_item, C_item, D_item, E_item, F_item, G_item});
tuple_folder!({A, B, C, D, E, F, G, H},
              {A_sm, B_sm, C_sm, D_sm, E_sm, F_sm, G_sm, H_sm},
              {A_acc, B_acc, C_acc, D_acc, E_acc, F_acc, G_acc, H_acc},
              {A_item, B_item, C_item, D_item, E_item, F_item, G_item, H_item});
tuple_folder!({A, B, C, D, E, F, G, H, I},
              {A_sm, B_sm, C_sm, D_sm, E_sm, F_sm, G_sm, H_sm, I_sm},
              {A_acc, B_acc, C_acc, D_acc, E_acc, F_acc, G_acc, H_acc, I_acc},
              {A_item, B_item, C_item, D_item, E_item, F_item, G_item, H_item, I_item});
tuple_folder!({A, B, C, D, E, F, G, H, I, J},
              {A_sm, B_sm, C_sm, D_sm, E_sm, F_sm, G_sm, H_sm, I_sm, J_sm},
              {A_acc, B_acc, C_acc, D_acc, E_acc, F_acc, G_acc, H_acc, I_acc, J_acc},
              {A_item, B_item, C_item, D_item, E_item, F_item, G_item, H_item, I_item, J_item});
tuple_folder!({A, B, C, D, E, F, G, H, I, J, K},
              {A_sm, B_sm, C_sm, D_sm, E_sm, F_sm, G_sm, H_sm, I_sm, J_sm, K_sm},
              {A_acc, B_acc, C_acc, D_acc, E_acc, F_acc, G_acc, H_acc, I_acc, J_acc, K_acc},
              {A_item, B_item, C_item, D_item, E_item, F_item, G_item, H_item, I_item, J_item, K_item});
tuple_folder!({A, B, C, D, E, F, G, H, I, J, K, L},
              {A_sm, B_sm, C_sm, D_sm, E_sm, F_sm, G_sm, H_sm, I_sm, J_sm, K_sm, L_sm},
              {A_acc, B_acc, C_acc, D_acc, E_acc, F_acc, G_acc, H_acc, I_acc, J_acc, K_acc, L_acc},
              {A_item, B_item, C_item, D_item, E_item, F_item, G_item, H_item, I_item, J_item, K_item, L_item});

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
    /// If the point is not in the region it gives back `None`.
    ///
    /// ```
    /// let region = space::LeveledRegion(0);
    /// // This is inside the bounds, so it gives back `Some(morton)`.
    /// let inside_bounds = nalgebra::Vector3::new(0.5, 0.5, 0.5);
    /// assert!(region.discretize::<f32, u64>(inside_bounds).is_some());
    /// // This is outside the bounds, so it gives back `None`.
    /// let outside_bounds = nalgebra::Vector3::new(1.5, 1.5, 1.5);
    /// assert!(region.discretize::<f32, u64>(outside_bounds).is_none());
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
