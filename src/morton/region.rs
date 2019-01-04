use crate::*;
use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};
use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::hash::{Hash, Hasher};

/// Defines a region by dividing finite space into a z-order curve of `level` and uses the upper bits of `morton`.
#[derive(Debug, Clone, Copy)]
pub struct MortonRegion<M> {
    /// The most significant `level * 3` bits of this morton encode the voxel of the z-order curve this is a part of.
    pub morton: M,
    /// This defines the level of the z-order curve.
    ///
    /// A `level` of `0` is the whole space.
    /// A `level` of `1` means the region is one of the 8 top level octants of the space.
    /// If the `level` is equal to `M::dim_bits()`, then the entire morton is used.
    /// Level cannot exceed `M::dim_bits()` or there wont be enough bits to encode the morton.
    pub level: usize,
}

impl<M> MortonRegion<M>
where
    M: Morton,
{
    /// This gets the top level region (everything in the finite space).
    #[inline]
    pub fn base() -> Self {
        MortonRegion {
            morton: M::zero(),
            level: 0,
        }
    }

    /// Get the bits that are actually used to encode different levels in the morton.
    #[inline]
    pub fn significant_bits(self) -> M {
        self.morton.get_significant_bits(self.level)
    }

    /// Enter an octant in the region.
    ///
    /// Note that this does not mutate the region, but returns a new one. This can be reversed by calling `exit()`.
    #[inline]
    pub fn enter(mut self, octant: usize) -> Self {
        self.morton.set_level(self.level, octant);
        self.level += 1;
        self
    }

    /// Changes the region to its parent region by going up one level.
    #[inline]
    pub fn exit(&mut self) -> usize {
        self.level -= 1;
        let old = self.morton.get_level(self.level);
        // This is not totally necessary, but it resets the level to ensure unused bits are `0`.
        self.morton.reset_level(self.level);
        old
    }

    /// Gets the least-significant octant of the region.
    #[inline]
    pub fn get(&self) -> usize {
        self.morton.get_level(self.level - 1)
    }

    /// Gets the next octant when iterating in z-order over the least significant octant.
    ///
    /// This gives back None when it is on the last octant or if the level is `0`, in which case it is the whole space.
    #[inline]
    pub fn next(mut self) -> Option<Self> {
        if self.level == 0 {
            None
        } else {
            let last = self.exit();
            if last == 7 {
                None
            } else {
                Some(self.enter(last + 1))
            }
        }
    }

    /// Produces a single number that has a canonically unique mapping to every given valid MortonRegion by using
    /// the unused bits to store the level information via shifting.
    #[inline]
    pub fn canonicalize(&self) -> M {
        if self.level == 0 {
            M::zero()
        } else {
            (self.morton | M::unused_bits()).get_significant_bits(self.level - 1)
        }
    }

    /// Iterates over subregions of a region. Uses `explore` to limit the exploration space.
    pub fn iter<E>(self, explore: E) -> MortonRegionIterator<M, E>
    where
        E: FnMut(MortonRegion<M>) -> bool,
    {
        MortonRegionIterator {
            nodes: vec![self],
            explore,
        }
    }
}

impl<M> PartialEq for MortonRegion<M>
where
    M: Morton,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.canonicalize().eq(&other.canonicalize())
    }
}

impl<M> Eq for MortonRegion<M> where M: Morton {}

impl<M> PartialOrd for MortonRegion<M>
where
    M: Morton,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.canonicalize().partial_cmp(&other.canonicalize())
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        self.canonicalize().lt(&other.canonicalize())
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        self.canonicalize().le(&other.canonicalize())
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        self.canonicalize().gt(&other.canonicalize())
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        self.canonicalize().ge(&other.canonicalize())
    }
}

impl<M> Ord for MortonRegion<M>
where
    M: Morton,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.canonicalize().cmp(&other.canonicalize())
    }
}

impl<M> Default for MortonRegion<M>
where
    M: Morton,
{
    #[inline]
    fn default() -> Self {
        MortonRegion::base()
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl<M> Hash for MortonRegion<M>
where
    M: Morton + Hash,
{
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.canonicalize().hash(state);
    }
}

impl<S, M> Into<Vector3<S>> for MortonRegion<M>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
    M: Morton,
{
    #[inline]
    fn into(self) -> Vector3<S> {
        let v = self.morton;
        let cut = M::dim_bits() - self.level;
        let point = (v >> (3 * cut)).decode();
        let scale = (S::one() + S::one()).powi(-(self.level as i32));

        point.map(|d| {
            (S::from_u64(d.to_u64().unwrap()).unwrap() + S::from_f32(0.5).unwrap()) * scale
        })
    }
}

/// Generates regions over every level of this morton from the first octant (`level` `1`)
/// to the least significant level (`level` `M::dim_bits()`). This does not include the root region (`level` `0`).
#[inline]
pub fn morton_levels<M>(m: M) -> impl Iterator<Item = MortonRegion<M>>
where
    M: Morton,
{
    std::iter::once(MortonRegion::default()).chain((1..=M::dim_bits()).map(move |i| MortonRegion {
        morton: m.get_significant_bits(i - 1) << (3 * (M::dim_bits() - i)),
        level: i,
    }))
}

/// An `Iterator` over a `MortonRegion` that uses a closure to limit the exploration space.
///
/// Produced by `MortonRegion::iter`.
pub struct MortonRegionIterator<M, E> {
    nodes: Vec<MortonRegion<M>>,
    explore: E,
}

impl<M, E> MortonRegionIterator<M, E>
where
    E: FnMut(MortonRegion<M>) -> bool,
{
    /// Takes a region to iterate over and a closure to limit the exploration space.
    /// This will traverse through `8/7 * 8^(limit - region.level)` nodes, so mind the limit.
    pub fn new(region: MortonRegion<M>, explore: E) -> Self {
        MortonRegionIterator {
            nodes: vec![region],
            explore,
        }
    }
}

impl<M, E> Iterator for MortonRegionIterator<M, E>
where
    M: Morton,
    E: FnMut(MortonRegion<M>) -> bool,
{
    type Item = MortonRegion<M>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.nodes.pop().map(|region| {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push(next);
            }

            // Check if we should explore this sub region.
            if region.level < M::dim_bits() && (self.explore)(region) {
                self.nodes.push(region.enter(0));
            }
            region
        })
    }
}
