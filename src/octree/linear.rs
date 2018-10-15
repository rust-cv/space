use super::{Morton, MortonMap, MortonRegion, MortonRegionMap};
use smallvec::SmallVec;
use std::cmp::Eq;
use std::hash::Hash;

/// An octree that starts with a cube from [-1, 1] in each dimension and will only expand.
#[derive(Clone)]
pub struct Linear<T, N> {
    /// The leaves of the octree. Uses `SmallVec` because in most cases this shouldn't have more than one element.
    leaves: MortonMap<SmallVec<[T; 1]>, N>,
    /// The internal nodes of the octree.
    internal: MortonRegionMap<[MortonRegion<N>; 8], N>,
}

impl<T, N> Default for Linear<T, N>
where
    Morton<N>: Hash,
    MortonRegion<N>: Hash,
    N: Eq,
{
    fn default() -> Self {
        Linear {
            leaves: MortonMap::default(),
            internal: MortonRegionMap::default(),
        }
    }
}

impl<T> Linear<T, u128> {
    pub fn new() -> Self {
        Default::default()
    }
}
