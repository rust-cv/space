use super::{Morton, MortonMap, MortonRegion, MortonRegionSet};
use smallvec::SmallVec;
use std::cmp::Eq;
use std::hash::Hash;

/// An octree that starts with a cube from [-1, 1] in each dimension and will only expand.
#[derive(Clone)]
pub struct Linear<T, N> {
    /// The leaves of the octree. Uses `SmallVec` because in most cases this shouldn't have more than one element.
    leaves: MortonMap<SmallVec<[T; 1]>, N>,
    /// The empty regions in the tree.
    empty: MortonRegionSet<N>,
}

impl<T, N> Default for Linear<T, N>
where
    Morton<N>: Hash,
    MortonRegion<N>: Hash + Default,
    N: Eq,
{
    fn default() -> Self {
        let mut empty = MortonRegionSet::default();
        empty.insert(MortonRegion::default());
        Linear {
            leaves: MortonMap::default(),
            empty,
        }
    }
}

impl<T> Linear<T, u128> {
    pub fn new() -> Self {
        Default::default()
    }
}
