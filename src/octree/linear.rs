use super::{
    Morton, MortonMap, MortonRegion, MortonRegionMap, NUM_BITS_PER_DIM_128, NUM_BITS_PER_DIM_64,
};
use smallvec::{smallvec, SmallVec};

/// An octree that starts with a cube from [-1, 1] in each dimension and will only expand.
#[derive(Clone)]
pub struct Linear<T, N> {
    /// The leaves of the octree. Uses `SmallVec` because in most cases this shouldn't have more than one element.
    leaves: MortonMap<SmallVec<[T; 1]>, N>,
    /// The each internal node either contains a `null` Morton or a non-null Morton which points to a leaf.
    /// Nodes which are not explicity stated implicitly indicate that it must be traversed deeper.
    internals: MortonRegionMap<Morton<N>, N>,
}

impl<T> Default for Linear<T, u128> {
    fn default() -> Self {
        let mut internals = MortonRegionMap::<_, u128>::default();
        internals.insert(MortonRegion::default(), Morton::<u128>::null());
        Linear {
            leaves: MortonMap::default(),
            internals,
        }
    }
}

impl<T> Linear<T, u128> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn insert(&mut self, morton: Morton<u128>, item: T) {
        use std::collections::hash_map::Entry::*;
        // First we must insert the node into the leaves.
        match self.leaves.entry(morton) {
            Occupied(mut o) => o.get_mut().push(item),
            Vacant(v) => {
                v.insert(smallvec![item]);

                // Because it was vacant, we need to adjust the tree's internal nodes.
                for mut region in morton.levels() {
                    // Check if the region is in the map.
                    if let Occupied(mut o) = self.internals.entry(region) {
                        // It was in the map. Check if it was null or not.
                        if o.get().is_null() {
                            // It was null, so just replace the null with the leaf.
                            *o.get_mut() = morton;
                            // Now return because we are done.
                            return;
                        } else {
                            // It was not null, so it is a leaf.
                            // This means that we need to move the leaf to its sub-region.
                            // We also need to populate the other 6 null nodes created by this operation.
                            let leaf = o.remove_entry().1;
                            // Keep making the tree deeper until both leaves differ.
                            // TODO: Some bittwiddling with mortons might be able to get the number of traversals.
                            for level in region.level..NUM_BITS_PER_DIM_128 {
                                let leaf_level = leaf.get_level(level);
                                let item_level = morton.get_level(level);
                                if leaf_level == item_level {
                                    // They were the same so set every other region to null.
                                    for i in 0..8 {
                                        if i != leaf_level {
                                            self.internals
                                                .insert(region.enter(i), Morton::<u128>::null());
                                        }
                                    }
                                    region = region.enter(leaf_level);
                                } else {
                                    // They were different, so set the other 6 regions null and make 2 leaves.
                                    for i in 0..8 {
                                        if i == leaf_level {
                                            self.internals.insert(region.enter(i), leaf);
                                        } else if i == item_level {
                                            self.internals.insert(region.enter(i), morton);
                                        } else {
                                            self.internals
                                                .insert(region.enter(i), Morton::<u128>::null());
                                        }
                                    }
                                    // Now we must return as we have added the leaves.
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
