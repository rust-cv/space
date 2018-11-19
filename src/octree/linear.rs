use crate::{morton::*, octree::Folder};

/// A linear hashed octree. This has constant time lookup for a given region or morton code.
#[derive(Clone)]
pub struct Linear<T, M> {
    /// The leaves of the octree.
    leaves: MortonMap<T, M>,
    /// The each internal node either contains a `null` Morton or a non-null Morton which points to a leaf.
    /// Nodes which are not explicity stated implicitly indicate that it must be traversed deeper.
    internals: MortonRegionMap<M, M>,
}

impl<T, M> Default for Linear<T, M>
where
    M: Morton,
{
    fn default() -> Self {
        let mut internals = MortonRegionMap::default();
        internals.insert(MortonRegion::default(), M::null());
        Linear {
            leaves: MortonMap::<_, M>::default(),
            internals,
        }
    }
}

impl<T, M> Linear<T, M>
where
    M: Morton,
{
    /// Create an empty linear octree.
    pub fn new() -> Self {
        Default::default()
    }

    /// Inserts the item into the octree.
    ///
    /// If another element occupied the exact same morton, it will be evicted and replaced.
    pub fn insert(&mut self, morton: M, item: T) {
        use std::collections::hash_map::Entry::*;
        // First we must insert the node into the leaves.
        match self.leaves.entry(MortonWrapper(morton)) {
            Occupied(mut o) => {
                o.insert(item);
            }
            Vacant(v) => {
                v.insert(item);

                // Because it was vacant, we need to adjust the tree's internal nodes.
                for mut region in morton_levels(morton) {
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
                            for level in region.level..M::dim_bits() {
                                let leaf_level = leaf.get_level(level);
                                let item_level = morton.get_level(level);
                                if leaf_level == item_level {
                                    // They were the same so set every other region to null.
                                    for i in 0..8 {
                                        if i != leaf_level {
                                            self.internals.insert(region.enter(i), M::null());
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
                                            self.internals.insert(region.enter(i), M::null());
                                        }
                                    }
                                    // Now we must return as we have added the leaves.
                                    return;
                                }
                            }
                            unreachable!();
                        }
                    }
                }
            }
        }
    }

    /// This gathers the octree in a tree fold by gathering leaves with `gatherer` and folding with `folder`.
    /// This allows information to be folded up the tree so it doesn't have to be computed multiple times.
    /// This has O(n) (exactly `n`) `gather` operations and O(n) (approximately `8/7 * n`) `fold` operations,
    /// with each gather operation always gathering `1` leaf and each `fold` operation gathering no more
    /// than `8` other fold sums.
    pub fn collect_fold<F>(&self, folder: &F) -> MortonRegionMap<F::Sum, M>
    where
        F: Folder<T, M>,
        F::Sum: Clone,
    {
        let mut map = MortonRegionMap::default();
        self.collect_fold_region(MortonRegion::base(), folder, &mut map);
        map
    }

    /// Same as `collect_fold`, but adds things to a morton region map and gives back the region.
    pub fn collect_fold_region<F>(
        &self,
        region: MortonRegion<M>,
        folder: &F,
        map: &mut MortonRegionMap<F::Sum, M>,
    ) -> Option<F::Sum>
    where
        F: Folder<T, M>,
        F::Sum: Clone,
    {
        match self.internals.get(&region) {
            Some(m) if !m.is_null() => {
                // This is a leaf node.
                let sum = folder.gather(*m, &self.leaves[&MortonWrapper(*m)]);
                map.insert(region, sum.clone());
                Some(sum)
            }
            None => {
                // This needs to be traversed deeper.
                let sum =
                    folder
                        .fold((0..8).filter_map(|i| {
                            self.collect_fold_region(region.enter(i), folder, map)
                        }));
                map.insert(region, sum.clone());
                Some(sum)
            }
            _ => None,
        }
    }
}

impl<T, M> Extend<(M, T)> for Linear<T, M>
where
    M: Morton + Default,
{
    fn extend<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = (M, T)>,
    {
        for (morton, item) in it.into_iter() {
            self.insert(morton, item);
        }
    }
}
