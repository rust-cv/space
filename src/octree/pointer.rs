use crate::morton::*;
use crate::octree::*;

use itertools::Itertools;

use rand::Rng;
use std::default::Default;

use log::*;

#[derive(Copy, Clone, Debug, Default)]
pub struct Oct<T> {
    pub children: [T; 8],
}

impl<T> Oct<T> {
    pub fn new(children: [T; 8]) -> Self {
        Oct { children }
    }
}

/// An octree that uses pointers for internal nodes.
pub struct Pointer<T, M> {
    tree: Internal<T, M>,
}

impl<T, M> Default for Pointer<T, M> {
    fn default() -> Self {
        Pointer {
            tree: Internal::default(),
        }
    }
}

impl<T, M> Pointer<T, M>
where
    M: Morton,
{
    /// Creates a new empty octree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an item with a point and return the existing item if they would both occupy the same space.
    pub fn insert(&mut self, morton: M, item: T) {
        // Traverse the tree down to the node we need to operate on.
        let (tree_part, level) = (0..M::dim_bits())
            .fold_while((&mut self.tree, 0), |(node, old_ix), i| {
                use itertools::FoldWhile::{Continue, Done};
                match node {
                    Internal::Node(box Oct { ref mut children }) => {
                        // The index into the array to access the next octree node
                        let subindex = morton.get_level(i);
                        Continue((&mut children[subindex], i))
                    }
                    Internal::Leaf(_, _) => Done((node, old_ix)),
                    Internal::None => Done((node, old_ix)),
                }
            })
            .into_inner();

        match tree_part {
            Internal::Leaf(ref mut leaf_item, dest_morton) => {
                // If they have the same code then replace it.
                if morton == *dest_morton {
                    *leaf_item = item;
                    return;
                }
                // Otherwise we must split them, which we must do outside of this scope due to the borrow.
            }
            Internal::None => {
                // Simply add a new leaf.
                *tree_part = Internal::Leaf(item, morton);
                return;
            }
            _ => {
                unreachable!(
                    "space::Octree::insert(): can only get None or Leaf in this code area"
                );
            }
        }

        let mut dest_old = Internal::empty_node();
        std::mem::swap(&mut dest_old, tree_part);

        if let Internal::Leaf(dest_item, dest_morton) = dest_old {
            // Set our initial reference to the default node in the dest.
            let mut building_node = tree_part;
            // Create deeper nodes till they differ at some level.
            for i in level + 1..M::dim_bits() {
                // We know for sure that the dest is a node.
                if let Internal::Node(box Oct { ref mut children }) = building_node {
                    if morton.get_level(i) == dest_morton.get_level(i) {
                        children[morton.get_level(i)] = Internal::empty_node();
                        building_node = &mut children[morton.get_level(i)];
                    } else {
                        // We reached the end where they differ, so put them both into the node.
                        children[morton.get_level(i)] = Internal::Leaf(item, morton);
                        children[dest_morton.get_level(i)] = Internal::Leaf(dest_item, dest_morton);
                        return;
                    }
                } else {
                    unreachable!("space::Octree::insert(): cant get a non-node in this section");
                }
            }
        } else {
            unreachable!("space::Octree::insert(): cant get a non-leaf in this code area")
        }
    }

    /// Iterate over all octree nodes and their morton codes.
    pub fn iter(&self) -> impl Iterator<Item = (M, &T)> {
        self.tree.iter()
    }

    /// Iterate over all octree nodes, but stop at `depth` to randomly sample a point.
    ///
    /// If `depth` is set to `0`, only one point will be returned, which will either be the only point or
    /// a random sampling (over space, not points) at the node at this point. If a `depth` of `1` is used,
    /// it will traverse down by one level and do `8` random samples at that octree level. This will give back
    /// an iterator of no more than `8` spots.
    fn iter_rand<'a, R: Rng>(
        &'a self,
        depth: usize,
        rng: &'a mut R,
    ) -> impl Iterator<Item = (M, &T)> + 'a {
        self.tree.iter_rand(depth, rng)
    }

    /// Iterates over the octree and, for every internal node in the tree, runs `further` to check if it should
    /// stop at this node or continue down to the leaves. If it stops at an internal node, it passes an iterator
    /// over every leaf that descends from that internal node to `gatherer.gather()`. If it reaches a leaf node,
    /// it passes an iterator over just that one leaf to the `gatherer`. Each invocation of the `gatherer.gather()`
    /// is one item in the returned iterator. This allows an operation to be called on every region in the tree
    /// using `further` to limit the traversal from iterating over the whole tree.
    pub fn iter_gather<'a, F, G>(
        &'a self,
        further: F,
        gatherer: G,
    ) -> impl Iterator<Item = (MortonRegion<M>, G::Sum)> + 'a
    where
        F: FnMut(MortonRegion<M>) -> bool + 'a,
        G: Gatherer<T, M> + 'a,
    {
        self.tree.iter_gather(further, gatherer)
    }

    /// This is a variant of `iter_gather` that addionally allows the caching of gathered nodes.
    ///
    /// Note that whenever a node changes it should invalidate the internal nodes above it in the cache.
    /// See `morton_levels` for more details.
    pub fn iter_gather_cached<'a, F, G>(
        &'a self,
        further: F,
        gatherer: G,
        cache: MortonRegionCache<G::Sum, M>,
    ) -> PointerFurtherGatherCacheIter<'a, T, M, F, G>
    where
        F: FnMut(MortonRegion<M>) -> bool + 'a,
        G: Gatherer<T, M> + 'a,
        G::Sum: Clone,
    {
        self.tree.iter_gather_cached(further, gatherer, cache)
    }

    /// This is a variant of `iter_gather_cached` that takes a `depth` to sample at and will always randomly sample
    /// once starting at that depth. This improves performance by avoiding gathering more that a number of nodes.
    /// For many tasks, choosing a depth of `2` or `64` samples is performant and sufficient.
    pub fn iter_gather_random_cached<'a, F, G, R>(
        &'a self,
        depth: usize,
        further: F,
        gatherer: G,
        rng: &'a mut R,
        cache: MortonRegionCache<G::Sum, M>,
    ) -> PointerFurtherGatherRandomCacheIter<'a, T, M, F, G, R>
    where
        R: Rng,
        F: FnMut(MortonRegion<M>) -> bool + 'a,
        G: Gatherer<T, M> + 'a,
        G::Sum: Clone,
    {
        self.tree
            .iter_gather_random_cached(depth, further, gatherer, rng, cache)
    }

    /// This gathers the tree into a linear hashed octree map. This map contained every internal and leaf node
    /// and the result of gathering them.
    pub fn iter_gather_deep_linear_hashed<G>(&self, gatherer: &G) -> MortonRegionMap<G::Sum, M>
    where
        G: Gatherer<T, M>,
    {
        self.tree.iter_gather_deep_linear_hashed(gatherer)
    }

    /// This gathers the octree in a tree fold by gathering leaves with `gatherer` and folding with `folder`.
    /// This allows information to be folded up the tree so it doesn't have to be computed multiple times.
    /// This has O(n) (exactly `n`) `gather` operations and O(n) (approximately `8/7 * n`) `fold` operations,
    /// with each gather operation always gathering `1` leaf and each `fold` operation gathering no more
    /// than `8` other fold sums.
    pub fn iter_gather_deep_linear_hashed_tree_fold<G, F>(
        &self,
        gatherer: &G,
        folder: &F,
    ) -> MortonRegionMap<G::Sum, M>
    where
        G: Gatherer<T, M>,
        F: Folder<Sum = G::Sum>,
        G::Sum: Clone,
    {
        let mut map = MortonRegionMap::default();
        self.tree.iter_gather_deep_linear_hashed_tree_fold(
            MortonRegion::default(),
            gatherer,
            folder,
            &mut map,
        );
        map
    }
}

impl<T, M> Extend<(M, T)> for Pointer<T, M>
where
    M: Morton,
{
    fn extend<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = (M, T)>,
    {
        for (m, item) in it.into_iter() {
            self.insert(m, item);
        }
    }
}

/// Internal node of a pointer octree.
#[derive(Clone, Debug)]
enum Internal<T, M> {
    Node(Box<Oct<Internal<T, M>>>),
    Leaf(T, M),
    None,
}

impl<T, M> Internal<T, M>
where
    M: Morton,
{
    /// Iterate over all octree nodes and their morton codes.
    pub fn iter(&self) -> impl Iterator<Item = (M, &T)> {
        use either::Either::*;
        match self {
            Internal::Node(box ref n) => Left(InternalIter::new(vec![(&n.children, 0)])),
            Internal::Leaf(ref item, morton) => Right(std::iter::once((*morton, item))),
            Internal::None => Left(InternalIter::new(vec![])),
        }
    }

    /// Iterate over all octree nodes, but stop at `depth` to randomly sample a point.
    ///
    /// If `depth` is set to `0`, only one point will be returned, which will either be the only point or
    /// a random sampling (over space, not points) at the node at this point. If a `depth` of `1` is used,
    /// it will traverse down by one level and do `8` random samples at that octree level. This will give back
    /// an iterator of no more than `8` spots.
    pub fn iter_rand<'a, R: Rng>(
        &'a self,
        depth: usize,
        rng: &'a mut R,
    ) -> impl Iterator<Item = (M, &T)> + 'a {
        use either::Either::*;
        match self {
            Internal::Node(box Oct { ref children }) => {
                if depth == 0 {
                    let mut choice = rng.gen_range(0, 8);
                    // Iterate until we find the first non-empty spot.
                    // This technically results in not completely random behavior
                    // since an octant that comes after more empty octants is more likely to be chosen.
                    while let Internal::None = children[choice] {
                        choice += 1;
                        choice %= 8;
                    }
                    Left({ InternalRandIter::new(vec![(children, choice, 1)], depth, rng) })
                } else {
                    Left({ InternalRandIter::new(vec![(children, 0, 1)], depth, rng) })
                }
            }
            Internal::Leaf(ref item, morton) => Right(std::iter::once((*morton, item))),
            Internal::None => Left(InternalRandIter::new(vec![], depth, rng)),
        }
    }

    /// Iterates over the octree and, for every internal node in the tree, runs `further` to check if it should
    /// stop at this node or continue down to the leaves. If it stops at an internal node, it passes an iterator
    /// over every leaf that descends from that internal node to `gatherer.gather()`. If it reaches a leaf node,
    /// it passes an iterator over just that one leaf to the `gatherer`. Each invocation of the `gatherer.gather()`
    /// is one item in the returned iterator. This allows an operation to be called on every region in the tree
    /// using `further` to limit the traversal from iterating over the whole tree.
    pub fn iter_gather<'a, F, G>(
        &'a self,
        mut further: F,
        gatherer: G,
    ) -> impl Iterator<Item = (MortonRegion<M>, G::Sum)> + 'a
    where
        F: FnMut(MortonRegion<M>) -> bool + 'a,
        G: Gatherer<T, M> + 'a,
    {
        use either::Either::*;
        let base_region = MortonRegion::default();
        match self {
            Internal::Node(box Oct { ref children }) => {
                if further(base_region) {
                    Left(InternalFurtherGatherIter::new(
                        vec![(children, base_region.enter(0))],
                        further,
                        gatherer,
                    ))
                } else {
                    Right(std::iter::once((
                        base_region,
                        gatherer.gather(children.iter().flat_map(|c| c.iter())),
                    )))
                }
            }
            Internal::Leaf(ref item, morton) => Right(std::iter::once((
                base_region,
                gatherer.gather(std::iter::once((*morton, item))),
            ))),
            Internal::None => Left(InternalFurtherGatherIter::new(vec![], further, gatherer)),
        }
    }

    /// This is a variant of `iter_gather` that addionally allows the caching of gathered nodes.
    ///
    /// Note that whenever a node changes it should invalidate the internal nodes above it in the cache.
    /// See `morton_levels` for more details.
    pub fn iter_gather_cached<'a, F, G>(
        &'a self,
        mut further: F,
        gatherer: G,
        mut cache: MortonRegionCache<G::Sum, M>,
    ) -> PointerFurtherGatherCacheIter<'a, T, M, F, G>
    where
        F: FnMut(MortonRegion<M>) -> bool + 'a,
        G: Gatherer<T, M> + 'a,
        G::Sum: Clone,
    {
        let base_region = MortonRegion::default();
        match self {
            Internal::Node(box Oct { ref children }) => {
                if further(base_region) {
                    PointerFurtherGatherCacheIter::Deep(InternalFurtherGatherCacheIter::new(
                        vec![(children, base_region.enter(0))],
                        further,
                        gatherer,
                        cache,
                    ))
                } else {
                    let item = cache.get_mut(&base_region).cloned().unwrap_or_else(|| {
                        let item = gatherer.gather(children.iter().flat_map(|c| c.iter()));
                        cache.insert(base_region, item.clone());
                        item
                    });
                    PointerFurtherGatherCacheIter::Shallow(Some((base_region, item)), cache)
                }
            }
            Internal::Leaf(ref item, morton) => {
                let item = cache.get_mut(&base_region).cloned().unwrap_or_else(|| {
                    let item = gatherer.gather(std::iter::once((*morton, item)));
                    cache.insert(base_region, item.clone());
                    item
                });
                PointerFurtherGatherCacheIter::Shallow(Some((base_region, item)), cache)
            }
            Internal::None => PointerFurtherGatherCacheIter::Deep(
                InternalFurtherGatherCacheIter::new(vec![], further, gatherer, cache),
            ),
        }
    }

    /// This is a variant of `iter_gather_cached` that takes a `depth` to sample at and will always randomly sample
    /// once starting at that depth. This improves performance by avoiding gathering more that a number of nodes.
    /// For many tasks, choosing a depth of `2` or `64` samples is performant and sufficient.
    pub fn iter_gather_random_cached<'a, F, G, R>(
        &'a self,
        depth: usize,
        mut further: F,
        gatherer: G,
        rng: &'a mut R,
        mut cache: MortonRegionCache<G::Sum, M>,
    ) -> PointerFurtherGatherRandomCacheIter<'a, T, M, F, G, R>
    where
        R: Rng,
        F: FnMut(MortonRegion<M>) -> bool + 'a,
        G: Gatherer<T, M> + 'a,
        G::Sum: Clone,
    {
        let base_region = MortonRegion::default();
        if !further(base_region) {
            trace!("chose shallow due to further");
            // If we reach the depth we want or `further` is false, then we must start the random sampling.
            PointerFurtherGatherRandomCacheIter::Shallow(
                cache
                    .get_mut(&base_region)
                    .cloned()
                    .or_else(|| {
                        // We have to make sure this node is not None or else we can't gather it.
                        // This is because `gather` must be guaranteed that its not passed an empty iterator.
                        if let Internal::None = self {
                            None
                        } else {
                            Some(self)
                        }
                        .map(|n| {
                            let item = gatherer.gather(n.iter_rand(depth, rng));
                            cache.insert(base_region, item.clone());
                            item
                        })
                    })
                    .map(|item| (base_region, item)),
                cache,
            )
        } else {
            match self {
                Internal::Node(box Oct { ref children }) => {
                    trace!("chose deep due to node");
                    PointerFurtherGatherRandomCacheIter::Deep(
                        InternalFurtherGatherRandomCacheIter::new(
                            vec![(children, base_region.enter(0))],
                            further,
                            gatherer,
                            depth,
                            rng,
                            cache,
                        ),
                    )
                }
                Internal::Leaf(ref item, morton) => {
                    trace!("chose shallow due to leaf");
                    let item = cache.get_mut(&base_region).cloned().unwrap_or_else(|| {
                        let item = gatherer.gather(std::iter::once((*morton, item)));
                        cache.insert(base_region, item.clone());
                        item
                    });
                    PointerFurtherGatherRandomCacheIter::Shallow(Some((base_region, item)), cache)
                }
                Internal::None => {
                    trace!("chose empty deep due to None");
                    PointerFurtherGatherRandomCacheIter::Deep(
                        InternalFurtherGatherRandomCacheIter::new(
                            vec![],
                            further,
                            gatherer,
                            depth,
                            rng,
                            cache,
                        ),
                    )
                }
            }
        }
    }

    /// This gathers the tree into a linear hashed octree map. This map contained every internal and leaf node
    /// and the result of gathering them.
    pub fn iter_gather_deep_linear_hashed<G>(&self, gatherer: &G) -> MortonRegionMap<G::Sum, M>
    where
        G: Gatherer<T, M>,
    {
        let mut map = MortonRegionMap::default();
        let base_region: MortonRegion<M> = MortonRegion::default();
        let mut nodes = Vec::new();
        match self {
            Internal::Node(box Oct { ref children }) => {
                map.insert(
                    base_region,
                    gatherer.gather(children.iter().flat_map(|c| c.iter())),
                );

                nodes.push((children, base_region.enter(0)));
            }
            Internal::Leaf(ref item, morton) => {
                map.insert(
                    base_region,
                    gatherer.gather(std::iter::once((*morton, item))),
                );
            }
            _ => {}
        }

        while let Some((node, region)) = nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                nodes.push((node, next));
            }

            match node[region.get()] {
                Internal::Node(box Oct { ref children }) => {
                    map.insert(
                        region,
                        gatherer.gather(children.iter().flat_map(|c| c.iter())),
                    );

                    if region.level < M::dim_bits() - 1 {
                        nodes.push((children, region.enter(0)));
                    }
                }
                Internal::Leaf(ref item, morton) => {
                    map.insert(region, gatherer.gather(std::iter::once((morton, item))));
                }
                _ => {}
            }
        }

        map
    }

    /// This gathers the octree in a tree fold by gathering leaves with `gatherer` and folding with `folder`.
    /// This allows information to be folded up the tree so it doesn't have to be computed multiple times.
    /// This has O(n) `gather` operations and O(n^(7/8)) `fold` operations, with each gather operation
    /// always gathering `1` leaf and each `fold` operation gathering no more than `8` other folds.
    pub fn iter_gather_deep_linear_hashed_tree_fold<G, F>(
        &self,
        region: MortonRegion<M>,
        gatherer: &G,
        folder: &F,
        map: &mut MortonRegionMap<G::Sum, M>,
    ) -> Option<G::Sum>
    where
        G: Gatherer<T, M>,
        F: Folder<Sum = G::Sum>,
        G::Sum: Clone,
    {
        match self {
            Internal::Node(box Oct { ref children }) => {
                if region.level < M::dim_bits() {
                    if let Some(sum) = folder.sum((0..8).filter_map(|i| {
                        children[i].iter_gather_deep_linear_hashed_tree_fold(
                            region.enter(i),
                            gatherer,
                            folder,
                            map,
                        )
                    })) {
                        map.insert(region, sum.clone());
                        Some(sum)
                    } else {
                        None
                    }
                } else {
                    panic!("iter_gather_deep_linear_hashed_tree_fold(): if we get here, then we let a leaf descend pass morton range");
                }
            }
            Internal::Leaf(ref item, morton) => {
                let sum = gatherer.gather(std::iter::once((*morton, item)));
                map.insert(region, sum.clone());
                Some(sum)
            }
            _ => None,
        }
    }

    /// Gives back a `Node` with 8 empty `None` nodes.
    #[inline]
    pub fn empty_node() -> Self {
        use self::Internal::*;
        Node(box Oct::new([
            None, None, None, None, None, None, None, None,
        ]))
    }
}

impl<T, M> Default for Internal<T, M> {
    fn default() -> Self {
        Internal::None
    }
}

struct InternalIter<'a, T, M> {
    nodes: Vec<(&'a [Internal<T, M>; 8], usize)>,
}

impl<'a, T, M> InternalIter<'a, T, M>
where
    M: Morton,
{
    fn new(nodes: Vec<(&'a [Internal<T, M>; 8], usize)>) -> Self {
        InternalIter { nodes }
    }
}

impl<'a, T, M> Iterator for InternalIter<'a, T, M>
where
    M: Morton,
{
    type Item = (M, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, ix)) = self.nodes.pop() {
            if ix != 7 {
                self.nodes.push((node, ix + 1));
            }
            match node[ix] {
                Internal::Node(box Oct { ref children }) => self.nodes.push((children, 0)),
                Internal::Leaf(ref item, morton) => {
                    return Some((morton, item));
                }
                _ => {}
            }
        }
        None
    }
}

type NodeIndexLevel<'a, T, M> = (&'a [Internal<T, M>; 8], usize, usize);

struct InternalRandIter<'a, T, M, R> {
    nodes: Vec<NodeIndexLevel<'a, T, M>>,
    depth: usize,
    rng: &'a mut R,
}

impl<'a, T, M, R> InternalRandIter<'a, T, M, R>
where
    M: Morton,
    R: Rng,
{
    fn new(nodes: Vec<NodeIndexLevel<'a, T, M>>, depth: usize, rng: &'a mut R) -> Self {
        InternalRandIter { nodes, depth, rng }
    }
}

impl<'a, T, M, R> Iterator for InternalRandIter<'a, T, M, R>
where
    M: Morton,
    R: Rng,
{
    type Item = (M, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, ix, level)) = self.nodes.pop() {
            if level <= self.depth && ix != 7 {
                self.nodes.push((node, ix + 1, level));
            }
            match node[ix] {
                Internal::Node(box Oct { ref children }) => self.nodes.push((
                    children,
                    if level >= self.depth {
                        let mut choice = self.rng.gen_range(0, 8);
                        // Iterate until we find the first non-empty spot.
                        // This technically results in not completely random behavior
                        // since an octant that comes after more empty octants is more likely to be chosen.
                        while let Internal::None = children[choice] {
                            choice += 1;
                            choice %= 8;
                        }
                        choice
                    } else {
                        0
                    },
                    level + 1,
                )),
                Internal::Leaf(ref item, morton) => {
                    return Some((morton, item));
                }
                _ => {}
            }
        }
        None
    }
}

type InternalFurtherGatherIterNodeStack<'a, T, M> = Vec<(&'a [Internal<T, M>; 8], MortonRegion<M>)>;

struct InternalFurtherGatherIter<'a, T, M, F, G> {
    nodes: InternalFurtherGatherIterNodeStack<'a, T, M>,
    further: F,
    gatherer: G,
}

impl<'a, T, M, F, G> InternalFurtherGatherIter<'a, T, M, F, G> {
    fn new(nodes: InternalFurtherGatherIterNodeStack<'a, T, M>, further: F, gatherer: G) -> Self {
        InternalFurtherGatherIter {
            nodes,
            further,
            gatherer,
        }
    }
}

impl<'a, T, M, F, G> Iterator for InternalFurtherGatherIter<'a, T, M, F, G>
where
    M: Morton,
    F: FnMut(MortonRegion<M>) -> bool,
    G: Gatherer<T, M>,
{
    type Item = (MortonRegion<M>, G::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, region)) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push((node, next));
            }

            match node[region.get()] {
                Internal::Node(box Oct { ref children }) => {
                    if (self.further)(region) {
                        self.nodes.push((children, region.enter(0)));
                    } else {
                        return Some((
                            region,
                            self.gatherer.gather(children.iter().flat_map(|c| c.iter())),
                        ));
                    }
                }
                Internal::Leaf(ref item, morton) => {
                    return Some((
                        region,
                        self.gatherer.gather(std::iter::once((morton, item))),
                    ));
                }
                _ => {}
            }
        }
        None
    }
}

pub enum PointerFurtherGatherCacheIter<'a, T, M, F, G>
where
    G: Gatherer<T, M>,
    M: Morton,
{
    Deep(InternalFurtherGatherCacheIter<'a, T, M, F, G>),
    Shallow(
        Option<(MortonRegion<M>, G::Sum)>,
        MortonRegionCache<G::Sum, M>,
    ),
}

impl<'a, T, M, F, G> Into<MortonRegionCache<G::Sum, M>>
    for PointerFurtherGatherCacheIter<'a, T, M, F, G>
where
    G: Gatherer<T, M>,
    M: Morton,
{
    fn into(self) -> MortonRegionCache<G::Sum, M> {
        use self::PointerFurtherGatherCacheIter::*;
        match self {
            Deep(d) => d.into_cache(),
            Shallow(_, cache) => cache,
        }
    }
}

impl<'a, T, M, F, G> Iterator for PointerFurtherGatherCacheIter<'a, T, M, F, G>
where
    M: Morton,
    F: FnMut(MortonRegion<M>) -> bool,
    G: Gatherer<T, M>,
    G::Sum: Clone,
{
    type Item = (MortonRegion<M>, G::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        use self::PointerFurtherGatherCacheIter::*;
        match self {
            Deep(d) => d.next(),
            Shallow(s, _) => s.take(),
        }
    }
}

type InternalFurtherGatherCacheIterNodeStack<'a, T, M> =
    Vec<(&'a [Internal<T, M>; 8], MortonRegion<M>)>;

pub struct InternalFurtherGatherCacheIter<'a, T, M, F, G>
where
    G: Gatherer<T, M>,
    M: Morton,
{
    nodes: InternalFurtherGatherCacheIterNodeStack<'a, T, M>,
    further: F,
    gatherer: G,
    cache: MortonRegionCache<G::Sum, M>,
}

impl<'a, T, M, F, G> InternalFurtherGatherCacheIter<'a, T, M, F, G>
where
    G: Gatherer<T, M>,
    M: Morton,
{
    fn new(
        nodes: InternalFurtherGatherCacheIterNodeStack<'a, T, M>,
        further: F,
        gatherer: G,
        cache: MortonRegionCache<G::Sum, M>,
    ) -> Self {
        InternalFurtherGatherCacheIter {
            nodes,
            further,
            gatherer,
            cache,
        }
    }

    pub fn into_cache(self) -> MortonRegionCache<G::Sum, M> {
        self.cache
    }
}

impl<'a, T, M, F, G> Iterator for InternalFurtherGatherCacheIter<'a, T, M, F, G>
where
    M: Morton,
    F: FnMut(MortonRegion<M>) -> bool,
    G: Gatherer<T, M>,
    G::Sum: Clone,
{
    type Item = (MortonRegion<M>, G::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, region)) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push((node, next));
            }

            match node[region.get()] {
                Internal::Node(box Oct { ref children }) => {
                    if (self.further)(region) {
                        self.nodes.push((children, region.enter(0)));
                    } else {
                        let item = self.cache.get_mut(&region).cloned().unwrap_or_else(|| {
                            let item = self.gatherer.gather(children.iter().flat_map(|c| c.iter()));
                            self.cache.insert(region, item.clone());
                            item
                        });
                        return Some((region, item));
                    }
                }
                Internal::Leaf(ref item, morton) => {
                    let item = self.cache.get_mut(&region).cloned().unwrap_or_else(|| {
                        let item = self.gatherer.gather(std::iter::once((morton, item)));
                        self.cache.insert(region, item.clone());
                        item
                    });

                    return Some((region, item));
                }
                _ => {}
            }
        }
        None
    }
}

pub enum PointerFurtherGatherRandomCacheIter<'a, T, M, F, G, R>
where
    G: Gatherer<T, M>,
    R: Rng,
    M: Morton,
{
    Deep(InternalFurtherGatherRandomCacheIter<'a, T, M, F, G, R>),
    Shallow(
        Option<(MortonRegion<M>, G::Sum)>,
        MortonRegionCache<G::Sum, M>,
    ),
}

impl<'a, T, M, F, G, R> Into<MortonRegionCache<G::Sum, M>>
    for PointerFurtherGatherRandomCacheIter<'a, T, M, F, G, R>
where
    G: Gatherer<T, M>,
    R: Rng,
    M: Morton,
{
    fn into(self) -> MortonRegionCache<G::Sum, M> {
        use self::PointerFurtherGatherRandomCacheIter::*;
        match self {
            Deep(d) => d.into_cache(),
            Shallow(_, cache) => cache,
        }
    }
}

impl<'a, T, M, F, G, R> Iterator for PointerFurtherGatherRandomCacheIter<'a, T, M, F, G, R>
where
    M: Morton,
    R: Rng,
    F: FnMut(MortonRegion<M>) -> bool,
    G: Gatherer<T, M>,
    G::Sum: Clone,
{
    type Item = (MortonRegion<M>, G::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        use self::PointerFurtherGatherRandomCacheIter::*;
        match self {
            Deep(d) => d.next(),
            Shallow(s, _) => s.take(),
        }
    }
}

type InternalFurtherGatherRandomCacheIterNodeStack<'a, T, M> =
    Vec<(&'a [Internal<T, M>; 8], MortonRegion<M>)>;

pub struct InternalFurtherGatherRandomCacheIter<'a, T, M, F, G, R>
where
    G: Gatherer<T, M>,
    R: Rng,
    M: Morton,
{
    nodes: InternalFurtherGatherRandomCacheIterNodeStack<'a, T, M>,
    further: F,
    gatherer: G,
    depth: usize,
    rng: &'a mut R,
    cache: MortonRegionCache<G::Sum, M>,
}

impl<'a, T, M, F, G, R> InternalFurtherGatherRandomCacheIter<'a, T, M, F, G, R>
where
    G: Gatherer<T, M>,
    R: Rng,
    M: Morton,
{
    fn new(
        nodes: InternalFurtherGatherRandomCacheIterNodeStack<'a, T, M>,
        further: F,
        gatherer: G,
        depth: usize,
        rng: &'a mut R,
        cache: MortonRegionCache<G::Sum, M>,
    ) -> Self {
        InternalFurtherGatherRandomCacheIter {
            nodes,
            further,
            gatherer,
            depth,
            rng,
            cache,
        }
    }

    pub fn into_cache(self) -> MortonRegionCache<G::Sum, M> {
        self.cache
    }
}

impl<'a, T, M, F, G, R> Iterator for InternalFurtherGatherRandomCacheIter<'a, T, M, F, G, R>
where
    M: Morton,
    F: FnMut(MortonRegion<M>) -> bool,
    G: Gatherer<T, M>,
    G::Sum: Clone,
    R: Rng,
{
    type Item = (MortonRegion<M>, G::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, region)) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push((node, next));
            }

            // If we shouldn't go further into the region, then its time to do a random sample starting here.
            if !(self.further)(region) {
                trace!("chose not to go further");
                // If we reach the depth we want or `further` is false, then we must start the random sampling.
                if let Some(r) = self
                    .cache
                    .get_mut(&region)
                    .cloned()
                    .or_else(|| {
                        // We have to make sure this node is not None or else we can't gather it.
                        // This is because `gather` must be guaranteed that its not passed an empty iterator.
                        if let Internal::None = node[region.get()] {
                            None
                        } else {
                            Some(&node[region.get()])
                        }
                        .map(|n| {
                            let item = self.gatherer.gather(n.iter_rand(self.depth, self.rng));
                            self.cache.insert(region, item.clone());
                            item
                        })
                    })
                    .map(|item| (region, item))
                {
                    return Some(r);
                }
            } else {
                match node[region.get()] {
                    Internal::Node(box Oct { ref children }) => {
                        trace!("traversing deeper due to node at level {}", region.level);
                        // Traverse deeper (we already checked if we didn't need to go further).
                        self.nodes.push((children, region.enter(0)));
                    }
                    Internal::Leaf(ref item, morton) => {
                        trace!("stopping due to leaf at level {}", region.level);
                        let item = self.cache.get_mut(&region).cloned().unwrap_or_else(|| {
                            let item = self.gatherer.gather(std::iter::once((morton, item)));
                            self.cache.insert(region, item.clone());
                            item
                        });

                        return Some((region, item));
                    }
                    _ => {}
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::izip;
    use nalgebra::Vector3;
    use rand::distributions::Open01;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_octree_insert_rand() {
        let mut xrng = SmallRng::from_seed([1; 16]);
        let mut yrng = SmallRng::from_seed([4; 16]);
        let mut zrng = SmallRng::from_seed([0; 16]);

        let mut octree = Pointer::<_, u128>::new();
        let space = LeveledRegion(0);
        octree.extend(
            izip!(
                xrng.sample_iter(&Open01),
                yrng.sample_iter(&Open01),
                zrng.sample_iter(&Open01)
            )
            .take(5000)
            .map(|(x, y, z)| (space.discretize(Vector3::<f64>::new(x, y, z)).unwrap(), 0)),
        );

        assert_eq!(octree.iter().count(), 5000);
    }
}
