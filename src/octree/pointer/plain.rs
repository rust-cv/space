use crate::octree::morton::*;
use crate::octree::*;
use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

use itertools::Itertools;

use std::default::Default;

#[allow(type_alias_bounds)]
type CacheGatherIter<'a, T, N, F, G: Gatherer<T, N>> = either::Either<
    MortonOctreeFurtherGatherCacheIter<'a, T, N, F, G>,
    std::iter::Once<(MortonRegion<N>, &'a mut G::Sum)>,
>;

/// An octree that starts with a cube from [-1, 1] in each dimension and will only expand.
pub struct Plain<T, N> {
    tree: MortonOctree<T, N>,
    /// Dimensions of the top level node are from [-2**level, 2**level].
    level: i32,
}

impl<T> Plain<T, u64> {
    /// Dimensions of the top level node are fixed in the range [-2**level, 2**level].
    pub fn new(level: i32) -> Self {
        Plain {
            tree: MortonOctree::default(),
            level,
        }
    }

    pub fn morton<S>(&self, point: &Vector3<S>) -> Morton<u64>
    where
        S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
    {
        let bound = (S::one() + S::one()).powi(self.level);
        if point.iter().any(|n| n.abs() > bound) {
            panic!("space::Octree::morton(): tried to compute a Morton outside the Octree bounds");
        }

        // Convert the point into normalized space.
        Morton::from(point.map(|n| (n + bound) / bound.powi(2)))
    }

    /// Insert an item with a point and return the existing item if they would both occupy the same space.
    pub fn insert(&mut self, morton: Morton<u64>, item: T) {
        // Traverse the tree down to the node we need to operate on.
        let (tree_part, level) = (0..NUM_BITS_PER_DIM_64)
            .fold_while((&mut self.tree, 0), |(node, old_ix), i| {
                use itertools::FoldWhile::{Continue, Done};
                match node {
                    MortonOctree::Node(ns) => {
                        // The index into the array to access the next octree node
                        let subindex = morton.get_level(i);
                        Continue((&mut ns[subindex], i))
                    }
                    MortonOctree::Leaf(_, _) => Done((node, old_ix)),
                    MortonOctree::None => Done((node, old_ix)),
                }
            })
            .into_inner();

        match tree_part {
            MortonOctree::Leaf(nodes, dest_morton) => {
                // If they have the same code then add it to the same Vec and be done with it.
                if morton == *dest_morton {
                    nodes.push(item);
                    return;
                }
                // Otherwise we must split them, which we must do outside of this scope due to the borrow.
            }
            MortonOctree::None => {
                // Simply add a new leaf.
                *tree_part = MortonOctree::Leaf(vec![item], morton);
                return;
            }
            _ => {
                unreachable!(
                    "space::Octree::insert(): can only get None or Leaf in this code area"
                );
            }
        }

        let mut dest_old = MortonOctree::<T, u64>::empty_node();
        std::mem::swap(&mut dest_old, tree_part);

        if let MortonOctree::Leaf(dest_vec, dest_morton) = dest_old {
            // Set our initial reference to the default node in the dest.
            let mut building_node = tree_part;
            // Create deeper nodes till they differ at some level.
            for i in level + 1..NUM_BITS_PER_DIM_64 {
                // We know for sure that the dest is a node.
                if let MortonOctree::Node(box ref mut children) = building_node {
                    if morton.get_level(i) == dest_morton.get_level(i) {
                        children[morton.get_level(i)] = MortonOctree::<T, u64>::empty_node();
                        building_node = &mut children[morton.get_level(i)];
                    } else {
                        // We reached the end where they differ, so put them both into the node.
                        children[morton.get_level(i)] = MortonOctree::Leaf(vec![item], morton);
                        children[dest_morton.get_level(i)] =
                            MortonOctree::Leaf(dest_vec, dest_morton);
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

    pub fn iter(&self) -> impl Iterator<Item = (Morton<u64>, &T)> {
        self.tree.iter()
    }

    pub fn iter_gather<'a, F, G>(
        &'a self,
        further: F,
        gatherer: G,
    ) -> impl Iterator<Item = (MortonRegion<u64>, G::Sum)> + 'a
    where
        F: FnMut(MortonRegion<u64>) -> bool + 'a,
        G: Gatherer<T, u64> + 'a,
    {
        self.tree.iter_gather(further, gatherer)
    }

    pub fn iter_gather_cached<'a, F, G>(
        &'a self,
        further: F,
        gatherer: G,
        cache: &'a mut MortonMap<G::Sum, u64>,
    ) -> CacheGatherIter<'a, T, u64, F, G>
    where
        F: FnMut(MortonRegion<u64>) -> bool + 'a,
        G: Gatherer<T, u64> + 'a,
    {
        self.tree.iter_gather_cached(further, gatherer, cache)
    }

    /// This gathers the tree into a linear hashed octree map.
    pub fn iter_gather_deep_linear_hashed<G>(&self, gatherer: &G) -> MortonMap<G::Sum, u64>
    where
        G: Gatherer<T, u64>,
    {
        self.tree.iter_gather_deep_linear_hashed(gatherer)
    }
}

impl<T> Plain<T, u128> {
    /// Dimensions of the top level node are fixed in the range [-2**level, 2**level].
    pub fn new(level: i32) -> Self {
        Plain {
            tree: MortonOctree::default(),
            level,
        }
    }

    pub fn morton<S>(&self, point: &Vector3<S>) -> Morton<u128>
    where
        S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
    {
        let bound = (S::one() + S::one()).powi(self.level);
        if point.iter().any(|n| n.abs() > bound) {
            panic!("space::Octree::morton(): tried to compute a Morton outside the Octree bounds");
        }

        // Convert the point into normalized space.
        Morton::from(point.map(|n| (n + bound) / bound.powi(2)))
    }

    /// Insert an item with a point and return the existing item if they would both occupy the same space.
    pub fn insert(&mut self, morton: Morton<u128>, item: T) {
        // Traverse the tree down to the node we need to operate on.
        let (tree_part, level) = (0..NUM_BITS_PER_DIM_128)
            .fold_while((&mut self.tree, 0), |(node, old_ix), i| {
                use itertools::FoldWhile::{Continue, Done};
                match node {
                    MortonOctree::Node(ns) => {
                        // The index into the array to access the next octree node
                        let subindex = morton.get_level(i);
                        Continue((&mut ns[subindex], i))
                    }
                    MortonOctree::Leaf(_, _) => Done((node, old_ix)),
                    MortonOctree::None => Done((node, old_ix)),
                }
            })
            .into_inner();

        match tree_part {
            MortonOctree::Leaf(nodes, dest_morton) => {
                // If they have the same code then add it to the same Vec and be done with it.
                if morton == *dest_morton {
                    nodes.push(item);
                    return;
                }
                // Otherwise we must split them, which we must do outside of this scope due to the borrow.
            }
            MortonOctree::None => {
                // Simply add a new leaf.
                *tree_part = MortonOctree::Leaf(vec![item], morton);
                return;
            }
            _ => {
                unreachable!(
                    "space::Octree::insert(): can only get None or Leaf in this code area"
                );
            }
        }

        let mut dest_old = MortonOctree::<T, u128>::empty_node();
        std::mem::swap(&mut dest_old, tree_part);

        if let MortonOctree::Leaf(dest_vec, dest_morton) = dest_old {
            // Set our initial reference to the default node in the dest.
            let mut building_node = tree_part;
            // Create deeper nodes till they differ at some level.
            for i in level + 1..NUM_BITS_PER_DIM_128 {
                // We know for sure that the dest is a node.
                if let MortonOctree::Node(box ref mut children) = building_node {
                    if morton.get_level(i) == dest_morton.get_level(i) {
                        children[morton.get_level(i)] = MortonOctree::<T, u128>::empty_node();
                        building_node = &mut children[morton.get_level(i)];
                    } else {
                        // We reached the end where they differ, so put them both into the node.
                        children[morton.get_level(i)] = MortonOctree::Leaf(vec![item], morton);
                        children[dest_morton.get_level(i)] =
                            MortonOctree::Leaf(dest_vec, dest_morton);
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

    pub fn iter(&self) -> impl Iterator<Item = (Morton<u128>, &T)> {
        self.tree.iter()
    }

    pub fn iter_gather<'a, F, G>(
        &'a self,
        further: F,
        gatherer: G,
    ) -> impl Iterator<Item = (MortonRegion<u128>, G::Sum)> + 'a
    where
        F: FnMut(MortonRegion<u128>) -> bool + 'a,
        G: Gatherer<T, u128> + 'a,
    {
        self.tree.iter_gather(further, gatherer)
    }

    pub fn iter_gather_cached<'a, F, G>(
        &'a self,
        further: F,
        gatherer: G,
        cache: &'a mut MortonMap<G::Sum, u128>,
    ) -> CacheGatherIter<'a, T, u128, F, G>
    where
        F: FnMut(MortonRegion<u128>) -> bool + 'a,
        G: Gatherer<T, u128> + 'a,
    {
        self.tree.iter_gather_cached(further, gatherer, cache)
    }

    /// This gathers the tree into a linear hashed octree map.
    pub fn iter_gather_deep_linear_hashed<G>(&self, gatherer: &G) -> MortonMap<G::Sum, u128>
    where
        G: Gatherer<T, u128>,
    {
        self.tree.iter_gather_deep_linear_hashed(gatherer)
    }
}

impl<T, S> Extend<(Vector3<S>, T)> for Plain<T, u64>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    fn extend<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = (Vector3<S>, T)>,
    {
        for (v, item) in it.into_iter() {
            self.insert(self.morton(&v), item);
        }
    }
}

impl<T, S> Extend<(Vector3<S>, T)> for Plain<T, u128>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    fn extend<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = (Vector3<S>, T)>,
    {
        for (v, item) in it.into_iter() {
            self.insert(self.morton(&v), item);
        }
    }
}

impl<'a, T, S> Extend<(&'a Vector3<S>, T)> for Plain<T, u64>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    fn extend<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = (&'a Vector3<S>, T)>,
    {
        for (v, item) in it.into_iter() {
            self.insert(self.morton(v), item);
        }
    }
}

impl<'a, T, S> Extend<(&'a Vector3<S>, T)> for Plain<T, u128>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    fn extend<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = (&'a Vector3<S>, T)>,
    {
        for (v, item) in it.into_iter() {
            self.insert(self.morton(v), item);
        }
    }
}

/// Tree with space implicitly divided based on a Morton code.
#[derive(Clone, Debug)]
enum MortonOctree<T, N> {
    Node(Box<[MortonOctree<T, N>; 8]>),
    Leaf(Vec<T>, Morton<N>),
    None,
}

impl<T> MortonOctree<T, u64> {
    fn iter(&self) -> impl Iterator<Item = (Morton<u64>, &T)> {
        use either::Either::*;
        match self {
            MortonOctree::Node(box ref n) => Left(MortonOctreeIter::new(vec![(n, 0)])),
            MortonOctree::Leaf(ref item, morton) => {
                Right(item.iter().map(move |item| (*morton, item)))
            }
            MortonOctree::None => Left(MortonOctreeIter::new(vec![])),
        }
    }

    fn iter_gather<'a, F, G>(
        &'a self,
        mut further: F,
        gatherer: G,
    ) -> impl Iterator<Item = (MortonRegion<u64>, G::Sum)> + 'a
    where
        F: FnMut(MortonRegion<u64>) -> bool + 'a,
        G: Gatherer<T, u64> + 'a,
    {
        use either::Either::*;
        let base_region = MortonRegion {
            morton: Morton(0),
            level: 0,
        };
        match self {
            MortonOctree::Node(box ref n) => {
                if further(base_region) {
                    Left(MortonOctreeFurtherGatherIter::new(
                        vec![(n, base_region.enter(0))],
                        further,
                        gatherer,
                    ))
                } else {
                    Right(std::iter::once((
                        base_region,
                        gatherer.gather(n.iter().flat_map(|c| c.iter())),
                    )))
                }
            }
            MortonOctree::Leaf(ref items, morton) => Right(std::iter::once((
                base_region,
                gatherer.gather(items.iter().map(|i| (*morton, i))),
            ))),
            MortonOctree::None => Left(MortonOctreeFurtherGatherIter::new(
                vec![],
                further,
                gatherer,
            )),
        }
    }

    fn iter_gather_cached<'a, F, G>(
        &'a self,
        mut further: F,
        gatherer: G,
        cache: &'a mut MortonMap<G::Sum, u64>,
    ) -> CacheGatherIter<'a, T, u64, F, G>
    where
        F: FnMut(MortonRegion<u64>) -> bool + 'a,
        G: Gatherer<T, u64> + 'a,
    {
        use either::Either::*;
        let base_region = MortonRegion {
            morton: Morton(0),
            level: 0,
        };
        match self {
            MortonOctree::Node(box ref n) => {
                if further(base_region) {
                    Left(MortonOctreeFurtherGatherCacheIter::new(
                        vec![(n, base_region.enter(0))],
                        further,
                        gatherer,
                        cache,
                    ))
                } else {
                    Right(std::iter::once((
                        base_region,
                        cache
                            .entry(base_region)
                            .or_insert_with(|| gatherer.gather(n.iter().flat_map(|c| c.iter()))),
                    )))
                }
            }
            MortonOctree::Leaf(ref items, morton) => Right(std::iter::once((
                base_region,
                cache
                    .entry(base_region)
                    .or_insert_with(|| gatherer.gather(items.iter().map(|i| (*morton, i)))),
            ))),
            MortonOctree::None => Left(MortonOctreeFurtherGatherCacheIter::new(
                vec![],
                further,
                gatherer,
                cache,
            )),
        }
    }

    /// This gathers the tree into a linear hashed octree map.
    pub fn iter_gather_deep_linear_hashed<G>(&self, gatherer: &G) -> MortonMap<G::Sum, u64>
    where
        G: Gatherer<T, u64>,
    {
        let mut map = MortonMap::default();
        let base_region: MortonRegion<u64> = MortonRegion {
            morton: Morton(0),
            level: 0,
        };
        let mut nodes = Vec::new();
        match self {
            MortonOctree::Node(box ref n) => {
                map.insert(
                    base_region,
                    gatherer.gather(n.iter().flat_map(|c| c.iter())),
                );

                nodes.push((n, base_region.enter(0)));
            }
            MortonOctree::Leaf(ref items, morton) => {
                map.insert(
                    base_region,
                    gatherer.gather(items.iter().map(|i| (*morton, i))),
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
                MortonOctree::Node(ref children) => {
                    map.insert(
                        region,
                        gatherer.gather(children.iter().flat_map(|c| c.iter())),
                    );

                    if region.level < NUM_BITS_PER_DIM_64 - 1 {
                        nodes.push((children, region.enter(0)));
                    }
                }
                MortonOctree::Leaf(ref items, morton) => {
                    map.insert(region, gatherer.gather(items.iter().map(|i| (morton, i))));
                }
                _ => {}
            }
        }

        map
    }

    #[inline]
    fn empty_node() -> Self {
        use self::MortonOctree::*;
        Node(box [None, None, None, None, None, None, None, None])
    }
}

impl<T> MortonOctree<T, u128> {
    fn iter(&self) -> impl Iterator<Item = (Morton<u128>, &T)> {
        use either::Either::*;
        match self {
            MortonOctree::Node(box ref n) => Left(MortonOctreeIter::new(vec![(n, 0)])),
            MortonOctree::Leaf(ref item, morton) => {
                Right(item.iter().map(move |item| (*morton, item)))
            }
            MortonOctree::None => Left(MortonOctreeIter::new(vec![])),
        }
    }

    fn iter_gather<'a, F, G>(
        &'a self,
        mut further: F,
        gatherer: G,
    ) -> impl Iterator<Item = (MortonRegion<u128>, G::Sum)> + 'a
    where
        F: FnMut(MortonRegion<u128>) -> bool + 'a,
        G: Gatherer<T, u128> + 'a,
    {
        use either::Either::*;
        let base_region = MortonRegion {
            morton: Morton(0),
            level: 0,
        };
        match self {
            MortonOctree::Node(box ref n) => {
                if further(base_region) {
                    Left(MortonOctreeFurtherGatherIter::new(
                        vec![(n, base_region.enter(0))],
                        further,
                        gatherer,
                    ))
                } else {
                    Right(std::iter::once((
                        base_region,
                        gatherer.gather(n.iter().flat_map(|c| c.iter())),
                    )))
                }
            }
            MortonOctree::Leaf(ref items, morton) => Right(std::iter::once((
                base_region,
                gatherer.gather(items.iter().map(|i| (*morton, i))),
            ))),
            MortonOctree::None => Left(MortonOctreeFurtherGatherIter::new(
                vec![],
                further,
                gatherer,
            )),
        }
    }

    fn iter_gather_cached<'a, F, G>(
        &'a self,
        mut further: F,
        gatherer: G,
        cache: &'a mut MortonMap<G::Sum, u128>,
    ) -> CacheGatherIter<'a, T, u128, F, G>
    where
        F: FnMut(MortonRegion<u128>) -> bool + 'a,
        G: Gatherer<T, u128> + 'a,
    {
        use either::Either::*;
        let base_region = MortonRegion {
            morton: Morton(0),
            level: 0,
        };
        match self {
            MortonOctree::Node(box ref n) => {
                if further(base_region) {
                    Left(MortonOctreeFurtherGatherCacheIter::new(
                        vec![(n, base_region.enter(0))],
                        further,
                        gatherer,
                        cache,
                    ))
                } else {
                    Right(std::iter::once((
                        base_region,
                        cache
                            .entry(base_region)
                            .or_insert_with(|| gatherer.gather(n.iter().flat_map(|c| c.iter()))),
                    )))
                }
            }
            MortonOctree::Leaf(ref items, morton) => Right(std::iter::once((
                base_region,
                cache
                    .entry(base_region)
                    .or_insert_with(|| gatherer.gather(items.iter().map(|i| (*morton, i)))),
            ))),
            MortonOctree::None => Left(MortonOctreeFurtherGatherCacheIter::new(
                vec![],
                further,
                gatherer,
                cache,
            )),
        }
    }

    /// This gathers the tree into a linear hashed octree map.
    pub fn iter_gather_deep_linear_hashed<G>(&self, gatherer: &G) -> MortonMap<G::Sum, u128>
    where
        G: Gatherer<T, u128>,
    {
        let mut map = MortonMap::default();
        let base_region: MortonRegion<u128> = MortonRegion {
            morton: Morton(0),
            level: 0,
        };
        let mut nodes = Vec::new();
        match self {
            MortonOctree::Node(box ref n) => {
                map.insert(
                    base_region,
                    gatherer.gather(n.iter().flat_map(|c| c.iter())),
                );

                nodes.push((n, base_region.enter(0)));
            }
            MortonOctree::Leaf(ref items, morton) => {
                map.insert(
                    base_region,
                    gatherer.gather(items.iter().map(|i| (*morton, i))),
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
                MortonOctree::Node(ref children) => {
                    map.insert(
                        region,
                        gatherer.gather(children.iter().flat_map(|c| c.iter())),
                    );

                    if region.level < NUM_BITS_PER_DIM_128 - 1 {
                        nodes.push((children, region.enter(0)));
                    }
                }
                MortonOctree::Leaf(ref items, morton) => {
                    map.insert(region, gatherer.gather(items.iter().map(|i| (morton, i))));
                }
                _ => {}
            }
        }

        map
    }

    #[inline]
    fn empty_node() -> Self {
        use self::MortonOctree::*;
        Node(box [None, None, None, None, None, None, None, None])
    }
}

impl<T, N> Default for MortonOctree<T, N> {
    fn default() -> Self {
        MortonOctree::None
    }
}

struct MortonOctreeIter<'a, T, N> {
    nodes: Vec<(&'a [MortonOctree<T, N>; 8], usize)>,
    vec_iter: std::slice::Iter<'a, T>,
    vec_morton: Morton<N>,
}

impl<'a, T, N> MortonOctreeIter<'a, T, N>
where
    N: Default,
{
    fn new(nodes: Vec<(&'a [MortonOctree<T, N>; 8], usize)>) -> Self {
        MortonOctreeIter {
            nodes,
            vec_iter: [].iter(),
            vec_morton: Morton::default(),
        }
    }
}

impl<'a, T> Iterator for MortonOctreeIter<'a, T, u64> {
    type Item = (Morton<u64>, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.vec_iter
            .next()
            .map(|item| (self.vec_morton, item))
            .or_else(|| {
                while let Some((node, ix)) = self.nodes.pop() {
                    if ix != 7 {
                        self.nodes.push((node, ix + 1));
                    }
                    match node[ix] {
                        MortonOctree::Node(ref children) => self.nodes.push((children, 0)),
                        MortonOctree::Leaf(ref item, morton) => {
                            self.vec_iter = item.iter();
                            self.vec_morton = morton;
                            // This wont work if there is ever an empty vec.
                            return self.vec_iter.next().map(|item| (self.vec_morton, item));
                        }
                        _ => {}
                    }
                }
                None
            })
    }
}

impl<'a, T> Iterator for MortonOctreeIter<'a, T, u128> {
    type Item = (Morton<u128>, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.vec_iter
            .next()
            .map(|item| (self.vec_morton, item))
            .or_else(|| {
                while let Some((node, ix)) = self.nodes.pop() {
                    if ix != 7 {
                        self.nodes.push((node, ix + 1));
                    }
                    match node[ix] {
                        MortonOctree::Node(ref children) => self.nodes.push((children, 0)),
                        MortonOctree::Leaf(ref item, morton) => {
                            self.vec_iter = item.iter();
                            self.vec_morton = morton;
                            // This wont work if there is ever an empty vec.
                            return self.vec_iter.next().map(|item| (self.vec_morton, item));
                        }
                        _ => {}
                    }
                }
                None
            })
    }
}

type MortonOctreeFurtherGatherIterNodeStack<'a, T, N> =
    Vec<(&'a [MortonOctree<T, N>; 8], MortonRegion<N>)>;

struct MortonOctreeFurtherGatherIter<'a, T, N, F, G> {
    nodes: MortonOctreeFurtherGatherIterNodeStack<'a, T, N>,
    further: F,
    gatherer: G,
}

impl<'a, T, N, F, G> MortonOctreeFurtherGatherIter<'a, T, N, F, G> {
    fn new(
        nodes: MortonOctreeFurtherGatherIterNodeStack<'a, T, N>,
        further: F,
        gatherer: G,
    ) -> Self {
        MortonOctreeFurtherGatherIter {
            nodes,
            further,
            gatherer,
        }
    }
}

impl<'a, T, F, G> Iterator for MortonOctreeFurtherGatherIter<'a, T, u64, F, G>
where
    F: FnMut(MortonRegion<u64>) -> bool,
    G: Gatherer<T, u64>,
{
    type Item = (MortonRegion<u64>, G::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, region)) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push((node, next));
            }

            match node[region.get()] {
                MortonOctree::Node(ref children) => {
                    if (self.further)(region) {
                        self.nodes.push((children, region.enter(0)));
                    } else {
                        return Some((
                            region,
                            self.gatherer.gather(children.iter().flat_map(|c| c.iter())),
                        ));
                    }
                }
                MortonOctree::Leaf(ref items, morton) => {
                    return Some((
                        region,
                        self.gatherer.gather(items.iter().map(|i| (morton, i))),
                    ));
                }
                _ => {}
            }
        }
        None
    }
}

impl<'a, T, F, G> Iterator for MortonOctreeFurtherGatherIter<'a, T, u128, F, G>
where
    F: FnMut(MortonRegion<u128>) -> bool,
    G: Gatherer<T, u128>,
{
    type Item = (MortonRegion<u128>, G::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, region)) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push((node, next));
            }

            match node[region.get()] {
                MortonOctree::Node(ref children) => {
                    if (self.further)(region) {
                        self.nodes.push((children, region.enter(0)));
                    } else {
                        return Some((
                            region,
                            self.gatherer.gather(children.iter().flat_map(|c| c.iter())),
                        ));
                    }
                }
                MortonOctree::Leaf(ref items, morton) => {
                    return Some((
                        region,
                        self.gatherer.gather(items.iter().map(|i| (morton, i))),
                    ));
                }
                _ => {}
            }
        }
        None
    }
}

type MortonOctreeFurtherGatherCacheIterNodeStack<'a, T, N> =
    Vec<(&'a [MortonOctree<T, N>; 8], MortonRegion<N>)>;

pub struct MortonOctreeFurtherGatherCacheIter<'a, T, N, F, G>
where
    G: Gatherer<T, N>,
{
    nodes: MortonOctreeFurtherGatherCacheIterNodeStack<'a, T, N>,
    further: F,
    gatherer: G,
    cache: &'a mut MortonMap<G::Sum, N>,
}

impl<'a, T, N, F, G> MortonOctreeFurtherGatherCacheIter<'a, T, N, F, G>
where
    G: Gatherer<T, N>,
{
    fn new(
        nodes: MortonOctreeFurtherGatherCacheIterNodeStack<'a, T, N>,
        further: F,
        gatherer: G,
        cache: &'a mut MortonMap<G::Sum, N>,
    ) -> Self {
        MortonOctreeFurtherGatherCacheIter {
            nodes,
            further,
            gatherer,
            cache,
        }
    }
}

/// The reason `Iterator` is only implemented for `&'a mut` is because the returned reference cannot live beyond
/// the next call to `next()`. If it does, then the cache the iterator is borrowing could potentially reallocate
/// and cause a use-after-free. Some unsafe code exists in this implementation that assumes this soundness.
/// This implementation cannot be written soundly in a way that allows multiple items from `next()` to
/// exist simultaneously without changing the method of caching.
impl<'a, T, F, G> Iterator for &'a mut MortonOctreeFurtherGatherCacheIter<'a, T, u64, F, G>
where
    F: FnMut(MortonRegion<u64>) -> bool,
    G: Gatherer<T, u64>,
{
    type Item = (MortonRegion<u64>, &'a mut G::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, region)) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push((node, next));
            }

            // NOTE: This unsafe code is thought to be sound. By making the lifetime of the returned
            // G::Sum tied to the iterator, it is thought that the cache it is in cannot be mutated and
            // cause a potential data race.
            let cache: &'a mut MortonMap<G::Sum, u64> = unsafe { &mut *(self.cache as *mut _) };

            match node[region.get()] {
                MortonOctree::Node(ref children) => {
                    if (self.further)(region) {
                        self.nodes.push((children, region.enter(0)));
                    } else {
                        return Some((
                            region,
                            cache.entry(region).or_insert_with(|| {
                                self.gatherer.gather(children.iter().flat_map(|c| c.iter()))
                            }),
                        ));
                    }
                }
                MortonOctree::Leaf(ref items, morton) => {
                    return Some((
                        region,
                        cache.entry(region).or_insert_with(|| {
                            self.gatherer.gather(items.iter().map(|i| (morton, i)))
                        }),
                    ));
                }
                _ => {}
            }
        }
        None
    }
}

/// The reason `Iterator` is only implemented for `&'a mut` is because the returned reference cannot live beyond
/// the next call to `next()`. If it does, then the cache the iterator is borrowing could potentially reallocate
/// and cause a use-after-free. Some unsafe code exists in this implementation that assumes this soundness.
/// This implementation cannot be written soundly in a way that allows multiple items from `next()` to
/// exist simultaneously without changing the method of caching.
impl<'a, T, F, G> Iterator for &'a mut MortonOctreeFurtherGatherCacheIter<'a, T, u128, F, G>
where
    F: FnMut(MortonRegion<u128>) -> bool,
    G: Gatherer<T, u128>,
{
    type Item = (MortonRegion<u128>, &'a mut G::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, region)) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push((node, next));
            }

            // NOTE: This unsafe code is thought to be sound. By making the lifetime of the returned
            // G::Sum tied to the iterator, it is thought that the cache it is in cannot be mutated and
            // cause a potential data race.
            let cache: &'a mut MortonMap<G::Sum, u128> = unsafe { &mut *(self.cache as *mut _) };

            match node[region.get()] {
                MortonOctree::Node(ref children) => {
                    if (self.further)(region) {
                        self.nodes.push((children, region.enter(0)));
                    } else {
                        return Some((
                            region,
                            cache.entry(region).or_insert_with(|| {
                                self.gatherer.gather(children.iter().flat_map(|c| c.iter()))
                            }),
                        ));
                    }
                }
                MortonOctree::Leaf(ref items, morton) => {
                    return Some((
                        region,
                        cache.entry(region).or_insert_with(|| {
                            self.gatherer.gather(items.iter().map(|i| (morton, i)))
                        }),
                    ));
                }
                _ => {}
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

        let mut octree = Plain::<_, u128>::new(0);
        octree.extend(
            izip!(
                xrng.sample_iter(&Open01),
                yrng.sample_iter(&Open01),
                zrng.sample_iter(&Open01)
            )
            .take(5000)
            .map(|(x, y, z)| (Vector3::<f64>::new(x, y, z), 0)),
        );

        assert_eq!(octree.iter().count(), 5000);
    }
}
