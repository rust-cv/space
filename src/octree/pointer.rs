use crate::octree::morton::*;
use crate::octree::*;
use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

use itertools::Itertools;

use std::default::Default;

#[allow(type_alias_bounds)]
type CacheGatherIter<'a, T, M, F, G: Gatherer<T, M>> = either::Either<
    MortonOctreeFurtherGatherCacheIter<'a, T, M, F, G>,
    std::iter::Once<(MortonRegion<M>, &'a mut G::Sum)>,
>;

/// An octree that starts with a cube from [-1, 1] in each dimension and will only expand.
pub struct Pointer<T, M> {
    tree: MortonOctree<T, M>,
    /// Dimensions of the top level node are from [-2**level, 2**level].
    level: i32,
}

impl<T, M> Pointer<T, M>
where
    M: Morton,
{
    /// Dimensions of the top level node are fixed in the range [-2**level, 2**level].
    pub fn new(level: i32) -> Self {
        Pointer {
            tree: MortonOctree::default(),
            level,
        }
    }

    pub fn morton<S>(&self, point: &Vector3<S>) -> M
    where
        S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
        M: std::fmt::Debug + 'static,
    {
        let bound = (S::one() + S::one()).powi(self.level);
        if point.iter().any(|n| n.abs() > bound) {
            panic!("space::Octree::morton(): tried to compute a Morton outside the Octree bounds");
        }

        // Convert the point into normalized space.
        MortonWrapper::from(point.map(|n| (n + bound) / (S::one() + S::one()).powi(self.level + 1)))
            .0
    }

    /// Insert an item with a point and return the existing item if they would both occupy the same space.
    pub fn insert(&mut self, morton: M, item: T) {
        // Traverse the tree down to the node we need to operate on.
        let (tree_part, level) = (0..M::dim_bits())
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

        let mut dest_old = MortonOctree::empty_node();
        std::mem::swap(&mut dest_old, tree_part);

        if let MortonOctree::Leaf(dest_vec, dest_morton) = dest_old {
            // Set our initial reference to the default node in the dest.
            let mut building_node = tree_part;
            // Create deeper nodes till they differ at some level.
            for i in level + 1..M::dim_bits() {
                // We know for sure that the dest is a node.
                if let MortonOctree::Node(box ref mut children) = building_node {
                    if morton.get_level(i) == dest_morton.get_level(i) {
                        children[morton.get_level(i)] = MortonOctree::empty_node();
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

    pub fn iter(&self) -> impl Iterator<Item = (M, &T)> {
        self.tree.iter()
    }

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

    pub fn iter_gather_cached<'a, F, G>(
        &'a self,
        further: F,
        gatherer: G,
        cache: &'a mut MortonRegionMap<G::Sum, M>,
    ) -> CacheGatherIter<'a, T, M, F, G>
    where
        F: FnMut(MortonRegion<M>) -> bool + 'a,
        G: Gatherer<T, M> + 'a,
    {
        self.tree.iter_gather_cached(further, gatherer, cache)
    }

    /// This gathers the tree into a linear hashed octree map.
    pub fn iter_gather_deep_linear_hashed<G>(&self, gatherer: &G) -> MortonRegionMap<G::Sum, M>
    where
        G: Gatherer<T, M>,
    {
        self.tree.iter_gather_deep_linear_hashed(gatherer)
    }

    /// This gathers the octree in a tree fold by gathering leaves with `gatherer` and folding with `folder`.
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

impl<T, S, M> Extend<(Vector3<S>, T)> for Pointer<T, M>
where
    M: Morton + std::fmt::Debug + 'static,
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

impl<'a, T, M, S> Extend<(&'a Vector3<S>, T)> for Pointer<T, M>
where
    M: Morton + std::fmt::Debug + 'static,
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
enum MortonOctree<T, M> {
    Node(Box<[MortonOctree<T, M>; 8]>),
    Leaf(Vec<T>, M),
    None,
}

impl<T, M> MortonOctree<T, M>
where
    M: Morton,
{
    fn iter(&self) -> impl Iterator<Item = (M, &T)> {
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
    ) -> impl Iterator<Item = (MortonRegion<M>, G::Sum)> + 'a
    where
        F: FnMut(MortonRegion<M>) -> bool + 'a,
        G: Gatherer<T, M> + 'a,
    {
        use either::Either::*;
        let base_region = MortonRegion::default();
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
        cache: &'a mut MortonRegionMap<G::Sum, M>,
    ) -> CacheGatherIter<'a, T, M, F, G>
    where
        F: FnMut(MortonRegion<M>) -> bool + 'a,
        G: Gatherer<T, M> + 'a,
    {
        use either::Either::*;
        let base_region = MortonRegion::default();
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
    pub fn iter_gather_deep_linear_hashed<G>(&self, gatherer: &G) -> MortonRegionMap<G::Sum, M>
    where
        G: Gatherer<T, M>,
    {
        let mut map = MortonRegionMap::default();
        let base_region: MortonRegion<M> = MortonRegion::default();
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

                    if region.level < M::dim_bits() - 1 {
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

    /// This gathers the tree into a linear hashed octree map in a tree fold.
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
            MortonOctree::Node(box ref n) => {
                if region.level < M::dim_bits() {
                    if let Some(sum) = folder.sum((0..8).filter_map(|i| {
                        n[i].iter_gather_deep_linear_hashed_tree_fold(
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
            MortonOctree::Leaf(ref items, morton) => {
                let sum = gatherer.gather(items.iter().map(|i| (*morton, i)));
                map.insert(region, sum.clone());
                Some(sum)
            }
            _ => None,
        }
    }

    #[inline]
    fn empty_node() -> Self {
        use self::MortonOctree::*;
        Node(box [None, None, None, None, None, None, None, None])
    }
}

impl<T, M> Default for MortonOctree<T, M> {
    fn default() -> Self {
        MortonOctree::None
    }
}

struct MortonOctreeIter<'a, T, M> {
    nodes: Vec<(&'a [MortonOctree<T, M>; 8], usize)>,
    vec_iter: std::slice::Iter<'a, T>,
    vec_morton: M,
}

impl<'a, T, M> MortonOctreeIter<'a, T, M>
where
    M: Morton,
{
    fn new(nodes: Vec<(&'a [MortonOctree<T, M>; 8], usize)>) -> Self {
        MortonOctreeIter {
            nodes,
            vec_iter: [].iter(),
            vec_morton: M::zero(),
        }
    }
}

impl<'a, T, M> Iterator for MortonOctreeIter<'a, T, M>
where
    M: Morton,
{
    type Item = (M, &'a T);

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

type MortonOctreeFurtherGatherIterNodeStack<'a, T, M> =
    Vec<(&'a [MortonOctree<T, M>; 8], MortonRegion<M>)>;

struct MortonOctreeFurtherGatherIter<'a, T, M, F, G> {
    nodes: MortonOctreeFurtherGatherIterNodeStack<'a, T, M>,
    further: F,
    gatherer: G,
}

impl<'a, T, M, F, G> MortonOctreeFurtherGatherIter<'a, T, M, F, G> {
    fn new(
        nodes: MortonOctreeFurtherGatherIterNodeStack<'a, T, M>,
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

impl<'a, T, M, F, G> Iterator for MortonOctreeFurtherGatherIter<'a, T, M, F, G>
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

type MortonOctreeFurtherGatherCacheIterNodeStack<'a, T, M> =
    Vec<(&'a [MortonOctree<T, M>; 8], MortonRegion<M>)>;

pub struct MortonOctreeFurtherGatherCacheIter<'a, T, M, F, G>
where
    G: Gatherer<T, M>,
{
    nodes: MortonOctreeFurtherGatherCacheIterNodeStack<'a, T, M>,
    further: F,
    gatherer: G,
    cache: &'a mut MortonRegionMap<G::Sum, M>,
}

impl<'a, T, M, F, G> MortonOctreeFurtherGatherCacheIter<'a, T, M, F, G>
where
    G: Gatherer<T, M>,
{
    fn new(
        nodes: MortonOctreeFurtherGatherCacheIterNodeStack<'a, T, M>,
        further: F,
        gatherer: G,
        cache: &'a mut MortonRegionMap<G::Sum, M>,
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
impl<'a, T, M, F, G> Iterator for &'a mut MortonOctreeFurtherGatherCacheIter<'a, T, M, F, G>
where
    M: Morton,
    F: FnMut(MortonRegion<M>) -> bool,
    G: Gatherer<T, M>,
{
    type Item = (MortonRegion<M>, &'a mut G::Sum);

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
            let cache: &'a mut MortonRegionMap<G::Sum, M> = unsafe { &mut *(self.cache as *mut _) };

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

        let mut octree = Pointer::<_, u128>::new(0);
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
