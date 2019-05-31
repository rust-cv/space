use crate::*;

use itertools::Itertools;

use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use std::default::Default;

use num_traits::{Float, FromPrimitive, ToPrimitive};

use nalgebra::Vector3;

use std::ops::{Deref, DerefMut};

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
pub struct PointerOctree<T, M> {
    tree: Internal<T, M>,
    count: usize,
}

/// A pointer octree with the capability of resizing.
pub struct ResizingPointerOctree<T, M> {
    pub octree: PointerOctree<T, M>,
    pub region: LeveledRegion,
}

impl<T, M> Default for PointerOctree<T, M> {
    /// Create an empty octree.
    /// ```
    /// use space::PointerOctree;
    /// let mut tree = PointerOctree::<String, u64>::default();
    ///
    /// ```
    fn default() -> Self {
        PointerOctree {
            tree: Internal::default(),
            count: 0,
        }
    }
}

impl<T, M> PointerOctree<T, M>
where
    M: Morton,
{
    /// Create an empty octree. Calls Default impl.
    ///
    /// ```
    /// use space::PointerOctree;
    /// let mut tree = PointerOctree::<String, u64>::new();
    ///
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetches an immutable reference to the value of a specific coordinate in the octree
    /// ```
    /// use space::{PointerOctree, Morton};
    /// use nalgebra::Vector3;
    ///
    /// let mut tree = PointerOctree::<String, u64>::new();
    ///
    /// let fetched_value = tree.get(Morton::encode(Vector3::<u64>::new(1, 2, 3)));
    /// assert!(fetched_value.is_none());
    /// ```
    pub fn get(&self, morton: M) -> Option<&T> {
        // Traverse the tree down to the node we need to operate on.
        let (tree_part, _) = (0..M::dim_bits())
            .fold_while((&self.tree, 0), |(node, old_ix), i| {
                use itertools::FoldWhile::{Continue, Done};
                match node {
                    Internal::Node(box Oct { ref children }) => {
                        // The index into the array to access the next octree node
                        let subindex = morton.get_level(i);
                        Continue((&children[subindex], i))
                    }
                    Internal::Leaf(_, _) => Done((node, old_ix)),
                    Internal::None => Done((node, old_ix)),
                }
            })
            .into_inner();

        match tree_part {
            Internal::Leaf(ref leaf_item, dest_morton) => {
                // If they have the same code then replace it.
                if morton == *dest_morton {
                    Some(leaf_item)
                } else {
                    None
                }
                // Otherwise we must split them, which we must do outside of this scope due to the borrow.
            }
            Internal::None => None,
            _ => {
                unreachable!(
                    "space::PointerOctree::get(): can only get None or Leaf in this code area"
                );
            }
        }
    }

    /// Fetches a mututable reference to the value of a specific coordinate in the octree
    ///
    /// ```
    /// use space::{PointerOctree, Morton};
    /// use nalgebra::Vector3;
    ///
    /// let mut tree = PointerOctree::<String, u64>::new();
    ///
    /// let fetched_value = tree.get(Morton::encode(Vector3::<u64>::new(1, 2, 3)));
    /// assert!(fetched_value.is_none());
    /// ```
    pub fn get_mut(&mut self, morton: M) -> Option<&T> {
        // Traverse the tree down to the node we need to operate on.
        let (tree_part, _) = (0..M::dim_bits())
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
                    Some(leaf_item)
                } else {
                    None
                }
                // Otherwise we must split them, which we must do outside of this scope due to the borrow.
            }
            Internal::None => None,
            _ => {
                unreachable!(
                    "space::PointerOctree::get_mut(): can only get None or Leaf in this code area"
                );
            }
        }
    }

    /// Insert an item with a point and replace the existing item if they would both occupy the same space.
    ///
    /// ```
    /// use space::{PointerOctree, Morton};
    /// use nalgebra::Vector3;
    ///
    /// let mut tree = PointerOctree::<String, u64>::new();
    /// tree.insert(Morton::encode(Vector3::new(1, 2, 3)), "test1".to_string() );
    ///
    /// ```
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
                    // Don't increase the count here because we replaced only.
                    return;
                }
                // Otherwise we must split them, which we must do outside of this scope due to the borrow.
            }
            Internal::None => {
                // Simply add a new leaf.
                *tree_part = Internal::Leaf(item, morton);
                self.count += 1;
                return;
            }
            _ => {
                unreachable!(
                    "space::PointerOctree::insert(): can only get None or Leaf in this code area"
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
                        self.count += 1;
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

    /// Fetches a mututable reference to the value of a specific coordinate in the octree
    ///
    /// ```
    /// use space::{PointerOctree, Morton};
    /// use nalgebra::Vector3;
    ///
    /// let mut tree = PointerOctree::<String, u64>::new();
    /// let m = Morton::encode(Vector3::<u64>::new(1, 2, 3));
    ///
    /// let fetched_value = tree.remove(m);
    /// assert!(fetched_value.is_none());
    ///
    /// tree.insert(m, "hello".to_owned());
    ///
    /// let fetched_value = tree.remove(m);
    /// assert!(fetched_value == Some("hello".to_owned()));
    /// ```
    pub fn remove(&mut self, morton: M) -> Option<T> {
        // Traverse the tree down to the node we need to operate on.
        let (tree_part, _) = (0..M::dim_bits())
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

        let mut leaf = Internal::None;
        std::mem::swap(&mut leaf, tree_part);

        match leaf {
            Internal::Leaf(leaf_item, _) => Some(leaf_item),
            Internal::None => None,
            _ => {
                unreachable!(
                    "space::Octree::PointerOctree(): can only get None or Leaf in this code area"
                );
            }
        }
    }

    /// Iterate over all octree nodes and their morton codes.
    ///
    /// ```
    /// use space::{PointerOctree, Morton};
    /// use nalgebra::Vector3;
    ///
    /// let mut tree = PointerOctree::<String, u64>::new();
    /// let mut i = 0;
    /// for _ in tree.iter() {
    ///     i += 1;
    /// }
    /// assert!(i == 0);
    ///
    /// let m1 = Morton::encode(Vector3::<u64>::new(1, 2, 3));
    /// let m2 = Morton::encode(Vector3::<u64>::new(4, 5, 6));
    /// tree.insert(m1, "m1".to_owned());
    /// tree.insert(m2, "m2".to_owned());
    ///
    /// let mut i = 0;
    /// let mut items: Vec<String> = vec![];
    /// for (morton, item) in &tree.iter() {
    ///     i += 1;
    ///     items.push(item);
    /// }
    /// assert!(i == 2);
    /// assert!(items.contains("m1"));
    /// assert!(items.contains("m2"));
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (M, &T)> {
        self.tree.iter()
    }

    // pub fn iter_mut(&mut self) -> impl Iterator<Item = (M, T)> {
    //     self.tree.iter_mut();
    // }

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
        self.tree.iter_rand(depth, rng)
    }

    /// Iterates over the octree and, for every internal node in the tree, runs `explore` to check if it should
    /// continue down to the leaves or stop at this node. If it stops at an internal node, it passes each leaf
    /// that descends from that internal node to `folder.gather()` and then calls `folder.fold()` on every child
    /// internal node until it has propogated all the information up to the node we stopped on. If it reaches a
    /// leaf node, it passes an iterator over just that one leaf to `folder.gather()`. This allows an operation
    /// to be called on every region in the tree using `explore` to limit the traversal from iterating over the
    /// whole tree.
    ///
    /// Note that whenever a region changes it should invalidate all parent nodes and all child nodes in the cache.
    /// See `morton_levels` for how to generate the levels of a morton.
    ///
    /// If you want to ensure your cache can hold all results, it needs to have `len * 8 / 7` capacity.
    pub fn iter_fold<'a, F>(
        &'a self,
        folder: F,
        cache: MortonRegionCache<F::Sum, M>,
    ) -> FoldIter<'a, T, M, impl FnMut(MortonRegion<M>) -> bool + 'a, F, rand::ThreadRng>
    where
        F: Folder<T, M> + 'a,
        F::Sum: Clone,
        Standard: Distribution<M>,
    {
        // This uses `dim_bits` to avoid ever needing to use the rng (we cant go lower than that).
        self.tree.iter_fold_random(
            MortonRegion::base(),
            M::dim_bits(),
            |_| true,
            folder,
            rand::thread_rng(),
            cache,
        )
    }

    /// This is a variant of `iter_fold` that takes a `depth` to sample at and will always randomly sample
    /// once starting at that depth. This improves performance by avoiding calling `gather` and `fold` more than
    /// a finite number of times. For many tasks, choosing a depth of `2` or `64` samples is performant and sufficient.
    ///
    /// This will generate one morton per sample and is not perfectly randomly distributed since if it lands on an
    /// empty region, it will move in z-order to the next region to sample from (in a toroidal fashion) and thus is
    /// biased towards regions that come after more empty regions toroidally in z-order.
    pub fn iter_fold_random<'a, E, F, R>(
        &'a self,
        depth: usize,
        explore: E,
        folder: F,
        rng: R,
        cache: MortonRegionCache<F::Sum, M>,
    ) -> FoldIter<'a, T, M, E, F, R>
    where
        R: Rng + 'a,
        E: FnMut(MortonRegion<M>) -> bool + 'a,
        F: Folder<T, M> + 'a,
        F::Sum: Clone,
        Standard: Distribution<M>,
    {
        self.tree
            .iter_fold_random(MortonRegion::base(), depth, explore, folder, rng, cache)
    }

    /// Iterates over the octree and, for every internal node in the tree, runs `explore` to check if it should
    /// continue down to the leaves or stop at this node. If it stops at an internal node, it gets the first leaf
    /// node in z-order from that node and includes it in the iterator.
    pub fn iter_explore_simple<'a, E>(
        &'a self,
        explore: E,
    ) -> SimpleExploreIter<'a, T, M, impl FnMut(MortonRegion<M>) -> bool + 'a>
    where
        E: FnMut(MortonRegion<M>) -> bool + 'a,
    {
        // This uses `dim_bits` to avoid ever needing to use the rng (we cant go lower than that).
        self.tree.iter_explore_simple(MortonRegion::base(), explore)
    }

    /// This gathers the tree into a linear hashed octree map. This map contains every internal and leaf node
    /// as the sum type that the `folder` produces.
    pub fn collect_fold<E, F>(&self, folder: &F) -> E
    where
        F: Folder<T, M>,
        F::Sum: Clone,
        E: Extend<(MortonRegion<M>, F::Sum)> + Default,
    {
        let mut map = E::default();
        self.tree
            .collect_fold(MortonRegion::base(), folder, &mut map);
        map
    }

    /// Returns the number of leaves in the tree.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Checks if the octree is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl<T, M> IntoIterator for PointerOctree<T, M>
where
    M: Morton,
{
    type Item = (M, T);
    type IntoIter = InternalIntoIter<T, M>;

    fn into_iter(self) -> Self::IntoIter {
        self.tree.into_iter()
    }
}

impl<T, M> Extend<(M, T)> for PointerOctree<T, M>
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

#[allow(dead_code)]
/// Associates an octree with a LeveledRegion in order to enable resizing.
impl<T, M> ResizingPointerOctree<T, M>
where
    M: Morton,
{
    /// Create an empty resizing octree. Calls Default impl.
    /// The LeveledRegion parameter is initially set to zero.
    ///
    /// ```
    /// use space::ResizingPointerOctree;
    /// let mut tree = ResizingPointerOctree::<String, u64>::new();
    ///
    /// ```
    pub fn new() -> Self {
        Self {
            octree: PointerOctree::default(),
            region: LeveledRegion(0),
        }
    }

    /// Computes the octant the octree should expand in to add the given point.
    /// Note: performs only one iteration (does not recursively expand) and
    /// does not modify the actual ResizingPointerOctree in any way.
    ///
    ///
    /// In other words, the returned value is the octant the OLD coordinates should
    /// be place in.
    ///
    /// ```
    /// use space::ResizingPointerOctree;
    /// let mut tree = ResizingPointerOctree::<String, u64>::new();
    /// tree.insert(Morton::encode(Vector3::new(0, 0, 0)), String::from("test1"));
    ///
    /// let resize_loc = tree.resize_loc(Vector3::new(2, 2, 2));
    /// // This is outside all the bounds of the current octree, so the
    /// // point itself dictates the direction of expansion (i.e. 0b00000111)
    /// // Since the new point will be in octant 0b111, the returned octant will be 0b000.
    ///
    /// assert!(resize_loc == 0);
    ///
    pub fn resize_loc<S>(&self, point: Vector3<S>) -> Option<u8>
    where
        S: Float
            + ToPrimitive
            + FromPrimitive
            + Ord
            + PartialOrd
            + From<f64>
            + std::fmt::Debug
            + 'static,
    {
        // Track differences (which side the points are on)
        // weighting[0] corresponds to x, and so on
        let mut weighting: [i32; 3] = [0, 0, 0];

        // Region bounds (cube)
        // Specified in three dimensions for possible future expansion
        // (varying bounds)
        let num_bound: S = (2.0.powi(self.region.0) as f64).into();
        let bound: [S; 3] = [num_bound, num_bound, num_bound];

        // -1 if new point too negative, 0 if within, 1 if too positive
        let new_octant: Vec<i32> = vec![0, 1, 2]
            .iter()
            .map(|i| match (point[*i] < -bound[*i], point[*i] >= bound[*i]) {
                (true, _) => -1,
                (_, true) => 1,
                _ => 0,
            })
            .collect();

        // If every dimension is within the current region, no need to reside
        if (new_octant[0] == 0) && (new_octant[1] == 0) && (new_octant[2] == 0) {
            None
        } else {
            // Compute weightings, adding 1 if the leading coordinate is 1
            // and subtracting 1 if it is 0
            for (point, _) in self.octree.iter() {
                let significant_bits = point.decode();

                for i in 0..3 {
                    if significant_bits[i] == M::zero() {
                        weighting[i] -= 1;
                    } else {
                        weighting[i] += 1;
                    }
                }
            }

            // If it's negative, make 1, and vice versa
            // (or set direction of expansion to direction of new point)
            for i in 0..3 {
                if new_octant[i] != 0 {
                    weighting[i] = 1 - (new_octant[i] + 1) / 2;
                } else if weighting[i] <= 0 {
                    weighting[i] = 1;
                } else {
                    weighting[i] = 0;
                }
            }

            // Encode octant that we would like to expand in
            Some(((weighting[0] << 2) | (weighting[1] << 1) | weighting[2]) as u8)
        }
    }

    /// Resize the octree as many times as needed to accomodate the given point
    /// (we assume that the octree is centered at 0)
    ///
    /// ```
    /// use space::ResizingPointerOctree;
    /// let mut tree = ResizingPointerOctree::<String, u64>::new();
    /// tree.insert(Morton::encode(Vector3::new(0, 0, 0)), String::from("test1"));
    /// tree.resize(Vector3::new(1.5, 1.5, 1.5));
    /// // Since the region initially covered is [-1, 1) cubically,
    /// // exactly one expansion needed, so that [-2, 2) is covered
    /// // and (1.5, 1.5, 1.5) may be inserted.
    ///
    /// assert!(tree.region.0 == 1);
    ///
    pub fn resize<S>(&mut self, point: Vector3<S>)
    where
        S: Float
            + ToPrimitive
            + FromPrimitive
            + Ord
            + PartialOrd
            + From<f64>
            + std::fmt::Debug
            + 'static,
    {
        match self.resize_loc(point) {
            None => {}
            Some(octant) => {
                let old_octree = std::mem::replace(&mut self.octree, PointerOctree::<T, M>::new());

                self.octree
                    .extend(old_octree.into_iter().map(|(morton, item)| {
                        // Add modified morton to new octree
                        let octant: M = M::from_u8(octant).unwrap();
                        let new_morton: M =
                            (morton >> 3) | (octant << (M::highest_bits().to_usize().unwrap() - 3));
                        (new_morton, item)
                    }));
                self.region.0 += 1;
                self.resize(point);
            }
        }
    }

    /// Inserts a vector into the given tree, resizing as many times as necessary.
    ///
    /// ```
    /// use space::ResizingPointerOctree;
    /// let mut tree = ResizingPointerOctree::<String, u64>::new();
    /// tree.insert(Morton::encode(Vector3::new(0, 0, 0)), String::from("test1"));
    ///
    /// // This should execute without error, with exactly one expansion needed.
    /// tree.insert_vector(Vector3::new(1.5, 1.5, 1.5), String::from("test2"));
    /// assert!(tree.region.0 == 1);
    ///
    pub fn insert_vector<S>(&mut self, point: Vector3<S>, item: T)
    where
        S: Float
            + ToPrimitive
            + FromPrimitive
            + Ord
            + PartialOrd
            + From<f64>
            + std::fmt::Debug
            + 'static,
    {
        match self.resize_loc(point) {
            None => self
                .octree
                .insert(self.region.discretize(point).unwrap(), item),
            Some(_) => {
                self.resize(point);
                self.insert_vector(point, item);
            }
        }
    }
}

impl<T, M> Deref for ResizingPointerOctree<T, M> {
    type Target = PointerOctree<T, M>;

    fn deref(&self) -> &Self::Target {
        &self.octree
    }
}

impl<T, M> DerefMut for ResizingPointerOctree<T, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.octree
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
    fn iter(&self) -> impl Iterator<Item = (M, &T)> {
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
    fn iter_rand<'a, R: Rng>(
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

    /// Get a single random leaf sample from this node (cant be none).
    fn sample(&self, morton: M) -> (M, &T) {
        match self {
            Internal::Node(box Oct { ref children }) => {
                let mut choice = morton.get_level(0);
                // Iterate until we find the first non-empty spot.
                // This technically results in not completely random behavior
                // since an octant that comes after more empty octants is more likely to be chosen.
                while let Internal::None = children[choice] {
                    choice += 1;
                    choice %= 8;
                }
                children[choice].sample(morton << 3)
            }
            Internal::Leaf(ref item, morton) => (*morton, item),
            Internal::None => unreachable!("can't sample a none node"),
        }
    }

    fn iter_fold_random<'a, E, F, R>(
        &'a self,
        region: MortonRegion<M>,
        depth: usize,
        explore: E,
        folder: F,
        rng: R,
        cache: MortonRegionCache<F::Sum, M>,
    ) -> FoldIter<'a, T, M, E, F, R>
    where
        R: Rng + 'a,
        E: FnMut(MortonRegion<M>) -> bool + 'a,
        F: Folder<T, M> + 'a,
        F::Sum: Clone,
        Standard: Distribution<M>,
    {
        FoldIter::new(self, region, explore, folder, depth, rng, cache)
    }

    pub fn iter_explore_simple<'a, E>(
        &'a self,
        region: MortonRegion<M>,
        explore: E,
    ) -> SimpleExploreIter<'a, T, M, impl FnMut(MortonRegion<M>) -> bool + 'a>
    where
        E: FnMut(MortonRegion<M>) -> bool + 'a,
    {
        SimpleExploreIter::new(self, region, explore)
    }

    fn collect_fold<E, F>(&self, region: MortonRegion<M>, folder: &F, map: &mut E) -> Option<F::Sum>
    where
        F: Folder<T, M>,
        F::Sum: Clone,
        E: Extend<(MortonRegion<M>, F::Sum)> + Default,
    {
        match self {
            Internal::Node(box Oct { ref children }) => {
                if region.level < M::dim_bits() {
                    let sum = folder
                        .fold((0..8).filter_map(|i| {
                            children[i].collect_fold(region.enter(i), folder, map)
                        }));
                    map.extend(std::iter::once((region, sum.clone())));
                    Some(sum)
                } else {
                    panic!("collect_fold(): if we get here, then we let a leaf descend pass morton range");
                }
            }
            Internal::Leaf(ref item, morton) => {
                let sum = folder.gather(*morton, item);
                map.extend(std::iter::once((region, sum.clone())));
                Some(sum)
            }
            _ => None,
        }
    }

    fn fold_rand<F, R>(
        &self,
        region: MortonRegion<M>,
        depth: usize,
        folder: &F,
        cache: &mut MortonRegionCache<F::Sum, M>,
        rng: &mut R,
    ) -> Option<F::Sum>
    where
        F: Folder<T, M>,
        F::Sum: Clone,
        R: Rng,
        Standard: Distribution<M>,
    {
        match self {
            Internal::Node(box Oct { ref children }) => {
                if let Some(sum) = cache.get_mut(&region).cloned() {
                    return Some(sum);
                }
                if depth == 0 {
                    let morton = rng.gen();
                    match self {
                        Internal::Node(box Oct { ref children }) => {
                            let mut choice = morton.get_level(0);
                            // Iterate until we find the first non-empty spot.
                            // This technically results in not completely random behavior
                            // since an octant that comes after more empty octants is more likely to be chosen.
                            while let Internal::None = children[choice] {
                                choice += 1;
                                choice %= 8;
                            }
                            let (morton, item) = children[choice].sample(morton << 3);
                            let sum = folder.gather(morton, item);
                            cache.insert(region, sum.clone());
                            Some(sum)
                        }
                        Internal::Leaf(ref item, morton) => {
                            let sum = folder.gather(*morton, item);
                            cache.insert(region, sum.clone());
                            Some(sum)
                        }
                        Internal::None => None,
                    }
                } else {
                    let sum = folder.fold(
                        children
                            .iter()
                            .enumerate()
                            .map(|(ix, child)| {
                                child.fold_rand(region.enter(ix), depth - 1, folder, cache, rng)
                            })
                            .filter_map(|c| c),
                    );
                    cache.insert(region, sum.clone());
                    Some(sum)
                }
            }
            Internal::Leaf(ref item, morton) => {
                let sum = cache.get_mut(&region).cloned().unwrap_or_else(|| {
                    let sum = folder.gather(*morton, item);
                    cache.insert(region, sum.clone());
                    sum
                });
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

impl<T, M> IntoIterator for Internal<T, M>
where
    M: Morton,
{
    type Item = (M, T);
    type IntoIter = InternalIntoIter<T, M>;

    fn into_iter(self) -> Self::IntoIter {
        InternalIntoIter::new(self)
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

pub struct InternalIntoIter<T, M> {
    nodes: Vec<Internal<T, M>>,
}

impl<T, M> InternalIntoIter<T, M>
where
    M: Morton,
{
    fn new(node: Internal<T, M>) -> Self {
        InternalIntoIter { nodes: vec![node] }
    }
}

impl<T, M> Iterator for InternalIntoIter<T, M>
where
    M: Morton,
{
    type Item = (M, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.nodes.pop() {
            match node {
                Internal::Node(box Oct {
                    children: [c0, c1, c2, c3, c4, c5, c6, c7],
                }) => {
                    self.nodes.push(c0);
                    self.nodes.push(c1);
                    self.nodes.push(c2);
                    self.nodes.push(c3);
                    self.nodes.push(c4);
                    self.nodes.push(c5);
                    self.nodes.push(c6);
                    self.nodes.push(c7);
                }
                Internal::Leaf(item, morton) => {
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

type FoldStack<'a, T, M> = Vec<(&'a Internal<T, M>, MortonRegion<M>)>;

pub struct FoldIter<'a, T, M, E, F, R>
where
    F: Folder<T, M>,
    R: Rng,
    M: Morton,
{
    nodes: FoldStack<'a, T, M>,
    explore: E,
    folder: F,
    depth: usize,
    rng: R,
    cache: MortonRegionCache<F::Sum, M>,
}

impl<'a, T, M, E, F, R> FoldIter<'a, T, M, E, F, R>
where
    F: Folder<T, M>,
    R: Rng,
    M: Morton,
{
    fn new(
        node: &'a Internal<T, M>,
        region: MortonRegion<M>,
        explore: E,
        folder: F,
        depth: usize,
        rng: R,
        cache: MortonRegionCache<F::Sum, M>,
    ) -> Self {
        FoldIter {
            nodes: vec![(node, region)],
            explore,
            folder,
            depth,
            rng,
            cache,
        }
    }
}

impl<'a, T, M, E, F, R> Iterator for FoldIter<'a, T, M, E, F, R>
where
    M: Morton,
    E: FnMut(MortonRegion<M>) -> bool,
    F: Folder<T, M>,
    F::Sum: Clone,
    R: Rng,
    Standard: Distribution<M>,
{
    type Item = (MortonRegion<M>, F::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, region)) = self.nodes.pop() {
            // If we shouldn't go further into the region, then its time to do a random sample starting here.
            if !(self.explore)(region) {
                trace!("chose not to go further");
                // If we reach the depth we want or `further` is false, then we must start the random sampling.
                if let Some(r) = self.cache.get_mut(&region).cloned().or_else(|| {
                    // We have to make sure this node is not None or else we can't gather it.
                    // This is because `gather` must be guaranteed that its not passed an empty iterator.
                    node.fold_rand(
                        region,
                        self.depth,
                        &self.folder,
                        &mut self.cache,
                        &mut self.rng,
                    )
                    .map(|item| {
                        self.cache.insert(region, item.clone());
                        item
                    })
                }) {
                    return Some((region, r));
                }
            } else {
                match node {
                    Internal::Node(box Oct { ref children }) => {
                        trace!("traversing deeper due to node at level {}", region.level);
                        // Traverse deeper (we already checked if we didn't need to go further).
                        for (ix, child) in children.iter().enumerate() {
                            self.nodes.push((child, region.enter(ix)));
                        }
                    }
                    Internal::Leaf(ref item, morton) => {
                        trace!("stopping due to leaf at level {}", region.level);
                        let item = self.cache.get_mut(&region).cloned().unwrap_or_else(|| {
                            let item = self.folder.gather(*morton, item);
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

impl<'a, T, M, E, F, R> Into<MortonRegionCache<F::Sum, M>> for FoldIter<'a, T, M, E, F, R>
where
    F: Folder<T, M>,
    R: Rng,
    M: Morton,
{
    fn into(self) -> MortonRegionCache<F::Sum, M> {
        self.cache
    }
}

pub struct SimpleExploreIter<'a, T, M, E>
where
    M: Morton,
{
    nodes: FoldStack<'a, T, M>,
    explore: E,
}

impl<'a, T, M, E> SimpleExploreIter<'a, T, M, E>
where
    M: Morton,
{
    fn new(node: &'a Internal<T, M>, region: MortonRegion<M>, explore: E) -> Self {
        SimpleExploreIter {
            nodes: vec![(node, region)],
            explore,
        }
    }
}

impl<'a, T, M, E> Iterator for SimpleExploreIter<'a, T, M, E>
where
    M: Morton,
    E: FnMut(MortonRegion<M>) -> bool,
{
    type Item = (MortonRegion<M>, M, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, region)) = self.nodes.pop() {
            match node {
                Internal::Node(box Oct { ref children }) => {
                    // If we shouldn't go further into the region, then take the first thing from the iterator.
                    if !(self.explore)(region) {
                        trace!("chose not to go further");
                        return Some(
                            node.iter()
                                .next()
                                .map(|(m, t)| (region, m, t))
                                .expect("SimpleExploreIter::next(): internal node had no leaves"),
                        );
                    } else {
                        trace!("traversing deeper due to node at level {}", region.level);
                        // Traverse deeper (we already checked if we didn't need to go further).
                        for (ix, child) in children.iter().enumerate() {
                            self.nodes.push((child, region.enter(ix)));
                        }
                    }
                }
                Internal::Leaf(ref item, morton) => {
                    trace!("stopping due to leaf at level {}", region.level);

                    return Some((region, *morton, item));
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

        let mut octree = PointerOctree::<_, u128>::new();
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
