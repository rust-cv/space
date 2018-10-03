use super::{morton::*, *};
use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

use itertools::Itertools;

/// An octree that starts with a cube from [-1, 1] in each dimension and will only expand.
pub struct Octree<T> {
    tree: MortonOctree<T>,
    /// Dimensions of the top level node are from [-2**level, 2**level].
    level: i32,
}

impl<T> Octree<T> {
    /// Dimensions of the top level node are fixed in the range [-2**level, 2**level].
    pub fn new(level: i32) -> Octree<T> {
        Octree {
            tree: MortonOctree::default(),
            level: level,
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
        let (tree_part, level) = (0..NUM_BITS_PER_DIM)
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
            for i in level + 1..NUM_BITS_PER_DIM {
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

    pub fn iter(&self) -> impl Iterator<Item = (Morton<u64>, &T)> {
        self.tree.iter()
    }
}

impl<T, S> Extend<(Vector3<S>, T)> for Octree<T>
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

impl<'a, T, S> Extend<(&'a Vector3<S>, T)> for Octree<T>
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
enum MortonOctree<T> {
    Node(Box<[MortonOctree<T>; 8]>),
    Leaf(Vec<T>, Morton<u64>),
    None,
}

impl<T> MortonOctree<T> {
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
        mut gatherer: G,
    ) -> impl Iterator<Item = (MortonRegion<u64>, G::Sum)> + 'a
    where
        F: FnMut(MortonRegion<u64>) -> bool + 'a,
        G: Gatherer<T> + 'a,
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
                        vec![(n, 0)],
                        further,
                        gatherer,
                    ))
                } else {
                    Right(std::iter::once((
                        base_region,
                        gatherer.gather(n.iter().flat_map(|c| c.iter().map(|(_, t)| t))),
                    )))
                }
            }
            MortonOctree::Leaf(ref items, _) => Right(std::iter::once((
                base_region,
                gatherer.gather(items.iter()),
            ))),
            MortonOctree::None => Left(MortonOctreeFurtherGatherIter::new(
                vec![],
                further,
                gatherer,
            )),
        }
    }

    #[inline]
    fn empty_node() -> Self {
        use self::MortonOctree::*;
        Node(box [None, None, None, None, None, None, None, None])
    }
}

impl<T> Default for MortonOctree<T> {
    fn default() -> Self {
        MortonOctree::None
    }
}

struct MortonOctreeIter<'a, T> {
    nodes: Vec<(&'a [MortonOctree<T>; 8], usize)>,
    vec_iter: std::slice::Iter<'a, T>,
    vec_morton: Morton<u64>,
}

impl<'a, T> MortonOctreeIter<'a, T> {
    fn new(nodes: Vec<(&'a [MortonOctree<T>; 8], usize)>) -> Self {
        MortonOctreeIter {
            nodes,
            vec_iter: [].iter(),
            vec_morton: Morton(0),
        }
    }
}

impl<'a, T> Iterator for MortonOctreeIter<'a, T> {
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

struct MortonOctreeFurtherGatherIter<'a, T, F, G> {
    nodes: Vec<(&'a [MortonOctree<T>; 8], usize)>,
    region: MortonRegion<u64>,
    further: F,
    gatherer: G,
}

impl<'a, T, F, G> MortonOctreeFurtherGatherIter<'a, T, F, G> {
    fn new(nodes: Vec<(&'a [MortonOctree<T>; 8], usize)>, further: F, gatherer: G) -> Self {
        MortonOctreeFurtherGatherIter {
            nodes,
            region: MortonRegion {
                morton: Morton(0),
                level: 1,
            },
            further,
            gatherer,
        }
    }
}

impl<'a, T, F, G> Iterator for MortonOctreeFurtherGatherIter<'a, T, F, G>
where
    F: FnMut(MortonRegion<u64>) -> bool,
    G: Gatherer<T>,
{
    type Item = (MortonRegion<u64>, G::Sum);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, ix)) = self.nodes.pop() {
            // The previous region is the one we are in.
            let region = self.region;

            // Then update the region for the next iteration.
            self.region.exit();
            if ix != 7 {
                self.nodes.push((node, ix + 1));
                self.region.enter(ix + 1);
            }

            match node[ix] {
                MortonOctree::Node(ref children) => {
                    if (self.further)(region) {
                        self.nodes.push((children, 0));
                    } else {
                        return Some((
                            region,
                            self.gatherer
                                .gather(children.iter().flat_map(|c| c.iter().map(|(_, t)| t))),
                        ));
                    }
                }
                MortonOctree::Leaf(ref items, _) => {
                    return Some((region, self.gatherer.gather(items.iter())));
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

        let mut octree = Octree::new(0);
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
