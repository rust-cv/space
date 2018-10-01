use super::*;

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

    /// Insert an item with a point and return the existing item if they would both occupy the same space.
    pub fn insert<S>(&mut self, point: &Vector3<S>, item: T)
    where
        S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
    {
        let bound = (S::one() + S::one()).powi(self.level);
        if point.iter().any(|n| n.abs() > bound) {
            panic!("space::Octree::insert(): tried to add a point outside the Octree bounds");
        }

        // Convert the point into normalized space.
        let morton = Morton::from(point.map(|n| (n + bound) / bound.powi(2)));

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
            }).into_inner();

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

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.tree.iter()
    }
}

impl<'a, T, S> Extend<(Vector3<S>, T)> for Octree<T>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    fn extend<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = (Vector3<S>, T)>,
    {
        for (v, item) in it.into_iter() {
            self.insert(&v, item);
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
            self.insert(v, item);
        }
    }
}

/// Tree with space implicitly divided based on a Morton code.
#[derive(Clone, Debug)]
enum MortonOctree<T> {
    Node(Box<[MortonOctree<T>; 8]>),
    Leaf(Vec<T>, Morton),
    None,
}

impl<T> MortonOctree<T> {
    fn iter(&self) -> impl Iterator<Item = &T> {
        use either::Either::*;
        match self {
            MortonOctree::Node(box ref n) => Left(MortonOctreeIter {
                nodes: vec![(n, 0)],
                vec_iter: [].iter(),
            }),
            MortonOctree::Leaf(ref item, _) => Right(item.iter()),
            MortonOctree::None => Right([].iter()),
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
}

impl<'a, T> Iterator for MortonOctreeIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        self.vec_iter.next().or_else(|| {
            while let Some((node, ix)) = self.nodes.pop() {
                if ix != 7 {
                    self.nodes.push((node, ix + 1));
                }
                match node[ix] {
                    MortonOctree::Node(ref children) => self.nodes.push((children, 0)),
                    MortonOctree::Leaf(ref item, _) => {
                        self.vec_iter = item.iter();
                        // This wont work if there is ever an empty vec.
                        return self.vec_iter.next();
                    }
                    _ => {}
                }
            }
            None
        })
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
            ).take(5000)
            .map(|(x, y, z)| (Vector3::<f64>::new(x, y, z), 0)),
        );

        assert_eq!(octree.iter().count(), 5000);
    }
}
