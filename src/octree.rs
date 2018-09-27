use itertools::Itertools;
use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

use derive_more as dm;

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
    pub fn insert<S>(&mut self, point: Vector3<S>, item: T)
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

/// Also known as a Z-order encoding, this partitions a bounded space into finite, but localized, boxes.
#[derive(Debug, Clone, Copy, Eq, PartialEq, dm::BitOr, dm::BitAnd, dm::Shl, dm::Shr)]
struct Morton(u64);

const NUM_BITS_PER_DIM: usize = 64 / 3;
const MORTON_HIGHEST_BITS: Morton = Morton(0x7000_0000_0000_0000);

impl Morton {
    #[inline]
    fn get_level(self, level: usize) -> usize {
        ((self & (MORTON_HIGHEST_BITS >> (3 * level))) >> (3 * (NUM_BITS_PER_DIM - level - 1))).0
            as usize
    }
}

impl<S> From<Vector3<S>> for Morton
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn from(point: Vector3<S>) -> Morton {
        let point = point.map(|x| {
            (x * (S::one() + S::one()).powi(NUM_BITS_PER_DIM as i32))
                .to_u64()
                .unwrap()
        });
        split_by_3(point.x) << 0 | split_by_3(point.y) << 1 | split_by_3(point.z) << 2
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

/// This allows the lower bits to be spread into every third bit.
///
/// This comes from https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
#[inline]
fn split_by_3(x: u64) -> Morton {
    let x = x & 0x1fffff;
    let x = (x | x << 32) & 0x1f00000000ffff;
    let x = (x | x << 16) & 0x1f0000ff0000ff;
    let x = (x | x << 8) & 0x100f00f00f00f00f;
    let x = (x | x << 4) & 0x10c30c30c30c30c3;
    let x = (x | x << 2) & 0x1249249249249249;
    return Morton(x);
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
    fn test_octree_rand() {
        let mut xrng = SmallRng::from_seed([1; 16]);
        let mut yrng = SmallRng::from_seed([4; 16]);
        let mut zrng = SmallRng::from_seed([0; 16]);

        let mut octree = Octree::new(0);
        for (x, y, z) in izip!(
            xrng.sample_iter(&Open01).take(5000),
            yrng.sample_iter(&Open01).take(5000),
            zrng.sample_iter(&Open01).take(5000)
        ) {
            let v: Vector3<f64> = Vector3::new(x, y, z);
            octree.insert(v, 0);
        }

        assert_eq!(octree.iter().count(), 5000);
    }
}
