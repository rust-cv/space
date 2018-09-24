use crate::Contains;
use itertools::Itertools;
use nalgebra::{Scalar, Vector3};
use num::{Float, FromPrimitive, ToPrimitive};

use derive_more as dm;

/// An octree that starts with a cube from [-1, 1] in each dimension and will only expand.
pub struct Octree<T> {
    tree: Option<Box<MortonOctree<T>>>,
    /// Dimensions of the top level node are from [-2**level, 2**level].
    level: i32,
}

impl<T> Octree<T> {
    /// Insert an item with a point and return the existing item if they would both occupy the same space.
    pub fn insert<S>(&mut self, point: Vector3<S>, item: T) -> Option<T>
    where
        S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
    {
        let bound = (S::one() + S::one()).powi(self.level);
        while point.iter().any(|n| n.abs() > bound) {
            self.expand();
        }

        // Convert the point into normalized space.
        let morton = Morton::from(point.map(|n| (n + bound) / bound.powi(2)));

        // Traverse the tree down to the node we need to operate on.
        let (tree_part, level) = (0..NUM_BITS_PER_DIM)
            .fold_while((&mut self.tree, 0), |(node, old_ix), i| {
                use itertools::FoldWhile::{Continue, Done};
                match node {
                    Some(box MortonOctree::Node(ns)) => {
                        // The index into the array to access the next octree node
                        let subindex = morton.get_level(i);
                        Continue((&mut ns[subindex], i))
                    }
                    Some(box MortonOctree::Leaf(_, _)) => Done((node, old_ix)),
                    None => Done((node, old_ix)),
                }
            }).into_inner();

        match tree_part {
            Some(box ref mut other) => {
                match other {
                    MortonOctree::Leaf(other, other_morton) => {
                        // If they have the same code then we can't insert this so replace the existing one.
                        let mut item = item;
                        if morton == *other_morton {
                            std::mem::swap(other, &mut item);
                            Some(item)
                        } else {
                            // Keep expanding the octree down until they differ at some level.
                            for i in level + 1..NUM_BITS_PER_DIM {
                                // If they are in the same subsection.
                                if morton.get_level(i) == other_morton.get_level(i) {
                                    // Make another Node to place them in.
                                }
                            }
                        }
                    }
                    _ => panic!(
                        "space::Octree::insert(): we should never get a Node beyond the 21st level"
                    ),
                }
            }
            None => *tree_part = Some(box MortonOctree::Leaf(item, morton)),
        }
    }

    fn expand(&mut self) {
        self.level += 1;
        match &mut self.tree {
            Some(box MortonOctree::Node(ref mut ns)) => {
                // We must create 8 new nodes that each contain one of the old nodes in one of their corners.
                for i in 0..8 {
                    let mut temp = None;
                    std::mem::swap(&mut temp, &mut ns[i]);
                    ns[i] = if let Some(n) = temp {
                        let mut children = [None, None, None, None, None, None, None, None];
                        children[7 - i] = Some(n);
                        Some(Box::new(MortonOctree::Node(children)))
                    } else {
                        None
                    };
                }
            }
            _ => {}
        }
    }
}

/// Tree with space implicitly divided based on a Morton code.
#[derive(Clone, Debug)]
enum MortonOctree<T> {
    Node([Option<Box<MortonOctree<T>>>; 8]),
    Leaf(T, Morton),
}

impl<T> Default for MortonOctree<T> {
    fn default() -> Self {
        MortonOctree::Node([None, None, None, None, None, None, None, None])
    }
}

/// Also known as a Z-order encoding, this partitions a bounded space into finite, but localized, boxes.
#[derive(Debug, Clone, Copy, Eq, PartialEq, dm::BitOr, dm::BitAnd, dm::Shl, dm::Shr)]
struct Morton(u64);

const NUM_BITS_PER_DIM: usize = 64 / 3;
const MORTON_HIGHEST_BITS: Morton = Morton(0x7000_0000_0000_0000);

impl Morton {
    fn get_level(self, level: usize) -> usize {
        ((self & (MORTON_HIGHEST_BITS >> (3 * level))) >> (3 * (NUM_BITS_PER_DIM - level - 1))).0
            as usize
    }
}

impl<S> From<Vector3<S>> for Morton
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    fn from(point: Vector3<S>) -> Morton {
        let point = point.map(|x| {
            (x * (S::one() + S::one()).powi(NUM_BITS_PER_DIM as i32))
                .to_u64()
                .unwrap()
        });
        split_by_3(point.x) << 0 | split_by_3(point.y) << 1 | split_by_3(point.z) << 2
    }
}

/// This allows the lower bits to be spread into every third bit.
///
/// This comes from https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
fn split_by_3(x: u64) -> Morton {
    let x = x & 0x1fffff;
    let x = (x | x << 32) & 0x1f00000000ffff;
    let x = (x | x << 16) & 0x1f0000ff0000ff;
    let x = (x | x << 8) & 0x100f00f00f00f00f;
    let x = (x | x << 4) & 0x10c30c30c30c30c3;
    let x = (x | x << 2) & 0x1249249249249249;
    return Morton(x);
}
