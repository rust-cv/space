use crate::Contains;
use nalgebra::{Scalar, Vector3};
use num::{Float, FromPrimitive, ToPrimitive};

use derive_more as dm;

/// An octree that starts with a cube from [-1, 1] in each dimension and will only expand.
pub struct Octree<T> {
    tree: MortonOctree<T>,
    /// Dimensions of the top level node are from [-2**level, 2**level].
    level: u32,
}

impl<T> Octree<T> {
    pub fn insert<S>(&mut self, point: Vector3<S>, item: T)
    where
        S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
    {
        // TODO: Possibly improve performance.
        while point
            .iter()
            .any(|n| n.abs() > S::from_u64(1 << self.level).unwrap())
        {
            self.expand();
        }
    }

    fn expand(&mut self) {
        self.level += 1;
        match &mut self.tree {
            MortonOctree::Node(ref mut ns) => {
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
            MortonOctree::Leaf(_) => {}
        }
    }
}

/// Tree with space implicitly divided based on a Morton code.
#[derive(Clone, Debug)]
enum MortonOctree<T> {
    Node([Option<Box<MortonOctree<T>>>; 8]),
    Leaf(T),
}

impl<T> Default for MortonOctree<T> {
    fn default() -> Self {
        MortonOctree::Node([None, None, None, None, None, None, None, None])
    }
}

/// Also known as a Z-order encoding, this partitions a bounded space into finite, but localized, boxes.
#[derive(Debug, Clone, Copy, dm::BitOr, dm::Shl)]
struct Morton(u64);

const NUM_BITS_PER_DIM: usize = 64 / 3;

impl<S> From<Vector3<S>> for Morton
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    fn from(point: Vector3<S>) -> Morton {
        let point = point.map(|x| {
            (x * S::from_u64(1 << NUM_BITS_PER_DIM).unwrap())
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
