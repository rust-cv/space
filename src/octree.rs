pub mod pointer;

use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

use derive_more as dm;

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
