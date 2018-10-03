use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

use derive_more as dm;

/// Also known as a Z-order encoding, this partitions a bounded space into finite, but localized, boxes.
#[derive(
    Debug, Clone, Copy, Eq, PartialEq, Hash, dm::Not, dm::BitOr, dm::BitAnd, dm::Shl, dm::Shr,
)]
pub struct Morton<T>(pub T);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct MortonRegion<T> {
    pub morton: Morton<T>,
    pub level: usize,
}

impl MortonRegion<u64> {
    pub fn enter(&mut self, section: usize) {
        self.morton.set_level(self.level, section);
        self.level += 1;
    }

    pub fn exit(&mut self) {
        self.level -= 1;
    }
}

pub(crate) const NUM_BITS_PER_DIM: usize = 64 / 3;
const MORTON_HIGHEST_BITS: Morton<u64> = Morton(0x7000_0000_0000_0000);

impl Morton<u64> {
    #[inline]
    pub fn get_level(self, level: usize) -> usize {
        ((self & (MORTON_HIGHEST_BITS >> (3 * level))) >> (3 * (NUM_BITS_PER_DIM - level - 1))).0
            as usize
    }

    #[inline]
    pub fn set_level(&mut self, level: usize, val: usize) {
        *self = *self & !(MORTON_HIGHEST_BITS >> (3 * level))
            | Morton((val as u64) << (3 * (NUM_BITS_PER_DIM - level - 1)))
    }
}

impl<S> From<Vector3<S>> for Morton<u64>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn from(point: Vector3<S>) -> Self {
        let point = point.map(|x| {
            (x * (S::one() + S::one()).powi(NUM_BITS_PER_DIM as i32))
                .to_u64()
                .unwrap()
        });
        Morton(split_by_3(point.x) << 0 | split_by_3(point.y) << 1 | split_by_3(point.z) << 2)
    }
}

impl<S> Into<Vector3<S>> for Morton<u64>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn into(self) -> Vector3<S> {
        let Morton(v) = self;
        let x = gather_by_3(v >> 0);
        let y = gather_by_3(v >> 1);
        let z = gather_by_3(v >> 2);
        let scale = (S::one() + S::one()).powi(-(NUM_BITS_PER_DIM as i32));

        Vector3::new(
            (S::from_u64(x).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(y).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(z).unwrap() + S::from_f32(0.5).unwrap()) * scale,
        )
    }
}

/// This allows the lower bits to be spread into every third bit.
///
/// This derives from https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
#[inline]
fn split_by_3(x: u64) -> u64 {
    let x = x & 0x1fffff;
    let x = (x | x << 32) & 0x1f00000000ffff;
    let x = (x | x << 16) & 0x1f0000ff0000ff;
    let x = (x | x << 8) & 0x100f00f00f00f00f;
    let x = (x | x << 4) & 0x10c30c30c30c30c3;
    let x = (x | x << 2) & 0x1249249249249249;
    x
}

/// This allows every third bit to be gathered into the lowest bits.
///
/// This derives from https://stackoverflow.com/a/28358035
#[inline]
fn gather_by_3(x: u64) -> u64 {
    let x = x & 0x1249249249249249;
    let x = (x | (x >> 2)) & 0x10c30c30c30c30c3;
    let x = (x | (x >> 4)) & 0x100f00f00f00f00f;
    let x = (x | (x >> 8)) & 0x1f0000ff0000ff;
    let x = (x | (x >> 16)) & 0x1f00000000ffff;
    let x = (x | (x >> 32)) & 0x1fffff;
    x
}
