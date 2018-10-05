use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

use bitwise::morton;
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

impl<S> Into<Vector3<S>> for MortonRegion<u64>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn into(self) -> Vector3<S> {
        let Morton(v) = self.morton;
        let cut = NUM_BITS_PER_DIM - self.level;
        let (x, y, z) = morton::decode_3d(v >> (3 * cut));
        let scale = (S::one() + S::one()).powi(-(self.level as i32));

        Vector3::new(
            (S::from_u64(x).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(y).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(z).unwrap() + S::from_f32(0.5).unwrap()) * scale,
        )
    }
}

pub(crate) const NUM_BITS_PER_DIM: usize = 64 / 3;
const MORTON_HIGHEST_BITS: Morton<u64> = Morton(0x7000_0000_0000_0000);

impl Morton<u64> {
    #[inline]
    pub fn get_level(self, level: usize) -> usize {
        ((self >> (3 * (NUM_BITS_PER_DIM - level - 1))).0 & 0x7) as usize
    }

    #[inline]
    pub fn set_level(&mut self, level: usize, val: usize) {
        *self = (*self & !(MORTON_HIGHEST_BITS >> (3 * level)))
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
        Morton(morton::encode_3d(point.x, point.y, point.z))
    }
}

impl<S> Into<Vector3<S>> for Morton<u64>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn into(self) -> Vector3<S> {
        let Morton(v) = self;
        let (x, y, z) = morton::decode_3d(v);
        let scale = (S::one() + S::one()).powi(-(NUM_BITS_PER_DIM as i32));

        Vector3::new(
            (S::from_u64(x).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(y).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(z).unwrap() + S::from_f32(0.5).unwrap()) * scale,
        )
    }
}
