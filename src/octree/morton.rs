use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

use bitwise::morton;
use derive_more as dm;

use std::hash::{Hash, Hasher};

/// Also known as a Z-order encoding, this partitions a bounded space into finite, but localized, boxes.
#[derive(
    Debug,
    Clone,
    Copy,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    Hash,
    dm::Not,
    dm::BitOr,
    dm::BitAnd,
    dm::Shl,
    dm::Shr,
)]
pub struct Morton<T>(pub T);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct MortonRegion<T> {
    pub morton: Morton<T>,
    pub level: usize,
}

impl MortonRegion<u64> {
    #[inline]
    pub fn significant_bits(self) -> u64 {
        self.morton.get_significant_bits(self.level)
    }

    #[inline]
    pub(crate) fn enter(mut self, section: usize) -> Self {
        self.morton.set_level(self.level, section);
        self.level += 1;
        self
    }

    #[inline]
    pub(crate) fn exit(&mut self) -> usize {
        self.level -= 1;
        let old = self.morton.get_level(self.level);
        self.morton.reset_level(self.level);
        old
    }

    #[inline]
    pub(crate) fn get(&self) -> usize {
        self.morton.get_level(self.level - 1)
    }

    #[inline]
    pub(crate) fn next(mut self) -> Option<Self> {
        if self.level == 0 {
            None
        } else {
            let last = self.exit();
            if last == 7 {
                None
            } else {
                Some(self.enter(last + 1))
            }
        }
    }
}

impl Hash for MortonRegion<u64> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        state.write_u64((self.morton | MORTON_UNUSED_BIT).get_significant_bits(self.level))
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
const MORTON_UNUSED_BIT: Morton<u64> = Morton(0x8000_0000_0000_0000);

impl Morton<u64> {
    #[inline]
    pub fn get_significant_bits(self, level: usize) -> u64 {
        self.0 >> (3 * (NUM_BITS_PER_DIM - level - 1))
    }

    #[inline]
    pub fn get_level(self, level: usize) -> usize {
        (self.get_significant_bits(level) & 0x7) as usize
    }

    #[inline]
    pub fn set_level(&mut self, level: usize, val: usize) {
        *self = (*self & !(MORTON_HIGHEST_BITS >> (3 * level)))
            | Morton((val as u64) << (3 * (NUM_BITS_PER_DIM - level - 1)))
    }

    #[inline]
    pub fn reset_level(&mut self, level: usize) {
        *self = *self & !(MORTON_HIGHEST_BITS >> (3 * level))
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
        Morton(morton::encode_3d(point.x, point.y, point.z)) & !MORTON_UNUSED_BIT
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

pub type PassthroughMap<K, V> = std::collections::HashMap<K, V, PassthroughBuildHasher>;

pub type PassthroughBuildHasher = std::hash::BuildHasherDefault<PassthroughHash>;

/// This is not to be used with anything larger than 64-bit. This is not enforced presently.
#[derive(Copy, Clone, Default)]
pub struct PassthroughHash {
    value: u64,
}

impl Hasher for PassthroughHash {
    #[inline]
    fn finish(&self) -> u64 {
        self.value
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.value = bytes[0] as u64;
    }

    fn write_u8(&mut self, i: u8) {
        self.value = i as u64;
    }

    fn write_u16(&mut self, i: u16) {
        self.value = i as u64;
    }

    fn write_u32(&mut self, i: u32) {
        self.value = i as u64;
    }

    fn write_u64(&mut self, i: u64) {
        self.value = i as u64;
    }

    fn write_u128(&mut self, i: u128) {
        self.value = i as u64;
    }

    fn write_usize(&mut self, i: usize) {
        self.value = i as u64;
    }

    fn write_i8(&mut self, i: i8) {
        self.value = i as u64;
    }

    fn write_i16(&mut self, i: i16) {
        self.value = i as u64;
    }

    fn write_i32(&mut self, i: i32) {
        self.value = i as u64;
    }

    fn write_i64(&mut self, i: i64) {
        self.value = i as u64;
    }

    fn write_i128(&mut self, i: i128) {
        self.value = i as u64;
    }

    fn write_isize(&mut self, i: isize) {
        self.value = i as u64;
    }
}
