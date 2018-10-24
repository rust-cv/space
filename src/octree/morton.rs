mod region;
mod wrapper;

pub use self::morton::*;
pub use self::region::*;
pub use self::wrapper::*;

use bitwise::morton;
use num::{FromPrimitive, PrimInt, ToPrimitive};
use std::hash::{Hash, Hasher};

// I tried BTreeMap and its worse in every way, so don't bother.
pub type MortonRegionMap<T, M> = std::collections::HashMap<MortonRegion<M>, T, MortonBuildHasher>;
pub type MortonRegionSet<M> = std::collections::HashSet<MortonRegion<M>, MortonBuildHasher>;
pub type MortonMap<T, M> = std::collections::HashMap<MortonWrapper<M>, T, MortonBuildHasher>;
pub type MortonSet<M> = std::collections::HashSet<MortonWrapper<M>, MortonBuildHasher>;

pub type MortonRegionCache<T, M> = lru_cache::LruCache<MortonRegion<M>, T, MortonBuildHasher>;
pub type MortonCache<T, M> = lru_cache::LruCache<MortonWrapper<M>, T, MortonBuildHasher>;

/// Visits the values representing the difference, i.e. the keys that are in `primary` but not in `secondary`.
pub fn region_map_difference<'a, T, U, M>(
    primary: &'a MortonRegionMap<T, M>,
    secondary: &'a MortonRegionMap<U, M>,
) -> impl Iterator<Item = MortonRegion<M>> + 'a
where
    M: Morton,
{
    primary.keys().filter_map(move |&k| {
        if secondary.get(&k).is_none() {
            Some(k)
        } else {
            None
        }
    })
}

/// Also known as a Z-order encoding, this partitions a bounded space into finite, but localized, boxes.
pub trait Morton: PrimInt + FromPrimitive + ToPrimitive + Hash {
    const BITS: usize;

    fn encode(x: Self, y: Self, z: Self) -> Self;
    fn decode(self) -> (Self, Self, Self);

    #[inline]
    fn dim_bits() -> usize {
        Self::BITS / 3
    }

    #[inline]
    fn highest_bits() -> Self {
        Self::from_u8(0b111).unwrap() << (3 * (Self::dim_bits() - 1))
    }

    #[inline]
    fn used_bits() -> Self {
        (Self::one() << (3 * Self::dim_bits())) - Self::one()
    }

    #[inline]
    fn unused_bits() -> Self {
        !Self::used_bits()
    }

    #[inline]
    fn get_significant_bits(self, level: usize) -> Self {
        self >> (3 * (Self::dim_bits() - level - 1))
    }

    #[inline]
    fn get_level(self, level: usize) -> usize {
        (self.get_significant_bits(level) & Self::from_u8(0b111).unwrap())
            .to_usize()
            .unwrap()
    }

    #[inline]
    fn level_mask(level: usize) -> Self {
        Self::highest_bits() >> (3 * level)
    }

    #[inline]
    fn set_level(&mut self, level: usize, val: usize) {
        if Self::dim_bits() < level + 1 {
            panic!(
                "Morton::set_level: got invalid level {} (max is {})",
                level,
                Self::dim_bits() - 1
            );
        }
        *self = (*self & !Self::level_mask(level))
            | Self::from_usize(val).unwrap() << (3 * (Self::dim_bits() - level - 1))
    }

    #[inline]
    fn reset_level(&mut self, level: usize) {
        *self = *self & !Self::level_mask(level)
    }

    #[inline]
    fn null() -> Self {
        !Self::zero()
    }

    #[inline]
    fn is_null(self) -> bool {
        self == Self::null()
    }
}

impl Morton for u64 {
    const BITS: usize = 64;

    #[inline]
    fn encode(x: Self, y: Self, z: Self) -> Self {
        morton::encode_3d(x, y, z) & Self::used_bits()
    }

    #[inline]
    fn decode(self) -> (Self, Self, Self) {
        morton::decode_3d(self)
    }
}

impl Morton for u128 {
    const BITS: usize = 128;

    #[inline]
    #[allow(clippy::cast_lossless)]
    fn decode(self) -> (Self, Self, Self) {
        let low = self as u64;
        let high = (self >> 63) as u64;
        let (lowx, lowy, lowz) = morton::decode_3d(low);
        let (highx, highy, highz) = morton::decode_3d(high);
        (
            (highx << 21 | lowx) as u128,
            (highy << 21 | lowy) as u128,
            (highz << 21 | lowz) as u128,
        )
    }

    #[inline]
    #[allow(clippy::cast_lossless)]
    fn encode(x: Self, y: Self, z: Self) -> u128 {
        let highx = (x >> 21) & ((1 << 21) - 1);
        let lowx = x & ((1 << 21) - 1);
        let highy = (y >> 21) & ((1 << 21) - 1);
        let lowy = y & ((1 << 21) - 1);
        let highz = (z >> 21) & ((1 << 21) - 1);
        let lowz = z & ((1 << 21) - 1);
        let high = morton::encode_3d(highx as u64, highy as u64, highz as u64);
        let low = morton::encode_3d(lowx as u64, lowy as u64, lowz as u64);
        (high as u128) << 63 | low as u128
    }
}

pub type MortonBuildHasher = std::hash::BuildHasherDefault<MortonHash>;

/// This const determines how many significant bits from the morton get added into the hash instead of multiplied
/// by the FNV prime. This is done to improve cache locality for mortons and works to great effect. Unfortunately,
/// this has a slight impact on memory consumption ~1/6, but the performance is drastically better. Little is gained
/// by going to higher amounts of bits and the memory cost is too high.
const CACHE_LOCALITY_BITS: usize = 3;

/// This is not to be used with anything larger than 64-bit. This is not enforced presently.
#[derive(Copy, Clone, Default)]
pub struct MortonHash {
    value: u64,
}

#[allow(clippy::cast_lossless)]
impl Hasher for MortonHash {
    #[inline]
    fn finish(&self) -> u64 {
        self.value
    }

    #[inline]
    fn write(&mut self, _: &[u8]) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }

    fn write_u8(&mut self, _: u8) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }

    fn write_u16(&mut self, _: u16) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }

    fn write_u32(&mut self, _: u32) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }

    #[inline(always)]
    #[allow(clippy::unreadable_literal)]
    fn write_u64(&mut self, i: u64) {
        let bottom_mask = (1 << CACHE_LOCALITY_BITS) - 1;
        let bottom = i & bottom_mask;
        let top = (i & !bottom_mask) >> CACHE_LOCALITY_BITS;
        self.value = (top ^ 14695981039346656037).wrapping_mul(1099511628211) + bottom;
    }

    #[inline(always)]
    #[allow(clippy::unreadable_literal)]
    fn write_u128(&mut self, i: u128) {
        let bottom_mask = (1 << CACHE_LOCALITY_BITS) - 1;
        let bottom = i & bottom_mask;
        let top = (i & !bottom_mask) >> CACHE_LOCALITY_BITS;
        self.value = ((top ^ 14695981039346656037).wrapping_mul(1099511628211) + bottom) as u64;
    }

    fn write_usize(&mut self, _: usize) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }

    fn write_i8(&mut self, _: i8) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }

    fn write_i16(&mut self, _: i16) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }

    fn write_i32(&mut self, _: i32) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }

    fn write_i64(&mut self, _: i64) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }

    fn write_i128(&mut self, _: i128) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }

    fn write_isize(&mut self, _: isize) {
        panic!("Morton hash should only be used with a single 64 bit value");
    }
}
