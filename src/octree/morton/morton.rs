use num::{FromPrimitive, PrimInt, ToPrimitive};
use std::hash::Hash;

use bitwise::morton;

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
