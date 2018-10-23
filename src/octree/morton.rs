mod morton;
mod region;
mod wrapper;

pub use self::morton::*;
pub use self::region::*;
pub use self::wrapper::*;

use std::hash::Hasher;

// I tried BTreeMap and its worse in every way, so don't bother.
pub type MortonRegionMap<T, M> = std::collections::HashMap<MortonRegion<M>, T, MortonBuildHasher>;
pub type MortonRegionSet<M> = std::collections::HashSet<MortonRegion<M>, MortonBuildHasher>;
pub type MortonMap<T, M> = std::collections::HashMap<MortonWrapper<M>, T, MortonBuildHasher>;
pub type MortonSet<M> = std::collections::HashSet<MortonWrapper<M>, MortonBuildHasher>;

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
    fn write_u64(&mut self, i: u64) {
        let bottom_mask = (1 << CACHE_LOCALITY_BITS) - 1;
        let bottom = i & bottom_mask;
        let top = (i & !bottom_mask) >> CACHE_LOCALITY_BITS;
        self.value = (top ^ 14695981039346656037).wrapping_mul(1099511628211) + bottom;
    }

    #[inline(always)]
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
