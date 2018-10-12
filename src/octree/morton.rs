use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};

use bitwise::morton;
use derive_more as dm;

use std::hash::{Hash, Hasher};

pub(crate) const NUM_BITS_PER_DIM_64: usize = 64 / 3;
pub(crate) const NUM_BITS_PER_DIM_128: usize = 128 / 3;
const MORTON_HIGHEST_BITS_64: Morton<u64> = Morton(0x7000_0000_0000_0000);
const MORTON_HIGHEST_BITS_128: Morton<u128> = Morton(0x3800_0000_0000_0000_0000_0000_0000_0000);
const MORTON_UNUSED_BITS_64: Morton<u64> = Morton(0x8000_0000_0000_0000);
const MORTON_UNUSED_BITS_128: Morton<u128> = Morton(0xC000_0000_0000_0000_0000_0000_0000_0000);
const SINGLE_BITS_64: u64 = (1 << 21) - 1;

/// Also known as a Z-order encoding, this partitions a bounded space into finite, but localized, boxes.
#[derive(
    Debug,
    Default,
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

impl MortonRegion<u128> {
    #[inline]
    pub fn significant_bits(self) -> u128 {
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

impl Default for MortonRegion<u64> {
    fn default() -> Self {
        MortonRegion {
            morton: Morton(0),
            level: 0,
        }
    }
}

impl Default for MortonRegion<u128> {
    fn default() -> Self {
        MortonRegion {
            morton: Morton(0),
            level: 0,
        }
    }
}

impl Hash for MortonRegion<u64> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        state.write_u64(self.morton.get_significant_bits(self.level))
    }
}

impl Hash for MortonRegion<u128> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        let bits = self.morton.get_significant_bits(self.level);

        state.write_u64(bits as u64)
    }
}

impl<S> Into<Vector3<S>> for MortonRegion<u64>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn into(self) -> Vector3<S> {
        let Morton(v) = self.morton;
        let cut = NUM_BITS_PER_DIM_64 - self.level;
        let (x, y, z) = morton::decode_3d(v >> (3 * cut));
        let scale = (S::one() + S::one()).powi(-(self.level as i32));

        Vector3::new(
            (S::from_u64(x).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(y).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(z).unwrap() + S::from_f32(0.5).unwrap()) * scale,
        )
    }
}

impl<S> Into<Vector3<S>> for MortonRegion<u128>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn into(self) -> Vector3<S> {
        let Morton(v) = self.morton;
        let (x, y, z) = decode_128(v);
        let cut = NUM_BITS_PER_DIM_128 - self.level;
        let (x, y, z) = (x >> cut, y >> cut, z >> cut);
        let scale = (S::one() + S::one()).powi(-(self.level as i32));

        Vector3::new(
            (S::from_u64(x).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(y).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(z).unwrap() + S::from_f32(0.5).unwrap()) * scale,
        )
    }
}

pub struct MortonRegionIterator<'a, T, N> {
    nodes: Vec<MortonRegion<N>>,
    limit: usize,
    map: &'a MortonMap<T, N>,
}

impl<'a, T, N> MortonRegionIterator<'a, T, N> {
    /// Takes a region to iterate over the regions within it and a limit for the depth level.
    /// This will traverse through `8/7 * 8^(limit - region.level)` nodes, so mind the limit.
    pub fn new(region: MortonRegion<N>, limit: usize, map: &'a MortonMap<T, N>) -> Self {
        // Enough capacity for all the regions.
        let mut nodes = Vec::with_capacity(limit);
        nodes.push(region);
        MortonRegionIterator { nodes, limit, map }
    }
}

impl<'a, T> Iterator for MortonRegionIterator<'a, T, u64> {
    type Item = (MortonRegion<u64>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(region) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push(next);
            }

            // Now try to retrieve this region from the map.
            if let Some(item) = self.map.get(&region) {
                // It worked, so we need to descend into this region further.
                // Only do this so long as the level wouldn't exceed the limit.
                if region.level < self.limit {
                    self.nodes.push(region.enter(0));
                }
                return Some((region, item));
            }
        }
        None
    }
}

impl<'a, T> Iterator for MortonRegionIterator<'a, T, u128> {
    type Item = (MortonRegion<u128>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(region) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push(next);
            }

            // Now try to retrieve this region from the map.
            if let Some(item) = self.map.get(&region) {
                // It worked, so we need to descend into this region further.
                // Only do this so long as the level wouldn't exceed the limit.
                if region.level < self.limit {
                    self.nodes.push(region.enter(0));
                }
                return Some((region, item));
            }
        }
        None
    }
}

pub struct MortonRegionFurtherIterator<'a, T, N, F> {
    nodes: Vec<MortonRegion<N>>,
    further: F,
    map: &'a MortonMap<T, N>,
}

impl<'a, T, N, F> MortonRegionFurtherIterator<'a, T, N, F>
where
    F: FnMut(MortonRegion<N>) -> bool,
{
    /// Takes a region to iterate over the regions within it and a limit for the depth level.
    /// This will traverse through `8/7 * 8^(limit - region.level)` nodes, so mind the limit.
    pub fn new(region: MortonRegion<N>, further: F, map: &'a MortonMap<T, N>) -> Self {
        MortonRegionFurtherIterator {
            nodes: vec![region],
            further,
            map,
        }
    }
}

impl<'a, T, F> Iterator for MortonRegionFurtherIterator<'a, T, u64, F>
where
    F: FnMut(MortonRegion<u64>) -> bool,
{
    type Item = (MortonRegion<u64>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(region) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push(next);
            }

            // Now try to retrieve this region from the map.
            if let Some(item) = self.map.get(&region) {
                // It worked, so we need to descend into this region further.
                // Only do this so long as the level wouldn't exceed the limit.
                if (self.further)(region) {
                    self.nodes.push(region.enter(0));
                }
                return Some((region, item));
            }
        }
        None
    }
}

impl<'a, T, F> Iterator for MortonRegionFurtherIterator<'a, T, u128, F>
where
    F: FnMut(MortonRegion<u128>) -> bool,
{
    type Item = (MortonRegion<u128>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(region) = self.nodes.pop() {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push(next);
            }

            // Now try to retrieve this region from the map.
            if let Some(item) = self.map.get(&region) {
                // It worked, so we need to descend into this region further.
                // Only do this so long as the level wouldn't exceed the limit.
                if (self.further)(region) {
                    self.nodes.push(region.enter(0));
                }
                return Some((region, item));
            }
        }
        None
    }
}

pub struct MortonRegionFurtherLeavesIterator<'a, T, N, F> {
    nodes: Vec<(MortonRegion<N>, bool)>,
    further: F,
    map: &'a MortonMap<T, N>,
}

impl<'a, T, N, F> MortonRegionFurtherLeavesIterator<'a, T, N, F>
where
    F: FnMut(MortonRegion<N>) -> bool,
{
    /// Takes a region to iterate over the regions within it and a limit for the depth level.
    /// This will traverse through `8/7 * 8^(limit - region.level)` nodes, so mind the limit.
    pub fn new(region: MortonRegion<N>, further: F, map: &'a MortonMap<T, N>) -> Self {
        MortonRegionFurtherLeavesIterator {
            nodes: vec![(region, false)],
            further,
            map,
        }
    }
}

impl<'a, T, F> Iterator for MortonRegionFurtherLeavesIterator<'a, T, u64, F>
where
    F: FnMut(MortonRegion<u64>) -> bool,
{
    type Item = (MortonRegion<u64>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((region, had_child)) = self.nodes.pop() {
            // Then update the region for the next iteration.
            // Also get whether this is the last iteration of this child.
            let last = if let Some(next) = region.next() {
                self.nodes.push((next, had_child));
                false
            } else {
                true
            };

            // Now try to retrieve this region from the map.
            if let Some(item) = self.map.get(&region) {
                // Let the parent node know it had a child.
                if let Some((_, ref mut had_child)) = self.nodes.last_mut() {
                    *had_child = true;
                }
                // It worked, so we need to descend into this region further.
                // Only do this so long as the level wouldn't exceed the limit.
                if (self.further)(region) && region.level < NUM_BITS_PER_DIM_64 - 1 {
                    self.nodes.push((region.enter(0), false));
                } else {
                    return Some((region, item));
                }
            } else if last && !had_child {
                let mut parent_region = region;
                parent_region.exit();
                // If the parent failed to retrieve the child region and its the last child, it was a leaf.
                return Some((parent_region, self.map.get(&parent_region).unwrap()));
            }
        }
        None
    }
}

impl<'a, T, F> Iterator for MortonRegionFurtherLeavesIterator<'a, T, u128, F>
where
    F: FnMut(MortonRegion<u128>) -> bool,
{
    type Item = (MortonRegion<u128>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((region, had_child)) = self.nodes.pop() {
            // Then update the region for the next iteration.
            // Also get whether this is the last iteration of this child.
            let last = if let Some(next) = region.next() {
                self.nodes.push((next, had_child));
                false
            } else {
                true
            };

            // Now try to retrieve this region from the map.
            if let Some(item) = self.map.get(&region) {
                // Let the parent node know it had a child.
                if let Some((_, ref mut had_child)) = self.nodes.last_mut() {
                    *had_child = true;
                }
                // It worked, so we need to descend into this region further.
                // Only do this so long as the level wouldn't exceed the limit.
                if (self.further)(region) && region.level < NUM_BITS_PER_DIM_128 - 1 {
                    self.nodes.push((region.enter(0), false));
                } else {
                    return Some((region, item));
                }
            } else if last && !had_child {
                let mut parent_region = region;
                parent_region.exit();
                // If the parent failed to retrieve the child region and its the last child, it was a leaf.
                return Some((parent_region, self.map.get(&parent_region).unwrap()));
            }
        }
        None
    }
}

impl Morton<u64> {
    #[inline]
    pub fn get_significant_bits(self, level: usize) -> u64 {
        self.0 >> (3 * (NUM_BITS_PER_DIM_64 - level - 1))
    }

    #[inline]
    pub fn get_level(self, level: usize) -> usize {
        (self.get_significant_bits(level) & 0x7) as usize
    }

    #[inline]
    pub fn set_level(&mut self, level: usize, val: usize) {
        *self = (*self & !(MORTON_HIGHEST_BITS_64 >> (3 * level)))
            | Morton((val as u64) << (3 * (NUM_BITS_PER_DIM_64 - level - 1)))
    }

    #[inline]
    pub fn reset_level(&mut self, level: usize) {
        *self = *self & !(MORTON_HIGHEST_BITS_64 >> (3 * level))
    }
}

impl Morton<u128> {
    #[inline]
    pub fn get_significant_bits(self, level: usize) -> u128 {
        self.0 >> (3 * (NUM_BITS_PER_DIM_128 - level - 1))
    }

    #[inline]
    pub fn get_level(self, level: usize) -> usize {
        (self.get_significant_bits(level) & 0x7) as usize
    }

    #[inline]
    pub fn set_level(&mut self, level: usize, val: usize) {
        *self = (*self & !(MORTON_HIGHEST_BITS_128 >> (3 * level)))
            | Morton((val as u128) << (3 * (NUM_BITS_PER_DIM_128 - level - 1)))
    }

    #[inline]
    pub fn reset_level(&mut self, level: usize) {
        *self = *self & !(MORTON_HIGHEST_BITS_128 >> (3 * level))
    }
}

impl<S> From<Vector3<S>> for Morton<u64>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn from(point: Vector3<S>) -> Self {
        let point = point.map(|x| {
            (x * (S::one() + S::one()).powi(NUM_BITS_PER_DIM_64 as i32))
                .to_u64()
                .unwrap()
        });
        Morton(morton::encode_3d(point.x, point.y, point.z)) & !MORTON_UNUSED_BITS_64
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
        let scale = (S::one() + S::one()).powi(-(NUM_BITS_PER_DIM_64 as i32));

        Vector3::new(
            (S::from_u64(x).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(y).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(z).unwrap() + S::from_f32(0.5).unwrap()) * scale,
        )
    }
}

impl<S> From<Vector3<S>> for Morton<u128>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn from(point: Vector3<S>) -> Self {
        let point = point.map(|x| {
            (x * (S::one() + S::one()).powi(NUM_BITS_PER_DIM_128 as i32))
                .to_u64()
                .unwrap()
        });
        Morton(encode_128(point.x, point.y, point.z)) & !MORTON_UNUSED_BITS_128
    }
}

impl<S> Into<Vector3<S>> for Morton<u128>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn into(self) -> Vector3<S> {
        let Morton(v) = self;
        let (x, y, z) = decode_128(v);
        let scale = (S::one() + S::one()).powi(-(NUM_BITS_PER_DIM_128 as i32));

        Vector3::new(
            (S::from_u64(x).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(y).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(z).unwrap() + S::from_f32(0.5).unwrap()) * scale,
        )
    }
}

pub type MortonMap<T, N> = std::collections::HashMap<MortonRegion<N>, T, PassthroughBuildHasher>;
pub type MortonSet<N> = std::collections::HashSet<MortonRegion<N>, PassthroughBuildHasher>;

pub type PassthroughBuildHasher = std::hash::BuildHasherDefault<PassthroughHash>;

/// This is not to be used with anything larger than 64-bit. This is not enforced presently.
#[derive(Copy, Clone, Default)]
pub struct PassthroughHash {
    value: u64,
}

#[allow(clippy::cast_lossless)]
impl Hasher for PassthroughHash {
    #[inline]
    fn finish(&self) -> u64 {
        self.value
    }

    #[inline]
    fn write(&mut self, _: &[u8]) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_u8(&mut self, _: u8) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_u16(&mut self, _: u16) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_u32(&mut self, _: u32) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_u64(&mut self, i: u64) {
        self.value = i as u64;
    }

    fn write_u128(&mut self, _: u128) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_usize(&mut self, _: usize) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_i8(&mut self, _: i8) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_i16(&mut self, _: i16) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_i32(&mut self, _: i32) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_i64(&mut self, _: i64) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_i128(&mut self, _: i128) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }

    fn write_isize(&mut self, _: isize) {
        panic!("Passthrough hash should only be used with a single 64 bit value");
    }
}

#[inline]
fn decode_128(i: u128) -> (u64, u64, u64) {
    let low = i as u64;
    let high = (i >> 63) as u64;
    let (lowx, lowy, lowz) = morton::decode_3d(low);
    let (highx, highy, highz) = morton::decode_3d(high);
    (highx << 21 | lowx, highy << 21 | lowy, highz << 21 | lowz)
}

#[inline]
#[allow(clippy::cast_lossless)]
fn encode_128(x: u64, y: u64, z: u64) -> u128 {
    let highx = x >> 21;
    let lowx = x & SINGLE_BITS_64;
    let highy = y >> 21;
    let lowy = y & SINGLE_BITS_64;
    let highz = z >> 21;
    let lowz = z & SINGLE_BITS_64;
    let high = morton::encode_3d(highx, highy, highz);
    let low = morton::encode_3d(lowx, lowy, lowz);
    (high as u128) << 63 | low as u128
}
