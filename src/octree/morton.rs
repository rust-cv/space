use nalgebra::Vector3;
use num::{Float, FromPrimitive, PrimInt, ToPrimitive};

use bitwise::morton;

use std::hash::{Hash, Hasher};

/// Also known as a Z-order encoding, this partitions a bounded space into finite, but localized, boxes.
pub trait Morton: PrimInt + FromPrimitive + ToPrimitive {
    const BITS: usize;

    pub fn encode(x: Self, y: Self, z: Self) -> Self;
    pub fn decode(self) -> (Self, Self, Self);

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

#[inline]
pub fn morton_levels<M>(m: M) -> impl Iterator<Item = MortonRegion<M>>
where
    M: Morton,
{
    std::iter::once(MortonRegion::default()).chain((1..=M::dim_bits()).map(move |i| MortonRegion {
        morton: m.get_significant_bits(i - 1) << (3 * (M::dim_bits() - i)),
        level: i,
    }))
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct MortonRegion<M> {
    pub morton: M,
    pub level: usize,
}

impl<M> MortonRegion<M>
where
    M: Morton,
{
    #[inline]
    pub fn significant_bits(self) -> M {
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

impl<M> Default for MortonRegion<M>
where
    M: Morton,
{
    fn default() -> Self {
        MortonRegion {
            morton: M::zero(),
            level: 0,
        }
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl<M> Hash for MortonRegion<M>
where
    M: Morton,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        if self.level != 0 {
            let bits = self.morton.get_significant_bits(self.level - 1);

            state.write_u64((bits & M::from_u64(!0).unwrap()).to_u64().unwrap())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MortonWrapper<M>(pub M);

impl<M> Default for MortonWrapper<M>
where
    M: Morton,
{
    fn default() -> Self {
        MortonWrapper(M::zero())
    }
}

#[allow(clippy::derive_hash_xor_eq)]
impl<M> Hash for MortonWrapper<M>
where
    M: Morton,
{
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        state.write_u64((self.0 & M::from_u64(!0).unwrap()).to_u64().unwrap())
    }
}

impl<S, M> Into<Vector3<S>> for MortonRegion<M>
where
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
    M: Morton,
{
    #[inline]
    fn into(self) -> Vector3<S> {
        let v = self.morton;
        let cut = M::dim_bits() - self.level;
        let (x, y, z) = (v >> (3 * cut)).decode();
        let scale = (S::one() + S::one()).powi(-(self.level as i32));

        Vector3::new(
            (S::from_u64(x.to_u64().unwrap()).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(y.to_u64().unwrap()).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(z.to_u64().unwrap()).unwrap() + S::from_f32(0.5).unwrap()) * scale,
        )
    }
}

pub struct MortonRegionIterator<'a, T, M> {
    nodes: Vec<MortonRegion<M>>,
    limit: usize,
    map: &'a MortonRegionMap<T, M>,
}

impl<'a, T, M> MortonRegionIterator<'a, T, M> {
    /// Takes a region to iterate over the regions within it and a limit for the depth level.
    /// This will traverse through `8/7 * 8^(limit - region.level)` nodes, so mind the limit.
    pub fn new(region: MortonRegion<M>, limit: usize, map: &'a MortonRegionMap<T, M>) -> Self {
        // Enough capacity for all the regions.
        let mut nodes = Vec::with_capacity(limit);
        nodes.push(region);
        MortonRegionIterator { nodes, limit, map }
    }
}

impl<'a, T, M> Iterator for MortonRegionIterator<'a, T, M>
where
    M: Morton,
{
    type Item = (MortonRegion<M>, &'a T);

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

pub struct MortonRegionFurtherLinearIterator<M, F> {
    nodes: Vec<MortonRegion<M>>,
    further: F,
}

impl<M, F> MortonRegionFurtherLinearIterator<M, F>
where
    F: FnMut(MortonRegion<M>) -> bool,
{
    /// Takes a region to iterate over the regions within it and a limit for the depth level.
    /// This will traverse through `8/7 * 8^(limit - region.level)` nodes, so mind the limit.
    pub fn new(region: MortonRegion<M>, further: F) -> Self {
        MortonRegionFurtherLinearIterator {
            nodes: vec![region],
            further,
        }
    }
}

impl<M, F> Iterator for MortonRegionFurtherLinearIterator<M, F>
where
    M: Morton,
    F: FnMut(MortonRegion<M>) -> bool,
{
    type Item = MortonRegion<M>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.nodes.pop().map(|region| {
            // Then update the region for the next iteration.
            if let Some(next) = region.next() {
                self.nodes.push(next);
            }

            // Check if we should descend further.
            if region.level < M::dim_bits() && (self.further)(region) {
                self.nodes.push(region.enter(0));
            }
            region
        })
    }
}

pub struct MortonRegionFurtherIterator<'a, T, M, F> {
    nodes: Vec<MortonRegion<M>>,
    further: F,
    map: &'a MortonRegionMap<T, M>,
}

impl<'a, T, M, F> MortonRegionFurtherIterator<'a, T, M, F>
where
    F: FnMut(MortonRegion<M>) -> bool,
{
    /// Takes a region to iterate over the regions within it and a limit for the depth level.
    /// This will traverse through `8/7 * 8^(limit - region.level)` nodes, so mind the limit.
    pub fn new(region: MortonRegion<M>, further: F, map: &'a MortonRegionMap<T, M>) -> Self {
        MortonRegionFurtherIterator {
            nodes: vec![region],
            further,
            map,
        }
    }
}

impl<'a, T, M, F> Iterator for MortonRegionFurtherIterator<'a, T, M, F>
where
    M: Morton,
    F: FnMut(MortonRegion<M>) -> bool,
{
    type Item = (MortonRegion<M>, &'a T);

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

pub struct MortonRegionFurtherLeavesIterator<'a, T, M, F> {
    nodes: Vec<(MortonRegion<M>, bool)>,
    further: F,
    map: &'a MortonRegionMap<T, M>,
}

impl<'a, T, M, F> MortonRegionFurtherLeavesIterator<'a, T, M, F>
where
    F: FnMut(MortonRegion<M>) -> bool,
{
    /// Takes a region to iterate over the regions within it and a limit for the depth level.
    /// This will traverse through `8/7 * 8^(limit - region.level)` nodes, so mind the limit.
    pub fn new(region: MortonRegion<M>, further: F, map: &'a MortonRegionMap<T, M>) -> Self {
        MortonRegionFurtherLeavesIterator {
            nodes: vec![(region, false)],
            further,
            map,
        }
    }
}

impl<'a, T, M, F> Iterator for MortonRegionFurtherLeavesIterator<'a, T, M, F>
where
    M: Morton,
    F: FnMut(MortonRegion<M>) -> bool,
{
    type Item = (MortonRegion<M>, &'a T);

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
                if (self.further)(region) && region.level < M::dim_bits() - 1 {
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

impl<S, M> From<Vector3<S>> for MortonWrapper<M>
where
    M: Morton + std::fmt::Debug + 'static,
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn from(point: Vector3<S>) -> Self {
        let point = point.map(|x| {
            M::from_u64(
                (x * (S::one() + S::one()).powi(M::dim_bits() as i32))
                    .to_u64()
                    .unwrap(),
            )
            .unwrap()
        });
        MortonWrapper(M::encode(point.x, point.y, point.z))
    }
}

impl<S, M> Into<Vector3<S>> for MortonWrapper<M>
where
    M: Morton,
    S: Float + ToPrimitive + FromPrimitive + std::fmt::Debug + 'static,
{
    #[inline]
    fn into(self) -> Vector3<S> {
        let (x, y, z) = self.0.decode();
        let scale = (S::one() + S::one()).powi(-(M::dim_bits() as i32));

        Vector3::new(
            (S::from_u64(x.to_u64().unwrap()).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(y.to_u64().unwrap()).unwrap() + S::from_f32(0.5).unwrap()) * scale,
            (S::from_u64(z.to_u64().unwrap()).unwrap() + S::from_f32(0.5).unwrap()) * scale,
        )
    }
}

pub type MortonRegionMap<T, M> =
    std::collections::HashMap<MortonRegion<M>, T, PassthroughBuildHasher>;
pub type MortonRegionSet<M> = std::collections::HashSet<MortonRegion<M>, PassthroughBuildHasher>;
pub type MortonMap<T, M> = std::collections::HashMap<MortonWrapper<M>, T, PassthroughBuildHasher>;
pub type MortonSet<M> = std::collections::HashSet<MortonWrapper<M>, PassthroughBuildHasher>;

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
