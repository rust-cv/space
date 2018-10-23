use super::*;
use nalgebra::Vector3;
use num::{Float, FromPrimitive, ToPrimitive};
use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy)]
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

    /// Produces a single number that has a canonically unique mapping to every given valid MortonRegion by using
    /// the unused bits to store the level information via shifting.
    #[inline]
    pub(crate) fn canonicalize(&self) -> M {
        (self.morton | M::unused_bits()).get_significant_bits(self.level - 1)
    }
}

impl<M> PartialEq for MortonRegion<M>
where
    M: Morton,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.canonicalize().eq(&other.canonicalize())
    }

    #[inline]
    fn ne(&self, other: &Self) -> bool {
        self.canonicalize().ne(&other.canonicalize())
    }
}

impl<M> Eq for MortonRegion<M> where M: Morton {}

impl<M> PartialOrd for MortonRegion<M>
where
    M: Morton,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.canonicalize().partial_cmp(&other.canonicalize())
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        self.canonicalize().lt(&other.canonicalize())
    }

    #[inline]
    fn le(&self, other: &Self) -> bool {
        self.canonicalize().le(&other.canonicalize())
    }

    #[inline]
    fn gt(&self, other: &Self) -> bool {
        self.canonicalize().gt(&other.canonicalize())
    }

    #[inline]
    fn ge(&self, other: &Self) -> bool {
        self.canonicalize().ge(&other.canonicalize())
    }
}

impl<M> Ord for MortonRegion<M>
where
    M: Morton,
{
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.canonicalize().cmp(&other.canonicalize())
    }
}

impl<M> Default for MortonRegion<M>
where
    M: Morton,
{
    #[inline]
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
    M: Morton + Hash,
{
    #[inline]
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        if self.level != 0 {
            self.canonicalize().hash(state);
        }
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

    #[inline]
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

    #[inline]
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

    #[inline]
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
