mod morton;
mod region;
mod wrapper;

pub use self::morton::*;
pub use self::region::*;
pub use self::wrapper::*;

pub type MortonRegionMap<T, M> = fnv::FnvHashMap<MortonRegion<M>, T>;
pub type MortonRegionSet<M> = fnv::FnvHashSet<MortonRegion<M>>;
pub type MortonMap<T, M> = fnv::FnvHashMap<MortonWrapper<M>, T>;
pub type MortonSet<M> = fnv::FnvHashSet<MortonWrapper<M>>;

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
