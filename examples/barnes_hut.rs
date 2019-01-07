use itertools::izip;
use nalgebra::Vector3;
use rand::distributions::{Distribution, Standard};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::iter::repeat_with;
use std::ops::{Add, AddAssign};

use space::*;

const POINTS: usize = 10000;
const CACHE_SIZE: usize = 10000;

struct Center;

impl<M> Folder<(), M> for Center
where
    M: Morton + Add + AddAssign,
{
    type Sum = (u32, Vector3<M>);

    fn gather(&self, m: M, _: &()) -> Self::Sum {
        (1, m.decode())
    }

    fn fold<I>(&self, it: I) -> Self::Sum
    where
        I: Iterator<Item = Self::Sum>,
    {
        it.fold((0, Vector3::zeros()), |total, part| {
            (total.0 + part.0, total.1 + part.1)
        })
    }
}

fn octree_insertion<M: Morton, I: Iterator<Item = M>>(points: I) -> PointerOctree<(), M> {
    let mut octree = PointerOctree::new();
    octree.extend(points.map(|i| (i, ())));
    octree
}

fn random_points<M: Morton>(num: usize) -> impl Iterator<Item = M>
where
    Standard: Distribution<M>,
{
    let mut rng = SmallRng::from_seed([1; 16]);

    repeat_with(move || rng.gen())
        .map(|m| m & M::used_bits())
        .take(num)
}

fn main() {
    let mut octree = octree_insertion::<u64, _>(random_points(POINTS));
    let mut rng = SmallRng::from_seed([1; 16]);
    loop {
        let mut new_octree = PointerOctree::new();
        new_octree.extend(octree.iter().map(|(m, _)| (m, ())));

        std::mem::replace(&mut octree, new_octree);
    }
}
