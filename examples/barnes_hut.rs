use itertools::izip;
use nalgebra::Vector3;
use rand::distributions::Open01;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use space::*;

type Vect = Vector3<f64>;

struct Center;

impl<M> Folder<(), M> for Center
where
    M: Morton,
{
    type Sum = Vect;

    fn gather(&self, m: M, _: &()) -> Self::Sum {
        MortonWrapper(m).into()
    }

    fn fold<I>(&self, it: I) -> Self::Sum
    where
        I: Iterator<Item = Self::Sum>,
    {
        let mut count = 0u8;
        it.inspect(|_| count += 1).sum::<Vect>() / f64::from(count)
    }
}

fn octree_insertion<I: IntoIterator<Item = Vect>>(vecs: I) -> PointerOctree<(), u64> {
    let mut octree = PointerOctree::<_, u64>::new();
    let space = LeveledRegion(0);
    octree.extend(vecs.into_iter().map(|v| (space.discretize(v).unwrap(), ())));
    octree
}

fn random_points(num: usize) -> Vec<Vect> {
    let mut xrng = SmallRng::from_seed([1; 16]);
    let mut yrng = SmallRng::from_seed([4; 16]);
    let mut zrng = SmallRng::from_seed([0; 16]);

    izip!(
        xrng.sample_iter(&Open01),
        yrng.sample_iter(&Open01),
        zrng.sample_iter(&Open01)
    )
    .take(num)
    .map(|(x, y, z)| Vector3::new(x, y, z))
    .collect()
}

fn main() {
    let octree = octree_insertion(random_points(10000));
}
