use criterion::{criterion_group, criterion_main};
use criterion::{Criterion, ParameterizedBenchmark};

use itertools::izip;
use nalgebra::Vector3;
use rand::distributions::Open01;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use space::*;

struct PositiveX;

impl<M> Folder<i32, M> for PositiveX
where
    M: Morton,
{
    type Sum = (usize, usize);

    fn gather(&self, m: M, _: &i32) -> Self::Sum {
        if m.decode().0 > (M::one() << (M::dim_bits() - 1)) {
            (1, 0)
        } else {
            (0, 1)
        }
    }

    fn fold<I>(&self, it: I) -> Self::Sum
    where
        I: Iterator<Item = Self::Sum>,
    {
        it.fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }
}

fn octree_insertion<I: IntoIterator<Item = (Vector3<f64>, i32)>>(
    vecs: I,
) -> PointerOctree<i32, u64> {
    let mut octree = PointerOctree::<_, u64>::new();
    let space = LeveledRegion(0);
    octree.extend(
        vecs.into_iter()
            .map(|(v, i)| (space.discretize(v).unwrap(), i)),
    );
    octree
}

fn random_points(num: usize) -> Vec<Vector3<f64>> {
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

fn points(c: &mut Criterion) {
    c.bench(
        "octree",
        ParameterizedBenchmark::new(
            "insertion",
            |b, &n| {
                let points = random_points(n);
                b.iter(move || octree_insertion(points.iter().cloned().map(|v| (v, 0))))
            },
            (10..39).map(|n| 1.5f64.powi(n) as usize),
        )
        .with_function("iteration", |b, &n| {
            let points = random_points(n);
            let octree = octree_insertion(points.iter().cloned().map(|v| (v, 0)));
            b.iter(move || octree.iter().count())
        })
        .with_function("full_fold", |b, &n| {
            let points = random_points(n);
            let octree = octree_insertion(points.iter().cloned().map(|v| (v, 0)));
            b.iter(move || {
                octree
                    .iter_fold(
                        PositiveX,
                        MortonRegionCache::with_hasher(1, MortonBuildHasher::default()),
                    )
                    .count()
            })
        })
        .sample_size(5)
        .warm_up_time(std::time::Duration::from_millis(1000))
        .measurement_time(std::time::Duration::from_millis(5000)),
    );
}

criterion_group!(benches, points);
criterion_main!(benches);
