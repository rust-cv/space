use criterion::{criterion_group, criterion_main};
use criterion::{Criterion, ParameterizedBenchmark};

use itertools::izip;
use nalgebra::Vector3;
use rand::distributions::Open01;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use space::octree;

fn octree_insertion<'a, I: IntoIterator<Item = (&'a Vector3<f64>, i32)>>(
    vecs: I,
) -> octree::Pointer<i32, u128> {
    let mut octree = octree::Pointer::<_, u128>::new(0);
    octree.extend(vecs);
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

fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "octree",
        ParameterizedBenchmark::new(
            "insertion",
            |b, &n| {
                let points = random_points(n);
                b.iter(move || octree_insertion(points.iter().map(|v| (v, 0))))
            },
            (10..39).map(|n| 1.5f64.powi(n) as usize),
        )
        .with_function("iteration", |b, &n| {
            let points = random_points(n);
            let octree = octree_insertion(points.iter().map(|v| (v, 0)));
            b.iter(move || assert_eq!(octree.iter().count(), n))
        })
        .sample_size(5)
        .warm_up_time(std::time::Duration::from_millis(1000))
        .measurement_time(std::time::Duration::from_millis(5000)),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
