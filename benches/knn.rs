use criterion::{criterion_group, criterion_main, Criterion};
use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64;
use space::{Hamming, MetricPoint, Simd512};

// (lhs ^ rhs).count_ones().wrapping_sum() as u32

fn knn(
    feature: &Hamming<Simd512>,
    k: usize,
    search_space: &[Hamming<Simd512>],
    metric: impl Fn(&Hamming<Simd512>, &Hamming<Simd512>) -> u32,
) -> Vec<(usize, u32)> {
    let mut v = vec![];
    for (ix, other) in search_space.iter().enumerate() {
        let distance = metric(feature, other);
        let pos = v
            .binary_search_by_key(&distance, |&(_, distance)| distance)
            .unwrap_or_else(|e| e);
        v.insert(pos, (ix, distance));
        if v.len() > k {
            v.resize_with(k, || unreachable!());
        }
    }
    v
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = Pcg64::from_seed([1; 32]);
    let mut gen = || {
        let mut feature = Hamming(Simd512([0; 64]));
        rng.fill_bytes(&mut (feature.0).0);
        feature
    };
    let search = gen();
    let data = (0..16384).map(|_| gen()).collect::<Vec<_>>();
    c.bench_function("4 nearest neighbors in 16384", |b| {
        b.iter(|| knn(&search, 4, &data, |a, b| a.distance(b)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
