use criterion::{criterion_group, criterion_main, Criterion};
use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64;
use space::{Hamming, MetricPoint, Neighbor, Simd512};

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = Pcg64::from_seed([1; 32]);
    let mut gen = || {
        let mut feature = Hamming(Simd512([0; 64]));
        rng.fill_bytes(&mut (feature.0).0);
        feature
    };
    let search = gen();
    let data = (0..16384).map(|_| gen()).collect::<Vec<_>>();
    c.bench_function("space: 4-nn in 16384", |b| {
        let mut s = [Neighbor::invalid(); 4];
        b.iter(|| space::linear_knn(&search, &mut s, &data).len())
    })
    .bench_function("min_by_key: 1-nn in 16384", |b| {
        b.iter(|| {
            data.iter()
                .map(|f| f.distance(&search))
                .enumerate()
                .min_by_key(|&(_, d)| d)
        })
    })
    .bench_function("space: 1-nn in 16384", |b| {
        let mut s = [Neighbor::invalid(); 1];
        b.iter(|| space::linear_knn(&search, &mut s, &data).len())
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
