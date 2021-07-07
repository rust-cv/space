use criterion::{criterion_group, criterion_main, Criterion};
use rand_core::{RngCore, SeedableRng};
use rand_pcg::Pcg64;
use space::{Bits512, Knn, MetricPoint};

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = Pcg64::from_seed([1; 32]);
    let mut gen = || {
        let mut feature = Bits512([0; 64]);
        rng.fill_bytes(&mut *feature);
        feature
    };
    let search = gen();
    let data = (0..16384).map(|_| gen()).collect::<Vec<_>>();
    c.bench_function("space: 4-nn in 16384", |b| {
        b.iter(|| space::LinearKnn(data.iter()).knn(&search, 4).len())
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
        b.iter(|| space::LinearKnn(data.iter()).nn(&search))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
