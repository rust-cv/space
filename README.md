# space

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo] ![Tests][btl] ![Lints][bll] ![no_std][bnl]

[ci]: https://img.shields.io/crates/v/space.svg
[cl]: https://crates.io/crates/space/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/space/badge.svg
[dl]: https://docs.rs/space/

[lo]: https://tokei.rs/b1/github/rust-cv/space?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[btl]: https://github.com/rust-cv/space/workflows/tests/badge.svg
[bll]: https://github.com/rust-cv/space/workflows/lints/badge.svg
[bnl]: https://github.com/rust-cv/space/workflows/no-std/badge.svg

A library providing abstractions for spatial datastructures and search

If you use a kNN datastructure library and would like to have the `Knn`
trait implemented on its types natively, please raise an issue on that library.
Similarly, crates which define datapoints with specific distance metrics, and
not general linear algebra crates, can implement the `MetricPoint` trait.

See the [bitarray](https://crates.io/crates/bitarray) crate for an implementation
of `MetricPoint` using hamming distance (with optional, though unstable, 512-bit
SIMD support, and always-on 64-bit popcnt instruction support).

## Usage Examples

This example shows how to use the LinearSearch and LinearContainer that come built-in by default. You would use third party containers similarly and can abstract over them using the traits like SpatialContainer and Knn.

```rust
use pgat::ReferenceProxy;
use space::{Knn, LinearContainer, LinearSearch, Metric, SpatialContainer};

#[derive(Copy, Clone, Default)]
struct Hamming;

impl Metric<ReferenceProxy<u8>> for Hamming {
    type Unit = u8;

    fn distance(&self, &a: &u8, &b: &u8) -> Self::Unit {
        (a ^ b).count_ones() as u8
    }
}

// Use type aliases like below to get default proxy types (ReferenceView) on the container.
type Container = LinearContainer<Hamming, u8, u8>;
type Search<'a> = LinearSearch<'a, Hamming, u8, u8>;

let data = [
    (0b1010_1010, 12),
    (0b1111_1111, 13),
    (0b0000_0000, 14),
    (0b1111_0000, 16),
    (0b0000_1111, 10),
];

let search = Search::new(Hamming, &data);

assert_eq!(
    search.knn(&0b0101_0000, 3).as_slice(),
    &[
        (2, &data[2].0, &data[2].1),
        (2, &data[3].0, &data[3].1),
        (6, &data[0].0, &data[0].1)
    ]
);

let mut search = Container::from_metric_and_iterator(Hamming, data);

assert_eq!(
    search.knn(&0b0101_0000, 3).as_slice(),
    &[
        (2, &data[2].0, &data[2].1),
        (2, &data[3].0, &data[3].1),
        (6, &data[0].0, &data[0].1)
    ]
);

search.insert(0b0101_0001, 8);

assert_eq!(search.nn(&0b0101_0000), Some((1, &0b0101_0001, &8)));
```

For an example on how to create a container, a great reference may be found in tests/ndarray.rs in the repository. For brevity it is omitted here, but it shows how to create a specialized structure that uses an Array2 as the storage mechanism for Array1 points.

## Benchmarks

To run the benchmarks, use the following command:

```bash
cargo bench --all-features
```

If you do not pass `--all-features`, the benchmark wont run. Due to [this issue](https://github.com/rust-lang/cargo/issues/2911), the SIMD feature must be enabled. Cargo offers no way to automatically bring the SIMD feature in for the benchmark, and thus it must be passed at the command line.
