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

## Usage

```rust
use space::Metric;

struct Hamming;

impl Metric<u8> for Hamming {
    type Unit = u8;

    fn distance(&self, &a: &u8, &b: &u8) -> Self::Unit {
        (a ^ b).count_ones() as u8
    }
}
```

## Benchmarks

To run the benchmarks, use the following command:

```bash
cargo bench --all-features
```

If you do not pass `--all-features`, the benchmark wont run. Due to [this issue](https://github.com/rust-lang/cargo/issues/2911), the SIMD feature must be enabled. Cargo offers no way to automatically bring the SIMD feature in for the benchmark, and thus it must be passed at the command line.