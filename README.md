# space

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo]

[ci]: https://img.shields.io/crates/v/space.svg
[cl]: https://crates.io/crates/space/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/space/badge.svg
[dl]: https://docs.rs/space/

[lo]: https://tokei.rs/b1/github/rust-cv/space?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

A library providing abstractions for spatial datastructures and search

If you use a linear algebra or SIMD library and would like to have the [`MetricPoint`]
trait implemented on its types natively, please raise an issue on that library
and they can provide an appropriate implementation for each distance type.
It should be implemented on `L1<Point>`, `L2<Point>`, etc. This policy does not apply
if SIMD types are created in the standard library.

## Benchmarks

To run the benchmarks, use the following command:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --features simd
```

This ensures processor extensions are used. Due to [this issue](https://github.com/rust-lang/cargo/issues/2911) the SIMD feature must be enabled. Cargo offers no way to automatically bring the SIMD feature in for the benchmark, and thus it must be passed at the command line.