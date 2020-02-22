# space

[![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo]

[ci]: https://img.shields.io/crates/v/space.svg
[cl]: https://crates.io/crates/space/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/space/badge.svg
[dl]: https://docs.rs/space/

[lo]: https://tokei.rs/b1/github/vadixidav/space?category=code

A library providing abstractions for spatial datastructures and search

If you use a linear algebra library and would like to have the [`MetricPoint`]
trait implemented on its types natively, please raise an issue on that library
and they can provide an appropriate implementation for each distance type.
It should be implemented on `L1<Point>`, `L2<Point>`, etc.
