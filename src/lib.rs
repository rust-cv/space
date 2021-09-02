//! See the [Crates.io page](https://crates.io/crates/space) for the README.

#![no_std]
doc_comment::doctest!("../README.md");

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use num_traits::Zero;

/// This trait is implemented for metrics that form a metric space.
/// It is primarily used for keys in nearest neighbor searches.
/// When implementing this trait, it is recommended to choose the smallest unsigned integer that
/// represents your metric space, but you may also use a float so long as you wrap it in
/// a newtype that enforces the `Ord + Zero + Copy` trait bounds.
/// It is recommended to use
/// [`NoisyFloat`](https://docs.rs/noisy_float/0.2.0/noisy_float/struct.NoisyFloat.html)
/// for this purpose, as it implements the trait bound.
///
/// It is important that all metrics that implement this trait satisfy
/// the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality).
/// This requirement basically means that the sum of distances that start
/// at a point A and end at a point B can never be less than the distance
/// from A to B directly. Note that the metric is required to be an unsigned integer,
/// as distances can only be positive and must be fully ordered.
/// It is also required that two overlapping points (the same point in space) must return
/// a distance of [`Zero::zero`].
///
/// Floating point numbers can be converted to integer metrics by being interpreted as integers by design,
/// although some special patterns (like NaN) do not fit into this model. To be interpreted as an unsigned
/// integer, the float must be positive zero, subnormal, normal, or positive infinity. Any NaN needs
/// to be dealt with before converting into a metric, as they do NOT satisfy the triangle inequality,
/// and will lead to errors. You may want to check for positive infinity as well depending on your use case.
/// You must remove NaNs if you convert to integers, but you must also remove NaNs if you use an ordered
/// wrapper like [`NoisyFloat`](https://docs.rs/noisy_float/0.2.0/noisy_float/struct.NoisyFloat.html).
/// Be careful if you use a wrapper like
/// [`FloatOrd`](https://docs.rs/float-ord/0.3.2/float_ord/struct.FloatOrd.html) which does not
/// force you to remove NaNs. When implementing a metric, you must be sure that NaNs are not allowed, because
/// they may cause nearest neighbor algorithms to panic.
///
/// ## Example
///
/// ```
/// struct AbsDiff;
///
/// impl space::Metric<f64> for AbsDiff {
///     type Unit = u64;
///
///     fn distance(&self, &a: &f64, &b: &f64) -> Self::Unit {
///         let delta = (a - b).abs();
///         debug_assert!(!delta.is_nan());
///         delta.to_bits()
///     }
/// }
/// ```
pub trait Metric<P> {
    type Unit: Ord + Zero + Copy;

    fn distance(&self, a: &P, b: &P) -> Self::Unit;
}

/// Implement this trait on data structures (or wrappers) which perform KNN searches.
/// The data structure should maintain a key-value mapping between neighbour points and data
/// values.
///
/// The lifetime on the trait will be removed once GATs are stabilized.
pub trait Knn<'a> {
    type Point: 'a;
    type Value: 'a;
    type Metric: Metric<Self::Point>;
    type KnnIter: IntoIterator<
        Item = (
            <Self::Metric as Metric<Self::Point>>::Unit,
            &'a Self::Point,
            &'a Self::Value,
        ),
    >;

    /// Get `num` nearest neighbor keys and values of `target`.
    ///
    /// For many KNN search algorithms, the returned neighbors are approximate, and may not
    /// be the actual nearest neighbors.
    fn knn(&'a self, query: &Self::Point, num: usize) -> Self::KnnIter;

    /// Get the nearest neighbor key and values of `target`.
    ///
    /// For many KNN search algorithms, the returned neighbors are approximate, and may not
    /// be the actual nearest neighbors.
    #[allow(clippy::type_complexity)]
    fn nn(
        &'a self,
        query: &Self::Point,
    ) -> Option<(
        <Self::Metric as Metric<Self::Point>>::Unit,
        &'a Self::Point,
        &'a Self::Value,
    )>;
}

/// Implement this trait on data structures (or wrappers) which perform range queries.
/// The data structure should maintain a key-value mapping between neighbour points and data
/// values.
///
/// The lifetime on the trait will be removed once GATs are stabilized.
pub trait RangeQuery<'a>: Knn<'a> {
    type RangeIter: IntoIterator<
        Item = (
            <Self::Metric as Metric<Self::Point>>::Unit,
            &'a Self::Point,
            &'a Self::Value,
        ),
    >;

    /// Get all the points in the data structure that lie within a specified range of the query
    /// point. The points may or may not be sorted by distance.
    #[allow(clippy::type_complexity)]
    fn range_query(
        &self,
        query: &Self::Point,
        range: <Self::Metric as Metric<Self::Point>>::Unit,
    ) -> Self::RangeIter;
}

/// Implement this trait on KNN search data structures that map keys to values and which you can
/// insert new (key, value) pairs.
pub trait KnnInsert<'a>: Knn<'a> {
    /// Insert a (key, value) pair to the [`KnnMap`].
    ///
    /// Returns the index type
    fn insert(&mut self, key: Self::Point, value: Self::Value);
}

/// Create a data structure from a metric and a batch of data points, such as a vector.
/// For many algorithms, using batch initialization yields better results than inserting the points
/// one at a time.
pub trait KnnFromMetricAndBatch<M, B> {
    fn from_metric_and_batch(metric: M, batch: B) -> Self;
}

/// Create a data structure from a batch of data points, such as a vector.
/// For many algorithms, using batch initialization yields better results than inserting the points
/// one at a time.
pub trait KnnFromBatch<M, B>: KnnFromMetricAndBatch<M, B> {
    fn from_batch(batch: B) -> Self;
}

impl<M, B, T> KnnFromBatch<M, B> for T
where
    T: KnnFromMetricAndBatch<M, B>,
    M: Default,
{
    fn from_batch(batch: B) -> Self {
        Self::from_metric_and_batch(M::default(), batch)
    }
}

/// Performs a linear knn search by iterating over everything in the space
/// and performing a binary search on running set of neighbors.
///
/// ## Example
///
/// ```
/// use space::{Knn, LinearKnn, Metric, KnnFromBatch};
///
/// #[derive(Default)]
/// struct Hamming;
///
/// impl Metric<u8> for Hamming {
///     type Unit = u8;
///
///     fn distance(&self, &a: &u8, &b: &u8) -> Self::Unit {
///         (a ^ b).count_ones() as u8
///     }
/// }
///
/// let data = vec![
///     (0b1010_1010, 12),
///     (0b1111_1111, 13),
///     (0b0000_0000, 14),
///     (0b1111_0000, 16),
///     (0b0000_1111, 10),
/// ];
///
/// let search: LinearKnn<Hamming, _> = KnnFromBatch::from_batch(data.iter());
///
/// assert_eq!(
///     &search.knn(&0b0101_0000, 2),
///     &[
///         (2, &data[2].0, &data[2].1),
///         (2, &data[3].0, &data[3].1),
///     ]
/// );
/// ```
#[cfg(feature = "alloc")]
pub struct LinearKnn<M, I> {
    pub metric: M,
    pub points: I,
}

#[cfg(feature = "alloc")]
impl<'a, M: Metric<P>, I, P: 'a, V: 'a> Knn<'a> for LinearKnn<M, I>
where
    I: Iterator<Item = &'a (P, V)> + Clone,
{
    type Metric = M;
    type Point = P;
    type Value = V;
    type KnnIter = Vec<(M::Unit, &'a P, &'a V)>;

    fn knn(&'a self, query: &Self::Point, num: usize) -> Self::KnnIter {
        let mut dataset = self
            .points
            .clone()
            .map(|(pt, val)| (self.metric.distance(pt, query), pt, val));

        // Create a vector with the correct capacity in advance.
        let mut neighbors = Vec::with_capacity(num);

        // Extend the vector with the first `num` neighbors.
        neighbors.extend((&mut dataset).take(num));
        // Sort the vector by the neighbor distance.
        neighbors.sort_unstable_by_key(|n| n.0);

        // Iterate over each additional neighbor.
        for point in dataset {
            // Find the position at which it would be inserted.
            let position = neighbors.partition_point(|n| n.0 <= point.0);
            // If the point is closer than at least one of the points already in `neighbors`, add it
            // into its sorted position.
            if position != num {
                neighbors.pop();
                neighbors.insert(position, point);
            }
        }

        neighbors
    }

    #[allow(clippy::type_complexity)]
    fn nn(
        &self,
        query: &Self::Point,
    ) -> Option<(
        <Self::Metric as Metric<Self::Point>>::Unit,
        &'a Self::Point,
        &'a Self::Value,
    )> {
        // Map the input iterator into neighbors and then find the smallest one by distance.
        self.points
            .clone()
            .map(|(pt, val)| (self.metric.distance(pt, query), pt, val))
            .min_by_key(|n| n.0)
    }
}

#[cfg(feature = "alloc")]
impl<'a, M, I> KnnFromMetricAndBatch<M, I> for LinearKnn<M, I>
where
    M: Default,
{
    fn from_metric_and_batch(metric: M, points: I) -> Self {
        Self { metric, points }
    }
}
