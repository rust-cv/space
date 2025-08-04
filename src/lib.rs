//! See the [Crates.io page](https://crates.io/crates/space) for the README.

#![no_std]
doc_comment::doctest!("../README.md");

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
mod linear;

#[cfg(feature = "alloc")]
pub use linear::*;

use num_traits::Zero;
use pgat::{Owned, ProxyView, View};

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
/// use pgat::ReferenceProxy;
///
/// #[derive(Copy, Clone, Default)]
/// struct AbsDiff;
///
/// impl space::Metric<ReferenceProxy<f64>> for AbsDiff {
///     type Unit = u64;
///
///     fn distance(&self, &a: &f64, &b: &f64) -> Self::Unit {
///         let delta = (a - b).abs();
///         debug_assert!(!delta.is_nan());
///         delta.to_bits()
///     }
/// }
/// ```
pub trait Metric<P: ProxyView>: Copy {
    type Unit: Ord + Zero + Copy;

    fn distance<'a, 'b>(&self, a: View<'a, P>, b: View<'b, P>) -> Self::Unit;
}

pub type MetricUnit<M, P> = <M as Metric<P>>::Unit;

/// Implement this trait on data structures (or wrappers) which perform spatial searches.
///
/// Note that [`ApproximateSpace`] encompasses both exact and approximate searches.
/// Approximate searches may not always return the actual nearest neighbors or the entire set of neighbors in a region.
/// Returning the exact set of neighbors that belong in the query results is also known as 100% recall.
/// The amount of recall you get depends on the exact data structure and algorithm used to perform the search.
/// If you need exact nearest neighbor search (guaranteed 100% recall), instead depend on the [`ExactSpace`] trait.
pub trait ApproximateSpace {
    type PointProxy: ProxyView;
    type ValueProxy: ProxyView;
    type Metric: Metric<Self::PointProxy>;
}

/// This marker trait indicates that the methods provided by search algorithms are exact.
/// It has no further functionality at this time. Implement this on search data structures
/// that guarantee exact nearest neighbor search.
///
/// In this context, exact doesn't mean equidistant neighbors will always be returned, nor does it mean
/// that the same query will always return the same neighbors. However, it does mean that closer neighbors
/// will always be returned before farther neighbors under the ordering of the metric used.
pub trait ExactSpace: ApproximateSpace {}

/// Implement this trait on data structures (or wrappers) which perform KNN searches.
/// The data structure should maintain a key-value mapping between neighbour points and data
/// values. It must be able to output the distance between the query point and the neighbours,
/// which is included in the results.
///
/// Note that [`Knn`] encompasses both exact and approximate nearest neighbor searches.
/// Depend on the [`ExactSpace`] trait to ensure all searches are exact. See [`ExactSpace`] for more details.
pub trait Knn: ApproximateSpace {
    type KnnIter<'a>: Iterator<
        Item = (
            MetricUnit<Self::Metric, Self::PointProxy>,
            View<'a, Self::PointProxy>,
            View<'a, Self::ValueProxy>,
        ),
    >
    where
        Self: 'a;

    /// Get `num` nearest neighbors' distance, key, and value relative to the `target` position.
    ///
    /// The neighbors must be sorted by distance, with the closest neighbor first.
    fn knn<'a, 'b>(&'a self, query: View<'b, Self::PointProxy>, num: usize) -> Self::KnnIter<'a>;

    /// Get the nearest neighbor's distance, key, and value relative to the `target` position.
    #[allow(clippy::type_complexity)]
    fn nn<'a, 'b>(
        &'a self,
        query: View<'b, Self::PointProxy>,
    ) -> Option<(
        MetricUnit<Self::Metric, Self::PointProxy>,
        View<'a, Self::PointProxy>,
        View<'a, Self::ValueProxy>,
    )> {
        self.knn(query, 1).next()
    }
}

/// Implement this trait on data structures (or wrappers) which perform n-sphere range queries.
/// The data structure should maintain a key-value mapping between neighbour points and data
/// values. It must be able to output the distance between the query point and the neighbours,
/// which is included in the results.
///
/// Note that [`NSphereRangeQuery`] encompasses both exact and approximate n-sphere searches.
/// Depend on the [`ExactSpace`] trait to ensure all searches are exact. See [`ExactSpace`] for more details.
pub trait NSphereRangeQuery: ApproximateSpace {
    type NSphereIter<'a>: Iterator<
        Item = (
            MetricUnit<Self::Metric, Self::PointProxy>,
            View<'a, Self::PointProxy>,
            View<'a, Self::ValueProxy>,
        ),
    >
    where
        Self: 'a;

    /// Get all the neighbors in the data structure that lie within a specified range of the query n-sphere.
    ///
    /// The neighbors must be sorted by distance, with the closest neighbor first.
    fn nsphere_query<'a, 'b>(
        &'a self,
        query: View<'b, Self::PointProxy>,
        radius: MetricUnit<Self::Metric, Self::PointProxy>,
    ) -> Self::NSphereIter<'a> {
        self.nsphere_query_limited(query, radius, usize::MAX).0
    }

    /// Get all the neighbors in the data structure that lie within a specified range of the query n-sphere.
    /// You may also provide a `max_neighbors` to limit the number of neighbors returned. This is useful if you
    /// only need neighbors with a certain region, but you need to bail out to prevent excessive searching.
    ///
    /// The neighbors must be sorted by distance, with the closest neighbor first.
    ///
    /// This returns a tuple containing the neighbors in the region and a boolean indicating if we
    /// completely searched the region or if we stopped early due to the `max_neighbors` limit. This boolean
    /// doesn't indicate the neighbors are complete if the algorithm is approximate, only if exact, but
    /// it does indicate that the algorithm terminated its search. If it is `true`,
    /// then the limit was not hit. If the result is `false`, one should not assume that neighbors of a lower
    /// radius than the furthest found neighbor have been searched, but only that the search algorithm itself
    /// was terminated early, so that the results are incomplete. They are still sorted by distance, however,
    /// and search algorithms should still attempt to search closer neighbors first where possible, but it is
    /// up to the user to use the results based on the algorithm's guarantees.
    fn nsphere_query_limited<'a, 'b>(
        &'a self,
        query: View<'b, Self::PointProxy>,
        radius: MetricUnit<Self::Metric, Self::PointProxy>,
        max_neighbors: usize,
    ) -> (Self::NSphereIter<'a>, bool);
}

/// Implement this trait on spatial containers that map points to values.
pub trait SpatialContainer: ApproximateSpace + Sized {
    type SpatialIter<'a>: Iterator<Item = (View<'a, Self::PointProxy>, View<'a, Self::ValueProxy>)>
    where
        Self: 'a;

    /// Create a new instance of the data structure with the given metric.
    fn with_metric(metric: Self::Metric) -> Self;

    /// Insert a (point, value) pair into a spatial data structure.
    fn insert(&mut self, point: Owned<Self::PointProxy>, value: Owned<Self::ValueProxy>);

    /// Iterate over all the point, value pairs in the data structure.
    fn iter(&self) -> Self::SpatialIter<'_>;

    /// Extend the data structure with additional data from an iterator of (point, value) pairs.
    fn extend(
        &mut self,
        iter: impl IntoIterator<Item = (Owned<Self::PointProxy>, Owned<Self::ValueProxy>)>,
    ) {
        for (point, value) in iter {
            self.insert(point, value);
        }
    }

    /// Create a new instance of the data structure with the given metric and an iterator of (point, value) pairs.
    fn from_metric_and_iterator(
        metric: Self::Metric,
        batch: impl IntoIterator<Item = (Owned<Self::PointProxy>, Owned<Self::ValueProxy>)>,
    ) -> Self {
        let mut instance = Self::with_metric(metric);
        instance.extend(batch);
        instance
    }
}

/// This function performs exact linear nearest neighbor search.
///
/// This may be useful specifically when implementing spatial containers
/// where you need to abstract over ProxyView types.
pub fn linear_nn<'a, 'b, M, P, V>(
    metric: M,
    dataset: impl Iterator<Item = (View<'a, P>, View<'a, V>)>,
    query: View<'b, P>,
) -> Option<(M::Unit, View<'a, P>, View<'a, V>)>
where
    M: Metric<P>,
    P: ProxyView,
    V: ProxyView,
{
    dataset
        .map(|(pt, val)| (metric.distance(pt, query), pt, val))
        .min_by_key(|n| n.0)
}
