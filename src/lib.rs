//! See the [Crates.io page](https://crates.io/crates/space) for the README.

#![no_std]
doc_comment::doctest!("../README.md");

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use num_traits::Unsigned;

/// This trait is implemented for points that exist in a metric space.
/// It is primarily used for keys in nearest neighbor searches.
/// When implementing this trait, you should always choose the smallest unsigned integer that
/// represents your metric space.
///
/// It is important that all points that implement this trait satisfy
/// the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality).
/// This requirement basically means that the sum of distances that start
/// at a point A and end at a point B can never be less than the distance
/// from A to B directly. Note that the metric is required to be an unsigned integer,
/// as distances can only be positive.
///
/// Floating point numbers can be converted to integer metrics by being interpreted as integers by design,
/// although some special patterns (like NaN) do not fit into this model. To be interpreted as an unsigned
/// integer, the float must be positive zero, subnormal, normal, or positive infinity. Any NaN needs
/// to be dealt with before converting into a metric, as they do NOT satisfy the triangle inequality,
/// and will lead to errors. You may want to check for positive infinity as well depending on your use case.
///
/// ## Example
///
/// ```
/// struct A(f64);
///
/// impl space::MetricPoint for A {
///     type Metric = u64;
///
///     fn distance(&self, other: &Self) -> Self::Metric {
///         let delta = (self.0 - other.0).abs();
///         debug_assert!(!delta.is_nan());
///         delta.to_bits()
///     }
/// }
/// ```
pub trait MetricPoint {
    type Metric: Unsigned + Ord + Copy;

    fn distance(&self, other: &Self) -> Self::Metric;
}

/// For k-NN algorithms to return neighbors.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Neighbor<Metric, Ix = usize> {
    /// Index of the neighbor in the search space.
    pub index: Ix,
    /// The distance of the neighbor from the search feature.
    pub distance: Metric,
}

/// Implement this trait on data structures (or wrappers) which perform KNN searches.
pub trait Knn<P, Ix = usize>
where
    P: MetricPoint,
    Ix: Copy,
{
    type KnnIter: IntoIterator<Item = Neighbor<P::Metric, Ix>>;

    /// Get `num` nearest neighbors of `target`.
    ///
    /// For many KNN search algorithms, the returned neighbors are approximate, and may not
    /// be the actual nearest neighbors.
    fn knn(&self, query: &P, num: usize) -> Self::KnnIter;

    /// Get the nearest neighbor of `target`.
    ///
    /// For many KNN search algorithms, the returned neighbors are approximate, and may not
    /// be the actual nearest neighbors.
    fn nn(&self, query: &P) -> Option<Neighbor<P::Metric, Ix>> {
        self.knn(query, 1).into_iter().next()
    }
}

/// This trait gives knn search collections the ability to give the nearest neighbor keys back.
///
/// This is not the final API. Eventually, the iterator type will be chosen by the collection,
/// but for now it is a [`Vec`] until Rust stabilizes GATs.
pub trait KnnPoints<P, Ix = usize>: Knn<P, Ix>
where
    P: MetricPoint,
    Ix: Copy,
{
    /// Get a point using a neighbor index returned by [`Knn::knn`] or [`Knn::nn`].
    ///
    /// This should only be used directly after one of the mentioned methods are called to retrieve
    /// a point associated with a neighbor, and will panic if the index is incorrect due to
    /// mutating the data structure thereafter. The index is only valid up until the next mutation.
    fn get_point(&self, index: Ix) -> &'_ P;

    /// Get `num` nearest neighbor points of `target`.
    ///
    /// For many KNN search algorithms, the returned neighbors are approximate, and may not
    /// be the actual nearest neighbors.
    fn knn_points(&self, query: &P, num: usize) -> Vec<(Neighbor<P::Metric, Ix>, &'_ P)> {
        self.knn(query, num)
            .into_iter()
            .map(|n| (n, self.get_point(n.index)))
            .collect()
    }

    /// Get the nearest neighbor point of `target`.
    ///
    /// For many KNN search algorithms, the returned neighbors are approximate, and may not
    /// be the actual nearest neighbors.
    fn nn_point(&self, query: &P) -> Option<(Neighbor<P::Metric, Ix>, &'_ P)> {
        self.nn(query).map(|n| (n, self.get_point(n.index)))
    }
}

/// This trait gives knn search collections the ability to give the nearest neighbor values back.
///
/// This is not the final API. Eventually, the iterator type will be chosen by the collection,
/// but for now it is a [`Vec`] until Rust stabilizes GATs.
pub trait KnnMap<K, V, Ix = usize>: KnnPoints<K, Ix>
where
    K: MetricPoint,
    Ix: Copy,
{
    /// Get a value using a neighbor index returned by [`Knn::knn`] or [`Knn::nn`].
    ///
    /// This should only be used directly after one of the mentioned methods are called to retrieve
    /// a value associated with a neighbor, and will panic if the index is incorrect due to
    /// mutating the data structure thereafter. The index is only valid up until the next mutation.
    fn get_value(&self, index: Ix) -> &'_ V;

    /// Get `num` nearest neighbor keys of `target`.
    ///
    /// For many KNN search algorithms, the returned neighbors are approximate, and may not
    /// be the actual nearest neighbors.
    fn knn_values(&self, query: &K, num: usize) -> Vec<(Neighbor<K::Metric, Ix>, &'_ V)> {
        self.knn(query, num)
            .into_iter()
            .map(|n| (n, self.get_value(n.index)))
            .collect()
    }

    /// Get the nearest neighbor key of `target`.
    ///
    /// For many KNN search algorithms, the returned neighbors are approximate, and may not
    /// be the actual nearest neighbors.
    fn nn_value(&self, query: &K) -> Option<(Neighbor<K::Metric, Ix>, &'_ V)> {
        self.nn(query).map(|n| (n, self.get_value(n.index)))
    }

    /// Get `num` nearest neighbor keys of `target`.
    ///
    /// For many KNN search algorithms, the returned neighbors are approximate, and may not
    /// be the actual nearest neighbors.
    #[allow(clippy::type_complexity)]
    fn knn_keys_values(
        &self,
        query: &K,
        num: usize,
    ) -> Vec<(Neighbor<K::Metric, Ix>, &'_ K, &'_ V)> {
        self.knn(query, num)
            .into_iter()
            .map(|n| (n, self.get_point(n.index), self.get_value(n.index)))
            .collect()
    }

    /// Get the nearest neighbor key of `target`.
    ///
    /// For many KNN search algorithms, the returned neighbors are approximate, and may not
    /// be the actual nearest neighbors.
    #[allow(clippy::type_complexity)]
    fn nn_key_value(&self, query: &K) -> Option<(Neighbor<K::Metric, Ix>, &'_ K, &'_ V)> {
        self.nn(query)
            .map(|n| (n, self.get_point(n.index), self.get_value(n.index)))
    }
}

/// Performs a linear knn search by iterating over everything in the space
/// and performing a binary search on running set of neighbors.
///
/// ## Example
///
/// ```
/// use space::{Knn, LinearKnn, MetricPoint, Neighbor};
///
/// #[derive(PartialEq)]
/// struct Hamming(u8);
///
/// impl MetricPoint for Hamming {
///     type Metric = u8;
///
///     fn distance(&self, other: &Self) -> Self::Metric {
///         (self.0 ^ other.0).count_ones() as u8
///     }
/// }
///
/// let data = [
///     Hamming(0b1010_1010),
///     Hamming(0b1111_1111),
///     Hamming(0b0000_0000),
///     Hamming(0b1111_0000),
///     Hamming(0b0000_1111),
/// ];
///
/// let search = LinearKnn(data.iter());
/// assert_eq!(
///     &search.knn(&Hamming(0b0101_0000), 3),
///     &[
///         Neighbor { index: 2, distance: 2 },
///         Neighbor { index: 3, distance: 2 },
///         Neighbor { index: 0, distance: 6 },
///     ]
/// );
/// ```
#[cfg(feature = "alloc")]
pub struct LinearKnn<I>(pub I);

#[cfg(feature = "alloc")]
impl<'a, P: 'a, I> Knn<P> for LinearKnn<I>
where
    P: MetricPoint,
    I: Clone + Iterator<Item = &'a P>,
{
    type KnnIter = Vec<Neighbor<P::Metric>>;

    fn knn(&self, query: &P, num: usize) -> Self::KnnIter {
        // Create an iterator mapping the dataset into `Neighbor`.
        let mut dataset = self
            .0
            .clone()
            .map(|point| point.distance(query))
            .enumerate()
            .map(|(index, distance)| Neighbor { index, distance });

        // Create a vector with the correct capacity in advance.
        let mut neighbors = Vec::with_capacity(num);

        // Extend the vector with the first `num` neighbors.
        neighbors.extend((&mut dataset).take(num));
        // Sort the vector by the neighbor distance.
        neighbors.sort_unstable_by_key(|n| n.distance);

        // Iterate over each additional neighbor.
        for point in dataset {
            // Find the position at which it would be inserted.
            let position = neighbors.partition_point(|n| n.distance <= point.distance);
            // If the point is closer than at least one of the points already in `neighbors`, add it
            // into its sorted position.
            if position != num {
                neighbors.pop();
                neighbors.insert(position, point);
            }
        }

        neighbors
    }

    fn nn(&self, query: &P) -> Option<Neighbor<P::Metric>> {
        // Map the input iterator into neighbors and then find the smallest one by distance.
        self.0
            .clone()
            .map(|point| point.distance(query))
            .enumerate()
            .map(|(index, distance)| Neighbor { index, distance })
            .min_by_key(|n| n.distance)
    }
}
