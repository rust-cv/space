use core::{iter::Map, slice};

use alloc::vec::{self, Vec};
use pgat::{ProxyView, ReferenceProxy, View};

use crate::{ApproximateSpace, ExactSpace, Knn, Metric, SpatialContainer, linear_nn};

/// This function performs exact linear nearest neighbor search.
///
/// This may be useful specifically when implementing spatial containers
/// where you need to abstract over ProxyView types.
pub fn linear_knn<'a, 'b, M, P, V>(
    metric: M,
    dataset: impl Iterator<Item = (View<'a, P>, View<'a, V>)>,
    query: View<'b, P>,
    num: usize,
) -> Vec<(M::Unit, View<'a, P>, View<'a, V>)>
where
    M: Metric<P>,
    P: ProxyView,
    V: ProxyView,
{
    let mut dataset = dataset.map(|(pt, val)| (metric.distance(pt, query), pt, val));

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

/// Performs a linear knn search by iterating one-by-one over the dataset
/// and keeping a running set of neighbors which it searches through with binary search.
///
/// You may use the optional type parameters `PP` and `VP` to specify the proxy types for the point and value.
/// By default, it uses [`ReferenceProxy`] for both point and value, which uses &P and &V as the proxies.
pub struct LinearSearch<'a, M, P, V, PP = ReferenceProxy<P>, VP = ReferenceProxy<V>> {
    pub metric: M,
    pub data: &'a [(P, V)],
    pub _phantom: core::marker::PhantomData<(PP, VP)>,
}

impl<'a, M, P, V, PP, VP> LinearSearch<'a, M, P, V, PP, VP>
where
    M: Metric<PP>,
    PP: ProxyView<Owned = P>,
    VP: ProxyView<Owned = V>,
{
    pub fn new(metric: M, data: &'a [(P, V)]) -> Self {
        Self {
            metric,
            data,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<'a, M, P, V, PP, VP> ApproximateSpace for LinearSearch<'a, M, P, V, PP, VP>
where
    M: Metric<PP>,
    PP: ProxyView<Owned = P>,
    VP: ProxyView<Owned = V>,
{
    type PointProxy = PP;
    type ValueProxy = VP;
    type Metric = M;
}

/// This trait is implemented for linear search, which is an exact search algorithm.
impl<'a, M, P, V, PP, VP> ExactSpace for LinearSearch<'a, M, P, V, PP, VP>
where
    M: Metric<PP>,
    PP: ProxyView<Owned = P>,
    VP: ProxyView<Owned = V>,
{
}

impl<'c, M, P, V, PP, VP> Knn for LinearSearch<'c, M, P, V, PP, VP>
where
    M: Metric<PP>,
    PP: ProxyView<Owned = P>,
    VP: ProxyView<Owned = V>,
{
    type KnnIter<'a>
        = vec::IntoIter<(M::Unit, View<'a, PP>, View<'a, VP>)>
    where
        Self: 'a;

    fn knn<'a, 'b>(&'a self, query: View<'b, Self::PointProxy>, num: usize) -> Self::KnnIter<'a> {
        linear_knn::<M, PP, VP>(
            self.metric,
            self.data
                .iter()
                .map(|(pt, val)| (PP::view(pt), VP::view(val))),
            query,
            num,
        )
        .into_iter()
    }

    fn nn<'a, 'b>(
        &'a self,
        query: View<'b, Self::PointProxy>,
    ) -> Option<(M::Unit, View<'a, PP>, View<'a, VP>)> {
        linear_nn::<M, PP, VP>(
            self.metric,
            self.data
                .iter()
                .map(|(pt, val)| (PP::view(pt), VP::view(val))),
            query,
        )
    }
}

pub struct LinearContainer<M, P, V, PP = ReferenceProxy<P>, VP = ReferenceProxy<V>> {
    pub metric: M,
    pub data: Vec<(P, V)>,
    pub _phantom: core::marker::PhantomData<(PP, VP)>,
}

impl<M, P, V, PP, VP> ApproximateSpace for LinearContainer<M, P, V, PP, VP>
where
    M: Metric<PP>,
    PP: ProxyView<Owned = P>,
    VP: ProxyView<Owned = V>,
{
    type PointProxy = PP;
    type ValueProxy = VP;
    type Metric = M;
}

/// This trait is implemented for linear search, which is an exact search algorithm.
impl<M, P, V, PP, VP> ExactSpace for LinearContainer<M, P, V, PP, VP>
where
    M: Metric<PP>,
    PP: ProxyView<Owned = P>,
    VP: ProxyView<Owned = V>,
{
}

impl<M, P, V, PP, VP> SpatialContainer for LinearContainer<M, P, V, PP, VP>
where
    M: Metric<PP>,
    PP: ProxyView<Owned = P>,
    VP: ProxyView<Owned = V>,
{
    type SpatialIter<'a>
        = Map<slice::Iter<'a, (P, V)>, fn(&'a (P, V)) -> (View<'a, PP>, View<'a, VP>)>
    where
        Self: 'a;

    fn with_metric(metric: Self::Metric) -> Self {
        Self {
            metric,
            data: Vec::new(),
            _phantom: core::marker::PhantomData,
        }
    }

    fn insert(&mut self, point: P, value: V) {
        self.data.push((point, value));
    }

    fn iter(&self) -> Self::SpatialIter<'_> {
        self.data.iter().map(|(p, v)| (PP::view(p), VP::view(v)))
    }
}

impl<M, P, V, PP, VP> Knn for LinearContainer<M, P, V, PP, VP>
where
    M: Metric<PP>,
    PP: ProxyView<Owned = P>,
    VP: ProxyView<Owned = V>,
{
    type KnnIter<'a>
        = vec::IntoIter<(M::Unit, View<'a, PP>, View<'a, VP>)>
    where
        Self: 'a;

    fn knn<'a, 'b>(&'a self, query: View<'b, Self::PointProxy>, num: usize) -> Self::KnnIter<'a> {
        linear_knn::<M, PP, VP>(
            self.metric,
            self.data
                .iter()
                .map(|(pt, val)| (PP::view(pt), VP::view(val))),
            query,
            num,
        )
        .into_iter()
    }

    fn nn<'a, 'b>(
        &'a self,
        query: View<'b, Self::PointProxy>,
    ) -> Option<(M::Unit, View<'a, PP>, View<'a, VP>)> {
        linear_nn::<M, PP, VP>(
            self.metric,
            self.data
                .iter()
                .map(|(pt, val)| (PP::view(pt), VP::view(val))),
            query,
        )
    }
}
