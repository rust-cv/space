use std::vec;

use decorum::Total;
use ndarray::{Array1, Array2, ArrayView1, arr1, arr2};
use pgat::{ProxyView, ReferenceProxy, View, ViewInverse};
use space::{ApproximateSpace, ExactSpace, Knn, Metric, SpatialContainer, linear_knn, linear_nn};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ArrayViewWrapper<'a>(pub ArrayView1<'a, f32>);

pub struct ArrayViewProxy;

impl ProxyView for ArrayViewProxy {
    type Owned = Array1<f32>;
    type View<'a> = ArrayViewWrapper<'a>;

    fn view<'a>(owned: &'a Self::Owned) -> Self::View<'a> {
        ArrayViewWrapper(owned.view())
    }
}

impl<'a> ViewInverse<'a> for ArrayViewWrapper<'a> {
    type Owned = Array1<f32>;

    type Proxy = ArrayViewProxy;
}

#[derive(Copy, Clone, Default)]
struct L2;

impl Metric<ArrayViewProxy> for L2 {
    type Unit = Total<f32>;

    fn distance<'a, 'b>(&self, a: ArrayViewWrapper<'a>, b: ArrayViewWrapper<'b>) -> Self::Unit {
        a.0.iter()
            .zip(b.0.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
            .into()
    }
}

pub struct Array2Container<M, V, VP = ReferenceProxy<V>> {
    pub metric: M,
    pub points: Array2<f32>,
    pub values: Vec<V>,
    pub _phantom: core::marker::PhantomData<VP>,
}

impl<M, V, VP> Array2Container<M, V, VP>
where
    M: Metric<ArrayViewProxy>,
    VP: ProxyView<Owned = V>,
{
    fn with_metric_and_data(metric: M, points: Array2<f32>, values: Vec<V>) -> Self {
        assert_eq!(
            points.nrows(),
            values.len(),
            "Number of points must match number of values"
        );
        Self {
            metric,
            points,
            values,
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<M, V, VP> ApproximateSpace for Array2Container<M, V, VP>
where
    M: Metric<ArrayViewProxy>,
    VP: ProxyView<Owned = V>,
{
    type PointProxy = ArrayViewProxy;
    type ValueProxy = VP;
    type Metric = M;
}

/// This trait is implemented for linear search, which is an exact search algorithm.
impl<M, V, VP> ExactSpace for Array2Container<M, V, VP>
where
    M: Metric<ArrayViewProxy>,
    VP: ProxyView<Owned = V>,
{
}

impl<M, V, VP> SpatialContainer for Array2Container<M, V, VP>
where
    M: Metric<ArrayViewProxy>,
    VP: ProxyView<Owned = V>,
{
    type SpatialIter<'a>
        = std::iter::Map<
        std::iter::Zip<
            ndarray::iter::LanesIter<'a, f32, ndarray::Dim<[usize; 1]>>,
            std::slice::Iter<'a, V>,
        >,
        fn((ArrayView1<'a, f32>, &'a V)) -> (ArrayViewWrapper<'a>, View<'a, VP>),
    >
    where
        Self: 'a;

    fn with_metric(metric: Self::Metric) -> Self {
        Self {
            metric,
            points: Array2::zeros((0, 2)),
            values: Vec::new(),
            _phantom: core::marker::PhantomData,
        }
    }

    fn insert(&mut self, point: Array1<f32>, value: V) {
        self.points.push_row(point.view()).unwrap();
        self.values.push(value);
    }

    fn iter(&self) -> Self::SpatialIter<'_> {
        self.points
            .rows()
            .into_iter()
            .zip(self.values.iter())
            .map(|(p, v)| (ArrayViewWrapper(p), VP::view(v)))
    }
}

impl<M, V, VP> Knn for Array2Container<M, V, VP>
where
    M: Metric<ArrayViewProxy>,
    VP: ProxyView<Owned = V>,
{
    type KnnIter<'a>
        = vec::IntoIter<(M::Unit, ArrayViewWrapper<'a>, View<'a, VP>)>
    where
        Self: 'a;

    fn knn<'a, 'b>(&'a self, query: View<'b, Self::PointProxy>, num: usize) -> Self::KnnIter<'a> {
        linear_knn::<M, ArrayViewProxy, VP>(
            self.metric,
            self.points
                .rows()
                .into_iter()
                .zip(self.values.iter())
                .map(|(pt, val)| (ArrayViewWrapper(pt), VP::view(val))),
            query,
            num,
        )
        .into_iter()
    }

    fn nn<'a, 'b>(
        &'a self,
        query: View<'b, Self::PointProxy>,
    ) -> Option<(M::Unit, View<'a, ArrayViewProxy>, View<'a, VP>)> {
        linear_nn::<M, ArrayViewProxy, VP>(
            self.metric,
            self.points
                .rows()
                .into_iter()
                .zip(self.values.iter())
                .map(|(pt, val)| (ArrayViewWrapper(pt), VP::view(val))),
            query,
        )
    }
}

type Container = Array2Container<L2, i32>;

#[test]
fn test_ndarray_container() {
    let points = arr2(&[[1.0, 1.2], [4.4, 4.5], [5.0, -1.2], [2.0, 2.8], [-5.0, 1.3]]);
    let values = vec![1, 2, 3, 4, 5];

    let mut search = Container::with_metric_and_data(L2, points.clone(), values.clone());

    let result = search.knn(ArrayViewWrapper(points.row(0)), 2);
    let result = result.as_slice();
    assert_eq!(result[0].1, ArrayViewWrapper(points.row(0)));
    assert_eq!(result[0].2, &values[0]);
    assert_eq!(result[1].1, ArrayViewWrapper(points.row(3)));
    assert_eq!(result[1].2, &values[3]);

    let new_point = arr1(&[2.0, 2.0]);
    search.insert(new_point.clone(), 6);

    let result = search.knn(ArrayViewWrapper(points.row(0)), 3);
    let result = result.as_slice();
    assert_eq!(result[0].1, ArrayViewWrapper(points.row(0)));
    assert_eq!(result[0].2, &values[0]);
    assert_eq!(result[1].1, ArrayViewWrapper(new_point.view()));
    assert_eq!(result[1].2, &6);
    assert_eq!(result[2].1, ArrayViewWrapper(points.row(3)));
    assert_eq!(result[2].2, &values[3]);
}
