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

type Container = LinearContainer<Hamming, u8, u8>;
type Search<'a> = LinearSearch<'a, Hamming, u8, u8>;

#[test]
fn test_linear_search() {
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
}

#[test]
fn test_linear_container() {
    let data = [
        (0b1010_1010, 12),
        (0b1111_1111, 13),
        (0b0000_0000, 14),
        (0b1111_0000, 16),
        (0b0000_1111, 10),
    ];

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
}
