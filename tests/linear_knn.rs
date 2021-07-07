use space::{Knn, LinearKnn, MetricPoint, Neighbor};

#[derive(PartialEq)]
struct Hamming(u8);

impl MetricPoint for Hamming {
    type Metric = u8;

    fn distance(&self, other: &Self) -> Self::Metric {
        (self.0 ^ other.0).count_ones() as u8
    }
}

#[test]
fn test_linear_knn() {
    let data = [
        Hamming(0b1010_1010),
        Hamming(0b1111_1111),
        Hamming(0b0000_0000),
        Hamming(0b1111_0000),
        Hamming(0b0000_1111),
    ];

    let search = LinearKnn(data.iter());

    assert_eq!(
        &search.knn(&Hamming(0b0101_0000), 3),
        &[
            Neighbor {
                index: 2,
                distance: 2
            },
            Neighbor {
                index: 3,
                distance: 2
            },
            Neighbor {
                index: 0,
                distance: 6
            }
        ]
    );
}
