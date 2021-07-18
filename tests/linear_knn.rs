use space::{Knn, LinearKnn, Metric, Neighbor};

struct Hamming;

impl Metric<u8> for Hamming {
    type Unit = u8;

    fn distance(&self, &a: &u8, &b: &u8) -> Self::Unit {
        (a ^ b).count_ones() as u8
    }
}

#[test]
fn test_linear_knn() {
    let data = [
        0b1010_1010,
        0b1111_1111,
        0b0000_0000,
        0b1111_0000,
        0b0000_1111,
    ];

    let search = LinearKnn {
        metric: Hamming,
        iter: data.iter(),
    };

    assert_eq!(
        &search.knn(&0b0101_0000, 3),
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
