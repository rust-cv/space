use space::{Knn, KnnFromBatch, LinearKnn, Metric, Neighbor};

#[derive(Default)]
struct Hamming;

impl Metric<u8> for Hamming {
    type Unit = u8;

    fn distance(&self, &a: &u8, &b: &u8) -> Self::Unit {
        (a ^ b).count_ones() as u8
    }
}

#[test]
fn test_linear_knn() {
    let data = vec![
        (0b1010_1010, 12),
        (0b1111_1111, 13),
        (0b0000_0000, 14),
        (0b1111_0000, 16),
        (0b0000_1111, 10),
    ];

    let search: LinearKnn<Hamming, _> = KnnFromBatch::from_batch(data.iter());

    assert_eq!(
        &search.knn(&0b0101_0000, 3),
        &[
            (
                Neighbor {
                    index: 2,
                    distance: 2
                },
                &data[2].0,
                &data[2].1
            ),
            (
                Neighbor {
                    index: 3,
                    distance: 2
                },
                &data[3].0,
                &data[3].1
            ),
            (
                Neighbor {
                    index: 0,
                    distance: 6
                },
                &data[0].0,
                &data[0].1
            )
        ]
    );
}
