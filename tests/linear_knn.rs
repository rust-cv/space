use space::{Knn, KnnFromBatch, LinearKnn, Metric};

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
    let data = [
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
            (2, &data[2].0, &data[2].1),
            (2, &data[3].0, &data[3].1),
            (6, &data[0].0, &data[0].1)
        ]
    );
}
