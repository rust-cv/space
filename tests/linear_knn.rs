use space::{linear_knn, Hamming, Neighbor};

#[test]
fn test_linear_knn() {
    let data = [
        Hamming(0b1010_1010u32),
        Hamming(0b1111_1111),
        Hamming(0b0000_0000),
        Hamming(0b1111_0000),
        Hamming(0b0000_1111),
    ];

    let mut neighbors = [Neighbor::invalid(); 3];

    assert_eq!(
        &neighbors[..],
        &[
            Neighbor {
                index: 3,
                distance: 2,
            },
            Neighbor {
                index: 2,
                distance: 2,
            },
            Neighbor {
                index: 4,
                distance: 6,
            },
        ]
    );
}
