use space::{linear_knn, Hamming};

#[test]
fn test_linear_knn() {
    let data = [
        Hamming(0b1010_1010u32),
        Hamming(0b1111_1111),
        Hamming(0b0000_0000),
        Hamming(0b1111_0000),
        Hamming(0b0000_1111),
    ];

    let mut neighbors = [(0, 0); 3];
    assert_eq!(
        linear_knn(&Hamming(0b0101_0000), &mut neighbors, &data).len(),
        3
    );

    assert_eq!(&neighbors[..], &[(3, 2), (2, 2), (4, 6)]);
}
