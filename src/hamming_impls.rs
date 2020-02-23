use crate::{Hamming, MetricPoint};

macro_rules! hamming_impl {
    ($x:ty) => {
        impl MetricPoint for Hamming<$x> {
            fn distance(&self, rhs: &Self) -> u32 {
                let Hamming(a) = *self;
                let Hamming(b) = *rhs;
                (a ^ b).count_ones()
            }
        }
    };
}

hamming_impl!(u8);
hamming_impl!(u16);
hamming_impl!(u32);
hamming_impl!(u64);
hamming_impl!(u128);

#[cfg(feature = "hamming-vec")]
mod vec_impls {
    use crate::{Hamming, MetricPoint};
    use alloc::vec::Vec;
    macro_rules! hamming_vec_impl {
        ($x:ty) => {
            impl MetricPoint for Hamming<Vec<$x>> {
                fn distance(&self, rhs: &Self) -> u32 {
                    let Hamming(a) = self;
                    let Hamming(b) = rhs;
                    a.iter().zip(b).map(|(&a, &b)| (a ^ b).count_ones()).sum()
                }
            }
        };
    }

    hamming_vec_impl!(u8);
    hamming_vec_impl!(u16);
    hamming_vec_impl!(u32);
    hamming_vec_impl!(u64);
    hamming_vec_impl!(u128);
}
