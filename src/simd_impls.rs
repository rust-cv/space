use crate::{Hamming, MetricPoint};
use core::fmt::{Debug, Error, Formatter};
use core::hash::{Hash, Hasher};

macro_rules! simd_impl {
    ($name:ident, $bytes:expr) => {
        #[repr(align($bytes))]
        #[derive(Copy, Clone)]
        pub struct $name(pub [u8; $bytes]);

        impl MetricPoint for Hamming<$name> {
            #[inline]
            fn distance(&self, rhs: &Self) -> u32 {
                // Perform an XOR popcnt. The compiler is smart
                // enough to optimize this well.
                (self.0)
                    .0
                    .iter()
                    .zip((rhs.0).0.iter())
                    .map(|(&a, &b)| (a ^ b).count_ones())
                    .sum()
            }
        }

        impl Debug for $name {
            fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
                Debug::fmt(&self.0[..], f)
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0[..] == other.0[..]
            }
        }

        impl Eq for $name {}

        impl Hash for $name {
            fn hash<H>(&self, state: &mut H)
            where
                H: Hasher,
            {
                self.0.hash(state)
            }
            fn hash_slice<H>(data: &[Self], state: &mut H)
            where
                H: Hasher,
            {
                for s in data {
                    s.hash(state);
                }
            }
        }

        impl Into<[u8; $bytes]> for $name {
            fn into(self) -> [u8; $bytes] {
                self.0
            }
        }

        impl From<[u8; $bytes]> for $name {
            fn from(a: [u8; $bytes]) -> Self {
                Self(a)
            }
        }
    };
}

simd_impl!(Simd128, 16);
simd_impl!(Simd256, 32);
simd_impl!(Simd512, 64);
simd_impl!(Simd1024, 128);
simd_impl!(Simd2048, 256);
simd_impl!(Simd4096, 512);
