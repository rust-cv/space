use crate::MetricPoint;
use core::{
    cmp::Ordering,
    fmt::{Debug, Error, Formatter},
    hash::{Hash, Hasher},
    ops::{Deref, DerefMut},
};
#[cfg(feature = "serde")]
use serde::{
    de::{self, SeqAccess, Visitor},
    Deserialize, Deserializer, Serialize, Serializer,
};

macro_rules! simd_impl {
    ($name:ident, $bytes:expr, $metric:ty) => {
        #[repr(align($bytes))]
        #[derive(Copy, Clone)]
        pub struct $name(pub [u8; $bytes]);

        impl MetricPoint for $name {
            type Metric = $metric;

            #[inline]
            fn distance(&self, other: &Self) -> Self::Metric {
                // I benchmarked this with many different configurations
                // and determined that it was fastest this way.
                // It was tried with u128, u128x1, u128x2, u128x4, u32x16, u16x32,
                // u64x8, u32x4, and some others. For some reason summing the
                // popcounts from u128x1 in packed_simd_2 gave the best result.
                let simd_left_base = self as *const _ as *const packed_simd_2::u128x1;
                let simd_right_base = other as *const _ as *const packed_simd_2::u128x1;
                (0..$bytes / 16)
                    .map(|i| {
                        let left = unsafe { *simd_left_base.offset(i) };
                        let right = unsafe { *simd_right_base.offset(i) };
                        (left ^ right).count_ones().wrapping_sum() as $metric
                    })
                    .sum()
            }
        }

        impl Deref for $name {
            type Target = [u8; $bytes];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl Debug for $name {
            fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
                self.0.fmt(f)
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
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
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.0.partial_cmp(&other.0)
            }

            fn lt(&self, other: &Self) -> bool {
                self.0.lt(&other.0)
            }
            fn le(&self, other: &Self) -> bool {
                self.0.le(&other.0)
            }
            fn gt(&self, other: &Self) -> bool {
                self.0.gt(&other.0)
            }
            fn ge(&self, other: &Self) -> bool {
                self.0.ge(&other.0)
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0[..].cmp(&other.0[..])
            }

            fn max(self, other: Self) -> Self {
                Self(self.0.max(other.0))
            }
            fn min(self, other: Self) -> Self {
                Self(self.0.min(other.0))
            }
            fn clamp(self, min: Self, max: Self) -> Self {
                Self(self.0.clamp(min.0, max.0))
            }
        }

        impl From<$name> for [u8; $bytes] {
            fn from(a: $name) -> [u8; $bytes] {
                a.0
            }
        }

        impl From<[u8; $bytes]> for $name {
            fn from(a: [u8; $bytes]) -> Self {
                Self(a)
            }
        }

        #[cfg(feature = "serde")]
        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let a: [u8; $bytes] = self.clone().into();
                a.serialize(serializer)
            }
        }

        #[cfg(feature = "serde")]
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct SimdVisitor($name, usize);

                impl<'de> Visitor<'de> for SimdVisitor {
                    type Value = $name;

                    fn expecting(&self, formatter: &mut Formatter) -> Result<(), Error> {
                        formatter.write_str("a sequence of $bytes bytes")
                    }

                    fn visit_seq<S>(mut self, mut seq: S) -> Result<$name, S::Error>
                    where
                        S: SeqAccess<'de>,
                    {
                        // Continuously fill the array with more values.
                        while let Some(value) = seq.next_element()? {
                            if self.1 == $bytes {
                                return Err(de::Error::custom(
                                    "cannot have more than $bytes bytes in sequence",
                                ));
                            }
                            (self.0).0[self.1] = value;
                            self.1 += 1;
                        }

                        if self.1 != $bytes {
                            Err(de::Error::custom(
                                "must have exactly $bytes bytes in sequence",
                            ))
                        } else {
                            Ok(self.0)
                        }
                    }
                }

                // Create the visitor and ask the deserializer to drive it. The
                // deserializer will call visitor.visit_seq() if a seq is present in
                // the input data.
                let visitor = SimdVisitor(Self([0; $bytes]), 0);
                deserializer.deserialize_seq(visitor)
            }
        }
    };
}

simd_impl!(Bits128, 16, u8);
simd_impl!(Bits256, 32, u16);
simd_impl!(Bits512, 64, u16);
simd_impl!(Bits1024, 128, u16);
simd_impl!(Bits2048, 256, u16);
simd_impl!(Bits4096, 512, u16);
