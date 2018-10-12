mod morton;
pub mod pointer;

pub use self::morton::*;

pub trait Gatherer<Item, N> {
    type Sum;
    fn gather<'a, I>(&self, it: I) -> Self::Sum
    where
        Item: 'a,
        I: Iterator<Item = (Morton<N>, &'a Item)>;
}

pub trait Folder {
    type Sum;
    fn sum<I>(&self, it: I) -> Self::Sum
    where
        I: Iterator<Item = Self::Sum>;
}
