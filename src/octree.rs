mod morton;
pub mod pointer;

pub use self::morton::*;

pub trait Gatherer<Item> {
    type Sum;
    fn gather<'a, I>(&mut self, it: I) -> Self::Sum
    where
        Item: 'a,
        I: Iterator<Item = (Morton<u64>, &'a Item)>;
}
