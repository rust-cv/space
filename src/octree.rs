mod morton;
pub mod pointer;

pub use self::morton::*;

trait Gatherer<Item> {
    type Sum;
    fn gather<'a, I>(&mut self, it: I) -> Self::Sum
    where
        Item: 'a,
        I: Iterator<Item = &'a Item>;
}
