#![feature(box_syntax, box_patterns)]

mod octree;

pub use self::octree::*;

trait Contains<Point> {
    fn contains(&self, point: Point) -> bool;
}
