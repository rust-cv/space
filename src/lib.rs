#![feature(box_syntax, box_patterns)]

mod octree;

pub use self::octree::*;

trait Contains<Region> {
    fn contains(&self, region: Region) -> bool;
}
