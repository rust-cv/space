#![feature(box_syntax, box_patterns)]

pub mod octree;

pub struct CartesianRegion<T>(pub T, pub T);

trait Contains<Region> {
    type Iter;
    fn contains(&self, region: Region) -> Self::Iter;
}
