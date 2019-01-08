use itertools::izip;
use nalgebra::Vector3;
use rand::distributions::{Distribution, Standard};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::iter::repeat_with;
use std::ops::{Add, AddAssign};

use space::*;

const POINTS: usize = 10000;
const CACHE_SIZE: usize = 10000;
const THETA2: f32 = 0.001;
const EPS2: f64 = 1_000_000.0;
const G: f64 = 1.0e-7;

struct Center;

impl<M> Folder<Vector3<f64>, M> for Center
where
    M: Morton + Add + AddAssign,
{
    type Sum = (u32, Vector3<M>);

    fn gather(&self, m: M, _: &Vector3<f64>) -> Self::Sum {
        (1, m.decode())
    }

    fn fold<I>(&self, it: I) -> Self::Sum
    where
        I: Iterator<Item = Self::Sum>,
    {
        it.fold((0, Vector3::zeros()), |total, part| {
            (total.0 + part.0, total.1 + part.1)
        })
    }
}

fn octree_insertion<M: Morton, I: Iterator<Item = M>>(points: I) -> PointerOctree<Vector3<f64>, M> {
    let mut octree = PointerOctree::new();
    octree.extend(points.map(|i| (i, Vector3::zeros())));
    octree
}

fn random_points<M: Morton>(num: usize) -> impl Iterator<Item = M>
where
    Standard: Distribution<M>,
{
    let mut rng = SmallRng::from_seed([1; 16]);

    repeat_with(move || rng.gen())
        .map(|m| m & M::used_bits())
        .take(num)
}

// Width of the boxed region for a given depth.
fn depth_width(depth: usize) -> f32 {
    2.0f32.powi(-(depth as i32))
}

fn main() {
    let mut octree = octree_insertion::<u64, _>(random_points(POINTS));
    let mut rng = SmallRng::from_seed([1; 16]);
    loop {
        // The cache needs to expire every iteration.
        let (new_octree, _) = octree.iter().fold(
            (
                PointerOctree::new(),
                MortonRegionCache::with_hasher(CACHE_SIZE, MortonBuildHasher::default()),
            ),
            |(mut new_octree, cache), (m, old_vel)| {
                let position: Vector3<f32> = MortonWrapper(m).into();
                let mut it = octree.iter_fold_random(
                    1,
                    move |region| {
                        // s/d is used in barnes hut simulations to control simulation accuracy.
                        // Here we are using it to control granularity based on screen space.
                        // We compute the square because it is more efficient.
                        let region_location: Vector3<f32> = region.into();
                        let distance2 = (region_location - position).norm_squared();
                        let width2 = depth_width(region.level).powi(2);
                        width2 > THETA2 * distance2
                    },
                    &Center,
                    &mut rng,
                    cache,
                );

                // Decode the morton into its int vector.
                let v = m.decode().map(|n| n as f64);

                // Compute the net inverse force without any scaling.
                let acceleration = (&mut it).fold(Vector3::zeros(), |acc, (region, (n, pos))| {
                    // Divide the position sum by `n` and subtract the current position so the result is `r'`.
                    // `n`, the number of particles, is our "mass". This is our delta vector.
                    // Because it can go negative in a dimension, its necessary to use signed.
                    let delta = pos.map(|n| n as f64) / f64::from(n) - v;
                    // Now we need the dot product of this vector with itself. This produces `r^2`.
                    // The `EPS` is used to soften the interaction as if the two particles
                    // were a cluster of particles of radius `EPS`. It is squared in advance.
                    let r2 = delta.dot(&delta) as f64 + EPS2;
                    let r3 = (r2 * r2 * r2).sqrt();
                    // We want `n * r' / r^3` as our final solution.
                    acc + delta * f64::from(n) / r3
                });

                // Take the midway between the old and new velocity and apply that to the position.
                let new_position = (v + acceleration * 0.5 + old_vel)
                    .map(|n| (n as i64 & u64::used_bits() as i64) as u64);
                new_octree.insert(u64::encode(new_position), old_vel + acceleration);
                (new_octree, it.into())
            },
        );

        std::mem::replace(&mut octree, new_octree);
    }
}
