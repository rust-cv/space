use nalgebra::Vector3;
use rand::distributions::{Distribution, Standard};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::iter::repeat_with;
use std::ops::{Add, AddAssign};

use glium::*;
use space::*;

const POINTS: usize = 1000;
const CACHE_SIZE: usize = 20000;
const THETA2: f64 = 0.02;
const EPS2: f64 = 100_000_000.0;
const G: f64 = 5.0e11;
const DRAG: f64 = 2.0e-1;

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
fn depth_width(depth: usize) -> f64 {
    2.0f64.powi(-(depth as i32))
}

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

implement_vertex!(Vertex, position);

fn wrap_delta(pos: f64) -> f64 {
    // Bound must be positive
    let bound = (1u64 << (u64::dim_bits() - 1)) as f64;
    let twobound = 2.0 * bound;
    // Create shrunk_pos, which may still not be inside the space, but is within one stride of it
    let shrunk_pos = pos % twobound;

    if shrunk_pos < -bound {
        shrunk_pos + twobound
    } else if shrunk_pos > bound {
        shrunk_pos - twobound
    } else {
        shrunk_pos
    }
}

fn main() {
    let mut octree = octree_insertion::<u64, _>(random_points(POINTS));
    let mut rng = SmallRng::from_seed([1; 16]);

    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new();
    let context = glutin::ContextBuilder::new().with_vsync(true);
    let display = glium::Display::new(window, context, &events_loop).unwrap();

    let program = glium::Program::from_source(
        &display,
        "
        #version 150 core
        in vec2 position;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
        ",
        "
        #version 150 core
        void main() {
            gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
        ",
        None,
    )
    .unwrap();

    loop {
        // The cache needs to expire every iteration.
        let (new_octree, _, verts) = octree.iter().fold(
            (
                PointerOctree::new(),
                MortonRegionCache::with_hasher(CACHE_SIZE, MortonBuildHasher::default()),
                Vec::new(),
            ),
            |(mut new_octree, cache, mut verts), (m, old_vel)| {
                let position: Vector3<f64> = MortonWrapper(m).into();
                let position = position.map(|n| n + 0.5);
                let mut it = octree.iter_fold_random(
                    21,
                    move |region| {
                        // s/d is used in barnes hut simulations to control simulation accuracy.
                        // Here we are using it to control granularity based on screen space.
                        // We compute the square because it is more efficient.
                        let region_location: Vector3<f64> = region.into();
                        let distance2 = (region_location - position).map(wrap_delta).norm_squared();
                        let width2 = depth_width(region.level).powi(2);
                        width2 > THETA2 * distance2
                    },
                    &Center,
                    &mut rng,
                    cache,
                );

                // Decode the morton into its int vector.
                let v = m.decode().map(|n| n as f64 + 0.5);

                // Compute the net inverse force without any scaling.
                let acceleration = (&mut it).fold(Vector3::zeros(), |acc, (_, (n, pos))| {
                    // Divide the position sum by `n` and subtract the current position so the result is `r'`.
                    // `n`, the number of particles, is our "mass". This is our delta vector.
                    // Because it can go negative in a dimension, its necessary to use signed.
                    let delta = pos.map(|n| n as f64) / f64::from(n) - v;
                    // This wraps the delta so that it is toroidal.
                    let delta = delta.map(wrap_delta);
                    // Now we need the dot product of this vector with itself. This produces `r^2`.
                    // The `EPS` is used to soften the interaction as if the two particles
                    // were a cluster of particles of radius `EPS`. It is squared in advance.
                    let r2 = delta.dot(&delta) as f64 + EPS2;
                    let r3 = (r2 * r2 * r2).sqrt();
                    // We want `n * r' / r^3` as our final solution.
                    acc + delta * f64::from(n) / r3
                });

                // Scale the acceleration by `G` and include drag.
                let acceleration = G * acceleration - DRAG * old_vel;

                // Take the midway between the old and new velocity and apply that to the position.
                let new_morton = u64::encode(
                    (v + acceleration * 0.5 + old_vel)
                        .map(|n| (n as i64 & ((1i64 << u64::dim_bits()) - 1)) as u64),
                );
                new_octree.insert(new_morton, old_vel + acceleration);
                let position: Vector3<f32> = MortonWrapper(new_morton).into();
                verts.push(Vertex {
                    position: [position.x * 2.0 - 1.0, position.y * 2.0 - 1.0],
                });
                (new_octree, it.into(), verts)
            },
        );

        // Replace the old octree.
        std::mem::replace(&mut octree, new_octree);

        let vbo = glium::VertexBuffer::new(&display, &verts).unwrap();
        let ibo = glium::index::NoIndices(glium::index::PrimitiveType::Points);

        let mut target = display.draw();
        target.clear_color(0.0, 0.0, 0.0, 0.0);
        target
            .draw(
                &vbo,
                &ibo,
                &program,
                &glium::uniforms::EmptyUniforms,
                &Default::default(),
            )
            .unwrap();
        target.finish().unwrap();

        events_loop.poll_events(|event| {
            if let glutin::Event::WindowEvent {
                event: glutin::WindowEvent::CloseRequested,
                ..
            } = event
            {
                std::process::exit(0)
            }
        });

        println!("{} particles left", octree.len());
    }
}
