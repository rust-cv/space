//! `space` intends to define the necessary abstractions and implementations for working with spatial data.
//!
//! Uses of spatial data structures:
//! - Point clouds
//! - k-NN (finding nearest neighborn in N-dimensional space)
//! - Collision detection
//! - N-body simulations
//!
//! This crate will not be 1.0 until it has removed all dependencies on nightly features and const generics
//! are available in stable to allow the abstraction over N-dimensional trees.
#![feature(box_syntax, box_patterns)]
#![deny(missing_docs)]

pub mod morton;
pub mod octree;
