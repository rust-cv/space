use crate::Neighbor;
use alloc::vec::Vec;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CandidatesVec {
    candidates: Vec<Neighbor>,
    cap: usize,
}

impl CandidatesVec {
    /// Clears the struct without freeing the memory.
    pub fn clear(&mut self) {
        self.candidates.clear();
        self.cap = 0;
    }

    /// Gets the number of items in the candidate pool.
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    /// Checks if any candidates are present.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pushes a new neighbor to the candidate list.
    ///
    /// Returns if it was added.
    pub fn push(&mut self, n: Neighbor) -> bool {
        let better = self
            .candidates
            .last()
            .map(|&other| other.distance > n.distance)
            .unwrap_or(self.cap != 0);
        let full = self.len() == self.cap;
        let will_add = better | !full;
        if will_add {
            let pos = self
                .candidates
                .binary_search_by_key(&n.distance, |other| other.distance)
                .unwrap_or_else(|e| e);
            self.candidates.insert(pos, n);
            if full {
                self.pop();
            }
        }

        will_add
    }

    /// Pop the worst candidate list.
    ///
    /// This removes it from the candidates.
    pub fn pop(&mut self) -> Option<Neighbor> {
        self.candidates.pop()
    }

    /// Peek at the best candidate.
    ///
    /// This does not remove the candidate.
    pub fn best(&self) -> Option<Neighbor> {
        self.candidates.first().copied()
    }

    /// Sets the cap to `cap`. Resizes if necessary, removing the bottom elements.
    pub fn set_cap(&mut self, cap: usize) {
        self.cap = cap;
        if self.candidates.len() > cap {
            // Remove the items at the end.
            self.candidates.drain(self.candidates.len() - cap..);
        }
    }

    /// Fill a slice with the best elements and return the part of the slice written.
    pub fn fill_slice<'a>(&self, slice: &'a mut [Neighbor]) -> &'a mut [Neighbor] {
        let total_fill = core::cmp::min(slice.len(), self.len());
        for (d, &s) in slice[0..total_fill].iter_mut().zip(&self.candidates) {
            *d = s;
        }
        &mut slice[0..total_fill]
    }
}

#[cfg(test)]
#[test]
fn test_candidates() {
    let mut candidates = CandidatesVec::default();
    candidates.set_cap(3);

    let distances = [
        0.5f32,
        0.000_000_01f32,
        1.1f32,
        2.0f32,
        0.000_000_000_1f32,
        1_000_000.0f32,
        0.6f32,
        0.5f32,
        0.000_000_01f32,
        0.000_000_01f32,
    ];
    for (index, &distance) in distances.iter().enumerate() {
        let distance = distance.to_bits();
        candidates.push(Neighbor { index, distance });
    }
    let mut arr = [Neighbor::invalid(); 3];
    candidates.fill_slice(&mut arr);
    arr[0..2].sort_unstable();
    assert_eq!(
        arr,
        [
            Neighbor {
                index: 4,
                distance: 786_163_455,
            },
            Neighbor {
                index: 8,
                distance: 841_731_191,
            },
            Neighbor {
                index: 1,
                distance: 841_731_191,
            }
        ]
    );
}
