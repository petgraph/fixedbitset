//! `FixedBitSet` is a simple fixed size set of bits.
//!
//!
//! ### Crate features
//!
//! - `std` (default feature)  
//!   Disabling this feature disables using std and instead uses crate alloc.
//!   Requires Rust 1.36 to disable.
//!
//! ### Rust Version
//!
//! This version of fixedbitset requires Rust 1.39 or later.
//!
#![doc(html_root_url = "https://docs.rs/fixedbitset/0.4.0/")]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(not(feature = "std"))]
use core as std;

mod range;

#[cfg(feature = "serde")]
extern crate serde;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::fmt::Write;
use std::fmt::{Binary, Display, Error, Formatter};

pub use range::IndexRange;
use std::cmp::{Ord, Ordering};
use std::iter::{Chain, FromIterator};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Index};

const BITS: usize = 32;
type Block = u32;

#[inline]
fn div_rem(x: usize, d: usize) -> (usize, usize) {
    (x / d, x % d)
}

/// `FixedBitSet` is a simple fixed size set of bits that each can
/// be enabled (1 / **true**) or disabled (0 / **false**).
///
/// The bit set has a fixed capacity in terms of enabling bits (and the
/// capacity can grow using the `grow` method).
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FixedBitSet {
    data: Vec<Block>,
    /// length in bits
    length: usize,
}

impl FixedBitSet {
    /// Create a new empty **FixedBitSet**.
    pub const fn new() -> Self {
        FixedBitSet {
            data: Vec::new(),
            length: 0,
        }
    }

    /// Create a new **FixedBitSet** with a specific number of bits,
    /// all initially clear.
    pub fn with_capacity(bits: usize) -> Self {
        let (mut blocks, rem) = div_rem(bits, BITS);
        blocks += (rem > 0) as usize;
        FixedBitSet {
            data: vec![0; blocks],
            length: bits,
        }
    }

    /// Create a new **FixedBitSet** with a specific number of bits,
    /// initialized from provided blocks.
    ///
    /// If the blocks are not the exact size needed for the capacity
    /// they will be padded with zeros (if shorter) or truncated to
    /// the capacity (if longer).
    ///
    /// For example:
    /// ```
    /// let data = vec![4];
    /// let bs = fixedbitset::FixedBitSet::with_capacity_and_blocks(4, data);
    /// assert_eq!(format!("{:b}", bs), "0010");
    /// ```
    pub fn with_capacity_and_blocks<I: IntoIterator<Item = Block>>(bits: usize, blocks: I) -> Self {
        let (mut n_blocks, rem) = div_rem(bits, BITS);
        n_blocks += (rem > 0) as usize;
        let mut data: Vec<Block> = blocks.into_iter().collect();
        // Pad data with zeros if smaller or truncate if larger
        if data.len() != n_blocks {
            data.resize(n_blocks, 0);
        }
        // Disable bits in blocks beyond capacity
        let end = data.len() * 32;
        for (block, mask) in Masks::new(bits..end, end) {
            unsafe {
                *data.get_unchecked_mut(block) &= !mask;
            }
        }
        FixedBitSet {
            data: data,
            length: bits,
        }
    }

    /// Grow capacity to **bits**, all new bits initialized to zero
    pub fn grow(&mut self, bits: usize) {
        let (mut blocks, rem) = div_rem(bits, BITS);
        blocks += (rem > 0) as usize;
        if bits > self.length {
            self.length = bits;
            self.data.resize(blocks, 0);
        }
    }

    /// Return the length of the `FixedBitSet` in bits.
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Return **true** if the bit is enabled in the **FixedBitSet**,
    /// **false** otherwise.
    ///
    /// Note: bits outside the capacity are always disabled.
    ///
    /// Note: Also available with index syntax: `bitset[bit]`.
    #[inline]
    pub fn contains(&self, bit: usize) -> bool {
        let (block, i) = div_rem(bit, BITS);
        match self.data.get(block) {
            None => false,
            Some(b) => (b & (1 << i)) != 0,
        }
    }

    /// Clear all bits.
    #[inline]
    pub fn clear(&mut self) {
        for elt in &mut self.data[..] {
            *elt = 0
        }
    }

    /// Enable `bit`.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn insert(&mut self, bit: usize) {
        assert!(
            bit < self.length,
            "insert at index {} exceeds fixbitset size {}",
            bit,
            self.length
        );
        let (block, i) = div_rem(bit, BITS);
        unsafe {
            *self.data.get_unchecked_mut(block) |= 1 << i;
        }
    }

    /// Enable `bit`, and return its previous value.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn put(&mut self, bit: usize) -> bool {
        assert!(
            bit < self.length,
            "put at index {} exceeds fixbitset size {}",
            bit,
            self.length
        );
        let (block, i) = div_rem(bit, BITS);
        unsafe {
            let word = self.data.get_unchecked_mut(block);
            let prev = *word & (1 << i) != 0;
            *word |= 1 << i;
            prev
        }
    }
    /// Toggle `bit` (inverting its state).
    ///
    /// ***Panics*** if **bit** is out of bounds
    #[inline]
    pub fn toggle(&mut self, bit: usize) {
        assert!(
            bit < self.length,
            "toggle at index {} exceeds fixbitset size {}",
            bit,
            self.length
        );
        let (block, i) = div_rem(bit, BITS);
        unsafe {
            *self.data.get_unchecked_mut(block) ^= 1 << i;
        }
    }
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn set(&mut self, bit: usize, enabled: bool) {
        assert!(
            bit < self.length,
            "set at index {} exceeds fixbitset size {}",
            bit,
            self.length
        );
        let (block, i) = div_rem(bit, BITS);
        unsafe {
            let elt = self.data.get_unchecked_mut(block);
            if enabled {
                *elt |= 1 << i;
            } else {
                *elt &= !(1 << i);
            }
        }
    }

    /// Copies boolean value from specified bit to the specified bit.
    ///
    /// **Panics** if **to** is out of bounds.
    #[inline]
    pub fn copy_bit(&mut self, from: usize, to: usize) {
        assert!(
            to < self.length,
            "copy at index {} exceeds fixbitset size {}",
            to,
            self.length
        );
        let (to_block, t) = div_rem(to, BITS);
        let enabled = self.contains(from);
        unsafe {
            let to_elt = self.data.get_unchecked_mut(to_block);
            if enabled {
                *to_elt |= 1 << t;
            } else {
                *to_elt &= !(1 << t);
            }
        }
    }

    /// Count the number of set bits in the given bit range.
    ///
    /// Use `..` to count the whole content of the bitset.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn count_ones<T: IndexRange>(&self, range: T) -> usize {
        Masks::new(range, self.length)
            .map(|(block, mask)| unsafe {
                let value = *self.data.get_unchecked(block);
                (value & mask).count_ones() as usize
            })
            .sum()
    }

    /// Sets every bit in the given range to the given state (`enabled`)
    ///
    /// Use `..` to set the whole bitset.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn set_range<T: IndexRange>(&mut self, range: T, enabled: bool) {
        for (block, mask) in Masks::new(range, self.length) {
            unsafe {
                if enabled {
                    *self.data.get_unchecked_mut(block) |= mask;
                } else {
                    *self.data.get_unchecked_mut(block) &= !mask;
                }
            }
        }
    }

    /// Enables every bit in the given range.
    ///
    /// Use `..` to make the whole bitset ones.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn insert_range<T: IndexRange>(&mut self, range: T) {
        self.set_range(range, true);
    }

    /// Toggles (inverts) every bit in the given range.
    ///
    /// Use `..` to toggle the whole bitset.
    ///
    /// **Panics** if the range extends past the end of the bitset.
    #[inline]
    pub fn toggle_range<T: IndexRange>(&mut self, range: T) {
        for (block, mask) in Masks::new(range, self.length) {
            unsafe {
                *self.data.get_unchecked_mut(block) ^= mask;
            }
        }
    }

    /// View the bitset as a slice of `u32` blocks
    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.data
    }

    /// View the bitset as a mutable slice of `u32` blocks. Writing past the bitlength in the last
    /// will cause `contains` to return potentially incorrect results for bits past the bitlength.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        &mut self.data
    }

    /// Iterates over all enabled bits.
    ///
    /// Iterator element is the index of the `1` bit, type `usize`.
    #[inline]
    pub fn ones(&self) -> Ones {
        match self.as_slice().split_first() {
            Some((&block, rem)) => Ones {
                bitset: block,
                block_idx: 0,
                remaining_blocks: rem,
            },
            None => Ones {
                bitset: 0,
                block_idx: 0,
                remaining_blocks: &[],
            },
        }
    }

    /// Returns a lazy iterator over the intersection of two `FixedBitSet`s
    pub fn intersection<'a>(&'a self, other: &'a FixedBitSet) -> Intersection<'a> {
        Intersection {
            iter: self.ones(),
            other: other,
        }
    }

    /// Returns a lazy iterator over the union of two `FixedBitSet`s.
    pub fn union<'a>(&'a self, other: &'a FixedBitSet) -> Union<'a> {
        Union {
            iter: self.ones().chain(other.difference(self)),
        }
    }

    /// Returns a lazy iterator over the difference of two `FixedBitSet`s. The difference of `a`
    /// and `b` is the elements of `a` which are not in `b`.
    pub fn difference<'a>(&'a self, other: &'a FixedBitSet) -> Difference<'a> {
        Difference {
            iter: self.ones(),
            other: other,
        }
    }

    /// Returns a lazy iterator over the symmetric difference of two `FixedBitSet`s.
    /// The symmetric difference of `a` and `b` is the elements of one, but not both, sets.
    pub fn symmetric_difference<'a>(&'a self, other: &'a FixedBitSet) -> SymmetricDifference<'a> {
        SymmetricDifference {
            iter: self.difference(other).chain(other.difference(self)),
        }
    }

    /// In-place union of two `FixedBitSet`s.
    ///
    /// On calling this method, `self`'s capacity may be increased to match `other`'s.
    pub fn union_with(&mut self, other: &FixedBitSet) {
        if other.len() >= self.len() {
            self.grow(other.len());
        }
        for (x, y) in self.data.iter_mut().zip(other.data.iter()) {
            *x |= *y;
        }
    }

    /// In-place intersection of two `FixedBitSet`s.
    ///
    /// On calling this method, `self`'s capacity will remain the same as before.
    pub fn intersect_with(&mut self, other: &FixedBitSet) {
        for (x, y) in self.data.iter_mut().zip(other.data.iter()) {
            *x &= *y;
        }
        let mn = std::cmp::min(self.data.len(), other.data.len());
        for wd in &mut self.data[mn..] {
            *wd = 0;
        }
    }

    /// In-place difference of two `FixedBitSet`s.
    ///
    /// On calling this method, `self`'s capacity will remain the same as before.
    pub fn difference_with(&mut self, other: &FixedBitSet) {
        for (x, y) in self.data.iter_mut().zip(other.data.iter()) {
            *x &= !*y;
        }

        // There's no need to grow self or do any other adjustments.
        //
        // * If self is longer than other, the bits at the end of self won't be affected since other
        //   has them implicitly set to 0.
        // * If other is longer than self, the bits at the end of other are irrelevant since self
        //   has them set to 0 anyway.
    }

    /// In-place symmetric difference of two `FixedBitSet`s.
    ///
    /// On calling this method, `self`'s capacity may be increased to match `other`'s.
    pub fn symmetric_difference_with(&mut self, other: &FixedBitSet) {
        if other.len() >= self.len() {
            self.grow(other.len());
        }
        for (x, y) in self.data.iter_mut().zip(other.data.iter()) {
            *x ^= *y;
        }
    }

    /// Returns `true` if `self` has no elements in common with `other`. This
    /// is equivalent to checking for an empty intersection.
    pub fn is_disjoint(&self, other: &FixedBitSet) -> bool {
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(x, y)| x & y == 0)
    }

    /// Returns `true` if the set is a subset of another, i.e. `other` contains
    /// at least all the values in `self`.
    pub fn is_subset(&self, other: &FixedBitSet) -> bool {
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(x, y)| x & !y == 0)
            && self.data.iter().skip(other.data.len()).all(|x| *x == 0)
    }

    /// Returns `true` if the set is a superset of another, i.e. `self` contains
    /// at least all the values in `other`.
    pub fn is_superset(&self, other: &FixedBitSet) -> bool {
        other.is_subset(self)
    }
}

impl Binary for FixedBitSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        if f.alternate() {
            f.write_str("0b")?;
        }

        for i in 0..self.length {
            if self[i] {
                f.write_char('1')?;
            } else {
                f.write_char('0')?;
            }
        }
        Ok(())
    }
}

impl Display for FixedBitSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        Binary::fmt(&self, f)
    }
}

/// An iterator producing elements in the difference of two sets.
///
/// This struct is created by the [`FixedBitSet::difference`] method.
pub struct Difference<'a> {
    iter: Ones<'a>,
    other: &'a FixedBitSet,
}

impl<'a> Iterator for Difference<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(nxt) = self.iter.next() {
            if !self.other.contains(nxt) {
                return Some(nxt);
            }
        }
        None
    }
}

/// An iterator producing elements in the symmetric difference of two sets.
///
/// This struct is created by the [`FixedBitSet::symmetric_difference`] method.
pub struct SymmetricDifference<'a> {
    iter: Chain<Difference<'a>, Difference<'a>>,
}

impl<'a> Iterator for SymmetricDifference<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

/// An iterator producing elements in the intersection of two sets.
///
/// This struct is created by the [`FixedBitSet::intersection`] method.
pub struct Intersection<'a> {
    iter: Ones<'a>,
    other: &'a FixedBitSet,
}

impl<'a> Iterator for Intersection<'a> {
    type Item = usize; // the bit position of the '1'

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(nxt) = self.iter.next() {
            if self.other.contains(nxt) {
                return Some(nxt);
            }
        }
        None
    }
}

/// An iterator producing elements in the union of two sets.
///
/// This struct is created by the [`FixedBitSet::union`] method.
pub struct Union<'a> {
    iter: Chain<Ones<'a>, Difference<'a>>,
}

impl<'a> Iterator for Union<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

struct Masks {
    first_block: usize,
    first_mask: Block,
    last_block: usize,
    last_mask: Block,
}

impl Masks {
    #[inline]
    fn new<T: IndexRange>(range: T, length: usize) -> Masks {
        let start = range.start().unwrap_or(0);
        let end = range.end().unwrap_or(length);
        assert!(
            start <= end && end <= length,
            "invalid range {}..{} for a fixedbitset of size {}",
            start,
            end,
            length
        );

        let (first_block, first_rem) = div_rem(start, BITS);
        let (last_block, last_rem) = div_rem(end, BITS);

        Masks {
            first_block: first_block as usize,
            first_mask: Block::max_value() << first_rem,
            last_block: last_block as usize,
            last_mask: (Block::max_value() >> 1) >> (BITS - last_rem - 1),
            // this is equivalent to `MAX >> (BITS - x)` with correct semantics when x == 0.
        }
    }
}

impl Iterator for Masks {
    type Item = (usize, Block);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.first_block.cmp(&self.last_block) {
            Ordering::Less => {
                let res = (self.first_block, self.first_mask);
                self.first_block += 1;
                self.first_mask = !0;
                Some(res)
            }
            Ordering::Equal => {
                let mask = self.first_mask & self.last_mask;
                let res = if mask == 0 {
                    None
                } else {
                    Some((self.first_block, mask))
                };
                self.first_block += 1;
                res
            }
            Ordering::Greater => None,
        }
    }
}

/// An  iterator producing the indices of the set bit in a set.
///
/// This struct is created by the [`FixedBitSet::ones`] method.
pub struct Ones<'a> {
    bitset: Block,
    block_idx: usize,
    remaining_blocks: &'a [Block],
}

impl<'a> Iterator for Ones<'a> {
    type Item = usize; // the bit position of the '1'

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.bitset == 0 {
            if self.remaining_blocks.is_empty() {
                return None;
            }
            self.bitset = self.remaining_blocks[0];
            self.remaining_blocks = &self.remaining_blocks[1..];
            self.block_idx += 1;
        }
        let t = self.bitset & (0 as Block).wrapping_sub(self.bitset);
        let r = self.bitset.trailing_zeros() as usize;
        self.bitset ^= t;
        Some(self.block_idx * BITS + r)
    }
}

impl Clone for FixedBitSet {
    #[inline]
    fn clone(&self) -> Self {
        FixedBitSet {
            data: self.data.clone(),
            length: self.length,
        }
    }
}

/// Return **true** if the bit is enabled in the bitset,
/// or **false** otherwise.
///
/// Note: bits outside the capacity are always disabled, and thus
/// indexing a FixedBitSet will not panic.
impl Index<usize> for FixedBitSet {
    type Output = bool;

    #[inline]
    fn index(&self, bit: usize) -> &bool {
        if self.contains(bit) {
            &true
        } else {
            &false
        }
    }
}

/// Sets the bit at index **i** to **true** for each item **i** in the input **src**.
impl Extend<usize> for FixedBitSet {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, src: I) {
        let iter = src.into_iter();
        for i in iter {
            if i >= self.len() {
                self.grow(i + 1);
            }
            self.put(i);
        }
    }
}

/// Return a FixedBitSet containing bits set to **true** for every bit index in
/// the iterator, other bits are set to **false**.
impl FromIterator<usize> for FixedBitSet {
    fn from_iter<I: IntoIterator<Item = usize>>(src: I) -> Self {
        let mut fbs = FixedBitSet::with_capacity(0);
        fbs.extend(src);
        fbs
    }
}

impl<'a> BitAnd for &'a FixedBitSet {
    type Output = FixedBitSet;
    fn bitand(self, other: &FixedBitSet) -> FixedBitSet {
        let (short, long) = {
            if self.len() <= other.len() {
                (&self.data, &other.data)
            } else {
                (&other.data, &self.data)
            }
        };
        let mut data = short.clone();
        for (data, block) in data.iter_mut().zip(long.iter()) {
            *data &= *block;
        }
        let len = std::cmp::min(self.len(), other.len());
        FixedBitSet {
            data: data,
            length: len,
        }
    }
}

impl<'a> BitAndAssign for FixedBitSet {
    fn bitand_assign(&mut self, other: Self) {
        self.intersect_with(&other);
    }
}

impl<'a> BitAndAssign<&Self> for FixedBitSet {
    fn bitand_assign(&mut self, other: &Self) {
        self.intersect_with(other);
    }
}

impl<'a> BitOr for &'a FixedBitSet {
    type Output = FixedBitSet;
    fn bitor(self, other: &FixedBitSet) -> FixedBitSet {
        let (short, long) = {
            if self.len() <= other.len() {
                (&self.data, &other.data)
            } else {
                (&other.data, &self.data)
            }
        };
        let mut data = long.clone();
        for (data, block) in data.iter_mut().zip(short.iter()) {
            *data |= *block;
        }
        let len = std::cmp::max(self.len(), other.len());
        FixedBitSet {
            data: data,
            length: len,
        }
    }
}

impl<'a> BitOrAssign for FixedBitSet {
    fn bitor_assign(&mut self, other: Self) {
        self.union_with(&other);
    }
}

impl<'a> BitOrAssign<&Self> for FixedBitSet {
    fn bitor_assign(&mut self, other: &Self) {
        self.union_with(other);
    }
}

impl<'a> BitXor for &'a FixedBitSet {
    type Output = FixedBitSet;
    fn bitxor(self, other: &FixedBitSet) -> FixedBitSet {
        let (short, long) = {
            if self.len() <= other.len() {
                (&self.data, &other.data)
            } else {
                (&other.data, &self.data)
            }
        };
        let mut data = long.clone();
        for (data, block) in data.iter_mut().zip(short.iter()) {
            *data ^= *block;
        }
        let len = std::cmp::max(self.len(), other.len());
        FixedBitSet {
            data: data,
            length: len,
        }
    }
}

impl<'a> BitXorAssign for FixedBitSet {
    fn bitxor_assign(&mut self, other: Self) {
        self.symmetric_difference_with(&other);
    }
}

impl<'a> BitXorAssign<&Self> for FixedBitSet {
    fn bitxor_assign(&mut self, other: &Self) {
        self.symmetric_difference_with(other);
    }
}
