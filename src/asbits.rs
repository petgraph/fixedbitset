use std::ops;

use bitvec::slice::{BitSlice, AsBits};
use bitvec::order::{BitOrder, Lsb0};

use super::FixedBitSet;
use crate::range::IndexRange;

impl AsBits for FixedBitSet {
	type Store = u32;

	fn bits<O: BitOrder>(&self) -> &BitSlice<O, Self::Store> {
		BitSlice::from_slice(self.as_slice())
	}

	fn bits_mut<O: BitOrder>(&mut self) -> &mut BitSlice<O, Self::Store> {
		BitSlice::from_slice_mut(self.as_mut_slice())
	}
}

impl ops::Deref for FixedBitSet {
	type Target = BitSlice<Lsb0, u32>;
	fn deref(&self) -> &Self::Target {
		self.bits()
	}
}

impl ops::DerefMut for FixedBitSet {
	fn deref_mut(&mut self) -> &mut Self::Target {
		self.bits_mut()
	}
}

impl<R: IndexRange> ops::Index<R> for FixedBitSet {
	type Output = BitSlice<Lsb0, u32>;
	fn index(&self, range: R) -> &Self::Output {
		let start = range.start().unwrap_or(0);
		let end = range.end().unwrap_or(self.len());
		&self.bits()[start..end]
	}
}

impl<R: IndexRange> ops::IndexMut<R> for FixedBitSet {
	fn index_mut(&mut self, range: R) -> &mut Self::Output {
		let start = range.start().unwrap_or(0);
		let end = range.end().unwrap_or(self.len());
		&mut self.bits_mut()[start..end]
	}
}

#[test]
fn test_asbits() {
	let mut fb = FixedBitSet::with_capacity(50);
    fb.set(11, true);
    fb.set(12, true);
    fb.set(7, true);
    fb.set(35, true);
    fb.set(40, true);

	assert_eq!(fb.get(5), Some(&false));
	assert_eq!(fb[10..].get(1), Some(&true));
	fb[10..].set(2, false);
	assert_eq!(fb.get(12), Some(&false));

	assert_eq!(fb.count_ones(..), 4);
}
