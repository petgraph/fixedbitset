//! **FixedBitSet** is a simple fixed size set of bits.
#![doc(html_root_url="https://docs.rs/fixedbitset/0.1/")]

mod range;

use std::ops::Index;
pub use range::IndexRange;

static TRUE: bool = true;
static FALSE: bool = false;

const BITS: usize = 32;
type Block = u32;

#[inline]
fn div_rem(x: usize, d: usize) -> (usize, usize)
{
    (x / d, x % d)
}

/// **FixedBitSet** is a simple fixed size set of bits that can
/// be enabled (1 / **true**) or disabled (0 / **false**).
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct FixedBitSet {
    data: Vec<Block>,
    /// length in bits
    length: usize,
}

impl FixedBitSet
{
    /// Create a new **FixedBitSet** with a specific number of bits,
    /// all initially clear.
    pub fn with_capacity(bits: usize) -> Self
    {
        let (mut blocks, rem) = div_rem(bits, BITS);
        blocks += (rem > 0) as usize;
        FixedBitSet {
            data: vec![0; blocks],
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

    #[inline]
    /// Return the length of the **FixedBitSet** in bits.
    pub fn len(&self) -> usize { self.length }

    /// Return **true** if the bit is enabled in the **FixedBitSet**,
    /// **false** otherwise.
    ///
    /// Note: bits outside the capacity are always disabled.
    ///
    /// Note: Also available with index syntax: `bitset[bit]`.
    #[inline]
    pub fn contains(&self, bit: usize) -> bool
    {
        let (block, i) = div_rem(bit, BITS);
        match self.data.get(block) {
            None => false,
            Some(b) => (b & (1 << i)) != 0,
        }
    }

    /// Clear all bits.
    #[inline]
    pub fn clear(&mut self)
    {
        for elt in &mut self.data[..] {
            *elt = 0
        }
    }

    /// Enable `bit`.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn insert(&mut self, bit: usize)
    {
        assert!(bit < self.length);
        let (block, i) = div_rem(bit, BITS);
        unsafe {
            *self.data.get_unchecked_mut(block) |= 1 << i;
        }
    }

    /// Enable `bit`, and return its previous value.
    ///
    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn put(&mut self, bit: usize) -> bool
    {
        assert!(bit < self.length);
        let (block, i) = div_rem(bit, BITS);
        unsafe {
            let word = self.data.get_unchecked_mut(block);
            let prev = *word & (1 << i) != 0;
            *word |= 1 << i;
            prev
        }
    }

    /// **Panics** if **bit** is out of bounds.
    #[inline]
    pub fn set(&mut self, bit: usize, enabled: bool)
    {
        assert!(bit < self.length);
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
    pub fn copy_bit(&mut self, from: usize, to: usize)
    {
        assert!(to < self.length);
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
    pub fn count_ones<T: IndexRange>(&self, range: T) -> usize
    {
        let start = range.start().unwrap_or(0);
        let end = range.end().unwrap_or(self.length);
        assert!(start <= end && end <= self.length);
        self.count_ones_impl(start, end)
    }

    #[inline]
    fn count_ones_impl(&self, start: usize, end: usize) -> usize {
        let (first_block, first_rem) = div_rem(start, BITS);
        let (last_block, last_rem) = div_rem(end, BITS);
        let mut sum = 0usize;
        // we can't skip first_block in case first_block == last_block
        for block in &self.data[first_block..last_block] {
            sum += block.count_ones() as usize;
        }
        // calculate masks; deals with overflowing shr problem when x == 0
        let mask = |x| if x != 0 { Block::max_value() >> (BITS - x) } else { 0 };
        let mask_first_block: Block = mask(first_rem);
        let mask_last_block: Block = mask(last_rem);
        sum += (self.data[last_block] & mask_last_block).count_ones() as usize;
        sum -= (self.data[first_block] & mask_first_block).count_ones() as usize;
        sum
    }

    /// View the bitset as a slice of `u32` blocks
    #[inline]
    pub fn as_slice(&self) -> &[u32]
    {
        &self.data
    }

    /// View the bitset as a mutable slice of `u32` blocks. Writing past the bitlength in the last
    /// will cause `contains` to return potentially incorrect results for bits past the bitlength.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u32]
    {
        &mut self.data
    }

    /// Iterates over all enabled bits.
    ///
    /// Iterator element is the index of the `1` bit, type `usize`.
    #[inline]
    pub fn ones(&self) -> Ones {
        match self.as_slice().split_first() {
            Some((&block, rem)) => {
                Ones {
                    current_bit_idx: 0,
                    current_block_idx: 0,
                    current_block: block,
                    remaining_blocks: rem
                }
            }
            None => {
                Ones {
                    current_bit_idx: 0,
                    current_block_idx: 0,
                    current_block: 0,
                    remaining_blocks: &[]
                }
            }
        }
    }
}

pub struct Ones<'a> {
    current_bit_idx: usize,
    current_block_idx: usize,
    remaining_blocks: &'a [Block],
    current_block: Block
}

impl<'a> Iterator for Ones<'a> {
    type Item = usize; // the bit position of the '1'

    fn next(&mut self) -> Option<Self::Item> {
        let mut block = self.current_block;
        let mut idx = self.current_bit_idx;

        loop {
            loop {
                if (block & 1) == 1 {
                    self.current_block = block >> 1;
                    self.current_bit_idx = idx + 1;
                    return Some(idx);
                }
                // reordering the two lines below makes a huge (2x) difference in performance!
                block = block >> 1;
                idx += 1;
                if block == 0 {
                    break;
                }
            }

            // go to next block
            match self.remaining_blocks.split_first() {
                Some((&next_block, rest)) => {
                    self.remaining_blocks = rest;
                    self.current_block_idx += 1;
                    idx = self.current_block_idx * BITS;
                    block = next_block;
                }
                None => {
                    // last block => done
                    return None;
                }
            }
        }
    }
}

impl Clone for FixedBitSet
{
    fn clone(&self) -> Self
    {
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
impl Index<usize> for FixedBitSet
{
    type Output = bool;

    #[inline]
    fn index(&self, bit: usize) -> &bool
    {
        if self.contains(bit) {
            &TRUE
        } else {
            &FALSE
        }
    }
}

#[test]
fn it_works() {
    const N: usize = 50;
    let mut fb = FixedBitSet::with_capacity(N);
    println!("{:?}", fb);

    for i in 0..(N + 10) {
        assert_eq!(fb.contains(i), false);
    }

    fb.insert(10);
    fb.set(11, false);
    fb.set(12, false);
    fb.set(12, true);
    fb.set(N-1, true);
    println!("{:?}", fb);
    assert!(fb.contains(10));
    assert!(!fb.contains(11));
    assert!(fb.contains(12));
    assert!(fb.contains(N-1));
    for i in 0..N {
        let contain = i == 10 || i == 12 || i == N - 1;
        assert_eq!(contain, fb[i]);
    }

    fb.clear();
}

#[test]
fn grow() {
    let mut fb = FixedBitSet::with_capacity(48);
    for i in 0..fb.len() {
        fb.set(i, true);
    }

    let old_len = fb.len();
    fb.grow(72);
    for j in 0..fb.len() {
        assert_eq!(fb.contains(j), j < old_len);
    }
    fb.set(64, true);
    assert!(fb.contains(64));
}

#[test]
fn copy_bit() {
    let mut fb = FixedBitSet::with_capacity(48);
    for i in 0..fb.len() {
        fb.set(i, true);
    }
    fb.set(42, false);
    fb.copy_bit(42, 2);
    assert!(!fb.contains(42));
    assert!(!fb.contains(2));
    assert!(fb.contains(1));
    fb.copy_bit(1, 42);
    assert!(fb.contains(42));
    fb.copy_bit(1024, 42);
    assert!(!fb[42]);
}

#[test]
fn count_ones() {
    let mut fb = FixedBitSet::with_capacity(100);
    fb.set(11, true);
    fb.set(12, true);
    fb.set(7, true);
    fb.set(35, true);
    fb.set(40, true);
    fb.set(77, true);
    fb.set(95, true);
    fb.set(50, true);
    fb.set(99, true);
    assert_eq!(fb.count_ones(..7), 0);
    assert_eq!(fb.count_ones(..8), 1);
    assert_eq!(fb.count_ones(..11), 1);
    assert_eq!(fb.count_ones(..12), 2);
    assert_eq!(fb.count_ones(..13), 3);
    assert_eq!(fb.count_ones(..35), 3);
    assert_eq!(fb.count_ones(..36), 4);
    assert_eq!(fb.count_ones(..40), 4);
    assert_eq!(fb.count_ones(..41), 5);
    assert_eq!(fb.count_ones(50..), 4);
    assert_eq!(fb.count_ones(70..95), 1);
    assert_eq!(fb.count_ones(70..96), 2);
    assert_eq!(fb.count_ones(70..99), 2);
    assert_eq!(fb.count_ones(..), 9);
    assert_eq!(fb.count_ones(0..100), 9);
    assert_eq!(fb.count_ones(0..0), 0);
    assert_eq!(fb.count_ones(100..100), 0);
}

#[test]
fn ones() {
    let mut fb = FixedBitSet::with_capacity(100);
    fb.set(11, true);
    fb.set(12, true);
    fb.set(7, true);
    fb.set(35, true);
    fb.set(40, true);
    fb.set(77, true);
    fb.set(95, true);
    fb.set(50, true);
    fb.set(99, true);

    let ones: Vec<_> = fb.ones().collect();

    assert_eq!(vec![7, 11, 12, 35, 40, 50, 77, 95, 99], ones);
}

#[test]
fn iter_ones_range() {
    fn test_range(from: usize, to: usize, capa: usize) {
        assert!(to <= capa);
        let mut fb = FixedBitSet::with_capacity(capa);
        for i in from..to {
            fb.insert(i);
        }
        let ones: Vec<_> = fb.ones().collect();
        let expected: Vec<_> = (from..to).collect();
        assert_eq!(expected, ones);
    }

    for i in 0..100 {
      test_range(i, 100, 100);
      test_range(0, i, 100);
    }
}

#[should_panic]
#[test]
fn count_ones_oob() {
    let fb = FixedBitSet::with_capacity(100);
    fb.count_ones(90..101);
}

#[should_panic]
#[test]
fn count_ones_negative_range() {
    let fb = FixedBitSet::with_capacity(100);
    fb.count_ones(90..80);
}

#[test]
fn count_ones_panic() {
    for i in 1..128 {
        let fb = FixedBitSet::with_capacity(i);
        for j in 0..fb.len() + 1 {
            for k in j..fb.len() + 1 {
                assert_eq!(fb.count_ones(j..k), 0);
            }
        }
    }
}


#[test]
fn default() {
    let fb = FixedBitSet::default();
    assert_eq!(fb.len(), 0);
}




