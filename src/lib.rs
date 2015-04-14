use std::ops::Index;

static TRUE: bool = true;
static FALSE: bool = false;

const BITS: usize = 32;
type Block = u32;

#[inline]
fn div_rem(x: usize, d: usize) -> (usize, usize)
{
    (x / d, x % d)
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedBitSet {
    data: Box<[Block]>,
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
        let mut data = Vec::with_capacity(blocks);
        unsafe {
            data.set_len(blocks);
            for elt in &mut data {
                *elt = 0;
            }
        }
        FixedBitSet {
            data: data.into_boxed_slice(),
            length: bits,
        }
    }

    /// Return the length of the **FixedBitSet**.
    pub fn len(&self) -> usize { self.length }

    /// Return **true** if the bit is enabled in the **FixedBitSet**,
    /// **false** otherwise.
    ///
    /// Note: bits outside the capacity are always disabled.
    #[inline]
    pub fn contains(&self, bit: usize) -> bool
    {
        let (block, i) = div_rem(bit, BITS);
        match self.data.get(block) {
            None => false,
            Some(b) => (b & (1 << i)) != 0,
        }
    }

    #[inline]
    pub fn clear(&mut self)
    {
        for elt in &mut self.data[..] {
            *elt = 0
        }
    }

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
}

impl Clone for FixedBitSet
{
    fn clone(&self) -> Self
    {
        FixedBitSet {
            data: self.data.to_vec().into_boxed_slice(),
            length: self.length,
        }
    }
}

impl Index<usize> for FixedBitSet
{
    type Output = bool;

    /// Return **true** if the bit is enabled in the bitset,
    /// or **false** otherwise.
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
    let N = 50;
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
