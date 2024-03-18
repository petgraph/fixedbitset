use core::iter::Iterator;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
#[repr(transparent)]
pub struct Block(usize);

impl Block {
    const _ASSERTION: () = {
        if core::mem::size_of::<Self>() % core::mem::size_of::<usize>() != 0 {
            panic!("vector is not a multiple size of usize");
        }
    };

    pub const USIZE_COUNT: usize = 1;
    pub const NONE: Self = Block(0);
    pub const ALL: Self = Block(!0);
    pub const BITS: usize = core::mem::size_of::<Self>() * 8;

    #[inline]
    pub fn create_buffer(iter: impl Iterator<Item = usize>) -> Vec<Self> {
        iter.map(Self).collect()
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == Self::NONE.0
    }

    #[inline]
    pub const fn count_ones(self) -> u32 {
        self.0.count_ones()
    }

    #[inline]
    pub const fn upper_mask(size: usize) -> Self {
        Self(core::usize::MAX << size)
    }

    #[inline]
    pub const fn lower_mask(size: usize) -> Self {
        Self((core::usize::MAX >> 1) >> (Self::BITS - size - 1))
    }

    #[inline]
    pub fn andnot(self, other: Self) -> Self {
        Self(!other.0 & self.0)
    }
}

impl Not for Block {
    type Output = Block;
    #[inline]
    fn not(self) -> Self::Output {
        Self(self.0.not())
    }
}

impl BitAnd for Block {
    type Output = Block;
    #[inline]
    fn bitand(self, other: Self) -> Self::Output {
        Self(self.0.bitand(other.0))
    }
}

impl BitAndAssign for Block {
    #[inline]
    fn bitand_assign(&mut self, other: Self) {
        self.0.bitand_assign(other.0);
    }
}

impl BitOr for Block {
    type Output = Block;
    #[inline]
    fn bitor(self, other: Self) -> Self::Output {
        Self(self.0.bitor(other.0))
    }
}

impl BitOrAssign for Block {
    #[inline]
    fn bitor_assign(&mut self, other: Self) {
        self.0.bitor_assign(other.0)
    }
}

impl BitXor for Block {
    type Output = Block;
    #[inline]
    fn bitxor(self, other: Self) -> Self::Output {
        Self(self.0.bitxor(other.0))
    }
}

impl BitXorAssign for Block {
    #[inline]
    fn bitxor_assign(&mut self, other: Self) {
        self.0.bitxor_assign(other.0)
    }
}
