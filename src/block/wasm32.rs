use core::{
    arch::wasm32::*,
    cmp::Ordering,
    hash::{Hash, Hasher},
    iter::Iterator,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Block(v128);

impl Block {
    const _ASSERTION: () = {
        if core::mem::size_of::<Self>() % core::mem::size_of::<usize>() != 0 {
            panic!("vector is not a multiple size of usize");
        }
    };

    pub const USIZE_COUNT: usize = core::mem::size_of::<Self>() / core::mem::size_of::<usize>();
    pub const NONE: Self = Self::from_usize_array([0; Self::USIZE_COUNT]);
    pub const ALL: Self = Self::from_usize_array([core::usize::MAX; Self::USIZE_COUNT]);
    pub const BITS: usize = core::mem::size_of::<Self>() * 8;

    #[inline]
    fn into_usize_array(self) -> [usize; Self::USIZE_COUNT] {
        unsafe { core::mem::transmute(self.0) }
    }

    #[inline]
    const fn from_usize_array(array: [usize; Self::USIZE_COUNT]) -> Self {
        Self(unsafe { core::mem::transmute(array) })
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        !v128_any_true(self.0)
    }

    #[inline]
    pub fn andnot(self, other: Self) -> Self {
        Self(unsafe { v128_andnot(self.0, other.0) })
    }
}

impl Not for Block {
    type Output = Block;
    #[inline]
    fn not(self) -> Self::Output {
        Self(v128_xor(self.0, Self::ALL.0))
    }
}

impl BitAnd for Block {
    type Output = Block;
    #[inline]
    fn bitand(self, other: Self) -> Self::Output {
        Self(v128_and(self.0, other.0))
    }
}

impl BitAndAssign for Block {
    #[inline]
    fn bitand_assign(&mut self, other: Self) {
        self.0 = v128_and(self.0, other.0);
    }
}

impl BitOr for Block {
    type Output = Block;
    #[inline]
    fn bitor(self, other: Self) -> Self::Output {
        Self(v128_or(self.0, other.0))
    }
}

impl BitOrAssign for Block {
    #[inline]
    fn bitor_assign(&mut self, other: Self) {
        self.0 = v128_or(self.0, other.0);
    }
}

impl BitXor for Block {
    type Output = Block;
    #[inline]
    fn bitxor(self, other: Self) -> Self::Output {
        Self(v128_xor(self.0, other.0))
    }
}

impl BitXorAssign for Block {
    #[inline]
    fn bitxor_assign(&mut self, other: Self) {
        self.0 = v128_xor(self.0, other.0)
    }
}

impl PartialEq for Block {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        !v128_any_true(v128_xor(self.0, other.0))
    }
}

impl Eq for Block {}

impl PartialOrd for Block {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let a = self.into_usize_array();
        let b = other.into_usize_array();
        for i in 0..Self::USIZE_COUNT {
            match a[i].cmp(&b[i]) {
                Ordering::Equal => continue,
                cmp => return Some(cmp),
            }
        }
        Some(Ordering::Equal)
    }
}

impl Ord for Block {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        let a = self.into_usize_array();
        let b = other.into_usize_array();
        for i in 0..Self::USIZE_COUNT {
            match a[i].cmp(&b[i]) {
                Ordering::Equal => continue,
                cmp => return cmp,
            }
        }
        Ordering::Equal
    }
}

impl Default for Block {
    #[inline]
    fn default() -> Self {
        Self::NONE
    }
}

impl Hash for Block {
    #[inline]
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.into_usize_array().hash(hasher)
    }
}
