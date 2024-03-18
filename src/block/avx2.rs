#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    iter::Iterator,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Block(__m256i);

impl Block {
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
        unsafe { _mm256_testz_si256(self.0, self.0) == 1 }
    }

    #[inline]
    pub fn andnot(self, other: Self) -> Self {
        Self(unsafe { _mm256_andnot_si256(other.0, self.0) })
    }
}

impl Not for Block {
    type Output = Block;
    #[inline]
    fn not(self) -> Self::Output {
        unsafe { Self(_mm256_xor_si256(self.0, Self::ALL.0)) }
    }
}

impl BitAnd for Block {
    type Output = Block;
    #[inline]
    fn bitand(self, other: Self) -> Self::Output {
        unsafe { Self(_mm256_and_si256(self.0, other.0)) }
    }
}

impl BitAndAssign for Block {
    #[inline]
    fn bitand_assign(&mut self, other: Self) {
        unsafe {
            self.0 = _mm256_and_si256(self.0, other.0);
        }
    }
}

impl BitOr for Block {
    type Output = Block;
    #[inline]
    fn bitor(self, other: Self) -> Self::Output {
        unsafe { Self(_mm256_or_si256(self.0, other.0)) }
    }
}

impl BitOrAssign for Block {
    #[inline]
    fn bitor_assign(&mut self, other: Self) {
        unsafe {
            self.0 = _mm256_or_si256(self.0, other.0);
        }
    }
}

impl BitXor for Block {
    type Output = Block;
    #[inline]
    fn bitxor(self, other: Self) -> Self::Output {
        unsafe { Self(_mm256_xor_si256(self.0, other.0)) }
    }
}

impl BitXorAssign for Block {
    #[inline]
    fn bitxor_assign(&mut self, other: Self) {
        unsafe { self.0 = _mm256_xor_si256(self.0, other.0) }
    }
}

impl PartialEq for Block {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let eq = _mm256_cmpeq_epi8(self.0, other.0);
            _mm256_movemask_epi8(eq) == !(0i32)
        }
    }
}
