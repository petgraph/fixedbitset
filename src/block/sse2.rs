use alloc::vec::Vec;
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
pub struct Block(__m128i);

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
    pub fn into_usize_array(self) -> [usize; Self::USIZE_COUNT] {
        unsafe { core::mem::transmute(self.0) }
    }

    #[inline]
    pub const fn from_usize_array(array: [usize; Self::USIZE_COUNT]) -> Self {
        Self(unsafe { core::mem::transmute(array) })
    }

    #[inline]
    pub fn create_buffer(iter: impl Iterator<Item = usize>) -> Vec<Self> {
        let (lower, _) = iter.size_hint();
        let mut output = Vec::with_capacity(lower / Self::USIZE_COUNT);
        let mut buffer = [0; Self::USIZE_COUNT];
        let mut index = 0;
        for chunk in iter {
            buffer[index] = chunk;
            index += 1;
            if index >= Self::USIZE_COUNT {
                output.push(Self::from_usize_array(buffer));
                index = 0;
            }
        }
        if index != 0 {
            #[allow(clippy::needless_range_loop)]
            for idx in index..Self::USIZE_COUNT {
                buffer[idx] = 0;
            }
            output.push(Self::from_usize_array(buffer));
        }
        output
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        #[cfg(not(target_feature = "sse4.1"))]
        {
            self == Self::NONE
        }
        #[cfg(target_feature = "sse4.1")]
        {
            unsafe { _mm_test_all_zeros(self.0, Self::ALL.0) == 1 }
        }
    }

    #[inline]
    pub fn count_ones(self) -> u32 {
        unsafe {
            let array: [usize; Self::USIZE_COUNT] = core::mem::transmute(self.0);
            array.iter().copied().map(usize::count_ones).sum()
        }
    }

    #[inline]
    pub const fn upper_mask(size: usize) -> Self {
        unsafe { Self(core::mem::transmute(core::u128::MAX << size)) }
    }

    #[inline]
    pub const fn lower_mask(size: usize) -> Self {
        unsafe {
            Self(core::mem::transmute(
                (core::u128::MAX >> 1) >> (Self::BITS - size - 1),
            ))
        }
    }

    #[inline]
    pub fn andnot(self, other: Self) -> Self {
        Self(unsafe { _mm_andnot_si128(other.0, self.0) })
    }
}

impl Not for Block {
    type Output = Block;
    #[inline]
    fn not(self) -> Self::Output {
        unsafe { Self(_mm_xor_si128(self.0, Self::ALL.0)) }
    }
}

impl BitAnd for Block {
    type Output = Block;
    #[inline]
    fn bitand(self, other: Self) -> Self::Output {
        unsafe { Self(_mm_and_si128(self.0, other.0)) }
    }
}

impl BitAndAssign for Block {
    #[inline]
    fn bitand_assign(&mut self, other: Self) {
        unsafe {
            self.0 = _mm_and_si128(self.0, other.0);
        }
    }
}

impl BitOr for Block {
    type Output = Block;
    #[inline]
    fn bitor(self, other: Self) -> Self::Output {
        unsafe { Self(_mm_or_si128(self.0, other.0)) }
    }
}

impl BitOrAssign for Block {
    #[inline]
    fn bitor_assign(&mut self, other: Self) {
        unsafe {
            self.0 = _mm_or_si128(self.0, other.0);
        }
    }
}

impl BitXor for Block {
    type Output = Block;
    #[inline]
    fn bitxor(self, other: Self) -> Self::Output {
        unsafe { Self(_mm_xor_si128(self.0, other.0)) }
    }
}

impl BitXorAssign for Block {
    #[inline]
    fn bitxor_assign(&mut self, other: Self) {
        unsafe { self.0 = _mm_xor_si128(self.0, other.0) }
    }
}

impl PartialEq for Block {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            #[cfg(not(target_feature = "sse4.1"))]
            {
                _mm_movemask_epi8(_mm_cmpeq_epi8(self.0, other.0)) == 0xffff
            }
            #[cfg(target_feature = "sse4.1")]
            {
                let neq = _mm_xor_si128(self.0, other.0);
                _mm_test_all_zeros(neq, neq) == 1
            }
        }
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
