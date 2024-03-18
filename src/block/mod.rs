use core::cmp::Ordering;
use core::hash::{Hash, Hasher};

#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_feature = "sse2"),
    not(target_feature = "avx2"),
))]
mod default;
#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_feature = "sse2"),
    not(target_feature = "avx2"),
))]
pub use self::default::*;

#[cfg(all(
    not(target_arch = "wasm32"),
    target_feature = "sse2",
    not(target_feature = "avx2"),
))]
mod sse2;
#[cfg(all(
    not(target_arch = "wasm32"),
    target_feature = "sse2",
    not(target_feature = "avx2"),
))]
pub use self::sse2::*;

#[cfg(all(not(target_arch = "wasm32"), target_feature = "avx2",))]
mod avx2;
#[cfg(all(not(target_arch = "wasm32"), target_feature = "avx2",))]
pub use self::avx2::*;

#[cfg(target_arch = "wasm32")]
mod wasm32;
#[cfg(target_arch = "wasm32")]
pub use self::wasm32::*;

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