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
