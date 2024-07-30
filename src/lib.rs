mod arithmetic;
pub mod ff_ext;
pub mod fft;
pub mod hash_to_curve;
pub mod msm;
pub mod multicore;
pub mod serde;

pub mod bls12_381;
pub mod bn256;
pub mod ed25519;
pub mod grumpkin;
pub mod pasta;
pub mod pluto_eris;
pub mod secp256k1;
pub mod secp256r1;
pub mod secq256k1;

#[macro_use]
mod derive;

// Re-export to simplify down stream dependencies
pub use arithmetic::CurveAffineExt;
pub use ff;
pub use group;
pub use pairing;
pub use pasta_curves::arithmetic::{Coordinates, CurveAffine, CurveExt};

#[cfg(test)]
pub mod tests;

#[cfg(all(feature = "prefetch", target_arch = "x86_64"))]
#[inline(always)]
pub fn prefetch<T>(data: &[T], offset: usize) {
    use core::arch::x86_64::_mm_prefetch;
    unsafe {
        _mm_prefetch(
            data.as_ptr().add(offset) as *const i8,
            core::arch::x86_64::_MM_HINT_T0,
        );
    }
}

#[cfg(feature = "gpu")]
fn u64_to_u32(limbs: &[u64]) -> Vec<u32> {
    limbs
        .iter()
        .flat_map(|limb| vec![(limb & 0xFFFF_FFFF) as u32, (limb >> 32) as u32])
        .collect()
}