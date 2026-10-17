#![allow(non_camel_case_types)]

mod bfloat16;
mod f32_ext;
mod f64_ext;
mod float16;
mod float32;
mod float4;
mod float6;
mod float8;
mod int_ext;
mod narrow;
// Named differently from the file, as a module cannot share the name of the `s2f6` type.
#[path = "s2f6.rs"]
mod s2f6_format;
mod vector;

pub use bfloat16::*;
pub use f32_ext::*;
pub use f64_ext::*;
pub use float4::*;
pub use float6::*;
pub use float8::*;
pub use float16::*;
pub use float32::*;
pub use int_ext::*;
pub use s2f6_format::*;
pub use vector::*;

use core::mem::MaybeUninit;

/// Like [`core::array::from_fn`], but the loop is fully unrolled, so that arrays of register
/// values are never placed in local memory (LLVM's own unrolling gives up on long loops).
#[inline(always)]
pub(crate) fn array_from_fn<T: Copy, const N: usize>(mut f: impl FnMut(usize) -> T) -> [T; N] {
    let mut out = [const { MaybeUninit::<T>::uninit() }; N];
    #[rustc_unroll(full)]
    for (i, slot) in out.iter_mut().enumerate() {
        slot.write(f(i));
    }
    // SAFETY: every element has been initialized, and `MaybeUninit<T>` has the layout of `T`.
    unsafe { core::mem::transmute_copy(&out) }
}

/// Elements `2 * i` and `2 * i + 1` of `x`, the second one being zero past the end of an odd
/// length. The `f32` operands of the packed instructions are read this way rather than with
/// `Vector::pair`, which reads the pair as one `u64` that LLVM then unpacks with extra moves.
/// Unused on the targets without the packed `f32` instructions and the narrow conversions.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn f32_pair<const N: usize>(x: &[f32; N], i: usize) -> [f32; 2] {
    [x[2 * i], x.get(2 * i + 1).copied().unwrap_or(0.0)]
}

/// Elements `4 * i` to `4 * i + 3` of `x`, zero past the end of the array.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn f32_quad<const N: usize>(x: &[f32; N], i: usize) -> [f32; 4] {
    core::array::from_fn(|j| x.get(4 * i + j).copied().unwrap_or(0.0))
}
