#![allow(non_camel_case_types)]

mod bfloat16;
mod f32_ext;
mod f64_ext;
mod float16;
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
pub use int_ext::*;
pub use s2f6_format::*;

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
