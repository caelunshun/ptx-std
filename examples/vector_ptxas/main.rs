// --- LLM-generated --- //
//! Instantiates the floating-point vector types of `ptx_std` in kernels, so that the code and the
//! inline PTX they generate can be validated by `ptxas` for several targets. No GPU is needed.
//!
//! Run `examples/vector_ptxas/check.sh [arch...]` from anywhere; it emits the PTX of this crate
//! for each target (default: `sm_80 sm_89 sm_90 sm_100a sm_103a sm_120a`) and assembles it with
//! `ptxas`.
//!
//! - `main.rs`: the container API of every vector type, written once over the element type and
//!   the length and instantiated per width by `container_kernels!` (which the other modules may
//!   reuse).
//! - `half.rs`: `f16` and `bf16` vectors, with the same structure: generic functions over the
//!   length, and thin kernels that instantiate them per width.
//! - `generic.rs`: kernels written the way users will write code generic over the vector length.
//! - `narrow.rs`: conversions of the narrow formats.
//! - `f32.rs`: `F32Ext`.
//! - `warp.rs`: warp collectives (shuffles, matches and reductions) on the vector types.
//!
//! Every kernel takes the operands as pointers so that nothing is constant-folded. Variants that
//! are only available on some targets are instantiated under the same `cfg`s as in the library.
//! Note that the `f32` module shadows the primitive type in this file, so it is not used here.

#![no_std]
#![feature(abi_ptx, portable_simd)]
#![allow(clippy::missing_safety_doc)]

use ptx_std::*;

/// Exercises the container API of `Vector<E, N>`: raw bits, elements, halves, pairs and arrays.
/// Writes the raw bits, the elements and the halves of `v`, and the vectors rebuilt from
/// `bits_in`, `arr_in`, `elem`, the swapped halves and the reversed pairs.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn container<E: VectorElement, const N: usize>(
    src: *const Vector<E, N>,
    elem: *const E,
    bits_in: *const Bits<E, N>,
    arr_in: *const [E; N],
    out: *mut [Vector<E, N>; 5],
    halves_out: *mut [Half<E, N>; 2],
    bits_out: *mut Bits<E, N>,
    arr_out: *mut [E; N],
) where
    VectorLen<N>: SupportedVectorLen,
{
    unsafe {
        let v = *src;
        *bits_out = v.to_bits();
        *arr_out = v.to_array();
        *halves_out = [v.lo(), v.hi()];
        let mut x2s = v.to_x2s();
        x2s.reverse();
        *out = [
            Vector::from_bits(*bits_in),
            Vector::from_array(*arr_in),
            Vector::splat(*elem),
            Vector::new(v.hi(), v.lo()),
            Vector::from_x2s(x2s),
        ];
    }
}

/// Instantiates [`container`] for the vectors of `$elem` of the given lengths, one kernel each.
macro_rules! container_kernels {
    ($elem:ident: $($name:ident => $n:literal),* $(,)?) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            src: *const Vector<$elem, $n>,
            elem: *const $elem,
            bits_in: *const Bits<$elem, $n>,
            arr_in: *const [$elem; $n],
            out: *mut [Vector<$elem, $n>; 5],
            halves_out: *mut [Half<$elem, $n>; 2],
            bits_out: *mut Bits<$elem, $n>,
            arr_out: *mut [$elem; $n],
        ) {
            unsafe {
                $crate::container::<$elem, $n>(
                    src,
                    elem,
                    bits_in,
                    arr_in,
                    out,
                    halves_out,
                    bits_out,
                    arr_out,
                )
            }
        }
    )*};
}

mod f32;
mod generic;
mod half;
mod narrow;
mod warp;

container_kernels!(
    f8e4m3:
    f8e4m3x2_container => 2,
    f8e4m3x4_container => 4,
    f8e4m3x8_container => 8,
    f8e4m3x16_container => 16,
    f8e4m3x32_container => 32,
);
container_kernels!(
    f8e5m2:
    f8e5m2x2_container => 2,
    f8e5m2x4_container => 4,
    f8e5m2x8_container => 8,
    f8e5m2x16_container => 16,
    f8e5m2x32_container => 32,
);
container_kernels!(
    f8e8m0:
    f8e8m0x2_container => 2,
    f8e8m0x4_container => 4,
    f8e8m0x8_container => 8,
    f8e8m0x16_container => 16,
    f8e8m0x32_container => 32,
);
container_kernels!(
    f6e2m3:
    f6e2m3x2_container => 2,
    f6e2m3x4_container => 4,
    f6e2m3x8_container => 8,
    f6e2m3x16_container => 16,
    f6e2m3x32_container => 32,
);
container_kernels!(
    f6e3m2:
    f6e3m2x2_container => 2,
    f6e3m2x4_container => 4,
    f6e3m2x8_container => 8,
    f6e3m2x16_container => 16,
    f6e3m2x32_container => 32,
);
container_kernels!(
    s2f6:
    s2f6x2_container => 2,
    s2f6x4_container => 4,
    s2f6x8_container => 8,
    s2f6x16_container => 16,
    s2f6x32_container => 32,
);
container_kernels!(
    f4e2m1:
    f4e2m1x2_container => 2,
    f4e2m1x4_container => 4,
    f4e2m1x8_container => 8,
    f4e2m1x16_container => 16,
    f4e2m1x32_container => 32,
);
