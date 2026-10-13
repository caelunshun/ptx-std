// --- LLM-generated --- //
//! Code generic over the vector length (and, where it fits, the element type), written the way
//! users write it: a function with a `const N: usize` parameter and the bound
//! `VectorLen<N>: SupportedVectorLen`, instantiated by thin kernels for a couple of widths.

use core::simd::Simd;
use ptx_std::{
    F32Ext, SupportedVectorLen, Vector, VectorElement, VectorLen, bf16xN, f4e2m1, f8e4m3, f16xN,
};

/// Converts `f32` values to `bf16`, rounding to nearest, ties to even.
#[inline(always)]
fn f32_to_bf16<const N: usize>(x: Simd<f32, N>) -> bf16xN<N>
where
    VectorLen<N>: SupportedVectorLen,
{
    bf16xN::from_f32(x)
}

/// The dot product of two `f16` vectors, accumulated in `f32`.
#[inline(always)]
fn dot<const N: usize>(a: f16xN<N>, b: f16xN<N>) -> f32
where
    VectorLen<N>: SupportedVectorLen,
{
    (a.to_f32() * b.to_f32()).reduce_sum()
}

/// The sum of squares of `count` vectors, with a vector accumulator of `f32` and one horizontal
/// reduction at the end.
#[inline(always)]
unsafe fn sum_squares<const N: usize>(vectors: *const f16xN<N>, count: usize) -> f32
where
    VectorLen<N>: SupportedVectorLen,
{
    let mut acc = Simd::<f32, N>::splat(0.0);
    for i in 0..count {
        let x = unsafe { vectors.add(i).read() }.to_f32();
        acc += x * x;
    }
    acc.reduce_sum()
}

/// Converts `f32` values to `bf16` with stochastic rounding, using one word of pseudo-random bits
/// per pair of elements derived from `seed` (a multiplicative hash, not a quality generator).
#[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
#[inline(always)]
fn f32_to_bf16_stochastic<const N: usize>(x: Simd<f32, N>, seed: u32) -> bf16xN<N>
where
    VectorLen<N>: SupportedVectorLen,
{
    use ptx_std::{Overflow, Pairs, Relu};

    let rbits: Pairs<u32, N> =
        core::array::from_fn(|i| (seed ^ i as u32).wrapping_mul(0x9e37_79b9).rotate_left(13));
    bf16xN::from_f32_stochastic::<{ Relu::Disabled }, { Overflow::Infinity }>(x, rbits)
}

/// Swaps the halves of a vector of any element type.
#[inline(always)]
fn swap_halves<E: VectorElement, const N: usize>(v: Vector<E, N>) -> Vector<E, N>
where
    VectorLen<N>: SupportedVectorLen,
{
    Vector::new(v.hi(), v.lo())
}

/// Reverses the order of the pairs of a vector of any element type.
#[inline(always)]
fn reverse_pairs<E: VectorElement, const N: usize>(v: Vector<E, N>) -> Vector<E, N>
where
    VectorLen<N>: SupportedVectorLen,
{
    let mut pairs = v.to_x2s();
    pairs.reverse();
    Vector::from_x2s(pairs)
}

macro_rules! kernels {
    ($n:literal: $convert:ident, $dot:ident, $sum_squares:ident, $stochastic:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $convert(x: *const Simd<f32, $n>, out: *mut bf16xN<$n>) {
            unsafe { *out = f32_to_bf16(*x) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $dot(
            a: *const f16xN<$n>,
            b: *const f16xN<$n>,
            out: *mut f32,
        ) {
            unsafe { *out = dot(*a, *b) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $sum_squares(
            vectors: *const f16xN<$n>,
            count: usize,
            out: *mut f32,
        ) {
            unsafe { *out = sum_squares(vectors, count) }
        }

        #[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $stochastic(
            x: *const Simd<f32, $n>,
            seed: u32,
            out: *mut bf16xN<$n>,
        ) {
            unsafe { *out = f32_to_bf16_stochastic(*x, seed) }
        }
    };
}

kernels!(8: generic_f32_to_bf16_x8, generic_dot_x8, generic_sum_squares_x8, generic_stochastic_x8);
kernels!(
    32: generic_f32_to_bf16_x32,
    generic_dot_x32,
    generic_sum_squares_x32,
    generic_stochastic_x32
);

macro_rules! element_kernels {
    ($elem:ident, $n:literal: $swap:ident, $reverse:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $swap(
            v: *const Vector<$elem, $n>,
            out: *mut Vector<$elem, $n>,
        ) {
            unsafe { *out = swap_halves(*v) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $reverse(
            v: *const Vector<$elem, $n>,
            out: *mut Vector<$elem, $n>,
        ) {
            unsafe { *out = reverse_pairs(*v) }
        }
    };
}

element_kernels!(f8e4m3, 16: generic_swap_halves_f8e4m3x16, generic_reverse_pairs_f8e4m3x16);
element_kernels!(f4e2m1, 32: generic_swap_halves_f4e2m1x32, generic_reverse_pairs_f4e2m1x32);
