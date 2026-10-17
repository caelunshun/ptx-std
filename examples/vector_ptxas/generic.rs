// --- LLM-generated --- //
//! Code generic over the vector length (and, where it fits, the element type), written the way
//! users write it: a function with a `const N: usize` parameter, with no bound on the length of a
//! concrete format unless the API needs one (`VectorLen<N>: SupportedVectorLen` for the types
//! that depend on `N / 2`, such as the random bits of the stochastic conversions, and
//! `EvenVectorLen` for the halves), instantiated by thin kernels for a few widths, among them
//! lengths that are not powers of two.

use ptx_std::{
    EvenVectorElement, EvenVectorLen, F32Ext, SupportedVectorLen, Vector, VectorElement, VectorLen,
    bf16xN, f4e2m1, f8e4m3, f16, f16xN, f32xN,
};

/// Converts `f32` values to `bf16`, rounding to nearest, ties to even.
#[inline(always)]
fn f32_to_bf16<const N: usize>(x: f32xN<N>) -> bf16xN<N> {
    bf16xN::from_f32(x)
}

/// The dot product of two `f16` vectors, accumulated in `f32`.
#[inline(always)]
fn dot<const N: usize>(a: f16xN<N>, b: f16xN<N>) -> f32 {
    (a.to_f32() * b.to_f32()).reduce_sum()
}

/// The sum of squares of `count` vectors, with a vector accumulator of `f32` and one horizontal
/// reduction at the end.
#[inline(always)]
unsafe fn sum_squares<const N: usize>(vectors: *const f16xN<N>, count: usize) -> f32 {
    let mut acc = f32xN::<N>::ZERO;
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
fn f32_to_bf16_stochastic<const N: usize>(x: f32xN<N>, seed: u32) -> bf16xN<N>
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
fn swap_halves<E: EvenVectorElement<N>, const N: usize>(v: Vector<E, N>) -> Vector<E, N>
where
    VectorLen<N>: EvenVectorLen,
{
    Vector::<E, N>::new(v.hi(), v.lo())
}

/// Reverses the order of the pairs of a vector of any element type.
#[inline(always)]
fn reverse_pairs<E, const N: usize>(v: Vector<E, N>) -> Vector<E, N>
where
    E: VectorElement<N> + VectorElement<2>,
    VectorLen<N>: SupportedVectorLen,
{
    let mut pairs = v.to_x2s();
    pairs.reverse();
    Vector::from_x2s(pairs)
}

/// Rebuilds a vector of any element type from its elements, which needs no more than
/// `E: VectorElement<N>`.
#[inline(always)]
fn roundtrip_array<E: VectorElement<N>, const N: usize>(v: Vector<E, N>) -> Vector<E, N> {
    Vector::from_array(v.to_array())
}

macro_rules! kernels {
    ($n:literal: $convert:ident, $dot:ident, $sum_squares:ident, $stochastic:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $convert(x: *const f32xN<$n>, out: *mut bf16xN<$n>) {
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
            x: *const f32xN<$n>,
            seed: u32,
            out: *mut bf16xN<$n>,
        ) {
            unsafe { *out = f32_to_bf16_stochastic(*x, seed) }
        }
    };
}

kernels!(1: generic_f32_to_bf16_x1, generic_dot_x1, generic_sum_squares_x1, generic_stochastic_x1);
kernels!(3: generic_f32_to_bf16_x3, generic_dot_x3, generic_sum_squares_x3, generic_stochastic_x3);
kernels!(6: generic_f32_to_bf16_x6, generic_dot_x6, generic_sum_squares_x6, generic_stochastic_x6);
kernels!(8: generic_f32_to_bf16_x8, generic_dot_x8, generic_sum_squares_x8, generic_stochastic_x8);
kernels!(
    32: generic_f32_to_bf16_x32,
    generic_dot_x32,
    generic_sum_squares_x32,
    generic_stochastic_x32
);

/// Instantiates [`dot`] with the operands passed by value, as a check of the lint on vectors in
/// the signature of a kernel (a 12-byte vector, aligned to 4 bytes).
#[allow(improper_ctypes_definitions)]
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn generic_dot_by_value_x6(a: f16xN<6>, b: f16xN<6>, out: *mut f32) {
    unsafe { *out = dot(a, b) }
}

macro_rules! swap_halves_kernel {
    ($elem:ident, $n:literal: $name:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            v: *const Vector<$elem, $n>,
            out: *mut Vector<$elem, $n>,
        ) {
            unsafe { *out = swap_halves(*v) }
        }
    };
}

macro_rules! reverse_pairs_kernel {
    ($elem:ident, $n:literal: $name:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            v: *const Vector<$elem, $n>,
            out: *mut Vector<$elem, $n>,
        ) {
            unsafe { *out = reverse_pairs(*v) }
        }
    };
}

macro_rules! roundtrip_array_kernel {
    ($elem:ident, $n:literal: $name:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            v: *const Vector<$elem, $n>,
            out: *mut Vector<$elem, $n>,
        ) {
            unsafe { *out = roundtrip_array(*v) }
        }
    };
}

swap_halves_kernel!(f8e4m3, 6: generic_swap_halves_f8e4m3x6);
swap_halves_kernel!(f8e4m3, 16: generic_swap_halves_f8e4m3x16);
swap_halves_kernel!(f4e2m1, 12: generic_swap_halves_f4e2m1x12);
swap_halves_kernel!(f4e2m1, 32: generic_swap_halves_f4e2m1x32);

reverse_pairs_kernel!(f8e4m3, 3: generic_reverse_pairs_f8e4m3x3);
reverse_pairs_kernel!(f8e4m3, 6: generic_reverse_pairs_f8e4m3x6);
reverse_pairs_kernel!(f8e4m3, 16: generic_reverse_pairs_f8e4m3x16);
reverse_pairs_kernel!(f4e2m1, 6: generic_reverse_pairs_f4e2m1x6);
reverse_pairs_kernel!(f4e2m1, 32: generic_reverse_pairs_f4e2m1x32);

roundtrip_array_kernel!(f16, 3: generic_roundtrip_array_f16x3);
roundtrip_array_kernel!(f8e4m3, 6: generic_roundtrip_array_f8e4m3x6);
roundtrip_array_kernel!(f4e2m1, 6: generic_roundtrip_array_f4e2m1x6);
