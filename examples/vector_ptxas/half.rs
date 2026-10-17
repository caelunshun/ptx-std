// --- LLM-generated --- //
//! `f16` and `bf16` vectors: container API, `f32` conversions, arithmetic, operators and
//! horizontal reductions. The operations are generic functions over the length (`f16_fns` and
//! `bf16_fns`), which the kernels instantiate once per width.

use ptx_std::{
    Bits, Denormals, Half, NanHandling, Overflow, Relu, Rounding, Saturation, Vector, XorSignAbs,
    bf16, f16, f32xN,
};

#[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
use ptx_std::{Pairs, SupportedVectorLen, VectorLen};

container_kernels!(
    f16:
    f16x4_container => 4,
    f16x6_container => 6,
    f16x8_container => 8,
    f16x12_container => 12,
    f16x16_container => 16,
    f16x32_container => 32,
);
container_basic_kernels!(
    f16:
    f16x1_container_basic => 1,
    f16x2_container_basic => 2,
    f16x3_container_basic => 3,
);
container_kernels!(
    bf16:
    bf16x4_container => 4,
    bf16x6_container => 6,
    bf16x8_container => 8,
    bf16x12_container => 12,
    bf16x16_container => 16,
    bf16x32_container => 32,
);
container_basic_kernels!(
    bf16:
    bf16x1_container_basic => 1,
    bf16x2_container_basic => 2,
    bf16x3_container_basic => 3,
);

/// Defines what `f16` and `bf16` vectors share, generic over the length: `from_f32`,
/// `from_f32_with` (several modifier combinations), `to_f32`, `from_f32_stochastic`, the
/// scalar-broadcast operators and `reduce_sum`.
macro_rules! half_shared {
    ($elem:ident) => {
        #[inline(always)]
        pub fn convert<const N: usize>(
            x: f32xN<N>,
            v: Vector<$elem, N>,
        ) -> ([Vector<$elem, N>; 5], f32xN<N>)
        {
            (
                [
                    Vector::<$elem, N>::from_f32(x),
                    Vector::<$elem, N>::from_f32_with::<
                        { Rounding::TowardZero },
                        { Relu::Disabled },
                        { Overflow::Infinity },
                    >(x),
                    Vector::<$elem, N>::from_f32_with::<
                        { Rounding::NearestEven },
                        { Relu::Enabled },
                        { Overflow::Infinity },
                    >(x),
                    Vector::<$elem, N>::from_f32_with::<
                        { Rounding::NearestEven },
                        { Relu::Disabled },
                        { Overflow::SaturateFinite },
                    >(x),
                    Vector::<$elem, N>::from_f32_with::<
                        { Rounding::TowardZero },
                        { Relu::Enabled },
                        { Overflow::SaturateFinite },
                    >(x),
                ],
                v.to_f32(),
            )
        }

        #[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
        #[inline(always)]
        pub fn stochastic<const N: usize>(
            x: f32xN<N>,
            rbits: Pairs<u32, N>,
        ) -> [Vector<$elem, N>; 4]
        where
            VectorLen<N>: SupportedVectorLen,
        {
            [
                Vector::<$elem, N>::from_f32_stochastic::<{ Relu::Disabled }, { Overflow::Infinity }>(x, rbits),
                Vector::<$elem, N>::from_f32_stochastic::<{ Relu::Enabled }, { Overflow::Infinity }>(x, rbits),
                Vector::<$elem, N>::from_f32_stochastic::<{ Relu::Disabled }, { Overflow::SaturateFinite }>(
                    x, rbits,
                ),
                Vector::<$elem, N>::from_f32_stochastic::<{ Relu::Enabled }, { Overflow::SaturateFinite }>(
                    x, rbits,
                ),
            ]
        }

        #[inline(always)]
        pub fn broadcast<const N: usize>(
            v: Vector<$elem, N>,
            s: $elem,
        ) -> ([Vector<$elem, N>; 7], f32xN<N>)
        {
            let mut acc = v;
            acc += s;
            acc -= s;
            acc *= s;
            (
                [v + s, v - s, v * s, s + v, s - v, s * v, acc],
                f32xN::from(v),
            )
        }

        #[inline(always)]
        pub fn sum<const N: usize>(a: Vector<$elem, N>) -> f32
        {
            a.reduce_sum()
        }
    };
}

/// Instantiates the shared kernels of the vectors of `$elem` with `$n` elements, defined in
/// `$fns`: the conversions, the scalar-broadcast operators and `reduce_sum`.
macro_rules! half_kernels {
    (
        $elem:ident, $fns:ident, $n:literal,
        convert: $convert:ident, stochastic: $stochastic:ident, broadcast: $broadcast:ident,
        sum: $sum:ident $(,)?
    ) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $convert(
            x: *const f32xN<$n>,
            v: *const Vector<$elem, $n>,
            out: *mut [Vector<$elem, $n>; 5],
            out_f32: *mut f32xN<$n>,
        ) {
            unsafe { (*out, *out_f32) = $fns::convert::<$n>(*x, *v) }
        }

        #[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $stochastic(
            x: *const f32xN<$n>,
            rbits: *const Pairs<u32, $n>,
            out: *mut [Vector<$elem, $n>; 4],
        ) {
            unsafe { *out = $fns::stochastic::<$n>(*x, *rbits) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $broadcast(
            v: *const Vector<$elem, $n>,
            s: *const $elem,
            out: *mut [Vector<$elem, $n>; 7],
            out_f32: *mut f32xN<$n>,
        ) {
            unsafe { (*out, *out_f32) = $fns::broadcast::<$n>(*v, *s) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $sum(a: *const Vector<$elem, $n>, out: *mut f32) {
            unsafe { *out = $fns::sum::<$n>(*a) }
        }
    };
}

mod f16_fns {
    use super::*;

    half_shared!(f16);

    #[inline(always)]
    pub fn arith<const N: usize>(a: Vector<f16, N>, b: Vector<f16, N>) -> [Vector<f16, N>; 14] {
        let mut acc = a;
        acc += b;
        acc -= b;
        acc *= b;
        [
            a + b,
            a - b,
            a * b,
            -a,
            a.add_with::<{ Denormals::Preserve }, { Saturation::None }>(b),
            a.add_with::<{ Denormals::Preserve }, { Saturation::ClampToUnit }>(b),
            a.add_with::<{ Denormals::FlushToZero }, { Saturation::ClampToUnit }>(b),
            a.sub_with::<{ Denormals::Preserve }, { Saturation::None }>(b),
            a.sub_with::<{ Denormals::FlushToZero }, { Saturation::ClampToUnit }>(b),
            a.mul_with::<{ Denormals::Preserve }, { Saturation::None }>(b),
            a.mul_with::<{ Denormals::Preserve }, { Saturation::ClampToUnit }>(b),
            a.mul_with::<{ Denormals::FlushToZero }, { Saturation::ClampToUnit }>(b),
            a.sub_with::<{ Denormals::Preserve }, { Saturation::ClampToUnit }>(b),
            acc,
        ]
    }

    #[inline(always)]
    pub fn fma<const N: usize>(
        a: Vector<f16, N>,
        b: Vector<f16, N>,
        c: Vector<f16, N>,
    ) -> [Vector<f16, N>; 6] {
        [
            a.mul_add(b, c),
            a.mul_add_with::<{ Denormals::FlushToZero }, { Saturation::None }, { Relu::Disabled }>(
                b, c,
            ),
            a.mul_add_with::<{ Denormals::Preserve }, { Saturation::ClampToUnit }, { Relu::Disabled }>(
                b, c,
            ),
            a.mul_add_with::<{ Denormals::FlushToZero }, { Saturation::ClampToUnit }, { Relu::Disabled }>(
                b, c,
            ),
            a.mul_add_with::<{ Denormals::Preserve }, { Saturation::None }, { Relu::Enabled }>(
                b, c,
            ),
            a.mul_add_with::<{ Denormals::FlushToZero }, { Saturation::None }, { Relu::Enabled }>(
                b, c,
            ),
        ]
    }

    #[cfg(target_feature = "sm_90")]
    #[inline(always)]
    pub fn oob<const N: usize>(
        a: Vector<f16, N>,
        b: Vector<f16, N>,
        c: Vector<f16, N>,
    ) -> [Vector<f16, N>; 2] {
        [
            a.mul_add_oob::<{ Relu::Disabled }>(b, c),
            a.mul_add_oob::<{ Relu::Enabled }>(b, c),
        ]
    }

    #[inline(always)]
    pub fn unary<const N: usize>(a: Vector<f16, N>) -> [Vector<f16, N>; 5] {
        [
            a.abs(),
            a.abs_with::<{ Denormals::FlushToZero }>(),
            a.exp2_approx(),
            a.tanh_approx(),
            -a,
        ]
    }

    #[inline(always)]
    pub fn minmax<const N: usize>(a: Vector<f16, N>, b: Vector<f16, N>) -> [Vector<f16, N>; 8] {
        [
            a.min(b),
            a.max(b),
            a.min_with::<{ Denormals::FlushToZero }, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(
                b,
            ),
            a.min_with::<{ Denormals::Preserve }, { NanHandling::Propagate }, { XorSignAbs::Disabled }>(
                b,
            ),
            a.min_with::<{ Denormals::FlushToZero }, { NanHandling::Propagate }, { XorSignAbs::Disabled }>(
                b,
            ),
            a.max_with::<{ Denormals::FlushToZero }, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(
                b,
            ),
            a.max_with::<{ Denormals::Preserve }, { NanHandling::Propagate }, { XorSignAbs::Disabled }>(
                b,
            ),
            a.max_with::<{ Denormals::FlushToZero }, { NanHandling::Propagate }, { XorSignAbs::Disabled }>(
                b,
            ),
        ]
    }

    #[cfg(target_feature = "sm_86")]
    #[inline(always)]
    pub fn xorsign<const N: usize>(a: Vector<f16, N>, b: Vector<f16, N>) -> [Vector<f16, N>; 4] {
        [
            a.min_with::<{ Denormals::Preserve }, { NanHandling::Ignore }, { XorSignAbs::Enabled }>(
                b,
            ),
            a.min_with::<{ Denormals::FlushToZero }, { NanHandling::Propagate }, { XorSignAbs::Enabled }>(
                b,
            ),
            a.max_with::<{ Denormals::Preserve }, { NanHandling::Ignore }, { XorSignAbs::Enabled }>(
                b,
            ),
            a.max_with::<{ Denormals::FlushToZero }, { NanHandling::Propagate }, { XorSignAbs::Enabled }>(
                b,
            ),
        ]
    }

    #[inline(always)]
    pub fn reduce<const N: usize>(a: Vector<f16, N>) -> [f16; 8] {
        [
            a.reduce_min(),
            a.reduce_max(),
            a.reduce_min_with::<{ Denormals::FlushToZero }, { NanHandling::Ignore }>(),
            a.reduce_max_with::<{ Denormals::FlushToZero }, { NanHandling::Ignore }>(),
            a.reduce_min_with::<{ Denormals::Preserve }, { NanHandling::Propagate }>(),
            a.reduce_max_with::<{ Denormals::Preserve }, { NanHandling::Propagate }>(),
            a.reduce_min_with::<{ Denormals::FlushToZero }, { NanHandling::Propagate }>(),
            a.reduce_max_with::<{ Denormals::FlushToZero }, { NanHandling::Propagate }>(),
        ]
    }
}

/// Instantiates the arithmetic and the reductions of the `f16` vectors with `$n` elements, with
/// kernels for the operators, the fused multiply-adds, the unary functions, `min` and `max` (the
/// `xorsign` kernel needs `sm_86` and the `oob` kernel `sm_90`).
macro_rules! f16_kernels {
    (
        $n:literal, $arith:ident, $fma:ident, $oob:ident, $unary:ident, $minmax:ident,
        $xorsign:ident, $reduce:ident $(,)?
    ) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $arith(
            a: *const Vector<f16, $n>,
            b: *const Vector<f16, $n>,
            out: *mut [Vector<f16, $n>; 14],
        ) {
            unsafe { *out = f16_fns::arith::<$n>(*a, *b) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $fma(
            a: *const Vector<f16, $n>,
            b: *const Vector<f16, $n>,
            c: *const Vector<f16, $n>,
            out: *mut [Vector<f16, $n>; 6],
        ) {
            unsafe { *out = f16_fns::fma::<$n>(*a, *b, *c) }
        }

        #[cfg(target_feature = "sm_90")]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $oob(
            a: *const Vector<f16, $n>,
            b: *const Vector<f16, $n>,
            c: *const Vector<f16, $n>,
            out: *mut [Vector<f16, $n>; 2],
        ) {
            unsafe { *out = f16_fns::oob::<$n>(*a, *b, *c) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $unary(
            a: *const Vector<f16, $n>,
            out: *mut [Vector<f16, $n>; 5],
        ) {
            unsafe { *out = f16_fns::unary::<$n>(*a) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $minmax(
            a: *const Vector<f16, $n>,
            b: *const Vector<f16, $n>,
            out: *mut [Vector<f16, $n>; 8],
        ) {
            unsafe { *out = f16_fns::minmax::<$n>(*a, *b) }
        }

        #[cfg(target_feature = "sm_86")]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $xorsign(
            a: *const Vector<f16, $n>,
            b: *const Vector<f16, $n>,
            out: *mut [Vector<f16, $n>; 4],
        ) {
            unsafe { *out = f16_fns::xorsign::<$n>(*a, *b) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $reduce(a: *const Vector<f16, $n>, out: *mut [f16; 8]) {
            unsafe { *out = f16_fns::reduce::<$n>(*a) }
        }
    };
}

mod bf16_fns {
    use super::*;

    half_shared!(bf16);

    #[inline(always)]
    pub fn arith<const N: usize>(a: Vector<bf16, N>, b: Vector<bf16, N>) -> [Vector<bf16, N>; 5] {
        let mut acc = a;
        acc += b;
        acc -= b;
        acc *= b;
        [a + b, a - b, a * b, -a, acc]
    }

    #[inline(always)]
    pub fn fma<const N: usize>(
        a: Vector<bf16, N>,
        b: Vector<bf16, N>,
        c: Vector<bf16, N>,
    ) -> [Vector<bf16, N>; 3] {
        [
            a.mul_add(b, c),
            a.mul_add_with::<{ Relu::Enabled }>(b, c),
            a.abs(),
        ]
    }

    #[cfg(target_feature = "sm_90")]
    #[inline(always)]
    pub fn sm90<const N: usize>(a: Vector<bf16, N>) -> [Vector<bf16, N>; 2] {
        [a.exp2_approx(), a.tanh_approx()]
    }

    #[inline(always)]
    pub fn minmax<const N: usize>(a: Vector<bf16, N>, b: Vector<bf16, N>) -> [Vector<bf16, N>; 4] {
        [
            a.min(b),
            a.max(b),
            a.min_with::<{ NanHandling::Propagate }, { XorSignAbs::Disabled }>(b),
            a.max_with::<{ NanHandling::Propagate }, { XorSignAbs::Disabled }>(b),
        ]
    }

    #[cfg(target_feature = "sm_86")]
    #[inline(always)]
    pub fn xorsign<const N: usize>(a: Vector<bf16, N>, b: Vector<bf16, N>) -> [Vector<bf16, N>; 4] {
        [
            a.min_with::<{ NanHandling::Ignore }, { XorSignAbs::Enabled }>(b),
            a.min_with::<{ NanHandling::Propagate }, { XorSignAbs::Enabled }>(b),
            a.max_with::<{ NanHandling::Ignore }, { XorSignAbs::Enabled }>(b),
            a.max_with::<{ NanHandling::Propagate }, { XorSignAbs::Enabled }>(b),
        ]
    }

    #[inline(always)]
    pub fn reduce<const N: usize>(a: Vector<bf16, N>) -> [bf16; 6] {
        [
            a.reduce_min(),
            a.reduce_max(),
            a.reduce_min_with::<{ NanHandling::Ignore }>(),
            a.reduce_max_with::<{ NanHandling::Ignore }>(),
            a.reduce_min_with::<{ NanHandling::Propagate }>(),
            a.reduce_max_with::<{ NanHandling::Propagate }>(),
        ]
    }
}

/// Instantiates the arithmetic and the reductions of the `bf16` vectors with `$n` elements;
/// `sm_90` is needed for `exp2_approx` and `tanh_approx` (the `sm90` kernel) and `sm_86` for the
/// `xorsign` kernel.
macro_rules! bf16_kernels {
    (
        $n:literal, $arith:ident, $fma:ident, $sm90:ident, $minmax:ident, $xorsign:ident,
        $reduce:ident $(,)?
    ) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $arith(
            a: *const Vector<bf16, $n>,
            b: *const Vector<bf16, $n>,
            out: *mut [Vector<bf16, $n>; 5],
        ) {
            unsafe { *out = bf16_fns::arith::<$n>(*a, *b) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $fma(
            a: *const Vector<bf16, $n>,
            b: *const Vector<bf16, $n>,
            c: *const Vector<bf16, $n>,
            out: *mut [Vector<bf16, $n>; 3],
        ) {
            unsafe { *out = bf16_fns::fma::<$n>(*a, *b, *c) }
        }

        #[cfg(target_feature = "sm_90")]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $sm90(
            a: *const Vector<bf16, $n>,
            out: *mut [Vector<bf16, $n>; 2],
        ) {
            unsafe { *out = bf16_fns::sm90::<$n>(*a) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $minmax(
            a: *const Vector<bf16, $n>,
            b: *const Vector<bf16, $n>,
            out: *mut [Vector<bf16, $n>; 4],
        ) {
            unsafe { *out = bf16_fns::minmax::<$n>(*a, *b) }
        }

        #[cfg(target_feature = "sm_86")]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $xorsign(
            a: *const Vector<bf16, $n>,
            b: *const Vector<bf16, $n>,
            out: *mut [Vector<bf16, $n>; 4],
        ) {
            unsafe { *out = bf16_fns::xorsign::<$n>(*a, *b) }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $reduce(a: *const Vector<bf16, $n>, out: *mut [bf16; 6]) {
            unsafe { *out = bf16_fns::reduce::<$n>(*a) }
        }
    };
}

/// Instantiates the scalar `tanh_approx` (`bf16` needs `sm_90`).
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn f16_tanh(a: *const f16, out: *mut f16) {
    unsafe { *out = (*a).tanh_approx() }
}

#[cfg(target_feature = "sm_90")]
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn bf16_tanh(a: *const bf16, out: *mut bf16) {
    unsafe { *out = (*a).tanh_approx() }
}

half_kernels!(
    f16, f16_fns, 1,
    convert: f16x1_convert,
    stochastic: f16x1_stochastic,
    broadcast: f16x1_broadcast,
    sum: f16x1_sum,
);
f16_kernels!(
    1,
    f16x1_arith,
    f16x1_fma,
    f16x1_oob,
    f16x1_unary,
    f16x1_minmax,
    f16x1_xorsign,
    f16x1_reduce,
);

half_kernels!(
    f16, f16_fns, 2,
    convert: f16x2_convert,
    stochastic: f16x2_stochastic,
    broadcast: f16x2_broadcast,
    sum: f16x2_sum,
);
f16_kernels!(
    2,
    f16x2_arith,
    f16x2_fma,
    f16x2_oob,
    f16x2_unary,
    f16x2_minmax,
    f16x2_xorsign,
    f16x2_reduce,
);

half_kernels!(
    f16, f16_fns, 3,
    convert: f16x3_convert,
    stochastic: f16x3_stochastic,
    broadcast: f16x3_broadcast,
    sum: f16x3_sum,
);
f16_kernels!(
    3,
    f16x3_arith,
    f16x3_fma,
    f16x3_oob,
    f16x3_unary,
    f16x3_minmax,
    f16x3_xorsign,
    f16x3_reduce,
);

half_kernels!(
    f16, f16_fns, 4,
    convert: f16x4_convert,
    stochastic: f16x4_stochastic,
    broadcast: f16x4_broadcast,
    sum: f16x4_sum,
);
f16_kernels!(
    4,
    f16x4_arith,
    f16x4_fma,
    f16x4_oob,
    f16x4_unary,
    f16x4_minmax,
    f16x4_xorsign,
    f16x4_reduce,
);

half_kernels!(
    f16, f16_fns, 6,
    convert: f16x6_convert,
    stochastic: f16x6_stochastic,
    broadcast: f16x6_broadcast,
    sum: f16x6_sum,
);
f16_kernels!(
    6,
    f16x6_arith,
    f16x6_fma,
    f16x6_oob,
    f16x6_unary,
    f16x6_minmax,
    f16x6_xorsign,
    f16x6_reduce,
);

half_kernels!(
    f16, f16_fns, 8,
    convert: f16x8_convert,
    stochastic: f16x8_stochastic,
    broadcast: f16x8_broadcast,
    sum: f16x8_sum,
);
f16_kernels!(
    8,
    f16x8_arith,
    f16x8_fma,
    f16x8_oob,
    f16x8_unary,
    f16x8_minmax,
    f16x8_xorsign,
    f16x8_reduce,
);

half_kernels!(
    f16, f16_fns, 12,
    convert: f16x12_convert,
    stochastic: f16x12_stochastic,
    broadcast: f16x12_broadcast,
    sum: f16x12_sum,
);
f16_kernels!(
    12,
    f16x12_arith,
    f16x12_fma,
    f16x12_oob,
    f16x12_unary,
    f16x12_minmax,
    f16x12_xorsign,
    f16x12_reduce,
);

half_kernels!(
    f16, f16_fns, 16,
    convert: f16x16_convert,
    stochastic: f16x16_stochastic,
    broadcast: f16x16_broadcast,
    sum: f16x16_sum,
);
f16_kernels!(
    16,
    f16x16_arith,
    f16x16_fma,
    f16x16_oob,
    f16x16_unary,
    f16x16_minmax,
    f16x16_xorsign,
    f16x16_reduce,
);

half_kernels!(
    f16, f16_fns, 32,
    convert: f16x32_convert,
    stochastic: f16x32_stochastic,
    broadcast: f16x32_broadcast,
    sum: f16x32_sum,
);
f16_kernels!(
    32,
    f16x32_arith,
    f16x32_fma,
    f16x32_oob,
    f16x32_unary,
    f16x32_minmax,
    f16x32_xorsign,
    f16x32_reduce,
);

half_kernels!(
    bf16, bf16_fns, 1,
    convert: bf16x1_convert,
    stochastic: bf16x1_stochastic,
    broadcast: bf16x1_broadcast,
    sum: bf16x1_sum,
);
bf16_kernels!(
    1,
    bf16x1_arith,
    bf16x1_fma,
    bf16x1_sm90,
    bf16x1_minmax,
    bf16x1_xorsign,
    bf16x1_reduce,
);

half_kernels!(
    bf16, bf16_fns, 2,
    convert: bf16x2_convert,
    stochastic: bf16x2_stochastic,
    broadcast: bf16x2_broadcast,
    sum: bf16x2_sum,
);
bf16_kernels!(
    2,
    bf16x2_arith,
    bf16x2_fma,
    bf16x2_sm90,
    bf16x2_minmax,
    bf16x2_xorsign,
    bf16x2_reduce,
);

half_kernels!(
    bf16, bf16_fns, 3,
    convert: bf16x3_convert,
    stochastic: bf16x3_stochastic,
    broadcast: bf16x3_broadcast,
    sum: bf16x3_sum,
);
bf16_kernels!(
    3,
    bf16x3_arith,
    bf16x3_fma,
    bf16x3_sm90,
    bf16x3_minmax,
    bf16x3_xorsign,
    bf16x3_reduce,
);

half_kernels!(
    bf16, bf16_fns, 4,
    convert: bf16x4_convert,
    stochastic: bf16x4_stochastic,
    broadcast: bf16x4_broadcast,
    sum: bf16x4_sum,
);
bf16_kernels!(
    4,
    bf16x4_arith,
    bf16x4_fma,
    bf16x4_sm90,
    bf16x4_minmax,
    bf16x4_xorsign,
    bf16x4_reduce,
);

half_kernels!(
    bf16, bf16_fns, 6,
    convert: bf16x6_convert,
    stochastic: bf16x6_stochastic,
    broadcast: bf16x6_broadcast,
    sum: bf16x6_sum,
);
bf16_kernels!(
    6,
    bf16x6_arith,
    bf16x6_fma,
    bf16x6_sm90,
    bf16x6_minmax,
    bf16x6_xorsign,
    bf16x6_reduce,
);

half_kernels!(
    bf16, bf16_fns, 8,
    convert: bf16x8_convert,
    stochastic: bf16x8_stochastic,
    broadcast: bf16x8_broadcast,
    sum: bf16x8_sum,
);
bf16_kernels!(
    8,
    bf16x8_arith,
    bf16x8_fma,
    bf16x8_sm90,
    bf16x8_minmax,
    bf16x8_xorsign,
    bf16x8_reduce,
);

half_kernels!(
    bf16, bf16_fns, 12,
    convert: bf16x12_convert,
    stochastic: bf16x12_stochastic,
    broadcast: bf16x12_broadcast,
    sum: bf16x12_sum,
);
bf16_kernels!(
    12,
    bf16x12_arith,
    bf16x12_fma,
    bf16x12_sm90,
    bf16x12_minmax,
    bf16x12_xorsign,
    bf16x12_reduce,
);

half_kernels!(
    bf16, bf16_fns, 16,
    convert: bf16x16_convert,
    stochastic: bf16x16_stochastic,
    broadcast: bf16x16_broadcast,
    sum: bf16x16_sum,
);
bf16_kernels!(
    16,
    bf16x16_arith,
    bf16x16_fma,
    bf16x16_sm90,
    bf16x16_minmax,
    bf16x16_xorsign,
    bf16x16_reduce,
);

half_kernels!(
    bf16, bf16_fns, 32,
    convert: bf16x32_convert,
    stochastic: bf16x32_stochastic,
    broadcast: bf16x32_broadcast,
    sum: bf16x32_sum,
);
bf16_kernels!(
    32,
    bf16x32_arith,
    bf16x32_fma,
    bf16x32_sm90,
    bf16x32_minmax,
    bf16x32_xorsign,
    bf16x32_reduce,
);
