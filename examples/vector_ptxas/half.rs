// --- LLM-generated --- //
//! `f16` and `bf16` vectors: container API, `f32` conversions, arithmetic, operators and
//! horizontal reductions.

use core::simd::Simd;
use ptx_std::{
    Denormals, NanHandling, Overflow, Relu, Rounding, Saturation, XorSignAbs, bf16, bf16x2, bf16x4,
    bf16x8, bf16x16, bf16x32, f16, f16x2, f16x4, f16x8, f16x16, f16x32,
};

container_all!(
    f16,
    f16x2(u32) f16x2_container,
    f16x4(u64) f16x4_container,
    f16x8(u128) f16x8_container,
    f16x16([u128; 2]) f16x16_container,
    f16x32([u128; 4]) f16x32_container,
);
container_all!(
    bf16,
    bf16x2(u32) bf16x2_container,
    bf16x4(u64) bf16x4_container,
    bf16x8(u128) bf16x8_container,
    bf16x16([u128; 2]) bf16x16_container,
    bf16x32([u128; 4]) bf16x32_container,
);

/// Instantiates `from_f32`, `from_f32_with` (several modifier combinations) and `to_f32` of `$ty`.
macro_rules! convert_kernel {
    ($name:ident, $ty:ident, $n:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            x: *const Simd<f32, $n>,
            v: *const $ty,
            out: *mut [$ty; 5],
            out_f32: *mut Simd<f32, $n>,
        ) {
            unsafe {
                let x = *x;
                *out = [
                    <$ty>::from_f32(x),
                    <$ty>::from_f32_with::<
                        { Rounding::TowardZero },
                        { Relu::Disabled },
                        { Overflow::Infinity },
                    >(x),
                    <$ty>::from_f32_with::<
                        { Rounding::NearestEven },
                        { Relu::Enabled },
                        { Overflow::Infinity },
                    >(x),
                    <$ty>::from_f32_with::<
                        { Rounding::NearestEven },
                        { Relu::Disabled },
                        { Overflow::SaturateFinite },
                    >(x),
                    <$ty>::from_f32_with::<
                        { Rounding::TowardZero },
                        { Relu::Enabled },
                        { Overflow::SaturateFinite },
                    >(x),
                ];
                *out_f32 = (*v).to_f32();
            }
        }
    };
}

convert_kernel!(f16x2_convert, f16x2, 2);
convert_kernel!(f16x4_convert, f16x4, 4);
convert_kernel!(f16x8_convert, f16x8, 8);
convert_kernel!(f16x16_convert, f16x16, 16);
convert_kernel!(f16x32_convert, f16x32, 32);
convert_kernel!(bf16x2_convert, bf16x2, 2);
convert_kernel!(bf16x4_convert, bf16x4, 4);
convert_kernel!(bf16x8_convert, bf16x8, 8);
convert_kernel!(bf16x16_convert, bf16x16, 16);
convert_kernel!(bf16x32_convert, bf16x32, 32);

/// Instantiates `from_f32_stochastic` of the pair types.
#[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
mod stochastic {
    use super::*;

    macro_rules! stochastic_kernel {
        ($name:ident, $ty:ident) => {
            #[unsafe(no_mangle)]
            pub unsafe extern "ptx-kernel" fn $name(
                x: *const Simd<f32, 2>,
                rbits: u32,
                out: *mut [$ty; 4],
            ) {
                unsafe {
                    let x = *x;
                    *out = [
                        <$ty>::from_f32_stochastic::<{ Relu::Disabled }, { Overflow::Infinity }>(
                            x, rbits,
                        ),
                        <$ty>::from_f32_stochastic::<{ Relu::Enabled }, { Overflow::Infinity }>(
                            x, rbits,
                        ),
                        <$ty>::from_f32_stochastic::<{ Relu::Disabled }, { Overflow::SaturateFinite }>(
                            x, rbits,
                        ),
                        <$ty>::from_f32_stochastic::<{ Relu::Enabled }, { Overflow::SaturateFinite }>(
                            x, rbits,
                        ),
                    ];
                }
            }
        };
    }

    stochastic_kernel!(f16x2_stochastic, f16x2);
    stochastic_kernel!(bf16x2_stochastic, bf16x2);
}

/// Instantiates `from_f32_stochastic` of the wide types; `rbits` has one word per pair.
#[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
mod stochastic_wide {
    use super::*;

    macro_rules! stochastic_wide_kernel {
        ($name:ident, $ty:ident, $n:literal, $pairs:literal) => {
            #[unsafe(no_mangle)]
            pub unsafe extern "ptx-kernel" fn $name(
                x: *const Simd<f32, $n>,
                rbits: *const Simd<u32, $pairs>,
                out: *mut [$ty; 4],
            ) {
                unsafe {
                    let (x, rbits) = (*x, *rbits);
                    *out = [
                        <$ty>::from_f32_stochastic::<{ Relu::Disabled }, { Overflow::Infinity }>(
                            x, rbits,
                        ),
                        <$ty>::from_f32_stochastic::<{ Relu::Enabled }, { Overflow::Infinity }>(
                            x, rbits,
                        ),
                        <$ty>::from_f32_stochastic::<{ Relu::Disabled }, { Overflow::SaturateFinite }>(
                            x, rbits,
                        ),
                        <$ty>::from_f32_stochastic::<{ Relu::Enabled }, { Overflow::SaturateFinite }>(
                            x, rbits,
                        ),
                    ];
                }
            }
        };
    }

    stochastic_wide_kernel!(f16x4_stochastic, f16x4, 4, 2);
    stochastic_wide_kernel!(f16x8_stochastic, f16x8, 8, 4);
    stochastic_wide_kernel!(f16x16_stochastic, f16x16, 16, 8);
    stochastic_wide_kernel!(f16x32_stochastic, f16x32, 32, 16);
    stochastic_wide_kernel!(bf16x4_stochastic, bf16x4, 4, 2);
    stochastic_wide_kernel!(bf16x8_stochastic, bf16x8, 8, 4);
    stochastic_wide_kernel!(bf16x16_stochastic, bf16x16, 16, 8);
    stochastic_wide_kernel!(bf16x32_stochastic, bf16x32, 32, 16);
}

/// Instantiates the scalar-broadcast operators and the conversion to `Simd<f32, N>` of a vector
/// `$ty` of `$elem`.
macro_rules! broadcast_kernel {
    ($name:ident, $ty:ident, $elem:ident, $n:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            v: *const $ty,
            s: *const $elem,
            out: *mut [$ty; 7],
            out_f32: *mut Simd<f32, $n>,
        ) {
            unsafe {
                let (v, s) = (*v, *s);
                let mut acc = v;
                acc += s;
                acc -= s;
                acc *= s;
                *out = [v + s, v - s, v * s, s + v, s - v, s * v, acc];
                *out_f32 = Simd::<f32, $n>::from(v);
            }
        }
    };
}

broadcast_kernel!(f16x2_broadcast, f16x2, f16, 2);
broadcast_kernel!(f16x4_broadcast, f16x4, f16, 4);
broadcast_kernel!(f16x8_broadcast, f16x8, f16, 8);
broadcast_kernel!(f16x16_broadcast, f16x16, f16, 16);
broadcast_kernel!(f16x32_broadcast, f16x32, f16, 32);
broadcast_kernel!(bf16x2_broadcast, bf16x2, bf16, 2);
broadcast_kernel!(bf16x4_broadcast, bf16x4, bf16, 4);
broadcast_kernel!(bf16x8_broadcast, bf16x8, bf16, 8);
broadcast_kernel!(bf16x16_broadcast, bf16x16, bf16, 16);
broadcast_kernel!(bf16x32_broadcast, bf16x32, bf16, 32);

/// Instantiates the arithmetic of a vector type `$ty` of `f16`, with kernels for the operators,
/// the fused multiply-adds, the unary functions, `min` and `max` (the `xorsign` kernel needs
/// `sm_86` and the `oob` kernel `sm_90`).
macro_rules! f16_kernels {
    (
        $ty:ident, $arith:ident, $fma:ident, $oob:ident, $unary:ident, $minmax:ident,
        $xorsign:ident
    ) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $arith(a: *const $ty, b: *const $ty, out: *mut [$ty; 14]) {
            unsafe {
                let (a, b) = (*a, *b);
                let mut acc = a;
                acc += b;
                acc -= b;
                acc *= b;
                *out = [
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
                ];
            }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $fma(
            a: *const $ty,
            b: *const $ty,
            c: *const $ty,
            out: *mut [$ty; 6],
        ) {
            unsafe {
                let (a, b, c) = (*a, *b, *c);
                *out = [
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
                ];
            }
        }

        #[cfg(target_feature = "sm_90")]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $oob(
            a: *const $ty,
            b: *const $ty,
            c: *const $ty,
            out: *mut [$ty; 2],
        ) {
            unsafe {
                let (a, b, c) = (*a, *b, *c);
                *out = [
                    a.mul_add_oob::<{ Relu::Disabled }>(b, c),
                    a.mul_add_oob::<{ Relu::Enabled }>(b, c),
                ];
            }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $unary(a: *const $ty, out: *mut [$ty; 5]) {
            unsafe {
                let a = *a;
                *out = [
                    a.abs(),
                    a.abs_with::<{ Denormals::FlushToZero }>(),
                    a.exp2_approx(),
                    a.tanh_approx(),
                    -a,
                ];
            }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $minmax(a: *const $ty, b: *const $ty, out: *mut [$ty; 8]) {
            unsafe {
                let (a, b) = (*a, *b);
                *out = [
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
                ];
            }
        }

        #[cfg(target_feature = "sm_86")]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $xorsign(a: *const $ty, b: *const $ty, out: *mut [$ty; 4]) {
            unsafe {
                let (a, b) = (*a, *b);
                *out = [
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
                ];
            }
        }
    };
}

f16_kernels!(
    f16x2,
    f16x2_arith,
    f16x2_fma,
    f16x2_oob,
    f16x2_unary,
    f16x2_minmax,
    f16x2_xorsign
);
f16_kernels!(
    f16x4,
    f16x4_arith,
    f16x4_fma,
    f16x4_oob,
    f16x4_unary,
    f16x4_minmax,
    f16x4_xorsign
);
f16_kernels!(
    f16x8,
    f16x8_arith,
    f16x8_fma,
    f16x8_oob,
    f16x8_unary,
    f16x8_minmax,
    f16x8_xorsign
);
f16_kernels!(
    f16x16,
    f16x16_arith,
    f16x16_fma,
    f16x16_oob,
    f16x16_unary,
    f16x16_minmax,
    f16x16_xorsign
);
f16_kernels!(
    f16x32,
    f16x32_arith,
    f16x32_fma,
    f16x32_oob,
    f16x32_unary,
    f16x32_minmax,
    f16x32_xorsign
);

/// Instantiates the arithmetic of a vector type `$ty` of `bf16`; `sm_90` is needed for
/// `exp2_approx` and `tanh_approx` (the `sm90` kernel) and `sm_86` for the `xorsign` kernel.
macro_rules! bf16_kernels {
    ($ty:ident, $arith:ident, $fma:ident, $sm90:ident, $minmax:ident, $xorsign:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $arith(a: *const $ty, b: *const $ty, out: *mut [$ty; 5]) {
            unsafe {
                let (a, b) = (*a, *b);
                let mut acc = a;
                acc += b;
                acc -= b;
                acc *= b;
                *out = [a + b, a - b, a * b, -a, acc];
            }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $fma(
            a: *const $ty,
            b: *const $ty,
            c: *const $ty,
            out: *mut [$ty; 3],
        ) {
            unsafe {
                let (a, b, c) = (*a, *b, *c);
                *out = [
                    a.mul_add(b, c),
                    a.mul_add_with::<{ Relu::Enabled }>(b, c),
                    a.abs(),
                ];
            }
        }

        #[cfg(target_feature = "sm_90")]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $sm90(a: *const $ty, out: *mut [$ty; 2]) {
            unsafe {
                let a = *a;
                *out = [a.exp2_approx(), a.tanh_approx()];
            }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $minmax(
            a: *const $ty,
            b: *const $ty,
            out: *mut [$ty; 4],
        ) {
            unsafe {
                let (a, b) = (*a, *b);
                *out = [
                    a.min(b),
                    a.max(b),
                    a.min_with::<{ NanHandling::Propagate }, { XorSignAbs::Disabled }>(b),
                    a.max_with::<{ NanHandling::Propagate }, { XorSignAbs::Disabled }>(b),
                ];
            }
        }

        #[cfg(target_feature = "sm_86")]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $xorsign(
            a: *const $ty,
            b: *const $ty,
            out: *mut [$ty; 4],
        ) {
            unsafe {
                let (a, b) = (*a, *b);
                *out = [
                    a.min_with::<{ NanHandling::Ignore }, { XorSignAbs::Enabled }>(b),
                    a.min_with::<{ NanHandling::Propagate }, { XorSignAbs::Enabled }>(b),
                    a.max_with::<{ NanHandling::Ignore }, { XorSignAbs::Enabled }>(b),
                    a.max_with::<{ NanHandling::Propagate }, { XorSignAbs::Enabled }>(b),
                ];
            }
        }
    };
}

bf16_kernels!(
    bf16x2,
    bf16x2_arith,
    bf16x2_fma,
    bf16x2_sm90,
    bf16x2_minmax,
    bf16x2_xorsign
);
bf16_kernels!(
    bf16x4,
    bf16x4_arith,
    bf16x4_fma,
    bf16x4_sm90,
    bf16x4_minmax,
    bf16x4_xorsign
);
bf16_kernels!(
    bf16x8,
    bf16x8_arith,
    bf16x8_fma,
    bf16x8_sm90,
    bf16x8_minmax,
    bf16x8_xorsign
);
bf16_kernels!(
    bf16x16,
    bf16x16_arith,
    bf16x16_fma,
    bf16x16_sm90,
    bf16x16_minmax,
    bf16x16_xorsign
);
bf16_kernels!(
    bf16x32,
    bf16x32_arith,
    bf16x32_fma,
    bf16x32_sm90,
    bf16x32_minmax,
    bf16x32_xorsign
);

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

/// Instantiates the horizontal reductions of the vector `$ty` of `$elem`, with `reduce_sum` in a
/// separate kernel. `$minmax` holds the modifier arguments of the `_with` variants.
macro_rules! reduce_kernels {
    ($ty:ident, $elem:ident, $sum:ident, $reduce:ident, [$($minmax:tt)*]) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $sum(a: *const $ty, out: *mut f32) {
            unsafe { *out = (*a).reduce_sum() }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $reduce(a: *const $ty, out: *mut [$elem; 6]) {
            unsafe {
                let a = *a;
                reduce_kernels!(@with a out [$($minmax)*]);
            }
        }
    };
    (@with $a:ident $out:ident [$(($($m:tt)*))*]) => {
        *$out = [
            $a.reduce_min(),
            $a.reduce_max(),
            $($a.reduce_min_with::<$($m)*>(), $a.reduce_max_with::<$($m)*>(),)*
        ];
    };
}

reduce_kernels!(
    f16x2,
    f16,
    f16x2_sum,
    f16x2_reduce,
    [({ Denormals::FlushToZero }, { NanHandling::Ignore })(
        { Denormals::Preserve },
        { NanHandling::Propagate }
    )]
);
reduce_kernels!(
    f16x4,
    f16,
    f16x4_sum,
    f16x4_reduce,
    [({ Denormals::FlushToZero }, { NanHandling::Ignore })(
        { Denormals::Preserve },
        { NanHandling::Propagate }
    )]
);
reduce_kernels!(
    f16x8,
    f16,
    f16x8_sum,
    f16x8_reduce,
    [({ Denormals::FlushToZero }, { NanHandling::Ignore })(
        { Denormals::Preserve },
        { NanHandling::Propagate }
    )]
);
reduce_kernels!(
    f16x16,
    f16,
    f16x16_sum,
    f16x16_reduce,
    [({ Denormals::FlushToZero }, { NanHandling::Propagate })(
        { Denormals::Preserve },
        { NanHandling::Propagate }
    )]
);
reduce_kernels!(
    f16x32,
    f16,
    f16x32_sum,
    f16x32_reduce,
    [({ Denormals::FlushToZero }, { NanHandling::Ignore })(
        { Denormals::Preserve },
        { NanHandling::Propagate }
    )]
);
reduce_kernels!(
    bf16x2,
    bf16,
    bf16x2_sum,
    bf16x2_reduce,
    [({ NanHandling::Ignore })({ NanHandling::Propagate })]
);
reduce_kernels!(
    bf16x4,
    bf16,
    bf16x4_sum,
    bf16x4_reduce,
    [({ NanHandling::Ignore })({ NanHandling::Propagate })]
);
reduce_kernels!(
    bf16x8,
    bf16,
    bf16x8_sum,
    bf16x8_reduce,
    [({ NanHandling::Ignore })({ NanHandling::Propagate })]
);
reduce_kernels!(
    bf16x16,
    bf16,
    bf16x16_sum,
    bf16x16_reduce,
    [({ NanHandling::Ignore })({ NanHandling::Propagate })]
);
reduce_kernels!(
    bf16x32,
    bf16,
    bf16x32_sum,
    bf16x32_reduce,
    [({ NanHandling::Ignore })({ NanHandling::Propagate })]
);
