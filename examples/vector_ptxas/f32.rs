// --- LLM-generated --- //
//! `F32Ext` on `f32` and `Simd<f32, N>`: every method with several modifier combinations, and the
//! horizontal reductions (in separate kernels, also for odd widths). On `sm_100` and later,
//! `add_with`, `mul_with` and `fma` on `Simd` use the packed `f32x2` instructions (`ClampToUnit`
//! saturation and the last lane of odd widths use the scalar ones).

use core::simd::Simd;
use ptx_std::{Denormals, F32Ext, NanHandling, Overflow, Relu, Rounding, Saturation, XorSignAbs};

const RN: Rounding = Rounding::NearestEven;
const RZ: Rounding = Rounding::TowardZero;
const RM: Rounding = Rounding::Down;
const RP: Rounding = Rounding::Up;
const KEEP: Denormals = Denormals::Preserve;
const FTZ: Denormals = Denormals::FlushToZero;
const NO_SAT: Saturation = Saturation::None;
const SAT: Saturation = Saturation::ClampToUnit;

/// Instantiates all methods of `F32Ext` on `$ty`. Float results are stored consecutively at `out`.
macro_rules! f32_ext_kernel {
    ($name:ident, $ty:ty) => {
        #[allow(unused_assignments)]
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            a: *const $ty,
            b: *const $ty,
            c: *const $ty,
            out: *mut $ty,
            out_i32: *mut <$ty as F32Ext>::I32,
            out_u32: *mut <$ty as F32Ext>::U32,
            out_i64: *mut <$ty as F32Ext>::I64,
            out_u64: *mut <$ty as F32Ext>::U64,
        ) {
            unsafe {
                let (a, b, c) = (*a, *b, *c);
                let mut i = 0;
                macro_rules! put {
                    ($v:expr) => {{
                        out.add(i).write($v);
                        i += 1;
                    }};
                }

                put!(a.add_with::<RN, KEEP, NO_SAT>(b));
                put!(a.add_with::<RZ, KEEP, NO_SAT>(b));
                put!(a.add_with::<RM, FTZ, NO_SAT>(b));
                put!(a.add_with::<RP, FTZ, NO_SAT>(b));
                put!(a.add_with::<RN, KEEP, SAT>(b));
                put!(a.add_with::<RZ, FTZ, SAT>(b));

                put!(a.mul_with::<RN, KEEP>(b));
                put!(a.mul_with::<RZ, KEEP>(b));
                put!(a.mul_with::<RM, FTZ>(b));
                put!(a.mul_with::<RP, FTZ>(b));

                put!(a.fma::<RN, KEEP, NO_SAT>(b, c));
                put!(a.fma::<RZ, KEEP, NO_SAT>(b, c));
                put!(a.fma::<RM, FTZ, NO_SAT>(b, c));
                put!(a.fma::<RP, FTZ, NO_SAT>(b, c));
                put!(a.fma::<RN, KEEP, SAT>(b, c));
                put!(a.fma::<RZ, FTZ, SAT>(b, c));

                put!(a.div_with::<RN, KEEP>(b));
                put!(a.div_with::<RP, FTZ>(b));
                put!(a.div_approx::<KEEP>(b));
                put!(a.div_approx::<FTZ>(b));
                put!(a.div_full_approx::<KEEP>(b));
                put!(a.div_full_approx::<FTZ>(b));

                put!(a.rcp_with::<RN, KEEP>());
                put!(a.rcp_with::<RM, FTZ>());
                put!(a.rcp_approx_ftz());

                put!(a.sqrt_with::<RN, KEEP>());
                put!(a.sqrt_with::<RZ, FTZ>());
                put!(a.sqrt_approx::<KEEP>());
                put!(a.sqrt_approx::<FTZ>());
                put!(a.rsqrt_approx::<KEEP>());
                put!(a.rsqrt_approx::<FTZ>());

                put!(a.sin_approx::<KEEP>());
                put!(a.sin_approx::<FTZ>());
                put!(a.cos_approx::<KEEP>());
                put!(a.cos_approx::<FTZ>());
                put!(a.log2_approx::<KEEP>());
                put!(a.log2_approx::<FTZ>());
                put!(a.exp2_approx::<KEEP>());
                put!(a.exp2_approx::<FTZ>());
                put!(a.tanh_approx());

                put!(a.abs_with::<KEEP>());
                put!(a.abs_with::<FTZ>());

                put!(a.min_with::<KEEP, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(b));
                put!(a.min_with::<FTZ, { NanHandling::Propagate }, { XorSignAbs::Disabled }>(b));
                put!(a.max_with::<KEEP, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(b));
                put!(a.max_with::<FTZ, { NanHandling::Propagate }, { XorSignAbs::Disabled }>(b));
                #[cfg(target_feature = "sm_86")]
                {
                    put!(a.min_with::<KEEP, { NanHandling::Ignore }, { XorSignAbs::Enabled }>(b));
                    put!(a.max_with::<FTZ, { NanHandling::Propagate }, { XorSignAbs::Enabled }>(b));
                }

                put!(a.round_to_integral::<RN, KEEP>());
                put!(a.round_to_integral::<RZ, FTZ>());
                put!(a.round_to_integral::<RM, KEEP>());
                put!(a.round_to_integral::<RP, FTZ>());
                put!(a.saturate::<KEEP>());
                put!(a.saturate::<FTZ>());

                put!(a.to_tf32::<{ Rounding::NearestAway }, { Relu::Disabled }, { Overflow::Infinity }>());
                put!(a.to_tf32::<{ Rounding::NearestAway }, { Relu::Disabled }, { Overflow::SaturateFinite }>());
                #[cfg(target_feature = "sm_90")]
                {
                    put!(a.to_tf32::<RN, { Relu::Disabled }, { Overflow::Infinity }>());
                    put!(a.to_tf32::<RZ, { Relu::Enabled }, { Overflow::Infinity }>());
                }
                #[cfg(target_feature = "sm_100")]
                {
                    put!(a.to_tf32::<RN, { Relu::Enabled }, { Overflow::SaturateFinite }>());
                    put!(a.to_tf32::<RZ, { Relu::Disabled }, { Overflow::SaturateFinite }>());
                }

                out_i32.write(a.to_i32::<RN, KEEP>());
                out_i32.add(1).write(a.to_i32::<RZ, FTZ>());
                out_i32.add(2).write(a.to_i32::<RM, KEEP>());
                out_i32.add(3).write(a.to_i32::<RP, FTZ>());
                out_u32.write(a.to_u32::<RN, KEEP>());
                out_u32.add(1).write(a.to_u32::<RZ, FTZ>());
                out_u32.add(2).write(a.to_u32::<RM, KEEP>());
                out_u32.add(3).write(a.to_u32::<RP, FTZ>());
                out_i64.write(a.to_i64::<RN, KEEP>());
                out_i64.add(1).write(a.to_i64::<RZ, FTZ>());
                out_i64.add(2).write(a.to_i64::<RM, KEEP>());
                out_i64.add(3).write(a.to_i64::<RP, FTZ>());
                out_u64.write(a.to_u64::<RN, KEEP>());
                out_u64.add(1).write(a.to_u64::<RZ, FTZ>());
                out_u64.add(2).write(a.to_u64::<RM, KEEP>());
                out_u64.add(3).write(a.to_u64::<RP, FTZ>());
            }
        }
    };
}

f32_ext_kernel!(f32_ext_scalar, f32);
f32_ext_kernel!(f32_ext_x2, Simd<f32, 2>);
f32_ext_kernel!(f32_ext_x3, Simd<f32, 3>);
f32_ext_kernel!(f32_ext_x4, Simd<f32, 4>);
f32_ext_kernel!(f32_ext_x8, Simd<f32, 8>);
f32_ext_kernel!(f32_ext_x16, Simd<f32, 16>);

/// Instantiates the horizontal reductions of `F32Ext` on `$ty`, with `reduce_sum` in a separate
/// kernel.
macro_rules! f32_reduce_kernels {
    ($sum:ident, $reduce:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $sum(a: *const $ty, out: *mut f32) {
            unsafe { *out = (*a).reduce_sum() }
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $reduce(a: *const $ty, out: *mut [f32; 6]) {
            unsafe {
                let a = *a;
                *out = [
                    a.reduce_min(),
                    a.reduce_max(),
                    a.reduce_min_with::<FTZ, { NanHandling::Propagate }>(),
                    a.reduce_max_with::<FTZ, { NanHandling::Ignore }>(),
                    a.reduce_min_with::<KEEP, { NanHandling::Propagate }>(),
                    a.reduce_max_with::<KEEP, { NanHandling::Propagate }>(),
                ];
            }
        }
    };
}

f32_reduce_kernels!(f32_sum_scalar, f32_reduce_scalar, f32);
f32_reduce_kernels!(f32_sum_x1, f32_reduce_x1, Simd<f32, 1>);
f32_reduce_kernels!(f32_sum_x2, f32_reduce_x2, Simd<f32, 2>);
f32_reduce_kernels!(f32_sum_x3, f32_reduce_x3, Simd<f32, 3>);
f32_reduce_kernels!(f32_sum_x4, f32_reduce_x4, Simd<f32, 4>);
f32_reduce_kernels!(f32_sum_x5, f32_reduce_x5, Simd<f32, 5>);
f32_reduce_kernels!(f32_sum_x7, f32_reduce_x7, Simd<f32, 7>);
f32_reduce_kernels!(f32_sum_x8, f32_reduce_x8, Simd<f32, 8>);
f32_reduce_kernels!(f32_sum_x16, f32_reduce_x16, Simd<f32, 16>);
f32_reduce_kernels!(f32_sum_x32, f32_reduce_x32, Simd<f32, 32>);
f32_ext_kernel!(f32_ext_x32, Simd<f32, 32>);
f32_ext_kernel!(f32_ext_x64, Simd<f32, 64>);
f32_reduce_kernels!(f32_sum_x64, f32_reduce_x64, Simd<f32, 64>);
