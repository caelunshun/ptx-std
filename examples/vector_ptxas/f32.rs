// --- LLM-generated --- //
//! `F32Ext` on `f32` and `f32xN<N>`: every method with several modifier combinations, and the
//! horizontal reductions (in separate kernels). On `sm_100` and later, `add_with`, `mul_with` and
//! `fma` on the vectors use the packed `f32x2` instructions (`ClampToUnit` saturation uses the
//! scalar ones). Also the operators, constants and inherent methods of `f32xN<N>`.

use ptx_std::{
    Bits, Denormals, F32Ext, Half, NanHandling, Overflow, Relu, Rounding, Saturation,
    SupportedVectorLen, Vector, VectorLen, XorSignAbs, f32xN,
};

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

container_kernels!(
    f32:
    f32x2_container => 2,
    f32x4_container => 4,
    f32x8_container => 8,
    f32x16_container => 16,
    f32x32_container => 32,
    f32x64_container => 64,
);

f32_ext_kernel!(f32_ext_scalar, f32);
f32_ext_kernel!(f32_ext_x2, f32xN<2>);
f32_ext_kernel!(f32_ext_x4, f32xN<4>);
f32_ext_kernel!(f32_ext_x8, f32xN<8>);
f32_ext_kernel!(f32_ext_x16, f32xN<16>);
f32_ext_kernel!(f32_ext_x32, f32xN<32>);
f32_ext_kernel!(f32_ext_x64, f32xN<64>);

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
f32_reduce_kernels!(f32_sum_x2, f32_reduce_x2, f32xN<2>);
f32_reduce_kernels!(f32_sum_x4, f32_reduce_x4, f32xN<4>);
f32_reduce_kernels!(f32_sum_x8, f32_reduce_x8, f32xN<8>);
f32_reduce_kernels!(f32_sum_x16, f32_reduce_x16, f32xN<16>);
f32_reduce_kernels!(f32_sum_x32, f32_reduce_x32, f32xN<32>);
f32_reduce_kernels!(f32_sum_x64, f32_reduce_x64, f32xN<64>);

/// The operators (with vector and broadcast scalar operands), the splatted constants, the inherent
/// methods and the `core::simd` conversions of `f32xN<N>`.
#[inline(always)]
fn vector_api<const N: usize>(a: f32xN<N>, b: f32xN<N>, c: f32xN<N>, s: f32) -> [f32xN<N>; 24]
where
    VectorLen<N>: SupportedVectorLen,
{
    let mut acc = a;
    acc += b;
    acc -= s;
    acc *= c;
    acc /= b;
    acc *= s;
    [
        a + b,
        a - b,
        a * b,
        a / b,
        -a,
        a + s,
        s - a,
        a * s,
        s / a,
        acc,
        a.abs(),
        a.mul_add(b, c),
        a.min(b),
        a.max(b),
        f32xN::from(core::simd::Simd::from(a) * core::simd::Simd::splat(2.0)),
        f32xN::ZERO,
        f32xN::NEG_ZERO,
        f32xN::ONE,
        f32xN::NEG_ONE,
        f32xN::INFINITY,
        f32xN::NAN,
        f32xN::MIN_POSITIVE_SUBNORMAL,
        f32xN::splat(f32::from(a == b)),
        f32xN::from_array(core::array::from_fn(|i| a.to_array()[i] * 0.5)),
    ]
}

macro_rules! vector_api_kernels {
    ($($name:ident => $n:literal),* $(,)?) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            a: *const f32xN<$n>,
            b: *const f32xN<$n>,
            c: *const f32xN<$n>,
            s: f32,
            out: *mut [f32xN<$n>; 24],
        ) {
            unsafe { *out = vector_api(*a, *b, *c, s) }
        }
    )*};
}

vector_api_kernels!(f32_api_x2 => 2, f32_api_x8 => 8, f32_api_x64 => 64);
