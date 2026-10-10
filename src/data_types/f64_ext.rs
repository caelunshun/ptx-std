// --- LLM-generated --- //
use core::arch::nvptx as nv;

use super::f32_ext::{with_ftz, with_rounding};
use crate::{Denormals, Rounding};

/// Extensions for using PTX intrinsics operating on f64 values.
///
/// Methods taking a [`Rounding`] accept [`Rounding::NearestEven`], [`Rounding::TowardZero`],
/// [`Rounding::Down`] and [`Rounding::Up`]; other modes are rejected at compile time.
///
/// Arithmetic methods are emitted with an explicit rounding modifier (e.g. `add.rn.f64`), which
/// the PTX optimizer treats conservatively: separate multiplications and additions are never
/// contracted into a fused multiply-add. Conversions to integers use the integer rounding
/// modifiers (`.rni`, `.rzi`, `.rmi`, `.rpi`). Subnormal values are always supported, except by
/// the `.ftz` approximations.
pub trait F64Ext: Sized {
    /// Adds two `f64` values, rounding with `RND`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add>
    fn add_with<const RND: Rounding>(self, rhs: f64) -> f64;

    /// Multiplies two `f64` values, rounding with `RND`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul>
    fn mul_with<const RND: Rounding>(self, rhs: f64) -> f64;

    /// Computes `self * b + c` with a single rounding, rounding with `RND`.
    ///
    /// The product is computed to infinite precision and `c` is added to it, also in infinite
    /// precision, before rounding.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma>
    fn fma<const RND: Rounding>(self, b: f64, c: f64) -> f64;

    /// Divides `self` by `rhs`, rounding with `RND`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div>
    fn div_with<const RND: Rounding>(self, rhs: f64) -> f64;

    /// Takes the reciprocal, rounding with `RND`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp>
    fn rcp_with<const RND: Rounding>(self) -> f64;

    /// Takes a fast, gross approximation of the reciprocal (`rcp.approx.ftz.f64`).
    ///
    /// Only the upper 32 bits of `self` are used (the approximation is computed from the top 20
    /// mantissa bits), and the lower 32 bits of the result are zero. The PTX ISA specifies no error
    /// bound. Subnormal inputs and results are flushed to sign-preserving zero, as `.ftz` is
    /// mandatory for `.f64`. `±0.0` and subnormals map to `±inf`, `±inf` to `±0.0`, and NaN to the
    /// canonical NaN `0x7fffffff00000000`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp-approx-ftz-f64>
    fn rcp_approx_ftz(self) -> f64;

    /// Takes the square root, rounding with `RND`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sqrt>
    fn sqrt_with<const RND: Rounding>(self) -> f64;

    /// Takes an approximate reciprocal square root (`rsqrt.approx.f64`).
    ///
    /// With [`Denormals::FlushToZero`], uses `rsqrt.approx.ftz.f64`, which computes from the upper
    /// 32 bits of `self` and zeroes the lower 32 bits of the result. Subnormal inputs and results
    /// are flushed to sign-preserving zero, and NaN maps to the canonical NaN
    /// `0x7fffffff00000000`. With [`Denormals::Preserve`], uses `rsqrt.approx.f64`, which is
    /// emulated in software and relatively slow. The PTX ISA specifies no error bound for either.
    ///
    /// For both, negative non-zero inputs and `-inf` give NaN, `±0.0` gives `±inf` and `+inf`
    /// gives `+0.0`.
    ///
    /// - <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt>
    /// - <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt-approx-ftz-f64>
    fn rsqrt_approx<const DEN: Denormals>(self) -> f64;

    /// Computes the absolute value (`abs.f64`).
    ///
    /// Named after C's `fabs` since it takes no modifiers and would otherwise be shadowed by
    /// `f64::abs`. A NaN input is passed through unchanged.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-abs>
    fn fabs(self) -> f64;

    /// Finds the minimum of two values (`min.f64`).
    ///
    /// `-0.0` is treated as less than `+0.0`. If one input is NaN, the other input is returned.
    /// The result is NaN only if both inputs are NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-min>
    fn fmin(self, rhs: f64) -> f64;

    /// Finds the maximum of two values (`max.f64`).
    ///
    /// `+0.0` is treated as greater than `-0.0`. If one input is NaN, the other input is
    /// returned. The result is NaN only if both inputs are NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max>
    fn fmax(self, rhs: f64) -> f64;

    /// Converts to a signed 32-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `i32::MIN` / `i32::MAX`, and NaN becomes `i32::MIN`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_i32<const RND: Rounding>(self) -> i32;

    /// Converts to an unsigned 32-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `0` / `u32::MAX`, and NaN becomes `1 << 31`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_u32<const RND: Rounding>(self) -> u32;

    /// Converts to a signed 64-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `i64::MIN` / `i64::MAX`, and NaN becomes `i64::MIN`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_i64<const RND: Rounding>(self) -> i64;

    /// Converts to an unsigned 64-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `0` / `u64::MAX`, and NaN becomes `1 << 63`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_u64<const RND: Rounding>(self) -> u64;

    /// Narrows to an `f32`, rounding with `RND`.
    ///
    /// `DEN` selects whether subnormal `f32` results are flushed to sign-preserving zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_f32<const RND: Rounding, const DEN: Denormals>(self) -> f32;

    /// Rounds to an integral value, returned as an `f64`.
    ///
    /// [`Rounding::NearestEven`] rounds to nearest with ties to even, [`Rounding::TowardZero`]
    /// truncates, [`Rounding::Down`] takes the floor and [`Rounding::Up`] the ceiling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn round_to_integral<const RND: Rounding>(self) -> f64;

    /// Clamps to the range `[0.0, 1.0]`. NaN becomes `+0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn saturate(self) -> f64;
}

impl F64Ext for f64 {
    #[inline]
    fn add_with<const RND: Rounding>(self, rhs: f64) -> f64 {
        with_rounding!(RND => R in unsafe { nv::add_f64::<R>(self, rhs) })
    }

    #[inline]
    fn mul_with<const RND: Rounding>(self, rhs: f64) -> f64 {
        with_rounding!(RND => R in unsafe { nv::mul_f64::<R>(self, rhs) })
    }

    #[inline]
    fn fma<const RND: Rounding>(self, b: f64, c: f64) -> f64 {
        with_rounding!(RND => R in unsafe { nv::fma_f64::<R>(self, b, c) })
    }

    #[inline]
    fn div_with<const RND: Rounding>(self, rhs: f64) -> f64 {
        with_rounding!(RND => R in unsafe { nv::div_f64::<R>(self, rhs) })
    }

    #[inline]
    fn rcp_with<const RND: Rounding>(self) -> f64 {
        with_rounding!(RND => R in unsafe { nv::rcp_f64::<R>(self) })
    }

    #[inline]
    fn rcp_approx_ftz(self) -> f64 {
        unsafe { nv::rcp_approx_f64(self) }
    }

    #[inline]
    fn sqrt_with<const RND: Rounding>(self) -> f64 {
        with_rounding!(RND => R in unsafe { nv::sqrt_f64::<R>(self) })
    }

    #[inline]
    fn rsqrt_approx<const DEN: Denormals>(self) -> f64 {
        with_ftz!(DEN => FTZ in unsafe { nv::rsqrt_approx_f64::<FTZ>(self) })
    }

    #[inline]
    fn fabs(self) -> f64 {
        unsafe { nv::abs_f64(self) }
    }

    #[inline]
    fn fmin(self, rhs: f64) -> f64 {
        unsafe { nv::min_f64(self, rhs) }
    }

    #[inline]
    fn fmax(self, rhs: f64) -> f64 {
        unsafe { nv::max_f64(self, rhs) }
    }

    #[inline]
    fn to_i32<const RND: Rounding>(self) -> i32 {
        with_rounding!(RND => R in unsafe { nv::cvt_s32_f64::<R>(self) })
    }

    #[inline]
    fn to_u32<const RND: Rounding>(self) -> u32 {
        with_rounding!(RND => R in unsafe { nv::cvt_u32_f64::<R>(self) })
    }

    #[inline]
    fn to_i64<const RND: Rounding>(self) -> i64 {
        with_rounding!(RND => R in unsafe { nv::cvt_s64_f64::<R>(self) })
    }

    #[inline]
    fn to_u64<const RND: Rounding>(self) -> u64 {
        with_rounding!(RND => R in unsafe { nv::cvt_u64_f64::<R>(self) })
    }

    #[inline]
    fn to_f32<const RND: Rounding, const DEN: Denormals>(self) -> f32 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in
            unsafe { nv::cvt_f32_f64::<R, FTZ>(self) }
        ))
    }

    #[inline]
    fn round_to_integral<const RND: Rounding>(self) -> f64 {
        with_rounding!(RND => R in unsafe { nv::cvt_f64_f64::<R>(self) })
    }

    #[inline]
    fn saturate(self) -> f64 {
        unsafe { nv::cvt_sat_f64_f64(self) }
    }
}
