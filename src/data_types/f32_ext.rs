// --- LLM-generated --- //
use core::arch::nvptx as nv;

use crate::{Denormals, NanHandling, Overflow, Relu, Rounding, Saturation, XorSignAbs};

/// Expands `$body` with a local `const $name: nv::Rounding` equal to the IEEE rounding mode
/// `$rnd`, rejecting other modes at compile time.
macro_rules! with_rounding {
    ($rnd:ident => $name:ident in $body:expr) => {{
        const {
            assert!(
                matches!(
                    $rnd,
                    $crate::Rounding::NearestEven
                        | $crate::Rounding::TowardZero
                        | $crate::Rounding::Down
                        | $crate::Rounding::Up
                ),
                "unsupported rounding mode: expected NearestEven, TowardZero, Down or Up"
            )
        };
        const_map!($rnd => $name: core::arch::nvptx::Rounding {
            $crate::Rounding::NearestEven => core::arch::nvptx::Rounding::Rn,
            $crate::Rounding::TowardZero => core::arch::nvptx::Rounding::Rz,
            $crate::Rounding::Down => core::arch::nvptx::Rounding::Rm,
            $crate::Rounding::Up => core::arch::nvptx::Rounding::Rp,
        } in $body)
    }};
}

/// Expands `$body` with a local `const $name: bool` that is `true` iff `$den` is
/// [`Denormals::FlushToZero`].
macro_rules! with_ftz {
    ($den:ident => $name:ident in $body:expr) => {
        const_bool!($name = matches!($den, $crate::Denormals::FlushToZero) => $body)
    };
}

/// Expands `$body` with a local `const $name: bool` that is `true` iff `$sat` is
/// [`Saturation::ClampToUnit`].
macro_rules! with_sat {
    ($sat:ident => $name:ident in $body:expr) => {
        const_bool!($name = matches!($sat, $crate::Saturation::ClampToUnit) => $body)
    };
}

/// Expands `$body` with a local `const $name: bool` that is `true` iff `$nan` is
/// [`NanHandling::Propagate`].
macro_rules! with_nan {
    ($nan:ident => $name:ident in $body:expr) => {
        const_bool!($name = matches!($nan, $crate::NanHandling::Propagate) => $body)
    };
}

/// Expands `$body` with a local `const $name: bool` that is `true` iff `$xor` is
/// [`XorSignAbs::Enabled`].
macro_rules! with_xorsign {
    ($xor:ident => $name:ident in $body:expr) => {
        const_bool!($name = matches!($xor, $crate::XorSignAbs::Enabled) => $body)
    };
}

/// Expands `$body` with a local `const $name: bool` that is `true` iff `$relu` is
/// [`Relu::Enabled`].
#[allow(unused_macros)]
macro_rules! with_relu {
    ($relu:ident => $name:ident in $body:expr) => {
        const_bool!($name = matches!($relu, $crate::Relu::Enabled) => $body)
    };
}

/// Expands `$body` with a local `const $name: bool` that is `true` iff `$ovf` is
/// [`Overflow::SaturateFinite`].
macro_rules! with_satfinite {
    ($ovf:ident => $name:ident in $body:expr) => {
        const_bool!($name = matches!($ovf, $crate::Overflow::SaturateFinite) => $body)
    };
}

pub(crate) use {with_ftz, with_rounding};

/// Extensions for using PTX intrinsics operating on f32 values.
///
/// Methods taking a [`Rounding`] accept [`Rounding::NearestEven`], [`Rounding::TowardZero`],
/// [`Rounding::Down`] and [`Rounding::Up`]; other modes are rejected at compile time. Methods
/// taking [`Denormals`] either preserve subnormal values or flush subnormal inputs and results
/// to sign-preserving zero.
///
/// The `.approx` instructions (methods with the `_approx` suffix) trade accuracy for speed. Where
/// the PTX ISA specifies an error bound, it is given in the method's documentation; some bounds
/// only hold over a restricted input range.
pub trait F32Ext: Sized {
    /// Adds two `f32` values.
    ///
    /// `RND` selects the rounding mode, `DEN` the subnormal handling, and `SAT` optionally clamps
    /// the result to `[0.0, 1.0]`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add>
    fn add_with<const RND: Rounding, const DEN: Denormals, const SAT: Saturation>(
        self,
        rhs: f32,
    ) -> f32;

    /// Multiplies two `f32` values.
    ///
    /// `RND` selects the rounding mode and `DEN` the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul>
    fn mul_with<const RND: Rounding, const DEN: Denormals>(self, rhs: f32) -> f32;

    /// Computes `self * b + c` with a single rounding.
    ///
    /// `RND` selects the rounding mode, `DEN` the subnormal handling, and `SAT` optionally clamps
    /// the result to `[0.0, 1.0]`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma>
    fn fma<const RND: Rounding, const DEN: Denormals, const SAT: Saturation>(
        self,
        b: f32,
        c: f32,
    ) -> f32;

    /// Divides `self` by `rhs`, with the IEEE 754 compliant `div.rnd` instruction.
    ///
    /// `RND` selects the rounding mode and `DEN` the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div>
    fn div_with<const RND: Rounding, const DEN: Denormals>(self, rhs: f32) -> f32;

    /// Divides `self` by `rhs` using the approximate `div.approx` instruction, computed as
    /// `self * (1 / rhs)`.
    ///
    /// For `|rhs|` in `[2^-126, 2^126]`, the maximum error is 2 ulp. For
    /// `2^126 < |rhs| < 2^128`, the result is NaN if `self` is infinite and a sign-preserving zero
    /// otherwise. `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div>
    fn div_approx<const DEN: Denormals>(self, rhs: f32) -> f32;

    /// Divides `self` by `rhs` using the `div.full` instruction, which is faster than IEEE 754
    /// division but less accurate (and not IEEE 754 compliant).
    ///
    /// The maximum error is 2 ulp over the full range of inputs. `DEN` selects the subnormal
    /// handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div>
    fn div_full_approx<const DEN: Denormals>(self, rhs: f32) -> f32;

    /// Takes the reciprocal, with the IEEE 754 compliant `rcp.rnd` instruction.
    ///
    /// `RND` selects the rounding mode and `DEN` the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp>
    fn rcp_with<const RND: Rounding, const DEN: Denormals>(self) -> f32;

    /// Takes a fast, approximate reciprocal.
    ///
    /// The maximum error is 1 ulp over the full range of inputs. Subnormal inputs and results are
    /// flushed to zero (`rcp.approx.ftz`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp>
    fn rcp_approx_ftz(self) -> f32;

    /// Takes the square root, with the IEEE 754 compliant `sqrt.rnd` instruction.
    ///
    /// `RND` selects the rounding mode and `DEN` the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sqrt>
    fn sqrt_with<const RND: Rounding, const DEN: Denormals>(self) -> f32;

    /// Takes a fast, approximate square root (`sqrt.approx`).
    ///
    /// The maximum relative error over the entire positive finite range is 2^-23. `DEN` selects
    /// the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sqrt>
    fn sqrt_approx<const DEN: Denormals>(self) -> f32;

    /// Takes a fast, approximate reciprocal square root (`rsqrt.approx`).
    ///
    /// The maximum relative error over the entire positive finite range is 2^-22.9. `DEN` selects
    /// the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt>
    fn rsqrt_approx<const DEN: Denormals>(self) -> f32;

    /// Computes the sine of `self` (in radians), approximately (`sin.approx`).
    ///
    /// The maximum absolute error is 2^-20.5 in the range `-2 * pi ..= 2 * pi` and 2^-14.7 in
    /// the range `-100 * pi ..= 100 * pi`. Outside of the latter, there is no error guarantee.
    /// `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sin>
    fn sin_approx<const DEN: Denormals>(self) -> f32;

    /// Computes the cosine of `self` (in radians), approximately (`cos.approx`).
    ///
    /// The maximum absolute error is 2^-20.4 in the range `-2 * pi ..= 2 * pi` and 2^-14.7 in
    /// the range `-100 * pi ..= 100 * pi`. Outside of the latter, there is no error guarantee.
    /// `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-cos>
    fn cos_approx<const DEN: Denormals>(self) -> f32;

    /// Computes the base-2 logarithm, approximately (`lg2.approx`).
    ///
    /// The maximum absolute error is 2^-22 for inputs in `(0.5, 2)`; for other positive finite
    /// inputs, the maximum relative error is 2^-22. `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-lg2>
    fn log2_approx<const DEN: Denormals>(self) -> f32;

    /// Computes the base-2 exponential `2^self`, approximately (`ex2.approx`).
    ///
    /// The maximum error is 2 ulp from the correctly rounded result over the full range of
    /// inputs. `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2>
    fn exp2_approx<const DEN: Denormals>(self) -> f32;

    /// Computes the absolute value.
    ///
    /// `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-abs>
    fn abs_with<const DEN: Denormals>(self) -> f32;

    /// Finds the minimum of two values.
    ///
    /// `-0.0` is treated as less than `+0.0`. `NAN` selects whether a NaN input is ignored
    /// ([`NanHandling::Ignore`]) or propagated ([`NanHandling::Propagate`]). `DEN` selects the
    /// subnormal handling.
    ///
    /// With `XOR` set to [`XorSignAbs::Enabled`] (`min.xorsign.abs`), the magnitude of the result
    /// is the minimum of the absolute values of the inputs, and its sign is the XOR of the input
    /// signs. This requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-min>
    fn min_with<const DEN: Denormals, const NAN: NanHandling, const XOR: XorSignAbs>(
        self,
        rhs: f32,
    ) -> f32;

    /// Finds the maximum of two values.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `NAN` selects whether a NaN input is ignored
    /// ([`NanHandling::Ignore`]) or propagated ([`NanHandling::Propagate`]). `DEN` selects the
    /// subnormal handling.
    ///
    /// With `XOR` set to [`XorSignAbs::Enabled`] (`max.xorsign.abs`), the magnitude of the result
    /// is the maximum of the absolute values of the inputs, and its sign is the XOR of the input
    /// signs. This requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max>
    fn max_with<const DEN: Denormals, const NAN: NanHandling, const XOR: XorSignAbs>(
        self,
        rhs: f32,
    ) -> f32;

    /// Converts to a signed 32-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `i32::MIN` / `i32::MAX`, and NaN becomes zero. If `DEN`
    /// is [`Denormals::FlushToZero`], subnormal inputs are flushed to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_i32<const RND: Rounding, const DEN: Denormals>(self) -> i32;

    /// Converts to an unsigned 32-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `0` / `u32::MAX`, and NaN becomes zero. If `DEN` is
    /// [`Denormals::FlushToZero`], subnormal inputs are flushed to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_u32<const RND: Rounding, const DEN: Denormals>(self) -> u32;

    /// Converts to a signed 64-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `i64::MIN` / `i64::MAX`, and NaN becomes zero. If `DEN`
    /// is [`Denormals::FlushToZero`], subnormal inputs are flushed to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_i64<const RND: Rounding, const DEN: Denormals>(self) -> i64;

    /// Converts to an unsigned 64-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `0` / `u64::MAX`, and NaN becomes zero. If `DEN` is
    /// [`Denormals::FlushToZero`], subnormal inputs are flushed to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_u64<const RND: Rounding, const DEN: Denormals>(self) -> u64;

    /// Rounds to an integral value, returned as an `f32`.
    ///
    /// [`Rounding::NearestEven`] rounds to nearest with ties to even, [`Rounding::TowardZero`]
    /// truncates, [`Rounding::Down`] takes the floor and [`Rounding::Up`] the ceiling. `DEN`
    /// selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn round_to_integral<const RND: Rounding, const DEN: Denormals>(self) -> f32;

    /// Clamps to the range `[0.0, 1.0]`. NaN becomes `+0.0`.
    ///
    /// `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn saturate<const DEN: Denormals>(self) -> f32;

    /// Converts to the `tf32` format, returning its bit pattern stored in an `f32`.
    ///
    /// The result has the low 13 mantissa bits cleared, so it can be used directly as an `f32`
    /// operand of `tf32` tensor core instructions.
    ///
    /// `RND` must be [`Rounding::NearestEven`], [`Rounding::TowardZero`] or
    /// [`Rounding::NearestAway`]. `RELU` clamps negative results to zero and `SAT` clamps
    /// out-of-range results to the largest finite `tf32` value.
    ///
    /// [`Rounding::NearestAway`] requires `sm_80` and does not support `RELU`. The other roundings
    /// require `sm_90`, and combining them with [`Overflow::SaturateFinite`] additionally requires
    /// `sm_100`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_tf32<const RND: Rounding, const RELU: Relu, const SAT: Overflow>(self) -> f32;
}

/// `min.xorsign.abs` is only emitted for `sm_86` and later, since the instruction cannot be
/// selected otherwise even if the call is dead code.
macro_rules! xorsign_variant {
    ($($name:ident => $core:ident),*) => {$(
        #[inline(always)]
        fn $name<const FTZ: bool, const NAN: bool, const XOR: bool>(a: f32, b: f32) -> f32 {
            #[cfg(target_feature = "sm_86")]
            unsafe {
                nv::$core::<FTZ, NAN, XOR>(a, b)
            }
            #[cfg(not(target_feature = "sm_86"))]
            unsafe {
                nv::$core::<FTZ, NAN, false>(a, b)
            }
        }
    )*};
}

xorsign_variant!(min_f32 => min_f32, max_f32 => max_f32);

/// `cvt.{rn,rz}{.relu}{.satfinite}.tf32.f32`. `.satfinite` is only emitted for `sm_100` and later,
/// since the instruction cannot be selected otherwise even if the call is dead code.
#[cfg(target_feature = "sm_90")]
#[inline(always)]
fn cvt_tf32_rn_rz<const RND: nv::Rounding, const RELU: bool, const SATFINITE: bool>(a: f32) -> u32 {
    #[cfg(target_feature = "sm_100")]
    unsafe {
        nv::cvt_tf32_f32::<RND, RELU, SATFINITE>(a)
    }
    #[cfg(not(target_feature = "sm_100"))]
    unsafe {
        nv::cvt_tf32_f32::<RND, RELU, false>(a)
    }
}

impl F32Ext for f32 {
    #[inline]
    fn add_with<const RND: Rounding, const DEN: Denormals, const SAT: Saturation>(
        self,
        rhs: f32,
    ) -> f32 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in with_sat!(SAT => S in
            unsafe { nv::add_f32::<R, FTZ, S>(self, rhs) }
        )))
    }

    #[inline]
    fn mul_with<const RND: Rounding, const DEN: Denormals>(self, rhs: f32) -> f32 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in
            unsafe { nv::mul_f32::<R, FTZ>(self, rhs) }
        ))
    }

    #[inline]
    fn fma<const RND: Rounding, const DEN: Denormals, const SAT: Saturation>(
        self,
        b: f32,
        c: f32,
    ) -> f32 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in with_sat!(SAT => S in
            unsafe { nv::fma_f32::<R, FTZ, S>(self, b, c) }
        )))
    }

    #[inline]
    fn div_with<const RND: Rounding, const DEN: Denormals>(self, rhs: f32) -> f32 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in
            unsafe { nv::div_f32::<R, FTZ>(self, rhs) }
        ))
    }

    #[inline]
    fn div_approx<const DEN: Denormals>(self, rhs: f32) -> f32 {
        with_ftz!(DEN => FTZ in unsafe { nv::div_approx_f32::<FTZ>(self, rhs) })
    }

    #[inline]
    fn div_full_approx<const DEN: Denormals>(self, rhs: f32) -> f32 {
        with_ftz!(DEN => FTZ in unsafe { nv::div_full_f32::<FTZ>(self, rhs) })
    }

    #[inline]
    fn rcp_with<const RND: Rounding, const DEN: Denormals>(self) -> f32 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in
            unsafe { nv::rcp_f32::<R, FTZ>(self) }
        ))
    }

    #[inline]
    fn rcp_approx_ftz(self) -> f32 {
        unsafe { nv::rcp_approx_ftz_f32(self) }
    }

    #[inline]
    fn sqrt_with<const RND: Rounding, const DEN: Denormals>(self) -> f32 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in
            unsafe { nv::sqrt_f32::<R, FTZ>(self) }
        ))
    }

    #[inline]
    fn sqrt_approx<const DEN: Denormals>(self) -> f32 {
        with_ftz!(DEN => FTZ in unsafe { nv::sqrt_approx_f32::<FTZ>(self) })
    }

    #[inline]
    fn rsqrt_approx<const DEN: Denormals>(self) -> f32 {
        with_ftz!(DEN => FTZ in unsafe { nv::rsqrt_approx_f32::<FTZ>(self) })
    }

    #[inline]
    fn sin_approx<const DEN: Denormals>(self) -> f32 {
        with_ftz!(DEN => FTZ in unsafe { nv::sin_f32::<FTZ>(self) })
    }

    #[inline]
    fn cos_approx<const DEN: Denormals>(self) -> f32 {
        with_ftz!(DEN => FTZ in unsafe { nv::cos_f32::<FTZ>(self) })
    }

    #[inline]
    fn log2_approx<const DEN: Denormals>(self) -> f32 {
        with_ftz!(DEN => FTZ in unsafe { nv::lg2_f32::<FTZ>(self) })
    }

    #[inline]
    fn exp2_approx<const DEN: Denormals>(self) -> f32 {
        with_ftz!(DEN => FTZ in unsafe { nv::ex2_f32::<FTZ>(self) })
    }

    #[inline]
    fn abs_with<const DEN: Denormals>(self) -> f32 {
        with_ftz!(DEN => FTZ in unsafe { nv::abs_f32::<FTZ>(self) })
    }

    #[inline]
    fn min_with<const DEN: Denormals, const NAN: NanHandling, const XOR: XorSignAbs>(
        self,
        rhs: f32,
    ) -> f32 {
        const {
            assert!(
                matches!(XOR, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "min with XorSignAbs::Enabled requires sm_86"
            )
        };
        with_ftz!(DEN => FTZ in with_nan!(NAN => N in
            with_xorsign!(XOR => X in min_f32::<FTZ, N, X>(self, rhs))
        ))
    }

    #[inline]
    fn max_with<const DEN: Denormals, const NAN: NanHandling, const XOR: XorSignAbs>(
        self,
        rhs: f32,
    ) -> f32 {
        const {
            assert!(
                matches!(XOR, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "max with XorSignAbs::Enabled requires sm_86"
            )
        };
        with_ftz!(DEN => FTZ in with_nan!(NAN => N in
            with_xorsign!(XOR => X in max_f32::<FTZ, N, X>(self, rhs))
        ))
    }

    #[inline]
    fn to_i32<const RND: Rounding, const DEN: Denormals>(self) -> i32 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in
            unsafe { nv::cvt_s32_f32::<R, FTZ>(self) }
        ))
    }

    #[inline]
    fn to_u32<const RND: Rounding, const DEN: Denormals>(self) -> u32 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in
            unsafe { nv::cvt_u32_f32::<R, FTZ>(self) }
        ))
    }

    #[inline]
    fn to_i64<const RND: Rounding, const DEN: Denormals>(self) -> i64 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in
            unsafe { nv::cvt_s64_f32::<R, FTZ>(self) }
        ))
    }

    #[inline]
    fn to_u64<const RND: Rounding, const DEN: Denormals>(self) -> u64 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in
            unsafe { nv::cvt_u64_f32::<R, FTZ>(self) }
        ))
    }

    #[inline]
    fn round_to_integral<const RND: Rounding, const DEN: Denormals>(self) -> f32 {
        with_rounding!(RND => R in with_ftz!(DEN => FTZ in
            unsafe { nv::cvt_f32_f32::<R, FTZ>(self) }
        ))
    }

    #[inline]
    fn saturate<const DEN: Denormals>(self) -> f32 {
        with_ftz!(DEN => FTZ in unsafe { nv::cvt_sat_f32_f32::<FTZ>(self) })
    }

    #[inline]
    fn to_tf32<const RND: Rounding, const RELU: Relu, const SAT: Overflow>(self) -> f32 {
        const {
            assert!(
                matches!(
                    RND,
                    Rounding::NearestEven | Rounding::TowardZero | Rounding::NearestAway
                ),
                "unsupported rounding mode: tf32 conversion supports NearestEven, TowardZero and NearestAway"
            );
            assert!(
                !matches!(RND, Rounding::NearestAway) || matches!(RELU, Relu::Disabled),
                "tf32 conversion with NearestAway rounding does not support Relu"
            );
            assert!(
                matches!(RND, Rounding::NearestAway) || cfg!(target_feature = "sm_90"),
                "tf32 conversion with NearestEven or TowardZero rounding requires sm_90"
            );
            assert!(
                matches!(RND, Rounding::NearestAway)
                    || matches!(SAT, Overflow::Infinity)
                    || cfg!(target_feature = "sm_100"),
                "tf32 conversion with NearestEven or TowardZero rounding and SaturateFinite requires sm_100"
            );
        }
        let bits = match RND {
            Rounding::NearestAway => {
                with_satfinite!(SAT => SATF in unsafe { nv::cvt_rna_tf32_f32::<SATF>(self) })
            }
            #[cfg(target_feature = "sm_90")]
            _ => const_map!(RND => R: nv::Rounding {
                Rounding::NearestEven => nv::Rounding::Rn,
                Rounding::TowardZero => nv::Rounding::Rz,
            } in with_relu!(RELU => RL in with_satfinite!(SAT => SATF in
                cvt_tf32_rn_rz::<R, RL, SATF>(self)
            ))),
            #[cfg(not(target_feature = "sm_90"))]
            _ => unreachable!(),
        };
        f32::from_bits(bits)
    }
}
