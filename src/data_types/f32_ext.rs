// --- LLM-generated --- //
use core::{
    arch::{asm, nvptx as nv},
    simd::{Simd, SimdElement},
};

use crate::{Denormals, NanHandling, Overflow, Relu, Rounding, Saturation, XorSignAbs, f32xN};

/// Rejects the rounding modes that are not IEEE 754 rounding modes at compile time.
pub(crate) const fn assert_ieee_rounding(rnd: Rounding) {
    assert!(
        matches!(
            rnd,
            Rounding::NearestEven | Rounding::TowardZero | Rounding::Down | Rounding::Up
        ),
        "unsupported rounding mode: expected NearestEven, TowardZero, Down or Up"
    );
}

/// Expands `$body` with a local `const $name: nv::Rounding` equal to the IEEE rounding mode
/// `$rnd`, rejecting other modes at compile time.
macro_rules! with_rounding {
    ($rnd:ident => $name:ident in $body:expr) => {{
        const { $crate::data_types::f32_ext::assert_ieee_rounding($rnd) };
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
/// The trait is implemented for [`f32`] and for the `f32` vectors [`f32xN<N>`](crate::f32xN) of
/// every length. Methods take and return `Self`, so on a vector they operate element-wise; the
/// integer conversions return the associated integer types, which are the scalars for `f32` and
/// [`Simd`](core::simd::Simd)s of the same length for `f32xN<N>`. The documentation of these
/// methods describes a single element. The horizontal reductions (`reduce_*`) combine all
/// elements into one `f32`.
///
/// Methods taking a [`Rounding`] accept [`Rounding::NearestEven`], [`Rounding::TowardZero`],
/// [`Rounding::Down`] and [`Rounding::Up`]; other modes are rejected at compile time. Methods
/// taking [`Denormals`] either preserve subnormal values or flush subnormal inputs and results
/// to sign-preserving zero.
///
/// Arithmetic methods are emitted with an explicit rounding modifier (e.g. `add.rn.f32`), which
/// the PTX optimizer treats conservatively: separate multiplications and additions are never
/// contracted into a fused multiply-add. Conversions to integers use the integer rounding
/// modifiers (`.rni`, `.rzi`, `.rmi`, `.rpi`).
///
/// The `.approx` instructions (methods with the `_approx` suffix) trade accuracy for speed. Where
/// the PTX ISA specifies an error bound, it is given in the method's documentation; some bounds
/// only hold over a restricted input range.
///
/// # `f32` vectors on Blackwell
///
/// With `sm_100` or later, [`add_with`](F32Ext::add_with), [`mul_with`](F32Ext::mul_with) and
/// [`fma`](F32Ext::fma) on `f32xN<N>` process pairs of elements with the packed `add`, `mul` and
/// `fma` `.f32x2` instructions. The packed instructions have no saturation modifier, so
/// [`Saturation::ClampToUnit`] uses the scalar instructions on every element.
/// [`reduce_sum`](F32Ext::reduce_sum) adds pairs of elements with `add.rn.f32x2`, and
/// [`reduce_min_with`](F32Ext::reduce_min_with) and [`reduce_max_with`](F32Ext::reduce_max_with)
/// use the three-input `min.f32` and `max.f32`. All other methods (and all methods on earlier
/// architectures) apply the scalar instruction to each element.
///
/// Note that LLVM can also lower the `+`, `-` and `*` operators of `f32xN<N>` to `f32x2`
/// instructions on `sm_100` and later (with round-to-nearest-even and subnormals preserved); use the
/// methods of this trait to select an explicit rounding mode or flush-to-zero.
pub trait F32Ext: Sized {
    /// The signed 32-bit integer type with the same shape as `Self`: `i32` for `f32` and
    /// `Simd<i32, N>` for `f32xN<N>`.
    type I32;

    /// The unsigned 32-bit integer type with the same shape as `Self`: `u32` for `f32` and
    /// `Simd<u32, N>` for `f32xN<N>`.
    type U32;

    /// The signed 64-bit integer type with the same shape as `Self`: `i64` for `f32` and
    /// `Simd<i64, N>` for `f32xN<N>`.
    type I64;

    /// The unsigned 64-bit integer type with the same shape as `Self`: `u64` for `f32` and
    /// `Simd<u64, N>` for `f32xN<N>`.
    type U64;

    /// Adds two values.
    ///
    /// `RND` selects the rounding mode, `DEN` the subnormal handling, and `SAT` optionally clamps
    /// the result to `[0.0, 1.0]` (NaN results become `+0.0`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add>
    fn add_with<const RND: Rounding, const DEN: Denormals, const SAT: Saturation>(
        self,
        rhs: Self,
    ) -> Self;

    /// Multiplies two values.
    ///
    /// `RND` selects the rounding mode and `DEN` the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-mul>
    fn mul_with<const RND: Rounding, const DEN: Denormals>(self, rhs: Self) -> Self;

    /// Computes `self * b + c` with a single rounding.
    ///
    /// The product is computed to infinite precision and `c` is added to it, also in infinite
    /// precision, before rounding to `f32` with `RND`. `DEN` selects the subnormal handling, and
    /// `SAT` optionally clamps the result to `[0.0, 1.0]` (NaN results become `+0.0`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma>
    fn fma<const RND: Rounding, const DEN: Denormals, const SAT: Saturation>(
        self,
        b: Self,
        c: Self,
    ) -> Self;

    /// Divides `self` by `rhs`, with the IEEE 754 compliant `div.rnd` instruction.
    ///
    /// `RND` selects the rounding mode and `DEN` the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div>
    fn div_with<const RND: Rounding, const DEN: Denormals>(self, rhs: Self) -> Self;

    /// Divides `self` by `rhs` using the approximate `div.approx` instruction, computed as
    /// `self * (1 / rhs)`.
    ///
    /// For `|rhs|` in `[2^-126, 2^126]`, the maximum error is 2 ulp. For
    /// `2^126 < |rhs| < 2^128`, the result is NaN if `self` is infinite and a sign-preserving zero
    /// otherwise. `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div>
    fn div_approx<const DEN: Denormals>(self, rhs: Self) -> Self;

    /// Divides `self` by `rhs` using the `div.full` instruction, which is faster than IEEE 754
    /// division but less accurate (and not IEEE 754 compliant).
    ///
    /// The maximum error is 2 ulp over the full range of inputs. `DEN` selects the subnormal
    /// handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-div>
    fn div_full_approx<const DEN: Denormals>(self, rhs: Self) -> Self;

    /// Takes the reciprocal, with the IEEE 754 compliant `rcp.rnd` instruction.
    ///
    /// `RND` selects the rounding mode and `DEN` the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp>
    fn rcp_with<const RND: Rounding, const DEN: Denormals>(self) -> Self;

    /// Takes a fast, approximate reciprocal.
    ///
    /// The maximum error is 1 ulp over the full range of inputs. Subnormal inputs and results are
    /// flushed to sign-preserving zero (`rcp.approx.ftz`). `±0.0` maps to `±inf`, `±inf` to
    /// `±0.0` and NaN to NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rcp>
    fn rcp_approx_ftz(self) -> Self;

    /// Takes the square root, with the IEEE 754 compliant `sqrt.rnd` instruction.
    ///
    /// `RND` selects the rounding mode and `DEN` the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sqrt>
    fn sqrt_with<const RND: Rounding, const DEN: Denormals>(self) -> Self;

    /// Takes a fast, approximate square root (`sqrt.approx`).
    ///
    /// The maximum relative error over the entire positive finite range is 2^-23. `DEN` selects
    /// the subnormal handling. Negative non-zero inputs and `-inf` give NaN, and `±0.0` gives
    /// `±0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sqrt>
    fn sqrt_approx<const DEN: Denormals>(self) -> Self;

    /// Takes a fast, approximate reciprocal square root (`rsqrt.approx`).
    ///
    /// The maximum relative error over the entire positive finite range is 2^-22.9. `DEN` selects
    /// the subnormal handling. Negative inputs and `-inf` give NaN, `±0.0` gives `±inf` and `+inf`
    /// gives `+0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-rsqrt>
    fn rsqrt_approx<const DEN: Denormals>(self) -> Self;

    /// Computes the sine of `self` (in radians), approximately (`sin.approx`).
    ///
    /// The maximum absolute error is 2^-20.5 in the range `-2 * pi ..= 2 * pi` and 2^-14.7 in
    /// the range `-100 * pi ..= 100 * pi`. Outside of the latter, there is no error guarantee.
    /// `±inf` gives NaN. `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-sin>
    fn sin_approx<const DEN: Denormals>(self) -> Self;

    /// Computes the cosine of `self` (in radians), approximately (`cos.approx`).
    ///
    /// The maximum absolute error is 2^-20.4 in the range `-2 * pi ..= 2 * pi` and 2^-14.7 in
    /// the range `-100 * pi ..= 100 * pi`. Outside of the latter, there is no error guarantee.
    /// `±inf` gives NaN. `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-cos>
    fn cos_approx<const DEN: Denormals>(self) -> Self;

    /// Computes the base-2 logarithm, approximately (`lg2.approx`).
    ///
    /// The maximum absolute error is 2^-22 for inputs in `(0.5, 2)`; for other positive finite
    /// inputs, the maximum relative error is 2^-22. Negative non-zero inputs and `-inf` give NaN,
    /// and `±0.0` gives `-inf`. `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-lg2>
    fn log2_approx<const DEN: Denormals>(self) -> Self;

    /// Computes the base-2 exponential `2^self`, approximately (`ex2.approx`).
    ///
    /// The maximum error is 2 ulp from the correctly rounded result over the full range of
    /// inputs. `-inf` gives `+0.0` and `+inf` gives `+inf`. `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-ex2>
    fn exp2_approx<const DEN: Denormals>(self) -> Self;

    /// Computes the hyperbolic tangent, approximately (`tanh.approx`).
    ///
    /// The maximum relative error over the entire floating point range is 2^-11. Subnormal inputs
    /// are supported. `-inf` gives `-1.0`, `+inf` gives `1.0`, `±0.0` gives `±0.0` and NaN gives
    /// NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-tanh>
    fn tanh_approx(self) -> Self;

    /// Computes the absolute value.
    ///
    /// `DEN` selects the subnormal handling. A NaN input yields an unspecified NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-abs>
    fn abs_with<const DEN: Denormals>(self) -> Self;

    /// Finds the minimum of two values.
    ///
    /// `-0.0` is treated as less than `+0.0`. `NAN` selects whether a NaN input is ignored
    /// ([`NanHandling::Ignore`]) or propagated ([`NanHandling::Propagate`], which yields a
    /// canonical NaN if either input is NaN). `DEN` selects the subnormal handling.
    ///
    /// With `XOR` set to [`XorSignAbs::Enabled`] (`min.xorsign.abs`), the magnitude of the result
    /// is the minimum of the absolute values of the inputs, and its sign is the XOR of the input
    /// signs (the sign handling is skipped if the result is NaN). This requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-min>
    fn min_with<const DEN: Denormals, const NAN: NanHandling, const XOR: XorSignAbs>(
        self,
        rhs: Self,
    ) -> Self;

    /// Finds the maximum of two values.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `NAN` selects whether a NaN input is ignored
    /// ([`NanHandling::Ignore`]) or propagated ([`NanHandling::Propagate`], which yields a
    /// canonical NaN if either input is NaN). `DEN` selects the subnormal handling.
    ///
    /// With `XOR` set to [`XorSignAbs::Enabled`] (`max.xorsign.abs`), the magnitude of the result
    /// is the maximum of the absolute values of the inputs, and its sign is the XOR of the input
    /// signs (the sign handling is skipped if the result is NaN). This requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max>
    fn max_with<const DEN: Denormals, const NAN: NanHandling, const XOR: XorSignAbs>(
        self,
        rhs: Self,
    ) -> Self;

    /// Sums all elements. For `f32`, this returns `self`.
    ///
    /// The elements are added with `add.rn.f32` (round to nearest, ties to even; subnormal values
    /// are preserved) in an unspecified order, which may differ between architectures. On
    /// `sm_100` and later, pairs of elements are added with `add.rn.f32x2`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add>
    fn reduce_sum(self) -> f32;

    /// Finds the minimum element. For `f32`, this returns `self`.
    ///
    /// `-0.0` is treated as less than `+0.0`. NaN elements are ignored; the result is NaN only if
    /// all elements are NaN. Subnormal values are preserved. See
    /// [`reduce_min_with`](F32Ext::reduce_min_with).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-min>
    #[inline]
    fn reduce_min(self) -> f32 {
        self.reduce_min_with::<{ Denormals::Preserve }, { NanHandling::Ignore }>()
    }

    /// Finds the maximum element. For `f32`, this returns `self`.
    ///
    /// `+0.0` is treated as greater than `-0.0`. NaN elements are ignored; the result is NaN only
    /// if all elements are NaN. Subnormal values are preserved. See
    /// [`reduce_max_with`](F32Ext::reduce_max_with).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max>
    #[inline]
    fn reduce_max(self) -> f32 {
        self.reduce_max_with::<{ Denormals::Preserve }, { NanHandling::Ignore }>()
    }

    /// Finds the minimum element with explicit modifiers. For `f32`, this returns `self`
    /// unchanged.
    ///
    /// `-0.0` is treated as less than `+0.0`. `NAN` selects whether NaN elements are ignored
    /// ([`NanHandling::Ignore`]; the result is NaN only if all elements are NaN) or propagated
    /// ([`NanHandling::Propagate`], which yields a canonical NaN if any element is NaN). `DEN`
    /// selects the subnormal handling.
    ///
    /// The elements are combined with `min` instructions in a tree. On `sm_100` and later, the
    /// three-input form of `min.f32` combines three values per instruction.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-min>
    fn reduce_min_with<const DEN: Denormals, const NAN: NanHandling>(self) -> f32;

    /// Finds the maximum element with explicit modifiers. For `f32`, this returns `self`
    /// unchanged.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `NAN` selects whether NaN elements are ignored
    /// ([`NanHandling::Ignore`]; the result is NaN only if all elements are NaN) or propagated
    /// ([`NanHandling::Propagate`], which yields a canonical NaN if any element is NaN). `DEN`
    /// selects the subnormal handling.
    ///
    /// The elements are combined with `max` instructions in a tree. On `sm_100` and later, the
    /// three-input form of `max.f32` combines three values per instruction.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max>
    fn reduce_max_with<const DEN: Denormals, const NAN: NanHandling>(self) -> f32;

    /// Converts to a signed 32-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `i32::MIN` / `i32::MAX`, and NaN becomes zero. If `DEN`
    /// is [`Denormals::FlushToZero`], subnormal inputs are flushed to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_i32<const RND: Rounding, const DEN: Denormals>(self) -> Self::I32;

    /// Converts to an unsigned 32-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `0` / `u32::MAX`, and NaN becomes zero. If `DEN` is
    /// [`Denormals::FlushToZero`], subnormal inputs are flushed to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_u32<const RND: Rounding, const DEN: Denormals>(self) -> Self::U32;

    /// Converts to a signed 64-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `i64::MIN` / `i64::MAX`, and NaN becomes `i64::MIN`. If
    /// `DEN` is [`Denormals::FlushToZero`], subnormal inputs are flushed to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_i64<const RND: Rounding, const DEN: Denormals>(self) -> Self::I64;

    /// Converts to an unsigned 64-bit integer, rounding to an integer with `RND`.
    ///
    /// Out-of-range values saturate to `0` / `u64::MAX`, and NaN becomes `1 << 63`. If `DEN` is
    /// [`Denormals::FlushToZero`], subnormal inputs are flushed to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_u64<const RND: Rounding, const DEN: Denormals>(self) -> Self::U64;

    /// Rounds to an integral value, returned as an `f32`.
    ///
    /// [`Rounding::NearestEven`] rounds to nearest with ties to even, [`Rounding::TowardZero`]
    /// truncates, [`Rounding::Down`] takes the floor and [`Rounding::Up`] the ceiling. `DEN`
    /// selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn round_to_integral<const RND: Rounding, const DEN: Denormals>(self) -> Self;

    /// Clamps to the range `[0.0, 1.0]`. NaN becomes `+0.0`.
    ///
    /// `DEN` selects the subnormal handling.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn saturate<const DEN: Denormals>(self) -> Self;

    /// Converts to the `tf32` format, returning its bit pattern stored in an `f32`.
    ///
    /// The PTX ISA leaves the internal layout of `tf32` implementation defined, so the result
    /// should only be passed to consumers of `tf32` values, such as `tf32` tensor core
    /// instructions, rather than used in `f32` arithmetic.
    ///
    /// `RND` must be [`Rounding::NearestEven`], [`Rounding::TowardZero`] or
    /// [`Rounding::NearestAway`]. `RELU` clamps negative results to zero (NaN results become
    /// canonical NaN) and `SAT` clamps results whose magnitude exceeds the largest finite `tf32`
    /// value to that value, keeping the sign (NaN inputs still produce NaN).
    ///
    /// [`Rounding::NearestAway`] requires `sm_80` and does not support `RELU`. The other roundings
    /// require `sm_90`, and combining them with [`Overflow::SaturateFinite`] additionally requires
    /// `sm_100`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_tf32<const RND: Rounding, const RELU: Relu, const SAT: Overflow>(self) -> Self;
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
    type I32 = i32;
    type U32 = u32;
    type I64 = i64;
    type U64 = u64;

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
    fn tanh_approx(self) -> f32 {
        let result: f32;
        // SAFETY: `tanh.approx.f32` only reads and writes registers.
        unsafe {
            asm!(
                "tanh.approx.f32 {d}, {a};",
                d = out(reg32) result,
                a = in(reg32) self,
                options(pure, nomem, nostack),
            );
        }
        result
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
    fn reduce_sum(self) -> f32 {
        self
    }

    #[inline]
    fn reduce_min_with<const DEN: Denormals, const NAN: NanHandling>(self) -> f32 {
        self
    }

    #[inline]
    fn reduce_max_with<const DEN: Denormals, const NAN: NanHandling>(self) -> f32 {
        self
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

/// Applies `f` to every element of `a`.
#[inline(always)]
fn map1<const N: usize>(a: f32xN<N>, f: impl Fn(f32) -> f32) -> f32xN<N> {
    let a = a.to_array();
    f32xN::from_array(crate::data_types::array_from_fn(|i| f(a[i])))
}

/// Converts every element of `a` with `f`.
#[inline(always)]
fn convert<const N: usize, R: SimdElement>(a: f32xN<N>, f: impl Fn(f32) -> R) -> Simd<R, N> {
    let a = a.to_array();
    Simd::from_array(crate::data_types::array_from_fn(|i| f(a[i])))
}

/// Applies `f` to every pair of elements of `a` and `b`.
#[inline(always)]
fn map2<const N: usize>(a: f32xN<N>, b: f32xN<N>, f: impl Fn(f32, f32) -> f32) -> f32xN<N> {
    let (a, b) = (a.to_array(), b.to_array());
    f32xN::from_array(crate::data_types::array_from_fn(|i| f(a[i], b[i])))
}

/// Applies `f` to every triple of elements of `a`, `b` and `c`.
#[inline(always)]
fn map3<const N: usize>(
    a: f32xN<N>,
    b: f32xN<N>,
    c: f32xN<N>,
    f: impl Fn(f32, f32, f32) -> f32,
) -> f32xN<N> {
    let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
    f32xN::from_array(crate::data_types::array_from_fn(|i| f(a[i], b[i], c[i])))
}

/// Reduces the elements of `x` to a single value. Elements `2 * i` and `2 * i + 1` form pair `i`;
/// the pairs are combined element-wise with `pair_op` in a tree, the two elements of the final pair
/// with `op`, and then the last element of an odd number of elements, again with `op`.
///
/// The pairs are kept in `x`, used as a ring buffer of `N / 2` pairs holding a FIFO queue: step
/// `k` pops the pairs at queue positions `2 * k` and `2 * k + 1` and pushes their combination at
/// queue position `N / 2 + k`, i.e. into pair `k`. All indices are constant after unrolling.
#[inline(always)]
fn reduce_pairs<const N: usize>(
    mut x: [f32; N],
    pair_op: impl Fn([f32; 2], [f32; 2]) -> [f32; 2],
    op: impl Fn(f32, f32) -> f32,
) -> f32 {
    if const { N == 1 } {
        return x[0];
    }
    let p = N / 2;
    #[rustc_unroll(full)]
    for k in 0..p - 1 {
        let (a, b) = (2 * k % p, (2 * k + 1) % p);
        [x[2 * k], x[2 * k + 1]] = pair_op([x[2 * a], x[2 * a + 1]], [x[2 * b], x[2 * b + 1]]);
    }
    let last = (2 * p - 2) % p;
    let reduced = op(x[2 * last], x[2 * last + 1]);
    if const { N % 2 == 1 } {
        op(reduced, x[N - 1])
    } else {
        reduced
    }
}

/// Reduces the elements of `x` to a single value, combining three values per step with `op3`, and
/// two values with `op2` in the last step if `N` is even.
///
/// `x` is used as a ring buffer holding a FIFO queue of values: step `k` pops the values at queue
/// positions `3 * k` to `3 * k + 2` and pushes their combination at queue position `N + k`, i.e.
/// into element `k`. All indices are constant after unrolling.
#[cfg(target_feature = "sm_100")]
#[inline(always)]
fn reduce_triples<const N: usize>(
    mut x: [f32; N],
    op3: impl Fn(f32, f32, f32) -> f32,
    op2: impl Fn(f32, f32) -> f32,
) -> f32 {
    let mut steps = (N - 1) / 2;
    #[rustc_unroll(full)]
    for k in 0..steps {
        x[k] = op3(x[3 * k % N], x[(3 * k + 1) % N], x[(3 * k + 2) % N]);
    }
    if N.is_multiple_of(2) {
        x[steps] = op2(x[3 * steps % N], x[(3 * steps + 1) % N]);
        steps += 1;
    }
    x[(N + steps - 1) % N]
}

/// The three-input `min` and `max` instructions of `sm_100`.
#[cfg(target_feature = "sm_100")]
mod minmax3 {
    use super::*;

    macro_rules! emit_minmax3 {
        ($op:literal $d:ident $a:ident $b:ident $c:ident $ftz:literal, $nan:literal) => {
            // SAFETY: the instruction only reads and writes registers.
            unsafe {
                asm!(
                    concat!($op, $ftz, $nan, ".f32 {d}, {a}, {b}, {c};"),
                    d = out(reg32) $d,
                    a = in(reg32) $a,
                    b = in(reg32) $b,
                    c = in(reg32) $c,
                    options(pure, nomem, nostack),
                )
            }
        };
    }

    macro_rules! minmax3_fn {
        ($name:ident, $op:literal) => {
            #[inline(always)]
            pub(super) fn $name<const DEN: Denormals, const NAN: NanHandling>(
                a: f32,
                b: f32,
                c: f32,
            ) -> f32 {
                let mut d = 0.0f32;
                asm_variants!(emit_minmax3!($op d a b c);
                    DEN => {
                        Denormals::Preserve => "",
                        Denormals::FlushToZero => ".ftz",
                    },
                    NAN => {
                        NanHandling::Ignore => "",
                        NanHandling::Propagate => ".NaN",
                    },
                );
                d
            }
        };
    }

    minmax3_fn!(min, "min");
    minmax3_fn!(max, "max");
}

/// The packed `f32x2` instructions of `sm_100`, whose operands are `.b64` registers holding two
/// `f32` values each, with the first lane in the low 32 bits.
#[cfg(target_feature = "sm_100")]
mod packed {
    use super::*;
    use crate::{data_types::f32_pair, f32x2};

    /// Emits `$op` on pairs of elements. The pairs are packed into `.b64` registers and unpacked
    /// again inside the assembly block, with `mov.b64`.
    macro_rules! emit_f32x2 {
        ($op:literal $d:ident $a:ident $b:ident $rnd:literal, $ftz:literal) => {
            // SAFETY: the instruction only reads and writes registers.
            unsafe {
                asm!(
                    "{{",
                    ".reg .b64 %pa, %pb, %pd;",
                    "mov.b64 %pa, {{{a0}, {a1}}};",
                    "mov.b64 %pb, {{{b0}, {b1}}};",
                    concat!($op, $rnd, $ftz, ".f32x2 %pd, %pa, %pb;"),
                    "mov.b64 {{{d0}, {d1}}}, %pd;",
                    "}}",
                    d0 = out(reg32) $d[0],
                    d1 = out(reg32) $d[1],
                    a0 = in(reg32) $a[0],
                    a1 = in(reg32) $a[1],
                    b0 = in(reg32) $b[0],
                    b1 = in(reg32) $b[1],
                    options(pure, nomem, nostack),
                )
            }
        };
        ($op:literal $d:ident $a:ident $b:ident $c:ident $rnd:literal, $ftz:literal) => {
            // SAFETY: the instruction only reads and writes registers.
            unsafe {
                asm!(
                    "{{",
                    ".reg .b64 %pa, %pb, %pc, %pd;",
                    "mov.b64 %pa, {{{a0}, {a1}}};",
                    "mov.b64 %pb, {{{b0}, {b1}}};",
                    "mov.b64 %pc, {{{c0}, {c1}}};",
                    concat!($op, $rnd, $ftz, ".f32x2 %pd, %pa, %pb, %pc;"),
                    "mov.b64 {{{d0}, {d1}}}, %pd;",
                    "}}",
                    d0 = out(reg32) $d[0],
                    d1 = out(reg32) $d[1],
                    a0 = in(reg32) $a[0],
                    a1 = in(reg32) $a[1],
                    b0 = in(reg32) $b[0],
                    b1 = in(reg32) $b[1],
                    c0 = in(reg32) $c[0],
                    c1 = in(reg32) $c[1],
                    options(pure, nomem, nostack),
                )
            }
        };
    }

    /// Defines a function emitting the packed instruction `$op` on pairs of elements, with the
    /// given source operands.
    macro_rules! packed_fn {
        ($name:ident, $op:literal, $($src:ident),+) => {
            #[inline(always)]
            pub(super) fn $name<const RND: Rounding, const DEN: Denormals>(
                $($src: [f32; 2]),+
            ) -> [f32; 2] {
                const { assert_ieee_rounding(RND) };
                let mut d = [0.0f32; 2];
                asm_variants!(emit_f32x2!($op d $($src)+);
                    RND => {
                        Rounding::NearestEven => ".rn",
                        Rounding::TowardZero => ".rz",
                        Rounding::Down => ".rm",
                        Rounding::Up => ".rp",
                    },
                    DEN => {
                        Denormals::Preserve => "",
                        Denormals::FlushToZero => ".ftz",
                    },
                );
                d
            }
        };
    }

    packed_fn!(add, "add", a, b);
    packed_fn!(mul, "mul", a, b);
    packed_fn!(fma, "fma", a, b, c);

    /// Applies `packed` to every pair of elements of `a` and `b`.
    #[inline(always)]
    pub(super) fn map2<const N: usize>(
        a: f32xN<N>,
        b: f32xN<N>,
        packed: impl Fn([f32; 2], [f32; 2]) -> [f32; 2],
    ) -> f32xN<N> {
        let (a, b) = (a.to_array(), b.to_array());
        f32xN::from_pairs_with(|i| f32x2::from_array(packed(f32_pair(&a, i), f32_pair(&b, i))))
    }

    /// Applies `packed` to every pair of elements of `a`, `b` and `c`.
    #[inline(always)]
    pub(super) fn map3<const N: usize>(
        a: f32xN<N>,
        b: f32xN<N>,
        c: f32xN<N>,
        packed: impl Fn([f32; 2], [f32; 2], [f32; 2]) -> [f32; 2],
    ) -> f32xN<N> {
        let (a, b, c) = (a.to_array(), b.to_array(), c.to_array());
        f32xN::from_pairs_with(|i| {
            f32x2::from_array(packed(f32_pair(&a, i), f32_pair(&b, i), f32_pair(&c, i)))
        })
    }
}

impl<const N: usize> F32Ext for f32xN<N> {
    type I32 = Simd<i32, N>;
    type U32 = Simd<u32, N>;
    type I64 = Simd<i64, N>;
    type U64 = Simd<u64, N>;

    #[inline]
    fn add_with<const RND: Rounding, const DEN: Denormals, const SAT: Saturation>(
        self,
        rhs: Self,
    ) -> Self {
        #[cfg(target_feature = "sm_100")]
        if const { !matches!(SAT, Saturation::ClampToUnit) } {
            return packed::map2(self, rhs, packed::add::<RND, DEN>);
        }
        map2(self, rhs, |a, b| a.add_with::<RND, DEN, SAT>(b))
    }

    #[inline]
    fn mul_with<const RND: Rounding, const DEN: Denormals>(self, rhs: Self) -> Self {
        #[cfg(target_feature = "sm_100")]
        return packed::map2(self, rhs, packed::mul::<RND, DEN>);
        #[cfg(not(target_feature = "sm_100"))]
        map2(self, rhs, |a, b| a.mul_with::<RND, DEN>(b))
    }

    #[inline]
    fn fma<const RND: Rounding, const DEN: Denormals, const SAT: Saturation>(
        self,
        b: Self,
        c: Self,
    ) -> Self {
        #[cfg(target_feature = "sm_100")]
        if const { !matches!(SAT, Saturation::ClampToUnit) } {
            return packed::map3(self, b, c, packed::fma::<RND, DEN>);
        }
        map3(self, b, c, |a, b, c| a.fma::<RND, DEN, SAT>(b, c))
    }

    #[inline]
    fn div_with<const RND: Rounding, const DEN: Denormals>(self, rhs: Self) -> Self {
        map2(self, rhs, |a, b| a.div_with::<RND, DEN>(b))
    }

    #[inline]
    fn div_approx<const DEN: Denormals>(self, rhs: Self) -> Self {
        map2(self, rhs, |a, b| a.div_approx::<DEN>(b))
    }

    #[inline]
    fn div_full_approx<const DEN: Denormals>(self, rhs: Self) -> Self {
        map2(self, rhs, |a, b| a.div_full_approx::<DEN>(b))
    }

    #[inline]
    fn rcp_with<const RND: Rounding, const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.rcp_with::<RND, DEN>())
    }

    #[inline]
    fn rcp_approx_ftz(self) -> Self {
        map1(self, |a| a.rcp_approx_ftz())
    }

    #[inline]
    fn sqrt_with<const RND: Rounding, const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.sqrt_with::<RND, DEN>())
    }

    #[inline]
    fn sqrt_approx<const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.sqrt_approx::<DEN>())
    }

    #[inline]
    fn rsqrt_approx<const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.rsqrt_approx::<DEN>())
    }

    #[inline]
    fn sin_approx<const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.sin_approx::<DEN>())
    }

    #[inline]
    fn cos_approx<const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.cos_approx::<DEN>())
    }

    #[inline]
    fn log2_approx<const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.log2_approx::<DEN>())
    }

    #[inline]
    fn exp2_approx<const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.exp2_approx::<DEN>())
    }

    #[inline]
    fn tanh_approx(self) -> Self {
        map1(self, |a| a.tanh_approx())
    }

    #[inline]
    fn abs_with<const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.abs_with::<DEN>())
    }

    #[inline]
    fn min_with<const DEN: Denormals, const NAN: NanHandling, const XOR: XorSignAbs>(
        self,
        rhs: Self,
    ) -> Self {
        map2(self, rhs, |a, b| a.min_with::<DEN, NAN, XOR>(b))
    }

    #[inline]
    fn max_with<const DEN: Denormals, const NAN: NanHandling, const XOR: XorSignAbs>(
        self,
        rhs: Self,
    ) -> Self {
        map2(self, rhs, |a, b| a.max_with::<DEN, NAN, XOR>(b))
    }

    #[inline]
    fn reduce_sum(self) -> f32 {
        let add = |a: f32, b: f32| {
            a.add_with::<{ Rounding::NearestEven }, { Denormals::Preserve }, { Saturation::None }>(
                b,
            )
        };
        #[cfg(target_feature = "sm_100")]
        let pair_add = packed::add::<{ Rounding::NearestEven }, { Denormals::Preserve }>;
        #[cfg(not(target_feature = "sm_100"))]
        let pair_add = |a: [f32; 2], b: [f32; 2]| [add(a[0], b[0]), add(a[1], b[1])];
        reduce_pairs(self.to_array(), pair_add, add)
    }

    #[inline]
    fn reduce_min_with<const DEN: Denormals, const NAN: NanHandling>(self) -> f32 {
        let min = |a: f32, b: f32| a.min_with::<DEN, NAN, { XorSignAbs::Disabled }>(b);
        #[cfg(target_feature = "sm_100")]
        return reduce_triples(self.to_array(), minmax3::min::<DEN, NAN>, min);
        #[cfg(not(target_feature = "sm_100"))]
        reduce_pairs(
            self.to_array(),
            |a, b| [min(a[0], b[0]), min(a[1], b[1])],
            min,
        )
    }

    #[inline]
    fn reduce_max_with<const DEN: Denormals, const NAN: NanHandling>(self) -> f32 {
        let max = |a: f32, b: f32| a.max_with::<DEN, NAN, { XorSignAbs::Disabled }>(b);
        #[cfg(target_feature = "sm_100")]
        return reduce_triples(self.to_array(), minmax3::max::<DEN, NAN>, max);
        #[cfg(not(target_feature = "sm_100"))]
        reduce_pairs(
            self.to_array(),
            |a, b| [max(a[0], b[0]), max(a[1], b[1])],
            max,
        )
    }

    #[inline]
    fn to_i32<const RND: Rounding, const DEN: Denormals>(self) -> Self::I32 {
        convert(self, |a| a.to_i32::<RND, DEN>())
    }

    #[inline]
    fn to_u32<const RND: Rounding, const DEN: Denormals>(self) -> Self::U32 {
        convert(self, |a| a.to_u32::<RND, DEN>())
    }

    #[inline]
    fn to_i64<const RND: Rounding, const DEN: Denormals>(self) -> Self::I64 {
        convert(self, |a| a.to_i64::<RND, DEN>())
    }

    #[inline]
    fn to_u64<const RND: Rounding, const DEN: Denormals>(self) -> Self::U64 {
        convert(self, |a| a.to_u64::<RND, DEN>())
    }

    #[inline]
    fn round_to_integral<const RND: Rounding, const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.round_to_integral::<RND, DEN>())
    }

    #[inline]
    fn saturate<const DEN: Denormals>(self) -> Self {
        map1(self, |a| a.saturate::<DEN>())
    }

    #[inline]
    fn to_tf32<const RND: Rounding, const RELU: Relu, const SAT: Overflow>(self) -> Self {
        map1(self, |a| a.to_tf32::<RND, RELU, SAT>())
    }
}
