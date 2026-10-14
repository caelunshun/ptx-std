// --- LLM-generated --- //
use crate::modifiers::*;
use bytemuck::{Pod, Zeroable};
use core::{
    arch::nvptx as nv,
    cmp::Ordering,
    fmt,
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    simd::Simd,
};

use super::{
    array_from_fn,
    vector::{Pairs, SupportedVectorLen, VectorLen, cast, half_vector, vector_element},
};

/// Alias for the unstable primitive `f16`, which our `f16` type shadows at the module root.
mod prim {
    pub type F16 = f16;
}

/// IEEE 754 binary16 half-precision float.
///
/// Arithmetic operators round to nearest, ties to even, and preserve subnormal values. Methods
/// suffixed `_with` select instruction modifiers (flush-to-zero, saturation, ...) via const
/// generics.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f16(u16);

impl f16 {
    /// Positive zero.
    pub const ZERO: Self = Self(0x0000);
    /// Negative zero.
    pub const NEG_ZERO: Self = Self(0x8000);
    /// One.
    pub const ONE: Self = Self(0x3C00);
    /// Negative one.
    pub const NEG_ONE: Self = Self(0xBC00);
    /// Positive infinity.
    pub const INFINITY: Self = Self(0x7C00);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(0xFC00);
    /// A quiet NaN.
    pub const NAN: Self = Self(0x7E00);
    /// The largest finite value (65504).
    pub const MAX: Self = Self(0x7BFF);
    /// The smallest finite value (-65504).
    pub const MIN: Self = Self(0xFBFF);
    /// The smallest positive normal value (2^-14).
    pub const MIN_POSITIVE: Self = Self(0x0400);
    /// The smallest positive subnormal value (2^-24).
    pub const MIN_POSITIVE_SUBNORMAL: Self = Self(0x0001);
    /// The difference between `1.0` and the next larger representable value (2^-10).
    pub const EPSILON: Self = Self(0x1400);

    /// Creates a value from its raw bit representation.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    #[inline]
    fn prim(self) -> prim::F16 {
        prim::F16::from_bits(self.0)
    }

    #[inline]
    fn from_prim(x: prim::F16) -> Self {
        Self(x.to_bits())
    }

    /// Converts an `f32` to `f16`, rounding to nearest, ties to even.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        Self::from_prim(x as prim::F16)
    }

    /// Converts to `f32`. This conversion is exact.
    #[inline]
    pub fn to_f32(self) -> f32 {
        self.prim() as f32
    }

    /// Converts to `f64`. This conversion is exact.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.prim() as f64
    }

    /// Converts an `f32` to `f16`, with explicit rounding and modifiers.
    ///
    /// - `RND` selects the rounding mode and must be [`Rounding::NearestEven`] or
    ///   [`Rounding::TowardZero`].
    /// - `D` selects whether subnormal `f32` inputs are flushed to sign-preserving zero (`.ftz`
    ///   applies only to single-precision inputs and results). [`Denormals::FlushToZero`] is only
    ///   supported with `NearestEven` rounding, [`Relu::Disabled`] and [`Overflow::Infinity`].
    /// - `R` optionally clamps negative results to zero (NaN results become canonical NaN).
    /// - `O` selects whether out-of-range results become infinity or are clamped to the finite
    ///   range of `f16` (`.satfinite`; NaN inputs still produce NaN).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32_with<
        const RND: Rounding,
        const D: Denormals,
        const R: Relu,
        const O: Overflow,
    >(
        x: f32,
    ) -> Self {
        const {
            assert!(
                matches!(RND, Rounding::NearestEven | Rounding::TowardZero),
                "f16 conversion only supports NearestEven and TowardZero rounding"
            );
            assert!(
                matches!(D, Denormals::Preserve)
                    || (matches!(RND, Rounding::NearestEven)
                        && matches!(R, Relu::Disabled)
                        && matches!(O, Overflow::Infinity)),
                "FlushToZero is only supported with NearestEven rounding, without Relu and without SaturateFinite"
            );
        }
        if matches!(D, Denormals::FlushToZero) {
            Self::from_prim(unsafe {
                nv::cvt_f16_f32::<{ nv::Rounding::Rn }, true, false, false>(x)
            })
        } else {
            const_map!(RND => ROUND: nv::Rounding {
                Rounding::NearestEven => nv::Rounding::Rn,
                Rounding::TowardZero => nv::Rounding::Rz,
            } in const_bool!(RELU = matches!(R, Relu::Enabled) =>
                const_bool!(SAT = matches!(O, Overflow::SaturateFinite) =>
                    Self::from_prim(unsafe { nv::cvt_f16_f32::<ROUND, false, RELU, SAT>(x) }))))
        }
    }

    /// Returns `true` if this value is NaN.
    #[inline]
    pub const fn is_nan(self) -> bool {
        self.0 & 0x7FFF > 0x7C00
    }

    /// Returns `true` if this value is positive or negative infinity.
    #[inline]
    pub const fn is_infinite(self) -> bool {
        self.0 & 0x7FFF == 0x7C00
    }

    /// Returns `true` if this value is neither infinite nor NaN.
    #[inline]
    pub const fn is_finite(self) -> bool {
        self.0 & 0x7C00 != 0x7C00
    }

    /// Returns `true` if the sign bit is set, including for `-0.0` and negative NaNs.
    #[inline]
    pub const fn is_sign_negative(self) -> bool {
        self.0 & 0x8000 != 0
    }

    /// Returns `true` if the sign bit is clear, including for `+0.0` and positive NaNs.
    #[inline]
    pub const fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    /// Computes the absolute value. A NaN input yields an unspecified NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs>
    #[inline]
    pub fn abs(self) -> Self {
        self.abs_with::<{ Denormals::Preserve }>()
    }

    /// Computes the absolute value, optionally flushing subnormal inputs and results to
    /// sign-preserving zero. A NaN input yields an unspecified NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs>
    #[inline]
    pub fn abs_with<const D: Denormals>(self) -> Self {
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
            Self::from_prim(unsafe { nv::abs_f16::<FTZ>(self.prim()) }))
    }

    /// Adds two values with explicit modifiers; rounds to nearest, ties to even.
    ///
    /// `D` selects whether subnormal inputs and results are flushed to sign-preserving zero; the
    /// underlying intrinsics only support this together with [`Saturation::ClampToUnit`]. `S`
    /// optionally clamps the result to `[0.0, 1.0]` (NaN results become `+0.0`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-add>
    #[inline]
    pub fn add_with<const D: Denormals, const S: Saturation>(self, rhs: Self) -> Self {
        const {
            assert!(
                matches!(D, Denormals::Preserve) || matches!(S, Saturation::ClampToUnit),
                "FlushToZero is only supported together with ClampToUnit saturation"
            );
        }
        let (a, b) = (self.prim(), rhs.prim());
        if matches!(S, Saturation::ClampToUnit) {
            const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
                Self::from_prim(unsafe { nv::add_f16::<FTZ, true>(a, b) }))
        } else {
            Self::from_prim(unsafe { nv::add_f16::<false, false>(a, b) })
        }
    }

    /// Subtracts two values with explicit modifiers; rounds to nearest, ties to even.
    ///
    /// Implemented as `self + (-rhs)`; see [`add_with`](Self::add_with) for the modifiers.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-sub>
    #[inline]
    pub fn sub_with<const D: Denormals, const S: Saturation>(self, rhs: Self) -> Self {
        self.add_with::<D, S>(-rhs)
    }

    /// Multiplies two values with explicit modifiers; rounds to nearest, ties to even.
    ///
    /// `D` selects whether subnormal inputs and results are flushed to sign-preserving zero; the
    /// underlying intrinsics only support this together with [`Saturation::ClampToUnit`]. `S`
    /// optionally clamps the result to `[0.0, 1.0]` (NaN results become `+0.0`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-mul>
    #[inline]
    pub fn mul_with<const D: Denormals, const S: Saturation>(self, rhs: Self) -> Self {
        const {
            assert!(
                matches!(D, Denormals::Preserve) || matches!(S, Saturation::ClampToUnit),
                "FlushToZero is only supported together with ClampToUnit saturation"
            );
        }
        let (a, b) = (self.prim(), rhs.prim());
        if matches!(S, Saturation::ClampToUnit) {
            const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
                Self::from_prim(unsafe { nv::mul_f16::<FTZ, true>(a, b) }))
        } else {
            Self::from_prim(unsafe { nv::mul_f16::<false, false>(a, b) })
        }
    }

    /// Fused multiply-add: computes `self * b + c` with a single rounding (to nearest, ties to
    /// even).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[inline]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        self.mul_add_with::<{ Denormals::Preserve }, { Saturation::None }, { Relu::Disabled }>(b, c)
    }

    /// Fused multiply-add with explicit modifiers.
    ///
    /// `D` selects whether subnormal inputs and results are flushed to sign-preserving zero. `S`
    /// clamps the result to `[0.0, 1.0]` (NaN results become `+0.0`), and `R` clamps negative
    /// results to zero (NaN results become canonical NaN); they cannot be combined.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[inline]
    pub fn mul_add_with<const D: Denormals, const S: Saturation, const R: Relu>(
        self,
        b: Self,
        c: Self,
    ) -> Self {
        const {
            assert!(
                matches!(S, Saturation::None) || matches!(R, Relu::Disabled),
                "ClampToUnit saturation and Relu cannot be combined"
            );
        }
        let (a, b, c) = (self.prim(), b.prim(), c.prim());
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
        if matches!(S, Saturation::ClampToUnit) {
            Self::from_prim(unsafe { nv::fma_f16::<FTZ, true, false>(a, b, c) })
        } else {
            const_bool!(RELU = matches!(R, Relu::Enabled) =>
                Self::from_prim(unsafe { nv::fma_f16::<FTZ, false, RELU>(a, b, c) }))
        })
    }

    /// Fused multiply-add with the out-of-bounds (OOB) modifier.
    ///
    /// The result is forced to `+0.0` if either operand is an [OOB-NaN], the special NaN produced
    /// by out-of-bounds elements of tensor accesses in the `OOB-NaN` fill mode. `R` optionally
    /// clamps negative results to zero.
    ///
    /// Requires `sm_90`.
    ///
    /// [OOB-NaN]: https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode-oob-access
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub fn mul_add_oob<const R: Relu>(self, b: Self, c: Self) -> Self {
        const_bool!(RELU = matches!(R, Relu::Enabled) =>
            Self::from_prim(unsafe { nv::fma_oob_f16::<RELU>(self.prim(), b.prim(), c.prim()) }))
    }

    /// Computes `2^self`, approximately (`ex2.approx.f16`).
    ///
    /// The maximum relative error is 2^-9.9. Subnormal inputs are supported. `-inf` gives `+0.0`
    /// and `+inf` gives `+inf`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-ex2>
    #[inline]
    pub fn exp2_approx(self) -> Self {
        Self::from_prim(unsafe { nv::ex2_f16(self.prim()) })
    }

    /// Computes the hyperbolic tangent, approximately (`tanh.approx.f16`).
    ///
    /// The maximum absolute error is 2^-10.987. Subnormal inputs are supported. `-inf` gives
    /// `-1.0`, `+inf` gives `+1.0`, and zeros are preserved including their sign.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-tanh>
    #[inline]
    pub fn tanh_approx(self) -> Self {
        let result: u16;
        // SAFETY: `tanh.approx.f16` only reads and writes registers.
        unsafe {
            core::arch::asm!(
                "tanh.approx.f16 {d}, {a};",
                d = out(reg16) result,
                a = in(reg16) self.to_bits(),
                options(pure, nomem, nostack),
            );
        }
        Self::from_bits(result)
    }

    /// Computes `acc + self` in `f32` with the mixed-precision `add.rn.f32.f16`.
    #[cfg(target_feature = "sm_100")]
    #[inline]
    fn add_to_f32(self, acc: f32) -> f32 {
        let result: f32;
        // SAFETY: `add.rn.f32.f16` only reads and writes registers.
        unsafe {
            core::arch::asm!(
                "add.rn.f32.f16 {d}, {a}, {c};",
                d = out(reg32) result,
                a = in(reg16) self.to_bits(),
                c = in(reg32) acc,
                options(pure, nomem, nostack),
            );
        }
        result
    }

    /// Returns the minimum of two values. If exactly one input is NaN, the other is returned.
    /// `-0.0` is treated as less than `+0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline]
    pub fn min(self, other: Self) -> Self {
        self.min_with::<{ Denormals::Preserve }, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(
            other,
        )
    }

    /// Returns the maximum of two values. If exactly one input is NaN, the other is returned.
    /// `+0.0` is treated as greater than `-0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn max(self, other: Self) -> Self {
        self.max_with::<{ Denormals::Preserve }, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(
            other,
        )
    }

    /// Returns the minimum of two values with explicit modifiers.
    ///
    /// `-0.0` is treated as less than `+0.0`. `D` selects whether subnormal inputs and results
    /// are flushed to sign-preserving zero. `NAN` selects the NaN handling: [`NanHandling::Ignore`]
    /// returns the other input if exactly one is NaN, [`NanHandling::Propagate`] yields a
    /// canonical NaN if either input is NaN. `X` selects whether the result has the minimum
    /// absolute value of the inputs and the XOR of their signs (skipped if the result is NaN);
    /// this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline]
    pub fn min_with<const D: Denormals, const NAN: NanHandling, const X: XorSignAbs>(
        self,
        other: Self,
    ) -> Self {
        const {
            assert!(
                matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "XorSignAbs requires sm_86"
            );
        }
        let (a, b) = (self.prim(), other.prim());
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
            const_bool!(PROP = matches!(NAN, NanHandling::Propagate) =>
                const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                    Self::from_prim(unsafe { nv::min_f16::<FTZ, PROP, XS>(a, b) }))))
    }

    /// Returns the maximum of two values with explicit modifiers.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `D` selects whether subnormal inputs and results
    /// are flushed to sign-preserving zero. `NAN` selects the NaN handling: [`NanHandling::Ignore`]
    /// returns the other input if exactly one is NaN, [`NanHandling::Propagate`] yields a
    /// canonical NaN if either input is NaN. `X` selects whether the result has the maximum
    /// absolute value of the inputs and the XOR of their signs (skipped if the result is NaN);
    /// this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn max_with<const D: Denormals, const NAN: NanHandling, const X: XorSignAbs>(
        self,
        other: Self,
    ) -> Self {
        const {
            assert!(
                matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "XorSignAbs requires sm_86"
            );
        }
        let (a, b) = (self.prim(), other.prim());
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
            const_bool!(PROP = matches!(NAN, NanHandling::Propagate) =>
                const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                    Self::from_prim(unsafe { nv::max_f16::<FTZ, PROP, XS>(a, b) }))))
    }
}

impl Default for f16 {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl PartialEq for f16 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.prim() == other.prim()
    }
}

impl PartialOrd for f16 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.prim().partial_cmp(&other.prim())
    }
}

impl fmt::Debug for f16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.to_f32(), f)
    }
}

impl fmt::Display for f16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_f32(), f)
    }
}

impl From<f16> for f32 {
    #[inline]
    fn from(x: f16) -> f32 {
        x.to_f32()
    }
}

impl From<f16> for f64 {
    #[inline]
    fn from(x: f16) -> f64 {
        x.to_f64()
    }
}

impl Add for f16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_prim(self.prim() + rhs.prim())
    }
}

impl Sub for f16 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_prim(self.prim() - rhs.prim())
    }
}

impl Mul for f16 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_prim(self.prim() * rhs.prim())
    }
}

impl Neg for f16 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::from_prim(unsafe { nv::neg_f16(self.prim()) })
    }
}

impl AddAssign for f16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for f16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for f16 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

vector_element! {
    elem: f16, pair: u32, packing: Bytes,
    consts: [
        ZERO, NEG_ZERO, ONE, NEG_ONE, INFINITY, NEG_INFINITY, NAN, MAX, MIN, MIN_POSITIVE,
        MIN_POSITIVE_SUBNORMAL, EPSILON
    ],
    note: "Arithmetic operators act element-wise, round to nearest, ties to even, and preserve subnormal values. Methods suffixed `_with` select instruction modifiers (flush-to-zero, saturation, ...) via const generics.",
    eq: elementwise,
    aliases: f16xN, f16x2, f16x4, f16x8, f16x16, f16x32, f16x64,
}

half_vector!(f16, "f16");

// The instructions of the vectors operate on pairs. Each operation is implemented once on the pair
// type, and then lifted to all lengths by the generic impl below.
impl f16x2 {
    #[inline(always)]
    fn to_core(self) -> nv::f16x2 {
        unsafe { transmute::<u32, nv::f16x2>(self.to_bits()) }
    }

    #[inline(always)]
    fn from_core(x: nv::f16x2) -> Self {
        Self::from_bits(unsafe { transmute::<nv::f16x2, u32>(x) })
    }

    #[inline(always)]
    fn pair_from_f32_with<const RND: Rounding, const R: Relu, const O: Overflow>(
        x: [f32; 2],
    ) -> Self {
        const_map!(RND => ROUND: nv::Rounding {
            Rounding::NearestEven => nv::Rounding::Rn,
            Rounding::TowardZero => nv::Rounding::Rz,
        } in const_bool!(RELU = matches!(R, Relu::Enabled) =>
            const_bool!(SAT = matches!(O, Overflow::SaturateFinite) =>
                Self::from_core(unsafe { nv::cvt_f16x2_f32::<ROUND, RELU, SAT>(x[1], x[0]) }))))
    }

    #[inline(always)]
    fn pair_to_f32(self) -> [f32; 2] {
        // A vector conversion lets LLVM unpack the pair with `mov.b32 {lo, hi}`, which `ptxas`
        // turns into half selectors.
        let x = unsafe { transmute::<u32, Simd<prim::F16, 2>>(self.to_bits()) };
        core::simd::num::SimdFloat::cast::<f32>(x).to_array()
    }

    #[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
    #[inline(always)]
    fn pair_from_f32_stochastic<const R: Relu, const O: Overflow>(x: [f32; 2], rbits: u32) -> Self {
        const_bool!(RELU = matches!(R, Relu::Enabled) =>
            const_bool!(SAT = matches!(O, Overflow::SaturateFinite) =>
                Self::from_core(unsafe { nv::cvt_rs_f16x2_f32::<RELU, SAT>(x[1], x[0], rbits) })))
    }

    #[inline(always)]
    fn pair_abs_with<const D: Denormals>(self) -> Self {
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
            Self::from_core(unsafe { nv::abs_f16x2::<FTZ>(self.to_core()) }))
    }

    #[inline(always)]
    fn pair_add_with<const D: Denormals, const S: Saturation>(self, rhs: Self) -> Self {
        let (a, b) = (self.to_core(), rhs.to_core());
        if matches!(S, Saturation::ClampToUnit) {
            const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
                Self::from_core(unsafe { nv::add_f16x2::<FTZ, true>(a, b) }))
        } else {
            Self::from_core(unsafe { nv::add_f16x2::<false, false>(a, b) })
        }
    }

    #[inline(always)]
    fn pair_sub(self, rhs: Self) -> Self {
        Self::from_core(unsafe { nv::sub_f16x2(self.to_core(), rhs.to_core()) })
    }

    #[inline(always)]
    fn pair_mul_with<const D: Denormals, const S: Saturation>(self, rhs: Self) -> Self {
        let (a, b) = (self.to_core(), rhs.to_core());
        if matches!(S, Saturation::ClampToUnit) {
            const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
                Self::from_core(unsafe { nv::mul_f16x2::<FTZ, true>(a, b) }))
        } else {
            Self::from_core(unsafe { nv::mul_f16x2::<false, false>(a, b) })
        }
    }

    #[inline(always)]
    fn pair_neg(self) -> Self {
        Self::from_core(unsafe { nv::neg_f16x2(self.to_core()) })
    }

    #[inline(always)]
    fn pair_mul_add_with<const D: Denormals, const S: Saturation, const R: Relu>(
        self,
        b: Self,
        c: Self,
    ) -> Self {
        let (a, b, c) = (self.to_core(), b.to_core(), c.to_core());
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
        if matches!(S, Saturation::ClampToUnit) {
            Self::from_core(unsafe { nv::fma_f16x2::<FTZ, true, false>(a, b, c) })
        } else {
            const_bool!(RELU = matches!(R, Relu::Enabled) =>
                Self::from_core(unsafe { nv::fma_f16x2::<FTZ, false, RELU>(a, b, c) }))
        })
    }

    #[cfg(target_feature = "sm_90")]
    #[inline(always)]
    fn pair_mul_add_oob<const R: Relu>(self, b: Self, c: Self) -> Self {
        const_bool!(RELU = matches!(R, Relu::Enabled) =>
        Self::from_core(unsafe {
            nv::fma_oob_f16x2::<RELU>(self.to_core(), b.to_core(), c.to_core())
        }))
    }

    #[inline(always)]
    fn pair_exp2_approx(self) -> Self {
        Self::from_core(unsafe { nv::ex2_f16x2(self.to_core()) })
    }

    #[inline(always)]
    fn pair_tanh_approx(self) -> Self {
        let result: u32;
        // SAFETY: `tanh.approx.f16x2` only reads and writes registers.
        unsafe {
            core::arch::asm!(
                "tanh.approx.f16x2 {d}, {a};",
                d = out(reg32) result,
                a = in(reg32) self.to_bits(),
                options(pure, nomem, nostack),
            );
        }
        Self::from_bits(result)
    }

    #[inline(always)]
    fn pair_min_with<const D: Denormals, const NAN: NanHandling, const X: XorSignAbs>(
        self,
        other: Self,
    ) -> Self {
        let (a, b) = (self.to_core(), other.to_core());
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
            const_bool!(PROP = matches!(NAN, NanHandling::Propagate) =>
                const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                    Self::from_core(unsafe { nv::min_f16x2::<FTZ, PROP, XS>(a, b) }))))
    }

    #[inline(always)]
    fn pair_max_with<const D: Denormals, const NAN: NanHandling, const X: XorSignAbs>(
        self,
        other: Self,
    ) -> Self {
        let (a, b) = (self.to_core(), other.to_core());
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
            const_bool!(PROP = matches!(NAN, NanHandling::Propagate) =>
                const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                    Self::from_core(unsafe { nv::max_f16x2::<FTZ, PROP, XS>(a, b) }))))
    }

    /// Computes `[acc[0] + lo, acc[1] + hi]` in `f32` with the mixed-precision
    /// `add.rn.f32.f16`. The pair is split inside the assembly, so that `ptxas` can select the
    /// halves with operand modifiers.
    #[cfg(target_feature = "sm_100")]
    #[inline(always)]
    fn add_to_f32x2(self, acc: [f32; 2]) -> [f32; 2] {
        let (lo, hi): (f32, f32);
        // SAFETY: the instructions only read and write registers.
        unsafe {
            core::arch::asm!(
                "{{",
                ".reg .b16 %lo, %hi;",
                "mov.b32 {{%lo, %hi}}, {p};",
                "add.rn.f32.f16 {d0}, %lo, {c0};",
                "add.rn.f32.f16 {d1}, %hi, {c1};",
                "}}",
                d0 = out(reg32) lo,
                d1 = out(reg32) hi,
                p = in(reg32) self.to_bits(),
                c0 = in(reg32) acc[0],
                c1 = in(reg32) acc[1],
                options(pure, nomem, nostack),
            );
        }
        [lo, hi]
    }
}

impl<const N: usize> f16xN<N>
where
    VectorLen<N>: SupportedVectorLen,
{
    /// Converts all elements to `f32`. This conversion is exact.
    #[inline(always)]
    pub fn to_f32(self) -> Simd<f32, N> {
        let x2s = self.to_x2s();
        let pairs: Pairs<[f32; 2], N> = array_from_fn(|i| x2s[i].pair_to_f32());
        // SAFETY: the arrays have the same size and hold the same elements.
        Simd::from_array(unsafe { cast(pairs) })
    }

    /// Converts `f32` values to `f16` using stochastic rounding.
    ///
    /// Element `i` of the result is converted from element `i` of `x`. Each result is rounded
    /// towards or away from zero depending on the carry out of adding the random bits to the
    /// mantissa bits discarded by the conversion. `rbits[i]` supplies the random bits of the pair
    /// formed by elements `2 * i` and `2 * i + 1`: the 13 least significant bits of its upper half
    /// are used for the odd element and those of its lower half for the even element; the
    /// remaining bits are ignored. `R` optionally clamps negative results to zero. `O` selects
    /// whether out-of-range results become infinity or are clamped to the finite range of `f16`.
    ///
    /// Requires `sm_100a` or `sm_103a` (`sm_107a` is also supported by the PTX ISA).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
    #[inline(always)]
    pub fn from_f32_stochastic<const R: Relu, const O: Overflow>(
        x: Simd<f32, N>,
        rbits: Pairs<u32, N>,
    ) -> Self {
        let x = x.to_array();
        Self::from_x2s(array_from_fn(|i| {
            f16x2::pair_from_f32_stochastic::<R, O>([x[2 * i], x[2 * i + 1]], rbits[i])
        }))
    }

    /// Computes the absolute value of each element. A NaN input yields an unspecified NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs>
    #[inline(always)]
    pub fn abs(self) -> Self {
        self.abs_with::<{ Denormals::Preserve }>()
    }

    /// Computes the absolute value of each element, optionally flushing subnormal inputs and
    /// results to sign-preserving zero. A NaN input yields an unspecified NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs>
    #[inline(always)]
    pub fn abs_with<const D: Denormals>(self) -> Self {
        self.map_pairs(|p| p.pair_abs_with::<D>())
    }

    /// Adds element-wise with explicit modifiers; rounds to nearest, ties to even.
    ///
    /// `D` selects whether subnormal inputs and results are flushed to sign-preserving zero; the
    /// underlying intrinsics only support this together with [`Saturation::ClampToUnit`]. `S`
    /// optionally clamps each result to `[0.0, 1.0]` (NaN results become `+0.0`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-add>
    #[inline(always)]
    pub fn add_with<const D: Denormals, const S: Saturation>(self, rhs: Self) -> Self {
        const {
            assert!(
                matches!(D, Denormals::Preserve) || matches!(S, Saturation::ClampToUnit),
                "FlushToZero is only supported together with ClampToUnit saturation"
            );
        }
        self.zip_pairs(rhs, |a, b| a.pair_add_with::<D, S>(b))
    }

    /// Subtracts element-wise with explicit modifiers; rounds to nearest, ties to even.
    ///
    /// Implemented as `self + (-rhs)`; see [`add_with`](Self::add_with) for the modifiers.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-sub>
    #[inline(always)]
    pub fn sub_with<const D: Denormals, const S: Saturation>(self, rhs: Self) -> Self {
        self.add_with::<D, S>(-rhs)
    }

    /// Multiplies element-wise with explicit modifiers; rounds to nearest, ties to even.
    ///
    /// `D` selects whether subnormal inputs and results are flushed to sign-preserving zero; the
    /// underlying intrinsics only support this together with [`Saturation::ClampToUnit`]. `S`
    /// optionally clamps each result to `[0.0, 1.0]` (NaN results become `+0.0`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-mul>
    #[inline(always)]
    pub fn mul_with<const D: Denormals, const S: Saturation>(self, rhs: Self) -> Self {
        const {
            assert!(
                matches!(D, Denormals::Preserve) || matches!(S, Saturation::ClampToUnit),
                "FlushToZero is only supported together with ClampToUnit saturation"
            );
        }
        self.zip_pairs(rhs, |a, b| a.pair_mul_with::<D, S>(b))
    }

    /// Element-wise fused multiply-add: computes `self * b + c` with a single rounding (to
    /// nearest, ties to even).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        self.mul_add_with::<{ Denormals::Preserve }, { Saturation::None }, { Relu::Disabled }>(b, c)
    }

    /// Element-wise fused multiply-add with explicit modifiers.
    ///
    /// `D` selects whether subnormal inputs and results are flushed to sign-preserving zero. `S`
    /// clamps each result to `[0.0, 1.0]` (NaN results become `+0.0`), and `R` clamps negative
    /// results to zero (NaN results become canonical NaN); they cannot be combined.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[inline(always)]
    pub fn mul_add_with<const D: Denormals, const S: Saturation, const R: Relu>(
        self,
        b: Self,
        c: Self,
    ) -> Self {
        const {
            assert!(
                matches!(S, Saturation::None) || matches!(R, Relu::Disabled),
                "ClampToUnit saturation and Relu cannot be combined"
            );
        }
        let (a, b, c) = (self.to_x2s(), b.to_x2s(), c.to_x2s());
        Self::from_x2s(array_from_fn(|i| {
            a[i].pair_mul_add_with::<D, S, R>(b[i], c[i])
        }))
    }

    /// Element-wise fused multiply-add with the out-of-bounds (OOB) modifier.
    ///
    /// Each result is forced to `+0.0` if either operand element is an [OOB-NaN], the special NaN
    /// produced by out-of-bounds elements of tensor accesses in the `OOB-NaN` fill mode. The test
    /// is performed independently for each element. `R` optionally clamps negative results to
    /// zero.
    ///
    /// Requires `sm_90`.
    ///
    /// [OOB-NaN]: https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode-oob-access
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[cfg(target_feature = "sm_90")]
    #[inline(always)]
    pub fn mul_add_oob<const R: Relu>(self, b: Self, c: Self) -> Self {
        let (a, b, c) = (self.to_x2s(), b.to_x2s(), c.to_x2s());
        Self::from_x2s(array_from_fn(|i| a[i].pair_mul_add_oob::<R>(b[i], c[i])))
    }

    /// Computes `2^x` of each element, approximately (`ex2.approx.f16x2`).
    ///
    /// The maximum relative error of each element is 2^-9.9 (as for `ex2.approx.f16`). Subnormal
    /// inputs are supported. `-inf` gives `+0.0` and `+inf` gives `+inf`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-ex2>
    #[inline(always)]
    pub fn exp2_approx(self) -> Self {
        self.map_pairs(|p| p.pair_exp2_approx())
    }

    /// Computes the hyperbolic tangent of each element, approximately (`tanh.approx.f16x2`).
    ///
    /// The maximum absolute error of each element is 2^-10.987 (as for `tanh.approx.f16`).
    /// Subnormal inputs are supported. `-inf` gives `-1.0`, `+inf` gives `+1.0`, and zeros are
    /// preserved including their sign.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-tanh>
    #[inline(always)]
    pub fn tanh_approx(self) -> Self {
        self.map_pairs(|p| p.pair_tanh_approx())
    }

    /// Returns the element-wise minimum. If exactly one input element is NaN, the other is
    /// returned. `-0.0` is treated as less than `+0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        self.min_with::<{ Denormals::Preserve }, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(
            other,
        )
    }

    /// Returns the element-wise maximum. If exactly one input element is NaN, the other is
    /// returned. `+0.0` is treated as greater than `-0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        self.max_with::<{ Denormals::Preserve }, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(
            other,
        )
    }

    /// Returns the element-wise minimum with explicit modifiers.
    ///
    /// `-0.0` is treated as less than `+0.0`. `D` selects whether subnormal inputs and results
    /// are flushed to sign-preserving zero. `NAN` selects the NaN handling: [`NanHandling::Ignore`]
    /// returns the other input if exactly one is NaN, [`NanHandling::Propagate`] yields a
    /// canonical NaN if either input is NaN. `X` selects whether each result has the minimum
    /// absolute value of the inputs and the XOR of their signs (skipped if the result is NaN);
    /// this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline(always)]
    pub fn min_with<const D: Denormals, const NAN: NanHandling, const X: XorSignAbs>(
        self,
        other: Self,
    ) -> Self {
        const {
            assert!(
                matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "XorSignAbs requires sm_86"
            );
        }
        self.zip_pairs(other, |a, b| a.pair_min_with::<D, NAN, X>(b))
    }

    /// Returns the element-wise maximum with explicit modifiers.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `D` selects whether subnormal inputs and results
    /// are flushed to sign-preserving zero. `NAN` selects the NaN handling: [`NanHandling::Ignore`]
    /// returns the other input if exactly one is NaN, [`NanHandling::Propagate`] yields a
    /// canonical NaN if either input is NaN. `X` selects whether each result has the maximum
    /// absolute value of the inputs and the XOR of their signs (skipped if the result is NaN);
    /// this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline(always)]
    pub fn max_with<const D: Denormals, const NAN: NanHandling, const X: XorSignAbs>(
        self,
        other: Self,
    ) -> Self {
        const {
            assert!(
                matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "XorSignAbs requires sm_86"
            );
        }
        self.zip_pairs(other, |a, b| a.pair_max_with::<D, NAN, X>(b))
    }

    /// Returns the minimum element. NaN elements are ignored; the result is NaN only if all
    /// elements are NaN. `-0.0` is treated as less than `+0.0`. See
    /// [`reduce_min_with`](Self::reduce_min_with).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline(always)]
    pub fn reduce_min(self) -> f16 {
        self.reduce_min_with::<{ Denormals::Preserve }, { NanHandling::Ignore }>()
    }

    /// Returns the maximum element. NaN elements are ignored; the result is NaN only if all
    /// elements are NaN. `+0.0` is treated as greater than `-0.0`. See
    /// [`reduce_max_with`](Self::reduce_max_with).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline(always)]
    pub fn reduce_max(self) -> f16 {
        self.reduce_max_with::<{ Denormals::Preserve }, { NanHandling::Ignore }>()
    }

    /// Returns the minimum element with explicit modifiers.
    ///
    /// `-0.0` is treated as less than `+0.0`. `D` selects whether subnormal elements and results
    /// are flushed to sign-preserving zero. `NAN` selects the NaN handling: [`NanHandling::Ignore`]
    /// ignores NaN elements (the result is NaN only if all elements are NaN),
    /// [`NanHandling::Propagate`] yields a canonical NaN if any element is NaN.
    ///
    /// The elements are combined with `min` instructions in a tree, using `min.f16x2` on pairs
    /// until a single pair remains.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline(always)]
    pub fn reduce_min_with<const D: Denormals, const NAN: NanHandling>(self) -> f16 {
        let mut acc = self.to_x2s();
        let mut n = acc.len();
        #[rustc_unroll(full)]
        while n > 1 {
            n /= 2;
            #[rustc_unroll(full)]
            for i in 0..n {
                acc[i] = acc[i].pair_min_with::<D, NAN, { XorSignAbs::Disabled }>(acc[i + n]);
            }
        }
        acc[0]
            .lo()
            .min_with::<D, NAN, { XorSignAbs::Disabled }>(acc[0].hi())
    }

    /// Returns the maximum element with explicit modifiers.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `D` selects whether subnormal elements and
    /// results are flushed to sign-preserving zero. `NAN` selects the NaN handling:
    /// [`NanHandling::Ignore`] ignores NaN elements (the result is NaN only if all elements are
    /// NaN), [`NanHandling::Propagate`] yields a canonical NaN if any element is NaN.
    ///
    /// The elements are combined with `max` instructions in a tree, using `max.f16x2` on pairs
    /// until a single pair remains.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline(always)]
    pub fn reduce_max_with<const D: Denormals, const NAN: NanHandling>(self) -> f16 {
        let mut acc = self.to_x2s();
        let mut n = acc.len();
        #[rustc_unroll(full)]
        while n > 1 {
            n /= 2;
            #[rustc_unroll(full)]
            for i in 0..n {
                acc[i] = acc[i].pair_max_with::<D, NAN, { XorSignAbs::Disabled }>(acc[i + n]);
            }
        }
        acc[0]
            .lo()
            .max_with::<D, NAN, { XorSignAbs::Disabled }>(acc[0].hi())
    }
}

impl<const N: usize> Add for f16xN<N>
where
    VectorLen<N>: SupportedVectorLen,
{
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.add_with::<{ Denormals::Preserve }, { Saturation::None }>(rhs)
    }
}

impl<const N: usize> Sub for f16xN<N>
where
    VectorLen<N>: SupportedVectorLen,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.zip_pairs(rhs, |a, b| a.pair_sub(b))
    }
}

impl<const N: usize> Mul for f16xN<N>
where
    VectorLen<N>: SupportedVectorLen,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.mul_with::<{ Denormals::Preserve }, { Saturation::None }>(rhs)
    }
}

impl<const N: usize> Neg for f16xN<N>
where
    VectorLen<N>: SupportedVectorLen,
{
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        self.map_pairs(|p| p.pair_neg())
    }
}
