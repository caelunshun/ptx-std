// --- LLM-generated --- //
use crate::modifiers::*;
use bytemuck::{Pod, Zeroable};
use core::{
    arch::nvptx as nv,
    cmp::Ordering,
    fmt,
    mem::transmute,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
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
    /// are flushed to sign-preserving zero. `N` selects the NaN handling: [`NanHandling::Ignore`]
    /// returns the other input if exactly one is NaN, [`NanHandling::Propagate`] yields a
    /// canonical NaN if either input is NaN. `X` selects whether the result has the minimum
    /// absolute value of the inputs and the XOR of their signs (skipped if the result is NaN);
    /// this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline]
    pub fn min_with<const D: Denormals, const N: NanHandling, const X: XorSignAbs>(
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
            const_bool!(NAN = matches!(N, NanHandling::Propagate) =>
                const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                    Self::from_prim(unsafe { nv::min_f16::<FTZ, NAN, XS>(a, b) }))))
    }

    /// Returns the maximum of two values with explicit modifiers.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `D` selects whether subnormal inputs and results
    /// are flushed to sign-preserving zero. `N` selects the NaN handling: [`NanHandling::Ignore`]
    /// returns the other input if exactly one is NaN, [`NanHandling::Propagate`] yields a
    /// canonical NaN if either input is NaN. `X` selects whether the result has the maximum
    /// absolute value of the inputs and the XOR of their signs (skipped if the result is NaN);
    /// this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn max_with<const D: Denormals, const N: NanHandling, const X: XorSignAbs>(
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
            const_bool!(NAN = matches!(N, NanHandling::Propagate) =>
                const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                    Self::from_prim(unsafe { nv::max_f16::<FTZ, NAN, XS>(a, b) }))))
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

/// Two packed [`f16`](crate::f16) values. Element 0 occupies the low 16 bits.
///
/// Arithmetic operators act element-wise, round to nearest, ties to even, and preserve subnormal
/// values. Methods suffixed `_with` select instruction modifiers (flush-to-zero, saturation, ...)
/// via const generics.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f16x2(u32);

impl f16x2 {
    /// Both elements positive zero.
    pub const ZERO: Self = Self::splat(f16::ZERO);
    /// Both elements negative zero.
    pub const NEG_ZERO: Self = Self::splat(f16::NEG_ZERO);
    /// Both elements one.
    pub const ONE: Self = Self::splat(f16::ONE);
    /// Both elements negative one.
    pub const NEG_ONE: Self = Self::splat(f16::NEG_ONE);
    /// Both elements positive infinity.
    pub const INFINITY: Self = Self::splat(f16::INFINITY);
    /// Both elements negative infinity.
    pub const NEG_INFINITY: Self = Self::splat(f16::NEG_INFINITY);
    /// Both elements a quiet NaN.
    pub const NAN: Self = Self::splat(f16::NAN);
    /// Both elements the largest finite `f16` value.
    pub const MAX: Self = Self::splat(f16::MAX);
    /// Both elements the smallest finite `f16` value.
    pub const MIN: Self = Self::splat(f16::MIN);

    /// Creates a value from its raw bit representation.
    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    #[inline]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    #[inline]
    fn to_core(self) -> nv::f16x2 {
        unsafe { transmute::<u32, nv::f16x2>(self.0) }
    }

    #[inline]
    fn from_core(x: nv::f16x2) -> Self {
        Self(unsafe { transmute::<nv::f16x2, u32>(x) })
    }

    /// Packs two values; `lo` becomes element 0 and `hi` element 1.
    #[inline]
    pub const fn new(lo: f16, hi: f16) -> Self {
        Self(lo.0 as u32 | (hi.0 as u32) << 16)
    }

    /// Creates a value with both elements set to `x`.
    #[inline]
    pub const fn splat(x: f16) -> Self {
        Self::new(x, x)
    }

    /// Returns element 0 (the low 16 bits).
    #[inline]
    pub const fn lo(self) -> f16 {
        f16(self.0 as u16)
    }

    /// Returns element 1 (the high 16 bits).
    #[inline]
    pub const fn hi(self) -> f16 {
        f16((self.0 >> 16) as u16)
    }

    /// Unpacks into an array `[lo, hi]`.
    #[inline]
    pub const fn to_array(self) -> [f16; 2] {
        [self.lo(), self.hi()]
    }

    /// Packs an array `[lo, hi]`.
    #[inline]
    pub const fn from_array(x: [f16; 2]) -> Self {
        Self::new(x[0], x[1])
    }

    /// Converts two `f32` values to `f16`, rounding to nearest, ties to even.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32x2(lo: f32, hi: f32) -> Self {
        Self::from_f32x2_with::<{ Rounding::NearestEven }, { Relu::Disabled }, { Overflow::Infinity }>(
            lo, hi,
        )
    }

    /// Converts both elements to `f32`. This conversion is exact.
    #[inline]
    pub fn to_f32x2(self) -> [f32; 2] {
        [self.lo().to_f32(), self.hi().to_f32()]
    }

    /// Converts two `f32` values to `f16`, with explicit rounding and modifiers.
    ///
    /// - `RND` selects the rounding mode and must be [`Rounding::NearestEven`] or
    ///   [`Rounding::TowardZero`].
    /// - `R` optionally clamps negative results to zero (NaN results become canonical NaN).
    /// - `O` selects whether out-of-range results become infinity or are clamped to the finite
    ///   range of `f16` (`.satfinite`; NaN inputs still produce NaN).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32x2_with<const RND: Rounding, const R: Relu, const O: Overflow>(
        lo: f32,
        hi: f32,
    ) -> Self {
        const {
            assert!(
                matches!(RND, Rounding::NearestEven | Rounding::TowardZero),
                "f16x2 conversion only supports NearestEven and TowardZero rounding"
            );
        }
        const_map!(RND => ROUND: nv::Rounding {
            Rounding::NearestEven => nv::Rounding::Rn,
            Rounding::TowardZero => nv::Rounding::Rz,
        } in const_bool!(RELU = matches!(R, Relu::Enabled) =>
            const_bool!(SAT = matches!(O, Overflow::SaturateFinite) =>
                Self::from_core(unsafe { nv::cvt_f16x2_f32::<ROUND, RELU, SAT>(hi, lo) }))))
    }

    /// Converts two `f32` values to `f16` using stochastic rounding.
    ///
    /// Each result is rounded towards or away from zero depending on the carry out of adding the
    /// random bits to the mantissa bits discarded by the conversion. The 13 least significant bits
    /// of the upper half of `rbits` are used for `hi` and those of the lower half for `lo`; the
    /// remaining bits are ignored. `R` optionally clamps negative results to zero. `O` selects
    /// whether out-of-range results become infinity or are clamped to the finite range of `f16`.
    ///
    /// Requires `sm_100a` or `sm_103a` (`sm_107a` is also supported by the PTX ISA).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
    #[inline]
    pub fn from_f32x2_stochastic<const R: Relu, const O: Overflow>(
        lo: f32,
        hi: f32,
        rbits: u32,
    ) -> Self {
        const_bool!(RELU = matches!(R, Relu::Enabled) =>
            const_bool!(SAT = matches!(O, Overflow::SaturateFinite) =>
                Self::from_core(unsafe { nv::cvt_rs_f16x2_f32::<RELU, SAT>(hi, lo, rbits) })))
    }

    /// Computes the absolute value of each element. A NaN input yields an unspecified NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs>
    #[inline]
    pub fn abs(self) -> Self {
        self.abs_with::<{ Denormals::Preserve }>()
    }

    /// Computes the absolute value of each element, optionally flushing subnormal inputs and
    /// results to sign-preserving zero. A NaN input yields an unspecified NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs>
    #[inline]
    pub fn abs_with<const D: Denormals>(self) -> Self {
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
            Self::from_core(unsafe { nv::abs_f16x2::<FTZ>(self.to_core()) }))
    }

    /// Adds element-wise with explicit modifiers; rounds to nearest, ties to even.
    ///
    /// `D` selects whether subnormal inputs and results are flushed to sign-preserving zero; the
    /// underlying intrinsics only support this together with [`Saturation::ClampToUnit`]. `S`
    /// optionally clamps each result to `[0.0, 1.0]` (NaN results become `+0.0`).
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
        let (a, b) = (self.to_core(), rhs.to_core());
        if matches!(S, Saturation::ClampToUnit) {
            const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
                Self::from_core(unsafe { nv::add_f16x2::<FTZ, true>(a, b) }))
        } else {
            Self::from_core(unsafe { nv::add_f16x2::<false, false>(a, b) })
        }
    }

    /// Subtracts element-wise with explicit modifiers; rounds to nearest, ties to even.
    ///
    /// Implemented as `self + (-rhs)`; see [`add_with`](Self::add_with) for the modifiers.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-sub>
    #[inline]
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
    #[inline]
    pub fn mul_with<const D: Denormals, const S: Saturation>(self, rhs: Self) -> Self {
        const {
            assert!(
                matches!(D, Denormals::Preserve) || matches!(S, Saturation::ClampToUnit),
                "FlushToZero is only supported together with ClampToUnit saturation"
            );
        }
        let (a, b) = (self.to_core(), rhs.to_core());
        if matches!(S, Saturation::ClampToUnit) {
            const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
                Self::from_core(unsafe { nv::mul_f16x2::<FTZ, true>(a, b) }))
        } else {
            Self::from_core(unsafe { nv::mul_f16x2::<false, false>(a, b) })
        }
    }

    /// Element-wise fused multiply-add: computes `self * b + c` with a single rounding (to
    /// nearest, ties to even).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[inline]
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
        let (a, b, c) = (self.to_core(), b.to_core(), c.to_core());
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
        if matches!(S, Saturation::ClampToUnit) {
            Self::from_core(unsafe { nv::fma_f16x2::<FTZ, true, false>(a, b, c) })
        } else {
            const_bool!(RELU = matches!(R, Relu::Enabled) =>
                Self::from_core(unsafe { nv::fma_f16x2::<FTZ, false, RELU>(a, b, c) }))
        })
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
    #[inline]
    pub fn mul_add_oob<const R: Relu>(self, b: Self, c: Self) -> Self {
        const_bool!(RELU = matches!(R, Relu::Enabled) =>
        Self::from_core(unsafe {
            nv::fma_oob_f16x2::<RELU>(self.to_core(), b.to_core(), c.to_core())
        }))
    }

    /// Computes `2^x` of each element, approximately (`ex2.approx.f16x2`).
    ///
    /// The maximum relative error of each element is 2^-9.9 (as for `ex2.approx.f16`). Subnormal
    /// inputs are supported. `-inf` gives `+0.0` and `+inf` gives `+inf`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-ex2>
    #[inline]
    pub fn exp2_approx(self) -> Self {
        Self::from_core(unsafe { nv::ex2_f16x2(self.to_core()) })
    }

    /// Returns the element-wise minimum. If exactly one input element is NaN, the other is
    /// returned. `-0.0` is treated as less than `+0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline]
    pub fn min(self, other: Self) -> Self {
        self.min_with::<{ Denormals::Preserve }, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(
            other,
        )
    }

    /// Returns the element-wise maximum. If exactly one input element is NaN, the other is
    /// returned. `+0.0` is treated as greater than `-0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn max(self, other: Self) -> Self {
        self.max_with::<{ Denormals::Preserve }, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(
            other,
        )
    }

    /// Returns the element-wise minimum with explicit modifiers.
    ///
    /// `-0.0` is treated as less than `+0.0`. `D` selects whether subnormal inputs and results
    /// are flushed to sign-preserving zero. `N` selects the NaN handling: [`NanHandling::Ignore`]
    /// returns the other input if exactly one is NaN, [`NanHandling::Propagate`] yields a
    /// canonical NaN if either input is NaN. `X` selects whether each result has the minimum
    /// absolute value of the inputs and the XOR of their signs (skipped if the result is NaN);
    /// this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline]
    pub fn min_with<const D: Denormals, const N: NanHandling, const X: XorSignAbs>(
        self,
        other: Self,
    ) -> Self {
        const {
            assert!(
                matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "XorSignAbs requires sm_86"
            );
        }
        let (a, b) = (self.to_core(), other.to_core());
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
            const_bool!(NAN = matches!(N, NanHandling::Propagate) =>
                const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                    Self::from_core(unsafe { nv::min_f16x2::<FTZ, NAN, XS>(a, b) }))))
    }

    /// Returns the element-wise maximum with explicit modifiers.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `D` selects whether subnormal inputs and results
    /// are flushed to sign-preserving zero. `N` selects the NaN handling: [`NanHandling::Ignore`]
    /// returns the other input if exactly one is NaN, [`NanHandling::Propagate`] yields a
    /// canonical NaN if either input is NaN. `X` selects whether each result has the maximum
    /// absolute value of the inputs and the XOR of their signs (skipped if the result is NaN);
    /// this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn max_with<const D: Denormals, const N: NanHandling, const X: XorSignAbs>(
        self,
        other: Self,
    ) -> Self {
        const {
            assert!(
                matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "XorSignAbs requires sm_86"
            );
        }
        let (a, b) = (self.to_core(), other.to_core());
        const_bool!(FTZ = matches!(D, Denormals::FlushToZero) =>
            const_bool!(NAN = matches!(N, NanHandling::Propagate) =>
                const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                    Self::from_core(unsafe { nv::max_f16x2::<FTZ, NAN, XS>(a, b) }))))
    }
}

impl Default for f16x2 {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl PartialEq for f16x2 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.lo() == other.lo() && self.hi() == other.hi()
    }
}

impl fmt::Debug for f16x2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("f16x2")
            .field(&self.lo())
            .field(&self.hi())
            .finish()
    }
}

impl From<[f16; 2]> for f16x2 {
    #[inline]
    fn from(x: [f16; 2]) -> Self {
        Self::from_array(x)
    }
}

impl From<f16x2> for [f16; 2] {
    #[inline]
    fn from(x: f16x2) -> Self {
        x.to_array()
    }
}

impl Add for f16x2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.add_with::<{ Denormals::Preserve }, { Saturation::None }>(rhs)
    }
}

impl Sub for f16x2 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_core(unsafe { nv::sub_f16x2(self.to_core(), rhs.to_core()) })
    }
}

impl Mul for f16x2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_with::<{ Denormals::Preserve }, { Saturation::None }>(rhs)
    }
}

impl Neg for f16x2 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::from_core(unsafe { nv::neg_f16x2(self.to_core()) })
    }
}

impl AddAssign for f16x2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for f16x2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for f16x2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
