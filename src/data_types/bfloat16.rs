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

use super::vector::{half_f32_wide, half_reduce_sum, vector_partial_eq, vector_types};

/// bfloat16 float (8 exponent bits, 7 mantissa bits).
///
/// Arithmetic operators round to nearest, ties to even. The PTX `fma` instruction is the only
/// native `bf16` arithmetic instruction available on all supported targets, so addition,
/// subtraction and multiplication are expressed as fused multiply-adds with exact constants.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct bf16(u16);

impl bf16 {
    /// Positive zero.
    pub const ZERO: Self = Self(0x0000);
    /// Negative zero.
    pub const NEG_ZERO: Self = Self(0x8000);
    /// One.
    pub const ONE: Self = Self(0x3F80);
    /// Negative one.
    pub const NEG_ONE: Self = Self(0xBF80);
    /// Positive infinity.
    pub const INFINITY: Self = Self(0x7F80);
    /// Negative infinity.
    pub const NEG_INFINITY: Self = Self(0xFF80);
    /// A quiet NaN.
    pub const NAN: Self = Self(0x7FC0);
    /// The largest finite value.
    pub const MAX: Self = Self(0x7F7F);
    /// The smallest finite value.
    pub const MIN: Self = Self(0xFF7F);
    /// The smallest positive normal value (2^-126).
    pub const MIN_POSITIVE: Self = Self(0x0080);
    /// The difference between `1.0` and the next larger representable value (2^-7).
    pub const EPSILON: Self = Self(0x3C00);

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
    fn to_core(self) -> nv::bf16 {
        nv::bf16::from_bits(self.0)
    }

    #[inline]
    fn from_core(x: nv::bf16) -> Self {
        Self(x.to_bits())
    }

    /// Converts an `f32` to `bf16`, rounding to nearest, ties to even.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32(x: f32) -> Self {
        Self::from_f32_with::<{ Rounding::NearestEven }, { Relu::Disabled }, { Overflow::Infinity }>(
            x,
        )
    }

    /// Converts to `f32`. This conversion is exact.
    #[inline]
    pub const fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// Converts to `f64`. This conversion is exact.
    #[inline]
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    /// Converts an `f32` to `bf16`, with explicit rounding and modifiers.
    ///
    /// - `RND` selects the rounding mode and must be [`Rounding::NearestEven`] or
    ///   [`Rounding::TowardZero`].
    /// - `R` optionally clamps negative results to zero (NaN results become canonical NaN).
    /// - `O` selects whether out-of-range results become infinity or are clamped to the finite
    ///   range of `bf16` (`.satfinite`; NaN inputs still produce NaN).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32_with<const RND: Rounding, const R: Relu, const O: Overflow>(x: f32) -> Self {
        const {
            assert!(
                matches!(RND, Rounding::NearestEven | Rounding::TowardZero),
                "bf16 conversion only supports NearestEven and TowardZero rounding"
            );
        }
        const_map!(RND => ROUND: nv::Rounding {
            Rounding::NearestEven => nv::Rounding::Rn,
            Rounding::TowardZero => nv::Rounding::Rz,
        } in const_bool!(RELU = matches!(R, Relu::Enabled) =>
            const_bool!(SAT = matches!(O, Overflow::SaturateFinite) =>
                Self::from_core(unsafe { nv::cvt_bf16_f32::<ROUND, RELU, SAT>(x) }))))
    }

    /// Returns `true` if this value is NaN.
    #[inline]
    pub const fn is_nan(self) -> bool {
        self.0 & 0x7FFF > 0x7F80
    }

    /// Returns `true` if this value is positive or negative infinity.
    #[inline]
    pub const fn is_infinite(self) -> bool {
        self.0 & 0x7FFF == 0x7F80
    }

    /// Returns `true` if this value is neither infinite nor NaN.
    #[inline]
    pub const fn is_finite(self) -> bool {
        self.0 & 0x7F80 != 0x7F80
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

    /// Computes the absolute value by clearing the sign bit.
    ///
    /// Unlike PTX `abs.bf16`, this preserves the NaN payload.
    #[inline]
    pub const fn abs(self) -> Self {
        Self(self.0 & 0x7FFF)
    }

    /// Fused multiply-add: computes `self * b + c` with a single rounding (to nearest, ties to
    /// even).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[inline]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        self.mul_add_with::<{ Relu::Disabled }>(b, c)
    }

    /// Fused multiply-add, optionally clamping negative results to zero (`R`; NaN results become
    /// canonical NaN).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[inline]
    pub fn mul_add_with<const R: Relu>(self, b: Self, c: Self) -> Self {
        const_bool!(RELU = matches!(R, Relu::Enabled) =>
        Self::from_core(unsafe {
            nv::fma_bf16::<RELU>(self.to_core(), b.to_core(), c.to_core())
        }))
    }

    /// Computes `acc + self` in `f32` with the mixed-precision `add.rn.f32.bf16`.
    #[cfg(target_feature = "sm_100")]
    #[inline]
    fn add_to_f32(self, acc: f32) -> f32 {
        let result: f32;
        // SAFETY: `add.rn.f32.bf16` only reads and writes registers.
        unsafe {
            core::arch::asm!(
                "add.rn.f32.bf16 {d}, {a}, {c};",
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
        self.min_with::<{ NanHandling::Ignore }, { XorSignAbs::Disabled }>(other)
    }

    /// Returns the maximum of two values. If exactly one input is NaN, the other is returned.
    /// `+0.0` is treated as greater than `-0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn max(self, other: Self) -> Self {
        self.max_with::<{ NanHandling::Ignore }, { XorSignAbs::Disabled }>(other)
    }

    /// Returns the minimum of two values with explicit modifiers.
    ///
    /// `-0.0` is treated as less than `+0.0`. `N` selects the NaN handling: [`NanHandling::Ignore`]
    /// returns the other input if exactly one is NaN, [`NanHandling::Propagate`] yields a
    /// canonical NaN if either input is NaN. `X` selects whether the result has the minimum
    /// absolute value of the inputs and the XOR of their signs (skipped if the result is NaN);
    /// this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline]
    pub fn min_with<const N: NanHandling, const X: XorSignAbs>(self, other: Self) -> Self {
        const {
            assert!(
                matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "XorSignAbs requires sm_86"
            );
        }
        let (a, b) = (self.to_core(), other.to_core());
        const_bool!(NAN = matches!(N, NanHandling::Propagate) =>
            const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                Self::from_core(unsafe { nv::min_bf16::<NAN, XS>(a, b) })))
    }

    /// Returns the maximum of two values with explicit modifiers.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `N` selects the NaN handling:
    /// [`NanHandling::Ignore`] returns the other input if exactly one is NaN,
    /// [`NanHandling::Propagate`] yields a canonical NaN if either input is NaN. `X` selects
    /// whether the result has the maximum absolute value of the inputs and the XOR of their signs
    /// (skipped if the result is NaN); this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn max_with<const N: NanHandling, const X: XorSignAbs>(self, other: Self) -> Self {
        const {
            assert!(
                matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "XorSignAbs requires sm_86"
            );
        }
        let (a, b) = (self.to_core(), other.to_core());
        const_bool!(NAN = matches!(N, NanHandling::Propagate) =>
            const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                Self::from_core(unsafe { nv::max_bf16::<NAN, XS>(a, b) })))
    }
}

#[cfg(target_feature = "sm_90")]
impl bf16 {
    /// Computes `2^self`, approximately (`ex2.approx.ftz.bf16`).
    ///
    /// The maximum relative error is 2^-7. Subnormal inputs and results are flushed to
    /// sign-preserving zero. `-inf` gives `+0.0`, subnormal inputs of either sign give `+1.0` and
    /// `+inf` gives `+inf`.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-ex2>
    #[inline]
    pub fn exp2_approx(self) -> Self {
        let result: u16;
        // SAFETY: `ex2.approx.ftz.bf16` only reads and writes registers.
        unsafe {
            core::arch::asm!(
                "ex2.approx.ftz.bf16 {d}, {a};",
                d = out(reg16) result,
                a = in(reg16) self.to_bits(),
                options(pure, nomem, nostack),
            );
        }
        Self::from_bits(result)
    }

    /// Computes the hyperbolic tangent, approximately (`tanh.approx.bf16`).
    ///
    /// The maximum absolute error is 2^-8. Subnormal inputs are supported. `-inf` gives `-1.0`,
    /// `+inf` gives `+1.0`, and zeros are preserved including their sign.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-tanh>
    #[inline]
    pub fn tanh_approx(self) -> Self {
        let result: u16;
        // SAFETY: `tanh.approx.bf16` only reads and writes registers.
        unsafe {
            core::arch::asm!(
                "tanh.approx.bf16 {d}, {a};",
                d = out(reg16) result,
                a = in(reg16) self.to_bits(),
                options(pure, nomem, nostack),
            );
        }
        Self::from_bits(result)
    }
}

impl Default for bf16 {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl PartialEq for bf16 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_f32() == other.to_f32()
    }
}

impl PartialOrd for bf16 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

impl fmt::Debug for bf16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.to_f32(), f)
    }
}

impl fmt::Display for bf16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_f32(), f)
    }
}

impl From<bf16> for f32 {
    #[inline]
    fn from(x: bf16) -> f32 {
        x.to_f32()
    }
}

impl From<bf16> for f64 {
    #[inline]
    fn from(x: bf16) -> f64 {
        x.to_f64()
    }
}

impl Add for bf16 {
    type Output = Self;

    /// Computed as `fma(self, 1.0, rhs)`, which is exact in the product.
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.mul_add(Self::ONE, rhs)
    }
}

impl Sub for bf16 {
    type Output = Self;

    /// Computed as `fma(rhs, -1.0, self)`, which is exact in the product.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        rhs.mul_add(Self::NEG_ONE, self)
    }
}

impl Mul for bf16 {
    type Output = Self;

    /// Computed as `fma(self, rhs, -0.0)`; the negative-zero addend preserves the sign of zero
    /// products.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_add(rhs, Self::NEG_ZERO)
    }
}

impl Neg for bf16 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::from_core(unsafe { nv::neg_bf16(self.to_core()) })
    }
}

impl AddAssign for bf16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for bf16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for bf16 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

vector_types! {
    elem: bf16, pack: bytes,
    consts: [
        ZERO, NEG_ZERO, ONE, NEG_ONE, INFINITY, NEG_INFINITY, NAN, MAX, MIN, MIN_POSITIVE, EPSILON
    ],
    note: "Arithmetic operators act element-wise and round to nearest, ties to even. Like [`bf16`](crate::bf16), addition, subtraction and multiplication are expressed as fused multiply-adds.",
    x2: bf16x2(u32),
    x4: bf16x4(u64),
    x8: bf16x8(u128),
    x16: bf16x16([u128; 2]),
    x32: bf16x32([u128; 4]),
}

vector_partial_eq!(bf16x2, bf16x4, bf16x8, bf16x16, bf16x32);

half_f32_wide!(bf16x4, bf16x2, bf16, n: 4, pairs: 2);
half_f32_wide!(bf16x8, bf16x2, bf16, n: 8, pairs: 4);
half_f32_wide!(bf16x16, bf16x2, bf16, n: 16, pairs: 8);
half_f32_wide!(bf16x32, bf16x2, bf16, n: 32, pairs: 16);

impl bf16x2 {
    #[inline]
    fn to_core(self) -> nv::bf16x2 {
        unsafe { transmute::<u32, nv::bf16x2>(self.0) }
    }

    #[inline]
    fn from_core(x: nv::bf16x2) -> Self {
        Self(unsafe { transmute::<nv::bf16x2, u32>(x) })
    }

    /// Converts two `f32` values to `bf16`, rounding to nearest, ties to even.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32(x: Simd<f32, 2>) -> Self {
        Self::from_f32_with::<{ Rounding::NearestEven }, { Relu::Disabled }, { Overflow::Infinity }>(
            x,
        )
    }

    /// Converts both elements to `f32`. This conversion is exact.
    #[inline]
    pub const fn to_f32(self) -> Simd<f32, 2> {
        Simd::from_array([self.lo().to_f32(), self.hi().to_f32()])
    }

    /// Converts two `f32` values to `bf16`, with explicit rounding and modifiers.
    ///
    /// - `RND` selects the rounding mode and must be [`Rounding::NearestEven`] or
    ///   [`Rounding::TowardZero`].
    /// - `R` optionally clamps negative results to zero (NaN results become canonical NaN).
    /// - `O` selects whether out-of-range results become infinity or are clamped to the finite
    ///   range of `bf16` (`.satfinite`; NaN inputs still produce NaN).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32_with<const RND: Rounding, const R: Relu, const O: Overflow>(
        x: Simd<f32, 2>,
    ) -> Self {
        const {
            assert!(
                matches!(RND, Rounding::NearestEven | Rounding::TowardZero),
                "bf16x2 conversion only supports NearestEven and TowardZero rounding"
            );
        }
        const_map!(RND => ROUND: nv::Rounding {
            Rounding::NearestEven => nv::Rounding::Rn,
            Rounding::TowardZero => nv::Rounding::Rz,
        } in const_bool!(RELU = matches!(R, Relu::Enabled) =>
            const_bool!(SAT = matches!(O, Overflow::SaturateFinite) =>
                Self::from_core(unsafe { nv::cvt_bf16x2_f32::<ROUND, RELU, SAT>(x[1], x[0]) }))))
    }

    /// Converts two `f32` values to `bf16` using stochastic rounding.
    ///
    /// Each result is rounded towards or away from zero depending on the carry out of adding the
    /// random bits to the mantissa bits discarded by the conversion. The upper 16 bits of `rbits`
    /// are used for `x[1]` and the lower 16 bits for `x[0]`. `R` optionally clamps negative results to
    /// zero. `O` selects whether out-of-range results become infinity or are clamped to the finite
    /// range of `bf16`.
    ///
    /// Requires `sm_100a` or `sm_103a` (`sm_107a` is also supported by the PTX ISA).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
    #[inline]
    pub fn from_f32_stochastic<const R: Relu, const O: Overflow>(
        x: Simd<f32, 2>,
        rbits: u32,
    ) -> Self {
        const_bool!(RELU = matches!(R, Relu::Enabled) =>
            const_bool!(SAT = matches!(O, Overflow::SaturateFinite) =>
                Self::from_core(unsafe { nv::cvt_rs_bf16x2_f32::<RELU, SAT>(x[1], x[0], rbits) })))
    }

    /// Computes the absolute value of each element by clearing the sign bits.
    ///
    /// Unlike PTX `abs.bf16x2`, this preserves NaN payloads.
    #[inline]
    pub const fn abs(self) -> Self {
        Self(self.0 & 0x7FFF_7FFF)
    }

    /// Element-wise fused multiply-add: computes `self * b + c` with a single rounding (to
    /// nearest, ties to even).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[inline]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        self.mul_add_with::<{ Relu::Disabled }>(b, c)
    }

    /// Element-wise fused multiply-add, optionally clamping negative results to zero (`R`; NaN
    /// results become canonical NaN).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
    #[inline]
    pub fn mul_add_with<const R: Relu>(self, b: Self, c: Self) -> Self {
        const_bool!(RELU = matches!(R, Relu::Enabled) =>
        Self::from_core(unsafe {
            nv::fma_bf16x2::<RELU>(self.to_core(), b.to_core(), c.to_core())
        }))
    }

    /// Returns the element-wise minimum. If exactly one input element is NaN, the other is
    /// returned. `-0.0` is treated as less than `+0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline]
    pub fn min(self, other: Self) -> Self {
        self.min_with::<{ NanHandling::Ignore }, { XorSignAbs::Disabled }>(other)
    }

    /// Returns the element-wise maximum. If exactly one input element is NaN, the other is
    /// returned. `+0.0` is treated as greater than `-0.0`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn max(self, other: Self) -> Self {
        self.max_with::<{ NanHandling::Ignore }, { XorSignAbs::Disabled }>(other)
    }

    /// Returns the element-wise minimum with explicit modifiers.
    ///
    /// `-0.0` is treated as less than `+0.0`. `N` selects the NaN handling: [`NanHandling::Ignore`]
    /// returns the other input if exactly one is NaN, [`NanHandling::Propagate`] yields a
    /// canonical NaN if either input is NaN. `X` selects whether each result has the minimum
    /// absolute value of the inputs and the XOR of their signs (skipped if the result is NaN);
    /// this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline]
    pub fn min_with<const N: NanHandling, const X: XorSignAbs>(self, other: Self) -> Self {
        const {
            assert!(
                matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "XorSignAbs requires sm_86"
            );
        }
        let (a, b) = (self.to_core(), other.to_core());
        const_bool!(NAN = matches!(N, NanHandling::Propagate) =>
            const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                Self::from_core(unsafe { nv::min_bf16x2::<NAN, XS>(a, b) })))
    }

    /// Returns the element-wise maximum with explicit modifiers.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `N` selects the NaN handling:
    /// [`NanHandling::Ignore`] returns the other input if exactly one is NaN,
    /// [`NanHandling::Propagate`] yields a canonical NaN if either input is NaN. `X` selects
    /// whether each result has the maximum absolute value of the inputs and the XOR of their signs
    /// (skipped if the result is NaN); this requires `sm_86`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn max_with<const N: NanHandling, const X: XorSignAbs>(self, other: Self) -> Self {
        const {
            assert!(
                matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                "XorSignAbs requires sm_86"
            );
        }
        let (a, b) = (self.to_core(), other.to_core());
        const_bool!(NAN = matches!(N, NanHandling::Propagate) =>
            const_bool!(XS = matches!(X, XorSignAbs::Enabled) =>
                Self::from_core(unsafe { nv::max_bf16x2::<NAN, XS>(a, b) })))
    }

    /// Computes `[acc[0] + lo, acc[1] + hi]` in `f32` with the mixed-precision
    /// `add.rn.f32.bf16`. The pair is split inside the assembly, so that `ptxas` can select the
    /// halves with operand modifiers.
    #[cfg(target_feature = "sm_100")]
    #[inline]
    fn add_to_f32x2(self, acc: [f32; 2]) -> [f32; 2] {
        let (lo, hi): (f32, f32);
        // SAFETY: the instructions only read and write registers.
        unsafe {
            core::arch::asm!(
                "{{",
                ".reg .b16 %lo, %hi;",
                "mov.b32 {{%lo, %hi}}, {p};",
                "add.rn.f32.bf16 {d0}, %lo, {c0};",
                "add.rn.f32.bf16 {d1}, %hi, {c1};",
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

    /// Returns the minimum element. NaN elements are ignored; the result is NaN only if all
    /// elements are NaN. `-0.0` is treated as less than `+0.0`. See
    /// [`reduce_min_with`](Self::reduce_min_with).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline]
    pub fn reduce_min(self) -> bf16 {
        self.reduce_min_with::<{ NanHandling::Ignore }>()
    }

    /// Returns the maximum element. NaN elements are ignored; the result is NaN only if all
    /// elements are NaN. `+0.0` is treated as greater than `-0.0`. See
    /// [`reduce_max_with`](Self::reduce_max_with).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn reduce_max(self) -> bf16 {
        self.reduce_max_with::<{ NanHandling::Ignore }>()
    }

    /// Returns the minimum element with explicit NaN handling.
    ///
    /// `-0.0` is treated as less than `+0.0`. `N` selects the NaN handling:
    /// [`NanHandling::Ignore`] ignores NaN elements (the result is NaN only if all elements are
    /// NaN), [`NanHandling::Propagate`] yields a canonical NaN if any element is NaN.
    ///
    /// The elements are combined with `min` instructions in a tree, using `min.bf16x2` on pairs
    /// until a single pair remains.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
    #[inline]
    pub fn reduce_min_with<const N: NanHandling>(self) -> bf16 {
        self.lo().min_with::<N, { XorSignAbs::Disabled }>(self.hi())
    }

    /// Returns the maximum element with explicit NaN handling.
    ///
    /// `+0.0` is treated as greater than `-0.0`. `N` selects the NaN handling:
    /// [`NanHandling::Ignore`] ignores NaN elements (the result is NaN only if all elements are
    /// NaN), [`NanHandling::Propagate`] yields a canonical NaN if any element is NaN.
    ///
    /// The elements are combined with `max` instructions in a tree, using `max.bf16x2` on pairs
    /// until a single pair remains.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
    #[inline]
    pub fn reduce_max_with<const N: NanHandling>(self) -> bf16 {
        self.lo().max_with::<N, { XorSignAbs::Disabled }>(self.hi())
    }
}

#[cfg(target_feature = "sm_90")]
impl bf16x2 {
    /// Computes `2^x` of each element, approximately (`ex2.approx.ftz.bf16x2`).
    ///
    /// The maximum relative error of each element is 2^-7 (as for `ex2.approx.ftz.bf16`).
    /// Subnormal inputs and results are flushed to sign-preserving zero. `-inf` gives `+0.0`,
    /// subnormal inputs of either sign give `+1.0` and `+inf` gives `+inf`.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-ex2>
    #[inline]
    pub fn exp2_approx(self) -> Self {
        let result: u32;
        // SAFETY: `ex2.approx.ftz.bf16x2` only reads and writes registers.
        unsafe {
            core::arch::asm!(
                "ex2.approx.ftz.bf16x2 {d}, {a};",
                d = out(reg32) result,
                a = in(reg32) self.to_bits(),
                options(pure, nomem, nostack),
            );
        }
        Self::from_bits(result)
    }

    /// Computes the hyperbolic tangent of each element, approximately (`tanh.approx.bf16x2`).
    ///
    /// The maximum absolute error of each element is 2^-8 (as for `tanh.approx.bf16`). Subnormal
    /// inputs are supported. `-inf` gives `-1.0`, `+inf` gives `+1.0`, and zeros are preserved
    /// including their sign.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-tanh>
    #[inline]
    pub fn tanh_approx(self) -> Self {
        let result: u32;
        // SAFETY: `tanh.approx.bf16x2` only reads and writes registers.
        unsafe {
            core::arch::asm!(
                "tanh.approx.bf16x2 {d}, {a};",
                d = out(reg32) result,
                a = in(reg32) self.to_bits(),
                options(pure, nomem, nostack),
            );
        }
        Self::from_bits(result)
    }
}

impl Add for bf16x2 {
    type Output = Self;

    /// Computed as `fma(self, 1.0, rhs)`, which is exact in the product.
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.mul_add(Self::ONE, rhs)
    }
}

impl Sub for bf16x2 {
    type Output = Self;

    /// Computed as `fma(rhs, -1.0, self)`, which is exact in the product.
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        rhs.mul_add(Self::NEG_ONE, self)
    }
}

impl Mul for bf16x2 {
    type Output = Self;

    /// Computed as `fma(self, rhs, -0.0)`; the negative-zero addend preserves the sign of zero
    /// products.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_add(rhs, Self::NEG_ZERO)
    }
}

impl Neg for bf16x2 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::from_core(unsafe { nv::neg_bf16x2(self.to_core()) })
    }
}

impl AddAssign for bf16x2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for bf16x2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for bf16x2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Implements the arithmetic of a wide (`N >= 4`) `bf16` vector pair-wise in terms of [`bf16x2`].
macro_rules! bf16_wide {
    ($ty:ident, n: $n:literal, pairs: $pairs:literal) => {
        impl $ty {
            #[doc = concat!("Converts ", stringify!($n), " `f32` values to `bf16` using stochastic rounding.")]
            ///
            /// Each pair of elements is converted with [`bf16x2::from_f32_stochastic`]; `rbits[i]`
            /// supplies the random bits of the pair formed by elements `2 * i` and `2 * i + 1`,
            /// with the upper 16 bits of its word used for the odd element and the lower 16 bits
            /// for the even element. See there for the meaning of the parameters.
            ///
            /// Requires `sm_100a` or `sm_103a` (`sm_107a` is also supported by the PTX ISA).
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
            #[inline]
            pub fn from_f32_stochastic<const R: Relu, const O: Overflow>(
                x: Simd<f32, $n>,
                rbits: Simd<u32, { $n / 2 }>,
            ) -> Self {
                let (x, rbits) = (x.to_array(), rbits.to_array());
                Self(crate::data_types::array_from_fn(|i| {
                    bf16x2::from_f32_stochastic::<R, O>(
                        Simd::from_array([x[2 * i], x[2 * i + 1]]),
                        rbits[i],
                    )
                }))
            }

            /// Computes the absolute value of each element by clearing the sign bits.
            ///
            /// Unlike PTX `abs.bf16x2`, this preserves NaN payloads.
            #[inline]
            pub const fn abs(self) -> Self {
                let mut pairs = self.0;
                let mut i = 0;
                #[rustc_unroll(full)]
                while i < $pairs {
                    pairs[i] = pairs[i].abs();
                    i += 1;
                }
                Self(pairs)
            }

            /// Element-wise fused multiply-add: computes `self * b + c` with a single rounding
            /// (to nearest, ties to even).
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
            #[inline]
            pub fn mul_add(self, b: Self, c: Self) -> Self {
                self.mul_add_with::<{ Relu::Disabled }>(b, c)
            }

            /// Element-wise fused multiply-add, optionally clamping negative results to zero
            /// (`R`; NaN results become canonical NaN).
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma>
            #[inline]
            pub fn mul_add_with<const R: Relu>(self, b: Self, c: Self) -> Self {
                Self(crate::data_types::array_from_fn(|i| {
                    self.0[i].mul_add_with::<R>(b.0[i], c.0[i])
                }))
            }

            /// Returns the element-wise minimum. If exactly one input element is NaN, the other
            /// is returned. `-0.0` is treated as less than `+0.0`.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
            #[inline]
            pub fn min(self, other: Self) -> Self {
                self.min_with::<{ NanHandling::Ignore }, { XorSignAbs::Disabled }>(other)
            }

            /// Returns the element-wise maximum. If exactly one input element is NaN, the other
            /// is returned. `+0.0` is treated as greater than `-0.0`.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
            #[inline]
            pub fn max(self, other: Self) -> Self {
                self.max_with::<{ NanHandling::Ignore }, { XorSignAbs::Disabled }>(other)
            }

            /// Returns the element-wise minimum with explicit modifiers.
            ///
            /// `-0.0` is treated as less than `+0.0`. `N` selects the NaN handling:
            /// [`NanHandling::Ignore`] returns the other input if exactly one is NaN,
            /// [`NanHandling::Propagate`] yields a canonical NaN if either input is NaN. `X`
            /// selects whether each result has the minimum absolute value of the inputs and the
            /// XOR of their signs (skipped if the result is NaN); this requires `sm_86`.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
            #[inline]
            pub fn min_with<const N: NanHandling, const X: XorSignAbs>(self, other: Self) -> Self {
                const {
                    assert!(
                        matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                        "XorSignAbs requires sm_86"
                    );
                }
                Self(crate::data_types::array_from_fn(|i| {
                    self.0[i].min_with::<N, X>(other.0[i])
                }))
            }

            /// Returns the element-wise maximum with explicit modifiers.
            ///
            /// `+0.0` is treated as greater than `-0.0`. `N` selects the NaN handling:
            /// [`NanHandling::Ignore`] returns the other input if exactly one is NaN,
            /// [`NanHandling::Propagate`] yields a canonical NaN if either input is NaN. `X`
            /// selects whether each result has the maximum absolute value of the inputs and the
            /// XOR of their signs (skipped if the result is NaN); this requires `sm_86`.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
            #[inline]
            pub fn max_with<const N: NanHandling, const X: XorSignAbs>(self, other: Self) -> Self {
                const {
                    assert!(
                        matches!(X, XorSignAbs::Disabled) || cfg!(target_feature = "sm_86"),
                        "XorSignAbs requires sm_86"
                    );
                }
                Self(crate::data_types::array_from_fn(|i| {
                    self.0[i].max_with::<N, X>(other.0[i])
                }))
            }

            /// Returns the minimum element. NaN elements are ignored; the result is NaN only if all
            /// elements are NaN. `-0.0` is treated as less than `+0.0`. See
            /// [`reduce_min_with`](Self::reduce_min_with).
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
            #[inline]
            pub fn reduce_min(self) -> bf16 {
                self.reduce_min_with::<{ NanHandling::Ignore }>()
            }

            /// Returns the maximum element. NaN elements are ignored; the result is NaN only if all
            /// elements are NaN. `+0.0` is treated as greater than `-0.0`. See
            /// [`reduce_max_with`](Self::reduce_max_with).
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
            #[inline]
            pub fn reduce_max(self) -> bf16 {
                self.reduce_max_with::<{ NanHandling::Ignore }>()
            }

            /// Returns the minimum element with explicit NaN handling.
            ///
            /// `-0.0` is treated as less than `+0.0`. `N` selects the NaN handling:
            /// [`NanHandling::Ignore`] ignores NaN elements (the result is NaN only if all elements are
            /// NaN), [`NanHandling::Propagate`] yields a canonical NaN if any element is NaN.
            ///
            /// The elements are combined with `min` instructions in a tree, using `min.bf16x2` on pairs
            /// until a single pair remains.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min>
            #[inline]
            pub fn reduce_min_with<const N: NanHandling>(self) -> bf16 {
                self.lo()
                    .min_with::<N, { XorSignAbs::Disabled }>(self.hi())
                    .reduce_min_with::<N>()
            }

            /// Returns the maximum element with explicit NaN handling.
            ///
            /// `+0.0` is treated as greater than `-0.0`. `N` selects the NaN handling:
            /// [`NanHandling::Ignore`] ignores NaN elements (the result is NaN only if all elements are
            /// NaN), [`NanHandling::Propagate`] yields a canonical NaN if any element is NaN.
            ///
            /// The elements are combined with `max` instructions in a tree, using `max.bf16x2` on pairs
            /// until a single pair remains.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max>
            #[inline]
            pub fn reduce_max_with<const N: NanHandling>(self) -> bf16 {
                self.lo()
                    .max_with::<N, { XorSignAbs::Disabled }>(self.hi())
                    .reduce_max_with::<N>()
            }
        }

        #[cfg(target_feature = "sm_90")]
        impl $ty {
            /// Computes `2^x` of each element, approximately (`ex2.approx.ftz.bf16x2`).
            ///
            /// The maximum relative error of each element is 2^-7 (as for
            /// `ex2.approx.ftz.bf16`). Subnormal inputs and results are flushed to
            /// sign-preserving zero. `-inf` gives `+0.0`, subnormal inputs of either sign give
            /// `+1.0` and `+inf` gives `+inf`.
            ///
            /// Requires `sm_90`.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-ex2>
            #[inline]
            pub fn exp2_approx(self) -> Self {
                Self(crate::data_types::array_from_fn(|i| self.0[i].exp2_approx()))
            }

            /// Computes the hyperbolic tangent of each element, approximately
            /// (`tanh.approx.bf16x2`).
            ///
            /// The maximum absolute error of each element is 2^-8 (as for `tanh.approx.bf16`).
            /// Subnormal inputs are supported. `-inf` gives `-1.0`, `+inf` gives `+1.0`, and
            /// zeros are preserved including their sign.
            ///
            /// Requires `sm_90`.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-tanh>
            #[inline]
            pub fn tanh_approx(self) -> Self {
                Self(crate::data_types::array_from_fn(|i| self.0[i].tanh_approx()))
            }
        }

        impl Add for $ty {
            type Output = Self;

            /// Computed as `fma(self, 1.0, rhs)`, which is exact in the product.
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self(crate::data_types::array_from_fn(|i| self.0[i] + rhs.0[i]))
            }
        }

        impl Sub for $ty {
            type Output = Self;

            /// Computed as `fma(rhs, -1.0, self)`, which is exact in the product.
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self(crate::data_types::array_from_fn(|i| self.0[i] - rhs.0[i]))
            }
        }

        impl Mul for $ty {
            type Output = Self;

            /// Computed as `fma(self, rhs, -0.0)`; the negative-zero addend preserves the sign
            /// of zero products.
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self(crate::data_types::array_from_fn(|i| self.0[i] * rhs.0[i]))
            }
        }

        impl Neg for $ty {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self {
                Self(crate::data_types::array_from_fn(|i| -self.0[i]))
            }
        }

        impl AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl MulAssign for $ty {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
    };
}

bf16_wide!(bf16x4, n: 4, pairs: 2);
bf16_wide!(bf16x8, n: 8, pairs: 4);
bf16_wide!(bf16x16, n: 16, pairs: 8);
bf16_wide!(bf16x32, n: 32, pairs: 16);

/// Implements the scalar-broadcast operators and the conversion to `Simd<f32, N>` of a vector.
macro_rules! bf16_broadcast {
    ($ty:ident, n: $n:literal) => {
        impl Add<bf16> for $ty {
            type Output = Self;

            #[inline]
            fn add(self, rhs: bf16) -> Self {
                self + Self::splat(rhs)
            }
        }

        impl Sub<bf16> for $ty {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: bf16) -> Self {
                self - Self::splat(rhs)
            }
        }

        impl Mul<bf16> for $ty {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: bf16) -> Self {
                self * Self::splat(rhs)
            }
        }

        impl Add<$ty> for bf16 {
            type Output = $ty;

            #[inline]
            fn add(self, rhs: $ty) -> $ty {
                $ty::splat(self) + rhs
            }
        }

        impl Sub<$ty> for bf16 {
            type Output = $ty;

            #[inline]
            fn sub(self, rhs: $ty) -> $ty {
                $ty::splat(self) - rhs
            }
        }

        impl Mul<$ty> for bf16 {
            type Output = $ty;

            #[inline]
            fn mul(self, rhs: $ty) -> $ty {
                $ty::splat(self) * rhs
            }
        }

        impl AddAssign<bf16> for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: bf16) {
                *self = *self + rhs;
            }
        }

        impl SubAssign<bf16> for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: bf16) {
                *self = *self - rhs;
            }
        }

        impl MulAssign<bf16> for $ty {
            #[inline]
            fn mul_assign(&mut self, rhs: bf16) {
                *self = *self * rhs;
            }
        }

        impl From<$ty> for Simd<f32, $n> {
            #[inline]
            fn from(x: $ty) -> Self {
                x.to_f32()
            }
        }
    };
}

bf16_broadcast!(bf16x2, n: 2);
bf16_broadcast!(bf16x4, n: 4);
bf16_broadcast!(bf16x8, n: 8);
bf16_broadcast!(bf16x16, n: 16);
bf16_broadcast!(bf16x32, n: 32);

half_reduce_sum!(bf16x2, "bf16", x2);
half_reduce_sum!(bf16x4, "bf16", wide);
half_reduce_sum!(bf16x8, "bf16", wide);
half_reduce_sum!(bf16x16, "bf16", wide);
half_reduce_sum!(bf16x32, "bf16", wide);
