// --- LLM-generated --- //
//! Packed 4-bit floating-point format (`e2m1`).
//!
//! In all packed types, element 0 occupies the least significant bits.

use bytemuck::{Pod, Zeroable};

// Only used by the conversions, which are gated on newer architectures.
#[allow(unused_imports)]
use super::narrow::{narrow_x2, narrow_x4};

/// Two packed FP4 E2M1 values (1 sign, 2 exponent and 1 mantissa bit; exponent bias 1).
///
/// The format has no infinities or NaNs; the largest finite magnitude is 6.0. Element 0 occupies the
/// low 4 bits.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f4e2m1x2(u8);

impl f4e2m1x2 {
    /// Creates a value from its raw bit representation.
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    pub const fn to_bits(self) -> u8 {
        self.0
    }
}

narrow_x2! {
    f4e2m1x2(u8), "FP4 E2M1",
    basic: any(target_feature = "sm_100f", target_feature = "sm_101f", target_feature = "sm_110f", target_feature = "sm_120f"),
    from_f16x2: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    core: {
        from_f32: cvt_e2m1x2_f32,
        from_f16x2: cvt_e2m1x2_f16x2,
        from_bf16x2: cvt_e2m1x2_bf16x2,
        to_f16x2: cvt_f16x2_e2m1x2,
        to_bf16x2_scaled: cvt_scale_bf16x2_e2m1x2,
    },
}

/// Four packed FP4 E2M1 values; see [`f4e2m1x2`] for the format. Element 0 occupies the low 4
/// bits.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f4e2m1x4(u16);

impl f4e2m1x4 {
    /// Creates a value from its raw bit representation.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

narrow_x4! {
    f4e2m1x4(u16), f4e2m1x2(u8), "FP4 E2M1",
    basic: any(target_feature = "sm_100f", target_feature = "sm_101f", target_feature = "sm_110f", target_feature = "sm_120f"),
    from_f16x2: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
    core_rs: cvt_rs_e2m1x4_f32,
}

/// Eight packed FP4 E2M1 values; see [`f4e2m1x2`] for the format. Element 0 occupies the low 4
/// bits.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f4e2m1x8(u32);

impl f4e2m1x8 {
    /// Creates a value from its raw bit representation.
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    pub const fn to_bits(self) -> u32 {
        self.0
    }
}

impl f4e2m1x8 {
    /// Packs two groups of four into eight values; `x4s[0]` becomes elements 0 to 3 and `x4s[1]`
    /// becomes elements 4 to 7.
    #[inline]
    pub const fn from_x4s(x4s: [f4e2m1x4; 2]) -> Self {
        Self((x4s[0].to_bits() as u32) | ((x4s[1].to_bits() as u32) << 16))
    }

    /// Splits the eight values into two groups of four; the first holds elements 0 to 3, the second
    /// elements 4 to 7.
    #[inline]
    pub const fn to_x4s(self) -> [f4e2m1x4; 2] {
        [
            f4e2m1x4::from_bits(self.0 as u16),
            f4e2m1x4::from_bits((self.0 >> 16) as u16),
        ]
    }
}

#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f",
    target_feature = "sm_120f"
))]
impl f4e2m1x8 {
    /// Converts eight `f32` values to FP4 E2M1 values, rounding to nearest even.
    ///
    /// `x[i]` becomes element `i`. Out-of-range values are clamped to the largest finite value of
    /// matching sign (`.satfinite`). `RELU` clamps negative results to zero.
    ///
    /// Implemented as four pair conversions; see [`Self::from_f32s_stochastic`] for stochastic
    /// rounding.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32s<const RELU: crate::Relu>(x: [f32; 8]) -> Self {
        Self::from_x4s([
            f4e2m1x4::from_f32s::<RELU>([x[0], x[1], x[2], x[3]]),
            f4e2m1x4::from_f32s::<RELU>([x[4], x[5], x[6], x[7]]),
        ])
    }

    /// Converts the eight FP4 E2M1 values to `f16` values, returned as four packed pairs.
    ///
    /// Element `i` of `self` becomes element `i % 2` of the result at index `i / 2`. `RELU` clamps
    /// negative results to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn to_f16x2s<const RELU: crate::Relu>(self) -> [crate::f16x2; 4] {
        let [lo, hi] = self.to_x4s();
        let [a, b] = lo.to_f16x2s::<RELU>();
        let [c, d] = hi.to_f16x2s::<RELU>();
        [a, b, c, d]
    }
}

#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_110f",
    target_feature = "sm_120f"
))]
impl f4e2m1x8 {
    /// Converts eight `f16` values, given as four packed pairs, to FP4 E2M1 values, rounding to
    /// nearest even.
    ///
    /// Element `i` of the result is converted from `x[i / 2]`, element `i % 2`. Out-of-range values
    /// are clamped to the largest finite value of matching sign (`.satfinite`). `RELU` clamps
    /// negative results to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f16x2s<const RELU: crate::Relu>(x: [crate::f16x2; 4]) -> Self {
        Self::from_x4s([
            f4e2m1x4::from_f16x2s::<RELU>([x[0], x[1]]),
            f4e2m1x4::from_f16x2s::<RELU>([x[2], x[3]]),
        ])
    }

    /// Converts eight `bf16` values, given as four packed pairs, to FP4 E2M1 values, rounding to
    /// nearest even.
    ///
    /// Element `i` of the result is converted from `x[i / 2]`, element `i % 2`. Out-of-range values
    /// are clamped to the largest finite value of matching sign (`.satfinite`). `RELU` clamps
    /// negative results to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_bf16x2s<const RELU: crate::Relu>(x: [crate::bf16x2; 4]) -> Self {
        Self::from_x4s([
            f4e2m1x4::from_bf16x2s::<RELU>([x[0], x[1]]),
            f4e2m1x4::from_bf16x2s::<RELU>([x[2], x[3]]),
        ])
    }

    /// Converts the eight FP4 E2M1 values to `bf16` values, returned as four packed pairs.
    ///
    /// Element `i` of `self` becomes element `i % 2` of the result at index `i / 2`. The conversion
    /// is exact; `RELU` clamps negative results to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn to_bf16x2s<const RELU: crate::Relu>(self) -> [crate::bf16x2; 4] {
        let [lo, hi] = self.to_x4s();
        let [a, b] = lo.to_bf16x2s::<RELU>();
        let [c, d] = hi.to_bf16x2s::<RELU>();
        [a, b, c, d]
    }

    /// Converts the eight FP4 E2M1 values to `bf16` values, returned as four packed pairs,
    /// multiplying each element by its `ue8m0` scale factor.
    ///
    /// Element `i` of `self` is multiplied by element `i % 4` of `scales[i / 4]` and becomes element
    /// `i % 2` of the result at index `i / 2`. `RELU` clamps negative results to zero. With
    /// [`Overflow::SaturateFinite`](crate::Overflow::SaturateFinite), results outside the finite
    /// range are clamped to the largest finite `bf16` value of matching sign.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn to_bf16x2s_scaled<const RELU: crate::Relu, const OVERFLOW: crate::Overflow>(
        self,
        scales: [crate::f8e8m0x4; 2],
    ) -> [crate::bf16x2; 4] {
        let [lo, hi] = self.to_x4s();
        let [a, b] = lo.to_bf16x2s_scaled::<RELU, OVERFLOW>(scales[0]);
        let [c, d] = hi.to_bf16x2s_scaled::<RELU, OVERFLOW>(scales[1]);
        [a, b, c, d]
    }
}

#[cfg(any(target_feature = "sm_100a", target_feature = "sm_103a"))]
impl f4e2m1x8 {
    /// Converts eight `f32` values to FP4 E2M1 values with stochastic rounding (`.rs`).
    ///
    /// `x[i]` becomes element `i`. Elements 0 to 3 are rounded using the random bits `rbits[0]`
    /// and elements 4 to 7 using `rbits[1]`, as described for [`f4e2m1x4::from_f32s_stochastic`].
    /// Out-of-range values are clamped to the largest finite value of matching sign
    /// (`.satfinite`). `RELU` clamps negative results to zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32s_stochastic<const RELU: crate::Relu>(x: [f32; 8], rbits: [u32; 2]) -> Self {
        Self::from_x4s([
            f4e2m1x4::from_f32s_stochastic::<RELU>([x[0], x[1], x[2], x[3]], rbits[0]),
            f4e2m1x4::from_f32s_stochastic::<RELU>([x[4], x[5], x[6], x[7]], rbits[1]),
        ])
    }
}
