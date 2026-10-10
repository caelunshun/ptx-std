// --- LLM-generated --- //
//! Packed 6-bit floating-point formats (`e2m3`, `e3m2`).
//!
//! Each 6-bit value is stored in the low 6 bits of a byte; the two most significant bits of every
//! byte are zero.
//!
//! In all packed types, element 0 occupies the least significant bits.

use bytemuck::{Pod, Zeroable};

// Only used by the conversions, which are gated on newer architectures.
#[allow(unused_imports)]
use super::narrow::{narrow_x2, narrow_x4};

/// Two packed FP6 E2M3 values (1 sign, 2 exponent and 3 mantissa bits; exponent bias 1)
///
/// The format has no infinities or NaNs; the largest finite magnitude is 7.5.
///
/// Each value is stored in the low 6 bits of a byte. Element 0 occupies the low byte.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f6e2m3x2(u16);

impl f6e2m3x2 {
    /// Creates a value from its raw bit representation.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

narrow_x2! {
    f6e2m3x2(u16), "FP6 E2M3",
    basic: any(target_feature = "sm_100f", target_feature = "sm_101f", target_feature = "sm_110f", target_feature = "sm_120f"),
    from_f16x2: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    core: {
        from_f32: cvt_e2m3x2_f32,
        from_f16x2: cvt_e2m3x2_f16x2,
        from_bf16x2: cvt_e2m3x2_bf16x2,
        to_f16x2: cvt_f16x2_e2m3x2,
        to_bf16x2_scaled: cvt_scale_bf16x2_e2m3x2,
    },
}

/// Four packed FP6 E2M3 values; see [`f6e2m3x2`] for the format. Each value is stored in the low 6 bits
/// of a byte. Element 0 occupies the low byte.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f6e2m3x4(u32);

impl f6e2m3x4 {
    /// Creates a value from its raw bit representation.
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    pub const fn to_bits(self) -> u32 {
        self.0
    }
}

narrow_x4! {
    f6e2m3x4(u32), f6e2m3x2(u16), "FP6 E2M3",
    basic: any(target_feature = "sm_100f", target_feature = "sm_101f", target_feature = "sm_110f", target_feature = "sm_120f"),
    from_f16x2: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
    core_rs: cvt_rs_e2m3x4_f32,
}

/// Two packed FP6 E3M2 values (1 sign, 3 exponent and 2 mantissa bits; exponent bias 3)
///
/// The format has no infinities or NaNs; the largest finite magnitude is 28.
///
/// Each value is stored in the low 6 bits of a byte. Element 0 occupies the low byte.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f6e3m2x2(u16);

impl f6e3m2x2 {
    /// Creates a value from its raw bit representation.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

narrow_x2! {
    f6e3m2x2(u16), "FP6 E3M2",
    basic: any(target_feature = "sm_100f", target_feature = "sm_101f", target_feature = "sm_110f", target_feature = "sm_120f"),
    from_f16x2: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    core: {
        from_f32: cvt_e3m2x2_f32,
        from_f16x2: cvt_e3m2x2_f16x2,
        from_bf16x2: cvt_e3m2x2_bf16x2,
        to_f16x2: cvt_f16x2_e3m2x2,
        to_bf16x2_scaled: cvt_scale_bf16x2_e3m2x2,
    },
}

/// Four packed FP6 E3M2 values; see [`f6e3m2x2`] for the format. Each value is stored in the low 6 bits
/// of a byte. Element 0 occupies the low byte.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f6e3m2x4(u32);

impl f6e3m2x4 {
    /// Creates a value from its raw bit representation.
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    pub const fn to_bits(self) -> u32 {
        self.0
    }
}

narrow_x4! {
    f6e3m2x4(u32), f6e3m2x2(u16), "FP6 E3M2",
    basic: any(target_feature = "sm_100f", target_feature = "sm_101f", target_feature = "sm_110f", target_feature = "sm_120f"),
    from_f16x2: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
    core_rs: cvt_rs_e3m2x4_f32,
}
