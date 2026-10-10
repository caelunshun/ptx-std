// --- LLM-generated --- //
//! Packed 8-bit floating-point formats: FP8 (`e4m3`, `e5m2`) and the `ue8m0` scale format.
//!
//! In all packed types, element 0 occupies the least significant bits.

use bytemuck::{Pod, Zeroable};

// Only used by the conversions, which are gated on newer architectures.
#[allow(unused_imports)]
use {
    super::narrow::{self, narrow_x2, narrow_x4},
    crate::{Overflow, Rounding, bf16x2},
    core::arch::nvptx as nv,
};

/// Two packed FP8 E4M3 values (1 sign, 4 exponent and 3 mantissa bits; exponent bias 7).
///
/// The format has no infinities and a single NaN encoding (`0x7f`/`0xff`); the largest finite
/// magnitude is 448. Element 0 occupies the low 8 bits.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f8e4m3x2(u16);

impl f8e4m3x2 {
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
    f8e4m3x2(u16), "FP8 E4M3",
    basic: target_feature = "sm_89",
    from_f16x2: target_feature = "sm_89",
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    core: {
        from_f32: cvt_e4m3x2_f32,
        from_f16x2: cvt_e4m3x2_f16x2,
        from_bf16x2: cvt_e4m3x2_bf16x2,
        to_f16x2: cvt_f16x2_e4m3x2,
        to_bf16x2_scaled: cvt_scale_bf16x2_e4m3x2,
    },
}

/// Four packed FP8 E4M3 values; see [`f8e4m3x2`] for the format. Element 0 occupies the low 8
/// bits.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f8e4m3x4(u32);

impl f8e4m3x4 {
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
    f8e4m3x4(u32), f8e4m3x2(u16), "FP8 E4M3",
    basic: target_feature = "sm_89",
    from_f16x2: target_feature = "sm_89",
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
    core_rs: cvt_rs_e4m3x4_f32,
}

/// Two packed FP8 E5M2 values (1 sign, 5 exponent and 2 mantissa bits; exponent bias 15).
///
/// The format follows IEEE 754 conventions, with the same encodings of infinity (`0x7c`/`0xfc`)
/// and NaN as the upper byte of an IEEE `binary16` value; the largest finite magnitude is 57344.
/// Element 0 occupies the low 8 bits.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f8e5m2x2(u16);

impl f8e5m2x2 {
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
    f8e5m2x2(u16), "FP8 E5M2",
    basic: target_feature = "sm_89",
    from_f16x2: target_feature = "sm_89",
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    core: {
        from_f32: cvt_e5m2x2_f32,
        from_f16x2: cvt_e5m2x2_f16x2,
        from_bf16x2: cvt_e5m2x2_bf16x2,
        to_f16x2: cvt_f16x2_e5m2x2,
        to_bf16x2_scaled: cvt_scale_bf16x2_e5m2x2,
    },
}

/// Four packed FP8 E5M2 values; see [`f8e5m2x2`] for the format. Element 0 occupies the low 8
/// bits.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f8e5m2x4(u32);

impl f8e5m2x4 {
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
    f8e5m2x4(u32), f8e5m2x2(u16), "FP8 E5M2",
    basic: target_feature = "sm_89",
    from_f16x2: target_feature = "sm_89",
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
    core_rs: cvt_rs_e5m2x4_f32,
}

/// Two packed unsigned E8M0 (`ue8m0`) scale factors.
///
/// Each element has 8 exponent bits and no sign or mantissa bits, so it represents the power of two
/// `2^(bits - 127)` (from `2^-127` to `2^127`); `0xff` is NaN. The format has no infinity, zero or
/// negative values. These are the scale factors of the block-scaled narrow formats and of the
/// `to_bf16x2_scaled` conversions. Element 0 occupies the low 8 bits.
///
/// The conversions to and from other formats require `sm_100f`, `sm_101f`, `sm_110f` or
/// `sm_120f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f8e8m0x2(u16);

impl f8e8m0x2 {
    /// All scale factors equal to `1.0` (`2^0`, encoded as `0x7f`).
    pub const ONES: Self = Self(0x7f7f);

    /// Creates a value from its raw bit representation.
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    pub const fn to_bits(self) -> u16 {
        self.0
    }
}

#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f",
    target_feature = "sm_120f"
))]
impl f8e8m0x2 {
    /// Converts two `f32` values to a packed pair of `ue8m0` scale factors.
    ///
    /// `x[0]` becomes element 0 and `x[1]` becomes element 1. Since the format only holds powers
    /// of two, `RND` selects rounding to a power of two toward zero ([`Rounding::TowardZero`]) or
    /// toward positive infinity ([`Rounding::Up`]); no other mode is supported. NaN becomes NaN.
    /// With [`Overflow::SaturateFinite`], out-of-range values are clamped to the largest finite
    /// value `2^127` (`.satfinite`, always positive as the format is unsigned); without it,
    /// the PTX ISA does not define the result of an overflow as the format has no infinity.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32s<const RND: Rounding, const OVERFLOW: Overflow>(x: [f32; 2]) -> Self {
        const {
            assert!(
                matches!(RND, Rounding::TowardZero | Rounding::Up),
                "conversion to ue8m0 only supports the TowardZero and Up rounding modes"
            )
        }
        const_map!(RND => R: nv::Rounding {
            Rounding::TowardZero => nv::Rounding::Rz,
            Rounding::Up => nv::Rounding::Rp,
        } in const_bool!(S = matches!(OVERFLOW, Overflow::SaturateFinite) => {
            Self(unsafe { nv::cvt_ue8m0x2_f32::<R, S>(x[1], x[0]) })
        }))
    }

    /// Converts a packed pair of `bf16` values to a packed pair of `ue8m0` scale factors.
    ///
    /// Element `i` of the result is converted from element `i` of `x`. Since the format only
    /// holds powers of two, `RND` selects rounding to a power of two toward zero
    /// ([`Rounding::TowardZero`]) or toward positive infinity ([`Rounding::Up`]); no other mode is
    /// supported. NaN becomes NaN. With [`Overflow::SaturateFinite`], out-of-range values are
    /// clamped to the largest finite value `2^127` (`.satfinite`, always positive as the format is
    /// unsigned); without it, the PTX ISA does not define the result of an overflow as the format
    /// has no infinity.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_bf16x2<const RND: Rounding, const OVERFLOW: Overflow>(x: bf16x2) -> Self {
        const {
            assert!(
                matches!(RND, Rounding::TowardZero | Rounding::Up),
                "conversion to ue8m0 only supports the TowardZero and Up rounding modes"
            )
        }
        const_map!(RND => R: nv::Rounding {
            Rounding::TowardZero => nv::Rounding::Rz,
            Rounding::Up => nv::Rounding::Rp,
        } in const_bool!(S = matches!(OVERFLOW, Overflow::SaturateFinite) => {
            Self(unsafe { nv::cvt_ue8m0x2_bf16x2::<R, S>(narrow::bf16x2_to_core(x)) })
        }))
    }

    /// Converts the packed pair of `ue8m0` scale factors to a packed pair of `bf16` values.
    ///
    /// Element `i` of the result is converted from element `i` of `self`. The conversion is
    /// exact; `0xff` becomes NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn to_bf16x2(self) -> bf16x2 {
        narrow::bf16x2_from_core(unsafe { nv::cvt_bf16x2_ue8m0x2(self.0) })
    }
}

/// Four packed unsigned E8M0 (`ue8m0`) scale factors; see [`f8e8m0x2`] for the format. Element 0
/// occupies the low 8 bits.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct f8e8m0x4(u32);

impl f8e8m0x4 {
    /// All scale factors equal to `1.0` (`2^0`, encoded as `0x7f`).
    pub const ONES: Self = Self(0x7f7f_7f7f);

    /// Creates a value from its raw bit representation.
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation.
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// Packs two pairs into four values; `x2s[0]` becomes elements 0 and 1, `x2s[1]` becomes
    /// elements 2 and 3.
    #[inline]
    pub const fn from_x2s(x2s: [f8e8m0x2; 2]) -> Self {
        Self((x2s[0].to_bits() as u32) | ((x2s[1].to_bits() as u32) << 16))
    }

    /// Splits the four values into two pairs; the first holds elements 0 and 1, the second
    /// elements 2 and 3.
    #[inline]
    pub const fn to_x2s(self) -> [f8e8m0x2; 2] {
        [
            f8e8m0x2::from_bits(self.0 as u16),
            f8e8m0x2::from_bits((self.0 >> 16) as u16),
        ]
    }
}

#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f",
    target_feature = "sm_120f"
))]
impl f8e8m0x4 {
    /// Converts four `f32` values to `ue8m0` scale factors.
    ///
    /// `x[i]` becomes element `i`. See [`f8e8m0x2::from_f32s`].
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_f32s<const RND: Rounding, const OVERFLOW: Overflow>(x: [f32; 4]) -> Self {
        Self::from_x2s([
            f8e8m0x2::from_f32s::<RND, OVERFLOW>([x[0], x[1]]),
            f8e8m0x2::from_f32s::<RND, OVERFLOW>([x[2], x[3]]),
        ])
    }

    /// Converts four `bf16` values, given as two packed pairs, to `ue8m0` scale factors.
    ///
    /// Element `i` of the result is converted from `x[i / 2]`, element `i % 2`. See
    /// [`f8e8m0x2::from_bf16x2`].
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn from_bf16x2s<const RND: Rounding, const OVERFLOW: Overflow>(x: [bf16x2; 2]) -> Self {
        Self::from_x2s([
            f8e8m0x2::from_bf16x2::<RND, OVERFLOW>(x[0]),
            f8e8m0x2::from_bf16x2::<RND, OVERFLOW>(x[1]),
        ])
    }

    /// Converts the four `ue8m0` scale factors to `bf16` values, returned as two packed pairs.
    ///
    /// Element `i` of `self` becomes element `i % 2` of the result at index `i / 2`. The
    /// conversion is exact; `0xff` becomes NaN.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn to_bf16x2s(self) -> [bf16x2; 2] {
        let [lo, hi] = self.to_x2s();
        [lo.to_bf16x2(), hi.to_bf16x2()]
    }
}
