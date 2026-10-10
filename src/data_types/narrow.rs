// --- LLM-generated --- //
//! Shared implementation of the conversions of the packed narrow floating-point formats
//! (FP8, FP6, FP4).
//!
//! All packed narrow types use the same element order: element 0 occupies the least significant
//! bits. The `cvt` instructions instead place their first operand `a` in the *upper* half of the
//! result, so the wrappers swap the operands (`x[0]` is PTX operand `b`, `x[1]` is `a`, and for the
//! four-element forms `x[3]` is `a` and `x[0]` is `f`).
//!
//! The conversion sets of the formats are identical but require different target architectures,
//! so the methods are generated per type by the macros below, with the `cfg` of each group passed
//! in by the invocation.

// The helpers are only used by conversions gated on architectures newer than the baseline.
#![allow(dead_code)]

use core::{arch::nvptx as nv, mem::transmute};

use crate::{bf16x2, f16x2};

#[inline]
pub(crate) fn f16x2_to_core(x: f16x2) -> nv::f16x2 {
    // SAFETY: both are plain 32-bit values holding two `f16`s.
    unsafe { transmute(x.to_bits()) }
}

#[inline]
pub(crate) fn f16x2_from_core(x: nv::f16x2) -> f16x2 {
    // SAFETY: see `f16x2_to_core`.
    f16x2::from_bits(unsafe { transmute::<nv::f16x2, u32>(x) })
}

#[inline]
pub(crate) fn bf16x2_to_core(x: bf16x2) -> nv::bf16x2 {
    // SAFETY: both are plain 32-bit values holding two `bf16`s.
    unsafe { transmute(x.to_bits()) }
}

#[inline]
pub(crate) fn bf16x2_from_core(x: nv::bf16x2) -> bf16x2 {
    // SAFETY: see `bf16x2_to_core`.
    bf16x2::from_bits(unsafe { transmute::<nv::bf16x2, u32>(x) })
}

/// Implements the conversions of a packed pair of narrow floats (`$ty`, stored in `$bits`).
///
/// - `basic`: `cfg` of the `f32` and `f16x2` up-conversions.
/// - `from_f16x2`: `cfg` of the `f16x2` down-conversion.
/// - `newer`: `cfg` of the `bf16x2` down-conversion and the scaled `bf16x2` up-conversion.
macro_rules! narrow_x2 {
    (
        $ty:ident($bits:ty), $name:literal,
        basic: $basic:meta,
        from_f16x2: $from_f16x2_cfg:meta,
        newer: $newer:meta,
        core: {
            from_f32: $core_from_f32:ident,
            from_f16x2: $core_from_f16x2:ident,
            from_bf16x2: $core_from_bf16x2:ident,
            to_f16x2: $core_to_f16x2:ident,
            to_bf16x2_scaled: $core_to_bf16x2_scaled:ident $(,)?
        } $(,)?
    ) => {
        impl $ty {
            #[inline]
            #[allow(clippy::unnecessary_cast, dead_code)]
            fn from_core(bits: u16) -> Self {
                Self::from_bits(bits as $bits)
            }

            #[inline]
            #[allow(clippy::unnecessary_cast, dead_code)]
            fn to_core(self) -> u16 {
                self.to_bits() as u16
            }

            #[doc = concat!("Converts two `f32` values to a packed pair of ", $name, " values, rounding to nearest even.")]
            ///
            /// `x[0]` becomes element 0 and `x[1]` becomes element 1. Out-of-range values are
            /// clamped to the largest finite value of matching sign (`.satfinite`). `RELU`
            /// clamps negative results to zero.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($basic)]
            #[inline]
            pub fn from_f32s<const RELU: crate::Relu>(x: [f32; 2]) -> Self {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    Self::from_core(unsafe { ::core::arch::nvptx::$core_from_f32::<R>(x[1], x[0]) })
                })
            }

            #[doc = concat!("Converts a packed pair of `f16` values to a packed pair of ", $name, " values, rounding to nearest even.")]
            ///
            /// Element `i` of the result is converted from element `i` of `x`. Out-of-range values
            /// are clamped to the largest finite value of matching sign (`.satfinite`). `RELU`
            /// clamps negative results to zero.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($from_f16x2_cfg)]
            #[inline]
            pub fn from_f16x2<const RELU: crate::Relu>(x: crate::f16x2) -> Self {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    Self::from_core(unsafe { ::core::arch::nvptx::$core_from_f16x2::<R>(crate::data_types::narrow::f16x2_to_core(x)) })
                })
            }

            #[doc = concat!("Converts a packed pair of `bf16` values to a packed pair of ", $name, " values, rounding to nearest even.")]
            ///
            /// Element `i` of the result is converted from element `i` of `x`. Out-of-range values
            /// are clamped to the largest finite value of matching sign (`.satfinite`). `RELU`
            /// clamps negative results to zero.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline]
            pub fn from_bf16x2<const RELU: crate::Relu>(x: crate::bf16x2) -> Self {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    Self::from_core(unsafe { ::core::arch::nvptx::$core_from_bf16x2::<R>(crate::data_types::narrow::bf16x2_to_core(x)) })
                })
            }

            #[doc = concat!("Converts the packed pair of ", $name, " values to a packed pair of `f16` values.")]
            ///
            /// Element `i` of the result is converted from element `i` of `self`. `RELU` clamps
            /// negative results to zero.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($basic)]
            #[inline]
            pub fn to_f16x2<const RELU: crate::Relu>(self) -> crate::f16x2 {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    crate::data_types::narrow::f16x2_from_core(unsafe { ::core::arch::nvptx::$core_to_f16x2::<R>(self.to_core()) })
                })
            }

            #[doc = concat!("Converts the packed pair of ", $name, " values to a packed pair of `bf16` values.")]
            ///
            /// Element `i` of the result is converted from element `i` of `self`. The conversion
            /// is exact; `RELU` clamps negative results to zero.
            ///
            /// Implemented as [`Self::to_bf16x2_scaled`] with unit scale factors.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline]
            pub fn to_bf16x2<const RELU: crate::Relu>(self) -> crate::bf16x2 {
                self.to_bf16x2_scaled::<RELU, { crate::Overflow::Infinity }>(crate::f8e8m0x2::ONES)
            }

            #[doc = concat!("Converts the packed pair of ", $name, " values to a packed pair of `bf16` values, multiplying each element by its `ue8m0` scale factor.")]
            ///
            /// Element `i` of the result is converted from element `i` of `self` and multiplied by
            /// element `i` of `scale` (`.scaled::n2::ue8m0`). `RELU` clamps negative results to
            /// zero. With [`Overflow::SaturateFinite`](crate::Overflow::SaturateFinite), results outside the finite range are
            /// clamped to the largest finite `bf16` value of matching sign.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline]
            pub fn to_bf16x2_scaled<const RELU: crate::Relu, const OVERFLOW: crate::Overflow>(
                self,
                scale: crate::f8e8m0x2,
            ) -> crate::bf16x2 {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    const_bool!(S = matches!(OVERFLOW, crate::Overflow::SaturateFinite) => {
                        crate::data_types::narrow::bf16x2_from_core(unsafe {
                            ::core::arch::nvptx::$core_to_bf16x2_scaled::<R, S>(self.to_core(), scale.to_bits())
                        })
                    })
                })
            }
        }
    };
}

/// Implements the packing, unpacking and conversions of four packed narrow floats (`$ty`, stored
/// in `$bits`) in terms of the pair type `$x2` (stored in `$x2_bits`).
///
/// The `cfg`s are those of the corresponding `$x2` conversions; `stochastic` is that of the
/// stochastic-rounding conversion.
macro_rules! narrow_x4 {
    (
        $ty:ident($bits:ty), $x2:ident($x2_bits:ty), $name:literal,
        basic: $basic:meta,
        from_f16x2: $from_f16x2_cfg:meta,
        newer: $newer:meta,
        stochastic: $stochastic:meta,
        core_rs: $core_rs:ident $(,)?
    ) => {
        impl $ty {
            /// Packs two pairs into four values; `x2s[0]` becomes elements 0 and 1, `x2s[1]`
            /// becomes elements 2 and 3.
            #[inline]
            pub const fn from_x2s(x2s: [$x2; 2]) -> Self {
                Self((x2s[0].to_bits() as $bits) | ((x2s[1].to_bits() as $bits) << <$x2_bits>::BITS))
            }

            /// Splits the four values into two pairs; the first holds elements 0 and 1, the second
            /// elements 2 and 3.
            #[inline]
            pub const fn to_x2s(self) -> [$x2; 2] {
                [
                    $x2::from_bits(self.0 as $x2_bits),
                    $x2::from_bits((self.0 >> <$x2_bits>::BITS) as $x2_bits),
                ]
            }

            #[doc = concat!("Converts four `f32` values to ", $name, " values, rounding to nearest even.")]
            ///
            /// `x[i]` becomes element `i`. Out-of-range values are clamped to the largest finite
            /// value of matching sign (`.satfinite`). `RELU` clamps negative results to zero.
            ///
            /// Implemented as two pair conversions; see [`Self::from_f32s_stochastic`] for
            /// stochastic rounding.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($basic)]
            #[inline]
            pub fn from_f32s<const RELU: crate::Relu>(x: [f32; 4]) -> Self {
                Self::from_x2s([
                    $x2::from_f32s::<RELU>([x[0], x[1]]),
                    $x2::from_f32s::<RELU>([x[2], x[3]]),
                ])
            }

            #[doc = concat!("Converts four `f32` values to ", $name, " values with stochastic rounding (`.rs`).")]
            ///
            /// `x[i]` becomes element `i`. `rbits` provides the random bits: the upper 16 bits
            /// are used for elements 2 and 3 and the lower 16 bits for elements 0 and 1 (see the
            /// PTX ISA for the exact bit layout). Out-of-range values are clamped to the largest
            /// finite value of matching sign (`.satfinite`). `RELU` clamps negative results to
            /// zero.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($stochastic)]
            #[inline]
            #[allow(clippy::unnecessary_cast)]
            pub fn from_f32s_stochastic<const RELU: crate::Relu>(x: [f32; 4], rbits: u32) -> Self {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    Self::from_bits(unsafe { ::core::arch::nvptx::$core_rs::<R>(x[3], x[2], x[1], x[0], rbits) } as $bits)
                })
            }

            #[doc = concat!("Converts four `f16` values, given as two packed pairs, to ", $name, " values, rounding to nearest even.")]
            ///
            /// Element `i` of the result is converted from `x[i / 2]`, element `i % 2`.
            /// Out-of-range values are clamped to the largest finite value of matching sign
            /// (`.satfinite`). `RELU` clamps negative results to zero.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($from_f16x2_cfg)]
            #[inline]
            pub fn from_f16x2s<const RELU: crate::Relu>(x: [crate::f16x2; 2]) -> Self {
                Self::from_x2s([
                    $x2::from_f16x2::<RELU>(x[0]),
                    $x2::from_f16x2::<RELU>(x[1]),
                ])
            }

            #[doc = concat!("Converts four `bf16` values, given as two packed pairs, to ", $name, " values, rounding to nearest even.")]
            ///
            /// Element `i` of the result is converted from `x[i / 2]`, element `i % 2`.
            /// Out-of-range values are clamped to the largest finite value of matching sign
            /// (`.satfinite`). `RELU` clamps negative results to zero.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline]
            pub fn from_bf16x2s<const RELU: crate::Relu>(x: [crate::bf16x2; 2]) -> Self {
                Self::from_x2s([
                    $x2::from_bf16x2::<RELU>(x[0]),
                    $x2::from_bf16x2::<RELU>(x[1]),
                ])
            }

            #[doc = concat!("Converts the four ", $name, " values to `f16` values, returned as two packed pairs.")]
            ///
            /// Element `i` of `self` becomes element `i % 2` of the result at index `i / 2`.
            /// `RELU` clamps negative results to zero.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($basic)]
            #[inline]
            pub fn to_f16x2s<const RELU: crate::Relu>(self) -> [crate::f16x2; 2] {
                let [lo, hi] = self.to_x2s();
                [lo.to_f16x2::<RELU>(), hi.to_f16x2::<RELU>()]
            }

            #[doc = concat!("Converts the four ", $name, " values to `bf16` values, returned as two packed pairs.")]
            ///
            /// Element `i` of `self` becomes element `i % 2` of the result at index `i / 2`. The
            /// conversion is exact; `RELU` clamps negative results to zero.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline]
            pub fn to_bf16x2s<const RELU: crate::Relu>(self) -> [crate::bf16x2; 2] {
                let [lo, hi] = self.to_x2s();
                [lo.to_bf16x2::<RELU>(), hi.to_bf16x2::<RELU>()]
            }

            #[doc = concat!("Converts the four ", $name, " values to `bf16` values, returned as two packed pairs, multiplying each element by its `ue8m0` scale factor.")]
            ///
            /// Element `i` of `self` is multiplied by element `i` of `scales` and becomes element
            /// `i % 2` of the result at index `i / 2`. `RELU` clamps negative results to zero.
            /// With [`Overflow::SaturateFinite`](crate::Overflow::SaturateFinite), results outside the finite range are clamped to
            /// the largest finite `bf16` value of matching sign.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline]
            pub fn to_bf16x2s_scaled<const RELU: crate::Relu, const OVERFLOW: crate::Overflow>(
                self,
                scales: crate::f8e8m0x4,
            ) -> [crate::bf16x2; 2] {
                let [lo, hi] = self.to_x2s();
                let [scale_lo, scale_hi] = scales.to_x2s();
                [
                    lo.to_bf16x2_scaled::<RELU, OVERFLOW>(scale_lo),
                    hi.to_bf16x2_scaled::<RELU, OVERFLOW>(scale_hi),
                ]
            }
        }
    };
}

pub(crate) use narrow_x2;
pub(crate) use narrow_x4;
