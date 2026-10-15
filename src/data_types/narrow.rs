// --- LLM-generated --- //
//! Shared implementation of the conversions of the narrow floating-point formats (FP8, FP6, FP4,
//! `ue8m0` and `s2f6`), for the scalars and all vector lengths.
//!
//! The PTX `cvt` instructions convert pairs (`x2`) of values, so the conversions are implemented
//! as private `pair_*` methods on the pair type and lifted to vectors of any length `N`:
//!
//! - the `pair_*` methods call the `core::arch::nvptx` intrinsics (or inline PTX) directly;
//! - the scalars use the pair instruction with the other lane set to zero and keep lane 0;
//! - the public methods of `Vector<E, N>` apply the pair conversion to each pair of elements. The
//!   stochastic-rounding conversions operate on four values instead and apply the four-element
//!   instruction to each group of four, so they require `VectorLen<N>: QuadVectorLen`.
//!
//! All packed narrow types use the same element order: element 0 occupies the least significant
//! bits. The `cvt` instructions instead place their first operand `a` in the *upper* half of the
//! result, so the pair methods swap the operands (`x[0]` is PTX operand `b`, `x[1]` is `a`, and for
//! the four-element forms `x[3]` is `a` and `x[0]` is `f`).
//!
//! The conversion sets of the formats are identical but require different target architectures,
//! so the methods are generated per format by the macros below, with the `cfg` of each group passed
//! in by the invocation:
//!
//! - `narrow_cvt!`: FP8, FP6 and FP4 (conversions with `f32`, `f16` and `bf16`);
//! - `ue8m0_cvt!`: the `ue8m0` scale format (conversions with `f32` and `bf16`);
//! - `s2f6_cvt!`: `s2f6` (conversions with `f32` and `bf16`, optionally scaled).
//!
//! The exact `From` conversions to `f16`, `bf16` and `f32` (with `RELU` disabled) are implemented
//! alongside the methods they use.

// The helpers are only used by conversions gated on architectures newer than the baseline.
#![allow(dead_code)]

use core::{arch::nvptx as nv, mem::transmute};

use crate::{bf16x2, f16x2};

#[inline(always)]
pub(crate) fn f16x2_to_core(x: f16x2) -> nv::f16x2 {
    // SAFETY: both are plain 32-bit values holding two `f16`s.
    unsafe { transmute(x.to_bits()) }
}

#[inline(always)]
pub(crate) fn f16x2_from_core(x: nv::f16x2) -> f16x2 {
    // SAFETY: see `f16x2_to_core`.
    f16x2::from_bits(unsafe { transmute::<nv::f16x2, u32>(x) })
}

#[inline(always)]
pub(crate) fn bf16x2_to_core(x: bf16x2) -> nv::bf16x2 {
    // SAFETY: both are plain 32-bit values holding two `bf16`s.
    unsafe { transmute(x.to_bits()) }
}

#[inline(always)]
pub(crate) fn bf16x2_from_core(x: nv::bf16x2) -> bf16x2 {
    // SAFETY: see `bf16x2_to_core`.
    bf16x2::from_bits(unsafe { transmute::<nv::bf16x2, u32>(x) })
}

/// Implements the conversions of an FP8, FP6 or FP4 format for its scalar and its vectors of all
/// lengths.
///
/// - `x2`, `x4`: the vector aliases of two and four elements, with the unsigned integer type of
///   the intrinsics returning them.
/// - `basic`: `cfg` of the `f32` and `f16` up-conversions.
/// - `from_f16`: `cfg` of the `f16` down-conversion.
/// - `newer`: `cfg` of the `bf16` down-conversion and of the `bf16` up-conversions.
/// - `stochastic`: `cfg` of the stochastic-rounding conversion.
/// - `rbits`: documentation of the layout of the random bits of the stochastic-rounding conversion.
/// - `core`: the intrinsics of the pair forms, and of the four-element stochastic-rounding form.
///
/// ```ignore
/// narrow_cvt! {
///     "FP8 E4M3",
///     scalar: f8e4m3,
///     x2: f8e4m3x2(u16), x4: f8e4m3x4(u32),
///     basic: target_feature = "sm_89",
///     from_f16: target_feature = "sm_89",
///     newer: any(target_feature = "sm_100f", target_feature = "sm_110f"),
///     stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
///     rbits: "the lower 16 bits are used for ...",
///     core: {
///         from_f32: cvt_e4m3x2_f32,
///         from_f16x2: cvt_e4m3x2_f16x2,
///         from_bf16x2: cvt_e4m3x2_bf16x2,
///         to_f16x2: cvt_f16x2_e4m3x2,
///         to_bf16x2_scaled: cvt_scale_bf16x2_e4m3x2,
///         rs_x4: cvt_rs_e4m3x4_f32,
///     },
/// }
/// ```
macro_rules! narrow_cvt {
    (
        $name:literal,
        scalar: $scalar:ident,
        x2: $x2:ident($b2:ty), x4: $x4:ident($b4:ty),
        basic: $basic:meta,
        from_f16: $from_f16:meta,
        newer: $newer:meta,
        stochastic: $stochastic:meta,
        rbits: $rbits:literal,
        core: {
            from_f32: $c_from_f32:ident,
            from_f16x2: $c_from_f16x2:ident,
            from_bf16x2: $c_from_bf16x2:ident,
            to_f16x2: $c_to_f16x2:ident,
            to_bf16x2_scaled: $c_to_bf16x2_scaled:ident,
            rs_x4: $c_rs:ident $(,)?
        } $(,)?
    ) => {
        narrow_cvt!(@x2 $x2($b2), $basic, $from_f16, $newer,
            $c_from_f32, $c_from_f16x2, $c_from_bf16x2, $c_to_f16x2, $c_to_bf16x2_scaled);
        narrow_cvt!(@x4 $x4($b4), $stochastic, $c_rs);
        narrow_cvt!(@scalar $scalar, $x2, $name, $basic, $from_f16, $newer);
        narrow_cvt!(@vector $scalar, $x2, $x4, $name, $basic, $from_f16, $newer, $stochastic, $rbits);
        narrow_cvt!(@from $scalar, $basic, $newer);
    };

    // Documentation paragraphs shared by the methods.
    (@doc down) => {
        "Out-of-range values, including infinities, are clamped to the largest finite value of \
         matching sign (`.satfinite`). NaN becomes NaN if the format has one (FP8) and the \
         largest positive finite value otherwise (FP4, FP6). `RELU` clamps negative results to \
         zero."
    };
    (@doc scaled) => {
        "`RELU` clamps negative results to zero. With \
         [`Overflow::SaturateFinite`](crate::Overflow::SaturateFinite), results outside the \
         finite range are clamped to the largest finite `bf16` value of matching sign."
    };

    // The pair type: the intrinsics are called directly.
    (
        @x2 $ty:ident($bits:ty), $basic:meta, $from_f16:meta, $newer:meta,
        $c_from_f32:ident, $c_from_f16x2:ident, $c_from_bf16x2:ident, $c_to_f16x2:ident,
        $c_to_bf16x2_scaled:ident
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

            #[cfg($basic)]
            #[inline]
            fn pair_from_f32<const RELU: crate::Relu>(x: [f32; 2]) -> Self {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    Self::from_core(unsafe { ::core::arch::nvptx::$c_from_f32::<R>(x[1], x[0]) })
                })
            }

            #[cfg($from_f16)]
            #[inline]
            fn pair_from_f16<const RELU: crate::Relu>(x: crate::f16x2) -> Self {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    Self::from_core(unsafe {
                        ::core::arch::nvptx::$c_from_f16x2::<R>(crate::data_types::narrow::f16x2_to_core(x))
                    })
                })
            }

            #[cfg($newer)]
            #[inline]
            fn pair_from_bf16<const RELU: crate::Relu>(x: crate::bf16x2) -> Self {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    Self::from_core(unsafe {
                        ::core::arch::nvptx::$c_from_bf16x2::<R>(crate::data_types::narrow::bf16x2_to_core(x))
                    })
                })
            }

            #[cfg($basic)]
            #[inline]
            fn pair_to_f16<const RELU: crate::Relu>(self) -> crate::f16x2 {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    crate::data_types::narrow::f16x2_from_core(unsafe {
                        ::core::arch::nvptx::$c_to_f16x2::<R>(self.to_core())
                    })
                })
            }

            #[cfg($newer)]
            #[inline]
            fn pair_to_bf16<const RELU: crate::Relu>(self) -> crate::bf16x2 {
                self.pair_to_bf16_scaled::<RELU, { crate::Overflow::Infinity }>(crate::f8e8m0x2::ONE)
            }

            #[cfg($newer)]
            #[inline]
            fn pair_to_bf16_scaled<const RELU: crate::Relu, const OVERFLOW: crate::Overflow>(
                self,
                scales: crate::f8e8m0x2,
            ) -> crate::bf16x2 {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    const_bool!(S = matches!(OVERFLOW, crate::Overflow::SaturateFinite) => {
                        crate::data_types::narrow::bf16x2_from_core(unsafe {
                            ::core::arch::nvptx::$c_to_bf16x2_scaled::<R, S>(self.to_core(), scales.to_bits())
                        })
                    })
                })
            }
        }
    };

    // The four-element form of the stochastic rounding.
    (@x4 $ty:ident($bits:ty), $stochastic:meta, $c_rs:ident) => {
        impl $ty {
            #[cfg($stochastic)]
            #[inline]
            #[allow(clippy::unnecessary_cast)]
            fn quad_from_f32_stochastic<const RELU: crate::Relu>(x: [f32; 4], rbits: u32) -> Self {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    Self::from_bits(unsafe {
                        ::core::arch::nvptx::$c_rs::<R>(x[3], x[2], x[1], x[0], rbits)
                    } as $bits)
                })
            }
        }
    };

    // The scalar: the pair conversion with the other lane set to zero.
    (@scalar $ty:ident, $x2:ident, $name:literal, $basic:meta, $from_f16:meta, $newer:meta) => {
        impl $ty {
            #[doc = concat!("Converts an `f32` value to ", $name, ", rounding to nearest even.")]
            ///
            #[doc = narrow_cvt!(@doc down)]
            ///
            #[doc = concat!("Available when `cfg(", stringify!($basic), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($basic)]
            #[inline]
            pub fn from_f32<const RELU: crate::Relu>(x: f32) -> Self {
                $x2::pair_from_f32::<RELU>([x, 0.0]).lo()
            }

            #[doc = concat!("Converts an `f16` value to ", $name, ", rounding to nearest even.")]
            ///
            #[doc = narrow_cvt!(@doc down)]
            ///
            #[doc = concat!("Available when `cfg(", stringify!($from_f16), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($from_f16)]
            #[inline]
            pub fn from_f16<const RELU: crate::Relu>(x: crate::f16) -> Self {
                $x2::pair_from_f16::<RELU>(crate::f16x2::new(x, crate::f16::ZERO)).lo()
            }

            #[doc = concat!("Converts a `bf16` value to ", $name, ", rounding to nearest even.")]
            ///
            #[doc = narrow_cvt!(@doc down)]
            ///
            #[doc = concat!("Available when `cfg(", stringify!($newer), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline]
            pub fn from_bf16<const RELU: crate::Relu>(x: crate::bf16) -> Self {
                $x2::pair_from_bf16::<RELU>(crate::bf16x2::new(x, crate::bf16::ZERO)).lo()
            }

            #[doc = concat!("Converts the ", $name, " value to `f16`.")]
            ///
            /// The conversion is exact; `RELU` clamps a negative result to zero.
            ///
            #[doc = concat!("Available when `cfg(", stringify!($basic), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($basic)]
            #[inline]
            pub fn to_f16<const RELU: crate::Relu>(self) -> crate::f16 {
                $x2::new(self, Self::ZERO).pair_to_f16::<RELU>().lo()
            }

            #[doc = concat!("Converts the ", $name, " value to `f32`.")]
            ///
            /// The conversion is exact; `RELU` clamps a negative result to zero.
            ///
            /// Implemented as [`Self::to_f16`] followed by the exact conversion of `f16` to `f32`.
            ///
            #[doc = concat!("Available when `cfg(", stringify!($basic), ")` holds.")]
            #[cfg($basic)]
            #[inline]
            pub fn to_f32<const RELU: crate::Relu>(self) -> f32 {
                self.to_f16::<RELU>().to_f32()
            }

            #[doc = concat!("Converts the ", $name, " value to `bf16`.")]
            ///
            /// The conversion is exact; `RELU` clamps a negative result to zero.
            ///
            #[doc = concat!("Available when `cfg(", stringify!($newer), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline]
            pub fn to_bf16<const RELU: crate::Relu>(self) -> crate::bf16 {
                $x2::new(self, Self::ZERO).pair_to_bf16::<RELU>().lo()
            }

            #[doc = concat!("Converts the ", $name, " value to `bf16`, multiplying it by the `ue8m0` scale factor `scale`.")]
            ///
            #[doc = narrow_cvt!(@doc scaled)]
            ///
            #[doc = concat!("Available when `cfg(", stringify!($newer), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline]
            pub fn to_bf16_scaled<const RELU: crate::Relu, const OVERFLOW: crate::Overflow>(
                self,
                scale: crate::f8e8m0,
            ) -> crate::bf16 {
                $x2::new(self, Self::ZERO)
                    .pair_to_bf16_scaled::<RELU, OVERFLOW>(crate::f8e8m0x2::new(scale, crate::f8e8m0::ONE))
                    .lo()
            }
        }
    };

    // The vectors of any length: pair-wise conversions, and groups of four for stochastic rounding.
    (
        @vector $scalar:ident, $x2:ident, $x4:ident, $name:literal,
        $basic:meta, $from_f16:meta, $newer:meta, $stochastic:meta, $rbits:literal
    ) => {
        impl<const N: usize> $crate::Vector<$scalar, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[doc = concat!("Converts `f32` values to ", $name, " values, rounding to nearest even.")]
            ///
            /// Element `i` of the result is converted from element `i` of `x`.
            ///
            #[doc = narrow_cvt!(@doc down)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            #[doc = concat!("Available when `cfg(", stringify!($basic), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($basic)]
            #[inline(always)]
            pub fn from_f32<const RELU: crate::Relu>(x: $crate::f32xN<N>) -> Self {
                let x = x.to_array();
                Self::from_x2s($crate::data_types::array_from_fn(|i| {
                    $x2::pair_from_f32::<RELU>([x[2 * i], x[2 * i + 1]])
                }))
            }

            #[doc = concat!("Converts `f16` values to ", $name, " values, rounding to nearest even.")]
            ///
            /// Element `i` of the result is converted from element `i` of `x`.
            ///
            #[doc = narrow_cvt!(@doc down)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            #[doc = concat!("Available when `cfg(", stringify!($from_f16), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($from_f16)]
            #[inline(always)]
            pub fn from_f16<const RELU: crate::Relu>(x: crate::f16xN<N>) -> Self {
                let x = x.to_x2s();
                Self::from_x2s($crate::data_types::array_from_fn(|i| $x2::pair_from_f16::<RELU>(x[i])))
            }

            #[doc = concat!("Converts `bf16` values to ", $name, " values, rounding to nearest even.")]
            ///
            /// Element `i` of the result is converted from element `i` of `x`.
            ///
            #[doc = narrow_cvt!(@doc down)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            #[doc = concat!("Available when `cfg(", stringify!($newer), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline(always)]
            pub fn from_bf16<const RELU: crate::Relu>(x: crate::bf16xN<N>) -> Self {
                let x = x.to_x2s();
                Self::from_x2s($crate::data_types::array_from_fn(|i| $x2::pair_from_bf16::<RELU>(x[i])))
            }

            #[doc = concat!("Converts the ", $name, " values to `f16` values.")]
            ///
            /// Element `i` of the result is converted from element `i` of `self`. The conversion
            /// is exact; `RELU` clamps negative results to zero.
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            #[doc = concat!("Available when `cfg(", stringify!($basic), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($basic)]
            #[inline(always)]
            pub fn to_f16<const RELU: crate::Relu>(self) -> crate::f16xN<N> {
                let x = self.to_x2s();
                crate::f16xN::<N>::from_x2s($crate::data_types::array_from_fn(|i| x[i].pair_to_f16::<RELU>()))
            }

            #[doc = concat!("Converts the ", $name, " values to `f32` values.")]
            ///
            /// Element `i` of the result is converted from element `i` of `self`. The conversion
            /// is exact; `RELU` clamps negative results to zero.
            ///
            /// Implemented as [`Self::to_f16`] followed by the exact conversion of `f16` to `f32`.
            ///
            #[doc = concat!("Available when `cfg(", stringify!($basic), ")` holds.")]
            #[cfg($basic)]
            #[inline(always)]
            pub fn to_f32<const RELU: crate::Relu>(self) -> $crate::f32xN<N> {
                self.to_f16::<RELU>().to_f32()
            }

            #[doc = concat!("Converts the ", $name, " values to `bf16` values.")]
            ///
            /// Element `i` of the result is converted from element `i` of `self`. The conversion
            /// is exact; `RELU` clamps negative results to zero.
            ///
            /// Each pair of elements is converted with a single packed instruction, using unit
            /// scale factors.
            ///
            #[doc = concat!("Available when `cfg(", stringify!($newer), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline(always)]
            pub fn to_bf16<const RELU: crate::Relu>(self) -> crate::bf16xN<N> {
                let x = self.to_x2s();
                crate::bf16xN::<N>::from_x2s($crate::data_types::array_from_fn(|i| x[i].pair_to_bf16::<RELU>()))
            }

            #[doc = concat!("Converts the ", $name, " values to `bf16` values, multiplying each element by its `ue8m0` scale factor.")]
            ///
            /// Element `i` of the result is converted from element `i` of `self` and multiplied by
            /// element `i` of `scales` (`.scaled::n2::ue8m0`).
            ///
            #[doc = narrow_cvt!(@doc scaled)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            #[doc = concat!("Available when `cfg(", stringify!($newer), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[cfg($newer)]
            #[inline(always)]
            pub fn to_bf16_scaled<const RELU: crate::Relu, const OVERFLOW: crate::Overflow>(
                self,
                scales: crate::f8e8m0xN<N>,
            ) -> crate::bf16xN<N> {
                let x = self.to_x2s();
                let scales = scales.to_x2s();
                crate::bf16xN::<N>::from_x2s($crate::data_types::array_from_fn(|i| {
                    x[i].pair_to_bf16_scaled::<RELU, OVERFLOW>(scales[i])
                }))
            }
        }

        // Stochastic rounding converts groups of four values with a single instruction.
        #[cfg($stochastic)]
        impl<const N: usize> $crate::Vector<$scalar, N>
        where
            $crate::VectorLen<N>: $crate::QuadVectorLen,
        {
            #[doc = concat!("Converts `f32` values to ", $name, " values with stochastic rounding (`.rs`).")]
            ///
            /// Element `i` of the result is converted from element `i` of `x`. Each value is
            /// rounded towards or away from zero depending on the carry out of adding its random
            /// bits to the discarded mantissa bits. The elements `4 * g` to `4 * g + 3` are
            /// converted together using the random bits `rbits[g]`; within that group, the
            /// elements are numbered from 0:
            #[doc = $rbits]
            ///
            #[doc = narrow_cvt!(@doc down)]
            ///
            #[doc = concat!("Available when `cfg(", stringify!($stochastic), ")` holds.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn from_f32_stochastic<const RELU: crate::Relu>(
                x: $crate::f32xN<N>,
                rbits: $crate::Quads<u32, N>,
            ) -> Self {
                let x = x.to_array();
                let quads: $crate::Quads<$x4, N> = $crate::data_types::array_from_fn(|i| {
                    $x4::quad_from_f32_stochastic::<RELU>(
                        [x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]],
                        rbits[i],
                    )
                });
                // SAFETY: both arrays hold the pairs of the result, in order.
                Self::from_pair_bits(unsafe { $crate::data_types::vector::cast(quads) })
            }
        }
    };

    // Exact widening conversions, without `RELU`.
    (@from $scalar:ident, $basic:meta, $newer:meta) => {
        #[cfg($basic)]
        impl ::core::convert::From<$scalar> for crate::f16 {
            #[inline(always)]
            fn from(x: $scalar) -> Self {
                x.to_f16::<{ crate::Relu::Disabled }>()
            }
        }

        #[cfg($newer)]
        impl ::core::convert::From<$scalar> for crate::bf16 {
            #[inline(always)]
            fn from(x: $scalar) -> Self {
                x.to_bf16::<{ crate::Relu::Disabled }>()
            }
        }

        #[cfg($basic)]
        impl ::core::convert::From<$scalar> for f32 {
            #[inline(always)]
            fn from(x: $scalar) -> Self {
                x.to_f32::<{ crate::Relu::Disabled }>()
            }
        }

        #[cfg($basic)]
        impl<const N: usize> ::core::convert::From<$crate::Vector<$scalar, N>> for crate::f16xN<N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn from(x: $crate::Vector<$scalar, N>) -> Self {
                x.to_f16::<{ crate::Relu::Disabled }>()
            }
        }

        #[cfg($newer)]
        impl<const N: usize> ::core::convert::From<$crate::Vector<$scalar, N>> for crate::bf16xN<N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn from(x: $crate::Vector<$scalar, N>) -> Self {
                x.to_bf16::<{ crate::Relu::Disabled }>()
            }
        }

        #[cfg($basic)]
        impl<const N: usize> ::core::convert::From<$crate::Vector<$scalar, N>>
            for $crate::f32xN<N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn from(x: $crate::Vector<$scalar, N>) -> Self {
                x.to_f32::<{ crate::Relu::Disabled }>()
            }
        }
    };
}

/// Implements the conversions of the `ue8m0` scale format for its scalar and its vectors of all
/// lengths, available when `cfg` holds.
macro_rules! ue8m0_cvt {
    (
        scalar: $scalar:ident,
        x2: $x2:ident,
        cfg: $cfg:meta $(,)?
    ) => {
        ue8m0_cvt!(@x2 $x2, $cfg);
        ue8m0_cvt!(@scalar $scalar, $x2, $cfg);
        ue8m0_cvt!(@vector $scalar, $x2, $cfg);
        ue8m0_cvt!(@from $scalar, $cfg);
    };

    (@doc down) => {
        "Since the format only holds powers of two, `RND` selects rounding to a power of two \
         toward zero ([`Rounding::TowardZero`](crate::Rounding::TowardZero)) or toward positive \
         infinity ([`Rounding::Up`](crate::Rounding::Up)); no other mode is supported. NaN \
         becomes NaN. With [`Overflow::SaturateFinite`](crate::Overflow::SaturateFinite), \
         out-of-range values are clamped to the largest finite value `2^127` (`.satfinite`, always \
         positive as the format is unsigned); without it, the PTX ISA does not define the result \
         of an overflow as the format has no infinity."
    };

    // The pair type: the intrinsics are called directly.
    (@x2 $ty:ident, $cfg:meta) => {
        #[cfg($cfg)]
        impl $ty {
            #[inline]
            fn pair_from_f32<const RND: crate::Rounding, const OVERFLOW: crate::Overflow>(
                x: [f32; 2],
            ) -> Self {
                const {
                    assert!(
                        matches!(RND, crate::Rounding::TowardZero | crate::Rounding::Up),
                        "conversion to ue8m0 only supports the TowardZero and Up rounding modes"
                    )
                }
                const_map!(RND => R: ::core::arch::nvptx::Rounding {
                    crate::Rounding::TowardZero => ::core::arch::nvptx::Rounding::Rz,
                    crate::Rounding::Up => ::core::arch::nvptx::Rounding::Rp,
                } in const_bool!(S = matches!(OVERFLOW, crate::Overflow::SaturateFinite) => {
                    Self::from_bits(unsafe {
                        ::core::arch::nvptx::cvt_ue8m0x2_f32::<R, S>(x[1], x[0])
                    })
                }))
            }

            #[inline]
            fn pair_from_bf16<const RND: crate::Rounding, const OVERFLOW: crate::Overflow>(
                x: crate::bf16x2,
            ) -> Self {
                const {
                    assert!(
                        matches!(RND, crate::Rounding::TowardZero | crate::Rounding::Up),
                        "conversion to ue8m0 only supports the TowardZero and Up rounding modes"
                    )
                }
                const_map!(RND => R: ::core::arch::nvptx::Rounding {
                    crate::Rounding::TowardZero => ::core::arch::nvptx::Rounding::Rz,
                    crate::Rounding::Up => ::core::arch::nvptx::Rounding::Rp,
                } in const_bool!(S = matches!(OVERFLOW, crate::Overflow::SaturateFinite) => {
                    Self::from_bits(unsafe {
                        ::core::arch::nvptx::cvt_ue8m0x2_bf16x2::<R, S>(
                            crate::data_types::narrow::bf16x2_to_core(x),
                        )
                    })
                }))
            }

            #[inline]
            fn pair_to_bf16(self) -> crate::bf16x2 {
                crate::data_types::narrow::bf16x2_from_core(unsafe {
                    ::core::arch::nvptx::cvt_bf16x2_ue8m0x2(self.to_bits())
                })
            }
        }
    };

    // The scalar: the pair conversion with the other lane set to a neutral value.
    (@scalar $ty:ident, $x2:ident, $cfg:meta) => {
        #[cfg($cfg)]
        impl $ty {
            /// Converts an `f32` value to a `ue8m0` scale factor.
            ///
            #[doc = ue8m0_cvt!(@doc down)]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn from_f32<const RND: crate::Rounding, const OVERFLOW: crate::Overflow>(
                x: f32,
            ) -> Self {
                $x2::pair_from_f32::<RND, OVERFLOW>([x, 0.0]).lo()
            }

            /// Converts a `bf16` value to a `ue8m0` scale factor.
            ///
            #[doc = ue8m0_cvt!(@doc down)]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn from_bf16<const RND: crate::Rounding, const OVERFLOW: crate::Overflow>(
                x: crate::bf16,
            ) -> Self {
                $x2::pair_from_bf16::<RND, OVERFLOW>(crate::bf16x2::new(x, crate::bf16::ONE)).lo()
            }

            /// Converts the `ue8m0` scale factor to `bf16`.
            ///
            /// The conversion is exact; `0x00` becomes `2^-127` (a `bf16` subnormal) and `0xff`
            /// becomes NaN.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn to_bf16(self) -> crate::bf16 {
                $x2::new(self, Self::ONE).pair_to_bf16().lo()
            }

            /// Converts the `ue8m0` scale factor to `f32`.
            ///
            /// The conversion is exact; `0x00` becomes `2^-127` (an `f32` subnormal) and `0xff`
            /// becomes NaN.
            ///
            /// Implemented as [`Self::to_bf16`] followed by the exact conversion of `bf16` to
            /// `f32`.
            #[inline]
            pub fn to_f32(self) -> f32 {
                self.to_bf16().to_f32()
            }
        }
    };

    // The vectors of any length: pair-wise conversions.
    (@vector $scalar:ident, $x2:ident, $cfg:meta) => {
        #[cfg($cfg)]
        impl<const N: usize> $crate::Vector<$scalar, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            /// Converts `f32` values to `ue8m0` scale factors.
            ///
            /// Element `i` of the result is converted from element `i` of `x`.
            ///
            #[doc = ue8m0_cvt!(@doc down)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn from_f32<const RND: crate::Rounding, const OVERFLOW: crate::Overflow>(
                x: $crate::f32xN<N>,
            ) -> Self {
                let x = x.to_array();
                Self::from_x2s($crate::data_types::array_from_fn(|i| {
                    $x2::pair_from_f32::<RND, OVERFLOW>([x[2 * i], x[2 * i + 1]])
                }))
            }

            /// Converts `bf16` values to `ue8m0` scale factors.
            ///
            /// Element `i` of the result is converted from element `i` of `x`.
            ///
            #[doc = ue8m0_cvt!(@doc down)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn from_bf16<const RND: crate::Rounding, const OVERFLOW: crate::Overflow>(
                x: crate::bf16xN<N>,
            ) -> Self {
                let x = x.to_x2s();
                Self::from_x2s($crate::data_types::array_from_fn(|i| {
                    $x2::pair_from_bf16::<RND, OVERFLOW>(x[i])
                }))
            }

            /// Converts the `ue8m0` scale factors to `bf16` values.
            ///
            /// Element `i` of the result is converted from element `i` of `self`. The conversion
            /// is exact; `0x00` becomes `2^-127` (a `bf16` subnormal) and `0xff` becomes NaN.
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn to_bf16(self) -> crate::bf16xN<N> {
                let x = self.to_x2s();
                crate::bf16xN::<N>::from_x2s($crate::data_types::array_from_fn(|i| x[i].pair_to_bf16()))
            }

            /// Converts the `ue8m0` scale factors to `f32` values.
            ///
            /// Element `i` of the result is converted from element `i` of `self`. The conversion
            /// is exact; `0x00` becomes `2^-127` (an `f32` subnormal) and `0xff` becomes NaN.
            ///
            /// Implemented as [`Self::to_bf16`] followed by the exact conversion of `bf16` to
            /// `f32`.
            #[inline(always)]
            pub fn to_f32(self) -> $crate::f32xN<N> {
                self.to_bf16().to_f32()
            }
        }
    };

    // Exact widening conversions.
    (@from $scalar:ident, $cfg:meta) => {
        #[cfg($cfg)]
        impl ::core::convert::From<$scalar> for crate::bf16 {
            #[inline(always)]
            fn from(x: $scalar) -> Self {
                x.to_bf16()
            }
        }

        #[cfg($cfg)]
        impl ::core::convert::From<$scalar> for f32 {
            #[inline(always)]
            fn from(x: $scalar) -> Self {
                x.to_f32()
            }
        }

        #[cfg($cfg)]
        impl<const N: usize> ::core::convert::From<$crate::Vector<$scalar, N>> for crate::bf16xN<N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn from(x: $crate::Vector<$scalar, N>) -> Self {
                x.to_bf16()
            }
        }

        #[cfg($cfg)]
        impl<const N: usize> ::core::convert::From<$crate::Vector<$scalar, N>>
            for $crate::f32xN<N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn from(x: $crate::Vector<$scalar, N>) -> Self {
                x.to_f32()
            }
        }
    };
}

/// Implements the conversions of `s2f6` for its scalar and its vectors of all lengths, available
/// when `cfg` holds.
///
/// The conversions to `s2f6` are not exposed by `core::arch::nvptx`, so they are inline PTX.
macro_rules! s2f6_cvt {
    (
        scalar: $scalar:ident,
        x2: $x2:ident,
        cfg: $cfg:meta $(,)?
    ) => {
        s2f6_cvt!(@x2 $x2, $cfg);
        s2f6_cvt!(@scalar $scalar, $x2, $cfg);
        s2f6_cvt!(@vector $scalar, $x2, $cfg);
        s2f6_cvt!(@from $scalar, $cfg);
    };

    (@doc down) => {
        "Rounds to nearest even. Out-of-range values, including infinities, are clamped to the \
         largest value of matching sign (`.satfinite`), and NaN becomes the largest positive \
         value as the format has neither. `RELU` clamps negative results to zero."
    };
    (@doc scaled_up) => {
        "`RELU` clamps negative results to zero. With \
         [`Overflow::SaturateFinite`](crate::Overflow::SaturateFinite), results outside the \
         finite range are clamped to the largest finite `bf16` value of matching sign."
    };

    // The pair type: the instructions are emitted directly.
    (@x2 $ty:ident, $cfg:meta) => {
        #[cfg($cfg)]
        impl $ty {
            #[inline]
            fn pair_from_f32<const RELU: crate::Relu>(x: [f32; 2]) -> Self {
                macro_rules! emit {
                    ($relu:literal) => {{
                        let d: u16;
                        // SAFETY: `cvt` only reads and writes registers.
                        unsafe {
                            ::core::arch::asm!(
                                concat!("cvt.rn.satfinite", $relu, ".s2f6x2.f32 {d}, {a}, {b};"),
                                d = out(reg16) d,
                                a = in(reg32) x[1],
                                b = in(reg32) x[0],
                                options(pure, nomem, nostack),
                            );
                        }
                        Self::from_bits(d)
                    }};
                }
                asm_variants!(emit;
                    RELU => { crate::Relu::Disabled => "", crate::Relu::Enabled => ".relu" },
                )
            }

            #[inline]
            fn pair_from_f32_scaled<const RELU: crate::Relu>(
                x: [f32; 2],
                scales: crate::f8e8m0x2,
            ) -> Self {
                macro_rules! emit {
                    ($relu:literal) => {{
                        let d: u16;
                        // SAFETY: `cvt` only reads and writes registers.
                        unsafe {
                            ::core::arch::asm!(
                                concat!(
                                    "cvt.rn.satfinite", $relu,
                                    ".scaled::n2::ue8m0.s2f6x2.f32 {d}, {a}, {b}, {s};"
                                ),
                                d = out(reg16) d,
                                a = in(reg32) x[1],
                                b = in(reg32) x[0],
                                s = in(reg16) scales.to_bits(),
                                options(pure, nomem, nostack),
                            );
                        }
                        Self::from_bits(d)
                    }};
                }
                asm_variants!(emit;
                    RELU => { crate::Relu::Disabled => "", crate::Relu::Enabled => ".relu" },
                )
            }

            #[inline]
            fn pair_from_bf16<const RELU: crate::Relu>(x: crate::bf16x2) -> Self {
                macro_rules! emit {
                    ($relu:literal) => {{
                        let d: u16;
                        // SAFETY: `cvt` only reads and writes registers.
                        unsafe {
                            ::core::arch::asm!(
                                concat!("cvt.rn.satfinite", $relu, ".s2f6x2.bf16x2 {d}, {a};"),
                                d = out(reg16) d,
                                a = in(reg32) x.to_bits(),
                                options(pure, nomem, nostack),
                            );
                        }
                        Self::from_bits(d)
                    }};
                }
                asm_variants!(emit;
                    RELU => { crate::Relu::Disabled => "", crate::Relu::Enabled => ".relu" },
                )
            }

            #[inline]
            fn pair_from_bf16_scaled<const RELU: crate::Relu>(
                x: crate::bf16x2,
                scales: crate::f8e8m0x2,
            ) -> Self {
                macro_rules! emit {
                    ($relu:literal) => {{
                        let d: u16;
                        // SAFETY: `cvt` only reads and writes registers.
                        unsafe {
                            ::core::arch::asm!(
                                concat!(
                                    "cvt.rn.satfinite", $relu,
                                    ".scaled::n2::ue8m0.s2f6x2.bf16x2 {d}, {a}, {s};"
                                ),
                                d = out(reg16) d,
                                a = in(reg32) x.to_bits(),
                                s = in(reg16) scales.to_bits(),
                                options(pure, nomem, nostack),
                            );
                        }
                        Self::from_bits(d)
                    }};
                }
                asm_variants!(emit;
                    RELU => { crate::Relu::Disabled => "", crate::Relu::Enabled => ".relu" },
                )
            }

            #[inline]
            fn pair_to_bf16<const RELU: crate::Relu>(self) -> crate::bf16x2 {
                self.pair_to_bf16_scaled::<RELU, { crate::Overflow::Infinity }>(crate::f8e8m0x2::ONE)
            }

            #[inline]
            fn pair_to_bf16_scaled<const RELU: crate::Relu, const OVERFLOW: crate::Overflow>(
                self,
                scales: crate::f8e8m0x2,
            ) -> crate::bf16x2 {
                const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
                    const_bool!(S = matches!(OVERFLOW, crate::Overflow::SaturateFinite) => {
                        crate::data_types::narrow::bf16x2_from_core(unsafe {
                            ::core::arch::nvptx::cvt_scale_bf16x2_s2f6x2::<R, S>(
                                self.to_bits(),
                                scales.to_bits(),
                            )
                        })
                    })
                })
            }
        }
    };

    // The scalar: the pair conversion with the other lane set to zero.
    (@scalar $ty:ident, $x2:ident, $cfg:meta) => {
        #[cfg($cfg)]
        impl $ty {
            /// Converts an `f32` value to `s2f6`.
            ///
            #[doc = s2f6_cvt!(@doc down)]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn from_f32<const RELU: crate::Relu>(x: f32) -> Self {
                $x2::pair_from_f32::<RELU>([x, 0.0]).lo()
            }

            /// Converts an `f32` value to `s2f6`, dividing it by the `ue8m0` scale factor `scale`
            /// first (`.scaled::n2::ue8m0`).
            ///
            #[doc = s2f6_cvt!(@doc down)]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn from_f32_scaled<const RELU: crate::Relu>(x: f32, scale: crate::f8e8m0) -> Self {
                $x2::pair_from_f32_scaled::<RELU>(
                    [x, 0.0],
                    crate::f8e8m0x2::new(scale, crate::f8e8m0::ONE),
                )
                .lo()
            }

            /// Converts a `bf16` value to `s2f6`.
            ///
            #[doc = s2f6_cvt!(@doc down)]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn from_bf16<const RELU: crate::Relu>(x: crate::bf16) -> Self {
                $x2::pair_from_bf16::<RELU>(crate::bf16x2::new(x, crate::bf16::ZERO)).lo()
            }

            /// Converts a `bf16` value to `s2f6`, dividing it by the `ue8m0` scale factor `scale`
            /// first (`.scaled::n2::ue8m0`).
            ///
            #[doc = s2f6_cvt!(@doc down)]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn from_bf16_scaled<const RELU: crate::Relu>(
                x: crate::bf16,
                scale: crate::f8e8m0,
            ) -> Self {
                $x2::pair_from_bf16_scaled::<RELU>(
                    crate::bf16x2::new(x, crate::bf16::ZERO),
                    crate::f8e8m0x2::new(scale, crate::f8e8m0::ONE),
                )
                .lo()
            }

            /// Converts the `s2f6` value to `bf16`.
            ///
            /// The conversion is exact; `RELU` clamps a negative result to zero.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn to_bf16<const RELU: crate::Relu>(self) -> crate::bf16 {
                $x2::new(self, Self::ZERO).pair_to_bf16::<RELU>().lo()
            }

            /// Converts the `s2f6` value to `bf16`, multiplying it by the `ue8m0` scale factor
            /// `scale` (`.scaled::n2::ue8m0`).
            ///
            #[doc = s2f6_cvt!(@doc scaled_up)]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn to_bf16_scaled<const RELU: crate::Relu, const OVERFLOW: crate::Overflow>(
                self,
                scale: crate::f8e8m0,
            ) -> crate::bf16 {
                $x2::new(self, Self::ZERO)
                    .pair_to_bf16_scaled::<RELU, OVERFLOW>(crate::f8e8m0x2::new(scale, crate::f8e8m0::ONE))
                    .lo()
            }

            /// Converts the `s2f6` value to `f32`.
            ///
            /// The conversion is exact; `RELU` clamps a negative result to zero.
            ///
            /// Implemented as [`Self::to_bf16`] followed by the exact conversion of `bf16` to
            /// `f32`.
            #[inline]
            pub fn to_f32<const RELU: crate::Relu>(self) -> f32 {
                self.to_bf16::<RELU>().to_f32()
            }
        }
    };

    // The vectors of any length: pair-wise conversions.
    (@vector $scalar:ident, $x2:ident, $cfg:meta) => {
        #[cfg($cfg)]
        impl<const N: usize> $crate::Vector<$scalar, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            /// Converts `f32` values to `s2f6` values.
            ///
            /// Element `i` of the result is converted from element `i` of `x`.
            ///
            #[doc = s2f6_cvt!(@doc down)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn from_f32<const RELU: crate::Relu>(x: $crate::f32xN<N>) -> Self {
                let x = x.to_array();
                Self::from_x2s($crate::data_types::array_from_fn(|i| {
                    $x2::pair_from_f32::<RELU>([x[2 * i], x[2 * i + 1]])
                }))
            }

            /// Converts `f32` values to `s2f6` values, dividing each by its `ue8m0` scale factor
            /// first.
            ///
            /// Element `i` of the result is converted from element `i` of `x`, divided by element
            /// `i` of `scales` (`.scaled::n2::ue8m0`).
            ///
            #[doc = s2f6_cvt!(@doc down)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn from_f32_scaled<const RELU: crate::Relu>(
                x: $crate::f32xN<N>,
                scales: crate::f8e8m0xN<N>,
            ) -> Self {
                let x = x.to_array();
                let scales = scales.to_x2s();
                Self::from_x2s($crate::data_types::array_from_fn(|i| {
                    $x2::pair_from_f32_scaled::<RELU>([x[2 * i], x[2 * i + 1]], scales[i])
                }))
            }

            /// Converts `bf16` values to `s2f6` values.
            ///
            /// Element `i` of the result is converted from element `i` of `x`.
            ///
            #[doc = s2f6_cvt!(@doc down)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn from_bf16<const RELU: crate::Relu>(x: crate::bf16xN<N>) -> Self {
                let x = x.to_x2s();
                Self::from_x2s($crate::data_types::array_from_fn(|i| $x2::pair_from_bf16::<RELU>(x[i])))
            }

            /// Converts `bf16` values to `s2f6` values, dividing each by its `ue8m0` scale factor
            /// first.
            ///
            /// Element `i` of the result is converted from element `i` of `x`, divided by element
            /// `i` of `scales` (`.scaled::n2::ue8m0`).
            ///
            #[doc = s2f6_cvt!(@doc down)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn from_bf16_scaled<const RELU: crate::Relu>(
                x: crate::bf16xN<N>,
                scales: crate::f8e8m0xN<N>,
            ) -> Self {
                let x = x.to_x2s();
                let scales = scales.to_x2s();
                Self::from_x2s($crate::data_types::array_from_fn(|i| {
                    $x2::pair_from_bf16_scaled::<RELU>(x[i], scales[i])
                }))
            }

            /// Converts the `s2f6` values to `bf16` values.
            ///
            /// Element `i` of the result is converted from element `i` of `self`. The conversion
            /// is exact; `RELU` clamps negative results to zero.
            ///
            /// Each pair of elements is converted with a single packed instruction, using unit
            /// scale factors.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn to_bf16<const RELU: crate::Relu>(self) -> crate::bf16xN<N> {
                let x = self.to_x2s();
                crate::bf16xN::<N>::from_x2s($crate::data_types::array_from_fn(|i| x[i].pair_to_bf16::<RELU>()))
            }

            /// Converts the `s2f6` values to `bf16` values, multiplying each element by its
            /// `ue8m0` scale factor.
            ///
            /// Element `i` of the result is converted from element `i` of `self` and multiplied by
            /// element `i` of `scales` (`.scaled::n2::ue8m0`).
            ///
            #[doc = s2f6_cvt!(@doc scaled_up)]
            ///
            /// Each pair of elements is converted with a single packed instruction.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn to_bf16_scaled<const RELU: crate::Relu, const OVERFLOW: crate::Overflow>(
                self,
                scales: crate::f8e8m0xN<N>,
            ) -> crate::bf16xN<N> {
                let x = self.to_x2s();
                let scales = scales.to_x2s();
                crate::bf16xN::<N>::from_x2s($crate::data_types::array_from_fn(|i| {
                    x[i].pair_to_bf16_scaled::<RELU, OVERFLOW>(scales[i])
                }))
            }

            /// Converts the `s2f6` values to `f32` values.
            ///
            /// Element `i` of the result is converted from element `i` of `self`. The conversion
            /// is exact; `RELU` clamps negative results to zero.
            ///
            /// Implemented as [`Self::to_bf16`] followed by the exact conversion of `bf16` to
            /// `f32`.
            #[inline(always)]
            pub fn to_f32<const RELU: crate::Relu>(self) -> $crate::f32xN<N> {
                self.to_bf16::<RELU>().to_f32()
            }
        }
    };

    // Exact widening conversions, without `RELU`.
    (@from $scalar:ident, $cfg:meta) => {
        #[cfg($cfg)]
        impl ::core::convert::From<$scalar> for crate::bf16 {
            #[inline(always)]
            fn from(x: $scalar) -> Self {
                x.to_bf16::<{ crate::Relu::Disabled }>()
            }
        }

        #[cfg($cfg)]
        impl ::core::convert::From<$scalar> for f32 {
            #[inline(always)]
            fn from(x: $scalar) -> Self {
                x.to_f32::<{ crate::Relu::Disabled }>()
            }
        }

        #[cfg($cfg)]
        impl<const N: usize> ::core::convert::From<$crate::Vector<$scalar, N>> for crate::bf16xN<N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn from(x: $crate::Vector<$scalar, N>) -> Self {
                x.to_bf16::<{ crate::Relu::Disabled }>()
            }
        }

        #[cfg($cfg)]
        impl<const N: usize> ::core::convert::From<$crate::Vector<$scalar, N>>
            for $crate::f32xN<N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn from(x: $crate::Vector<$scalar, N>) -> Self {
                x.to_f32::<{ crate::Relu::Disabled }>()
            }
        }
    };
}

pub(crate) use {narrow_cvt, s2f6_cvt, ue8m0_cvt};
