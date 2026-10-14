// --- LLM-generated --- //
//! 8-bit floating-point formats: FP8 (`e4m3`, `e5m2`) and the `ue8m0` scale format, as scalars and
//! packed vectors of 2, 4, 8, 16, 32 and 64 elements.
//!
//! In all packed types, element 0 occupies the least significant bits.

use super::vector::{Specials, minifloat_to_f32, narrow_scalar, vector_element};

// Only used by the conversions, which are gated on newer architectures.
#[allow(unused_imports)]
use super::narrow::{narrow_cvt, ue8m0_cvt};

narrow_scalar! {
    /// An FP8 E4M3 value (1 sign, 4 exponent and 3 mantissa bits; exponent bias 7).
    ///
    /// The format has no infinities and a single NaN encoding (`0x7f`/`0xff`); the largest finite
    /// magnitude is 448.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
    f8e4m3, width: 8,
    decode: |bits| minifloat_to_f32(bits, 4, 3, 7, Specials::NanOnly),
    consts: {
        /// Positive zero.
        ZERO = 0x00;
        /// Negative zero.
        NEG_ZERO = 0x80;
        /// One.
        ONE = 0x38;
        /// Negative one.
        NEG_ONE = 0xB8;
        /// The NaN encoding `0x7f`.
        NAN = 0x7F;
        /// The largest finite value (448).
        MAX = 0x7E;
        /// The smallest finite value (-448).
        MIN = 0xFE;
        /// The smallest positive normal value (2^-6).
        MIN_POSITIVE = 0x08;
        /// The smallest positive subnormal value (2^-9).
        MIN_POSITIVE_SUBNORMAL = 0x01;
    }
}

vector_element! {
    elem: f8e4m3, pair: u16, packing: Bytes,
    consts: [ZERO, NEG_ZERO, ONE, NEG_ONE, NAN, MAX, MIN, MIN_POSITIVE, MIN_POSITIVE_SUBNORMAL],
    note: "See [`f8e4m3`](crate::f8e4m3) for the format.",
    aliases: f8e4m3xN, f8e4m3x2, f8e4m3x4, f8e4m3x8, f8e4m3x16, f8e4m3x32, f8e4m3x64,
}

narrow_cvt! {
    "FP8 E4M3",
    scalar: f8e4m3,
    x2: f8e4m3x2(u16), x4: f8e4m3x4(u32),
    basic: target_feature = "sm_89",
    from_f16: target_feature = "sm_89",
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
    rbits: "the lower 16 bits are used for elements 0 and 1 and the upper 16 bits for elements 2 and 3.",
    core: {
        from_f32: cvt_e4m3x2_f32,
        from_f16x2: cvt_e4m3x2_f16x2,
        from_bf16x2: cvt_e4m3x2_bf16x2,
        to_f16x2: cvt_f16x2_e4m3x2,
        to_bf16x2_scaled: cvt_scale_bf16x2_e4m3x2,
        rs_x4: cvt_rs_e4m3x4_f32,
    },
}

narrow_scalar! {
    /// An FP8 E5M2 value (1 sign, 5 exponent and 2 mantissa bits; exponent bias 15).
    ///
    /// The format follows IEEE 754 conventions, with the same encodings of infinity
    /// (`0x7c`/`0xfc`) and NaN as the upper byte of an IEEE `binary16` value; the largest finite
    /// magnitude is 57344.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
    f8e5m2, width: 8,
    decode: |bits| minifloat_to_f32(bits, 5, 2, 15, Specials::Ieee),
    consts: {
        /// Positive zero.
        ZERO = 0x00;
        /// Negative zero.
        NEG_ZERO = 0x80;
        /// One.
        ONE = 0x3C;
        /// Negative one.
        NEG_ONE = 0xBC;
        /// Positive infinity.
        INFINITY = 0x7C;
        /// Negative infinity.
        NEG_INFINITY = 0xFC;
        /// A quiet NaN.
        NAN = 0x7E;
        /// The largest finite value (57344).
        MAX = 0x7B;
        /// The smallest finite value (-57344).
        MIN = 0xFB;
        /// The smallest positive normal value (2^-14).
        MIN_POSITIVE = 0x04;
        /// The smallest positive subnormal value (2^-16).
        MIN_POSITIVE_SUBNORMAL = 0x01;
    }
}

vector_element! {
    elem: f8e5m2, pair: u16, packing: Bytes,
    consts: [
        ZERO, NEG_ZERO, ONE, NEG_ONE, INFINITY, NEG_INFINITY, NAN, MAX, MIN, MIN_POSITIVE,
        MIN_POSITIVE_SUBNORMAL
    ],
    note: "See [`f8e5m2`](crate::f8e5m2) for the format.",
    aliases: f8e5m2xN, f8e5m2x2, f8e5m2x4, f8e5m2x8, f8e5m2x16, f8e5m2x32, f8e5m2x64,
}

narrow_cvt! {
    "FP8 E5M2",
    scalar: f8e5m2,
    x2: f8e5m2x2(u16), x4: f8e5m2x4(u32),
    basic: target_feature = "sm_89",
    from_f16: target_feature = "sm_89",
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
    rbits: "the lower 16 bits are used for elements 0 and 1 and the upper 16 bits for elements 2 and 3.",
    core: {
        from_f32: cvt_e5m2x2_f32,
        from_f16x2: cvt_e5m2x2_f16x2,
        from_bf16x2: cvt_e5m2x2_bf16x2,
        to_f16x2: cvt_f16x2_e5m2x2,
        to_bf16x2_scaled: cvt_scale_bf16x2_e5m2x2,
        rs_x4: cvt_rs_e5m2x4_f32,
    },
}

narrow_scalar! {
    /// An unsigned E8M0 (`ue8m0`) scale factor.
    ///
    /// The value has 8 exponent bits and no sign or mantissa bits, so it represents the power of
    /// two `2^(bits - 127)` (from `2^-127` to `2^127`); `0xff` is NaN. The format has no infinity,
    /// zero or negative values, so the `Default` (all-zero bits) is `2^-127`. These are the scale
    /// factors of the block-scaled narrow formats and of the `to_bf16_scaled` conversions.
    ///
    /// The conversions to and from other formats require `sm_100f`, `sm_101f`, `sm_110f` or
    /// `sm_120f`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
    f8e8m0, width: 8,
    decode: |bits| match bits {
        0xFF => f32::NAN,
        0 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits((bits as u32) << 23),
    },
    consts: {
        /// One (`2^0`, encoded as `0x7f`).
        ONE = 0x7F;
        /// The NaN encoding `0xff`.
        NAN = 0xFF;
        /// The largest finite value (`2^127`).
        MAX = 0xFE;
        /// The smallest value (`2^-127`).
        MIN = 0x00;
    }
}

vector_element! {
    elem: f8e8m0, pair: u16, packing: Bytes,
    consts: [ONE, NAN, MAX, MIN],
    note: "See [`f8e8m0`](crate::f8e8m0) for the format.",
    aliases: f8e8m0xN, f8e8m0x2, f8e8m0x4, f8e8m0x8, f8e8m0x16, f8e8m0x32, f8e8m0x64,
}

ue8m0_cvt! {
    scalar: f8e8m0,
    x2: f8e8m0x2,
    cfg: any(
        target_feature = "sm_100f",
        target_feature = "sm_101f",
        target_feature = "sm_110f",
        target_feature = "sm_120f"
    ),
}
