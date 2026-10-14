// --- LLM-generated --- //
//! 6-bit floating-point formats (`e2m3`, `e3m2`), as scalars and packed vectors of 2, 4, 8, 16, 32
//! and 64 elements.
//!
//! Each 6-bit value is stored in the low 6 bits of a byte; the two most significant bits of every
//! byte are zero.
//!
//! In all packed types, element 0 occupies the least significant bits.

use super::vector::{Specials, minifloat_to_f32, narrow_scalar, vector_element};

// Only used by the conversions, which are gated on newer architectures.
#[allow(unused_imports)]
use super::narrow::narrow_cvt;

narrow_scalar! {
    /// An FP6 E2M3 value (1 sign, 2 exponent and 3 mantissa bits; exponent bias 1).
    ///
    /// The format has no infinities or NaNs; the largest finite magnitude is 7.5.
    ///
    /// The value is stored in the low 6 bits of a byte; the two most significant bits are zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
    f6e2m3, width: 6,
    decode: |bits| minifloat_to_f32(bits, 2, 3, 1, Specials::None),
    consts: {
        /// Positive zero.
        ZERO = 0x00;
        /// Negative zero.
        NEG_ZERO = 0x20;
        /// One.
        ONE = 0x08;
        /// Negative one.
        NEG_ONE = 0x28;
        /// The largest finite value (7.5).
        MAX = 0x1F;
        /// The smallest finite value (-7.5).
        MIN = 0x3F;
        /// The smallest positive normal value (2^0).
        MIN_POSITIVE = 0x08;
        /// The smallest positive subnormal value (2^-3).
        MIN_POSITIVE_SUBNORMAL = 0x01;
    }
}

vector_element! {
    elem: f6e2m3, pair: u16, packing: Bytes,
    consts: [ZERO, NEG_ZERO, ONE, NEG_ONE, MAX, MIN, MIN_POSITIVE, MIN_POSITIVE_SUBNORMAL],
    note: "See [`f6e2m3`](crate::f6e2m3) for the format. Each element is stored in the low 6 bits of a byte; the two most significant bits of each byte must be zero when the value is used as a conversion source.",
    aliases: f6e2m3xN, f6e2m3x2, f6e2m3x4, f6e2m3x8, f6e2m3x16, f6e2m3x32, f6e2m3x64,
}

narrow_cvt! {
    "FP6 E2M3",
    scalar: f6e2m3,
    x2: f6e2m3x2(u16), x4: f6e2m3x4(u32),
    basic: any(target_feature = "sm_100f", target_feature = "sm_101f", target_feature = "sm_110f", target_feature = "sm_120f"),
    from_f16: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
    rbits: "the lower 16 bits are used for elements 0 and 1 and the upper 16 bits for elements 2 and 3.",
    core: {
        from_f32: cvt_e2m3x2_f32,
        from_f16x2: cvt_e2m3x2_f16x2,
        from_bf16x2: cvt_e2m3x2_bf16x2,
        to_f16x2: cvt_f16x2_e2m3x2,
        to_bf16x2_scaled: cvt_scale_bf16x2_e2m3x2,
        rs_x4: cvt_rs_e2m3x4_f32,
    },
}

narrow_scalar! {
    /// An FP6 E3M2 value (1 sign, 3 exponent and 2 mantissa bits; exponent bias 3).
    ///
    /// The format has no infinities or NaNs; the largest finite magnitude is 28.
    ///
    /// The value is stored in the low 6 bits of a byte; the two most significant bits are zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
    f6e3m2, width: 6,
    decode: |bits| minifloat_to_f32(bits, 3, 2, 3, Specials::None),
    consts: {
        /// Positive zero.
        ZERO = 0x00;
        /// Negative zero.
        NEG_ZERO = 0x20;
        /// One.
        ONE = 0x0C;
        /// Negative one.
        NEG_ONE = 0x2C;
        /// The largest finite value (28).
        MAX = 0x1F;
        /// The smallest finite value (-28).
        MIN = 0x3F;
        /// The smallest positive normal value (2^-2).
        MIN_POSITIVE = 0x04;
        /// The smallest positive subnormal value (2^-4).
        MIN_POSITIVE_SUBNORMAL = 0x01;
    }
}

vector_element! {
    elem: f6e3m2, pair: u16, packing: Bytes,
    consts: [ZERO, NEG_ZERO, ONE, NEG_ONE, MAX, MIN, MIN_POSITIVE, MIN_POSITIVE_SUBNORMAL],
    note: "See [`f6e3m2`](crate::f6e3m2) for the format. Each element is stored in the low 6 bits of a byte; the two most significant bits of each byte must be zero when the value is used as a conversion source.",
    aliases: f6e3m2xN, f6e3m2x2, f6e3m2x4, f6e3m2x8, f6e3m2x16, f6e3m2x32, f6e3m2x64,
}

narrow_cvt! {
    "FP6 E3M2",
    scalar: f6e3m2,
    x2: f6e3m2x2(u16), x4: f6e3m2x4(u32),
    basic: any(target_feature = "sm_100f", target_feature = "sm_101f", target_feature = "sm_110f", target_feature = "sm_120f"),
    from_f16: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
    rbits: "the lower 16 bits are used for elements 0 and 1 and the upper 16 bits for elements 2 and 3.",
    core: {
        from_f32: cvt_e3m2x2_f32,
        from_f16x2: cvt_e3m2x2_f16x2,
        from_bf16x2: cvt_e3m2x2_bf16x2,
        to_f16x2: cvt_f16x2_e3m2x2,
        to_bf16x2_scaled: cvt_scale_bf16x2_e3m2x2,
        rs_x4: cvt_rs_e3m2x4_f32,
    },
}
