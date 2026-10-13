// --- LLM-generated --- //
//! 4-bit floating-point format (`e2m1`), as a scalar and packed vectors of 2, 4, 8, 16 and
//! 32 elements.
//!
//! In all packed types, element 0 occupies the least significant bits; two elements share a byte,
//! with element 0 in the low nibble.

use super::vector::{Specials, minifloat_to_f32, narrow_scalar, vector_element};

// Only used by the conversions, which are gated on newer architectures.
#[allow(unused_imports)]
use super::narrow::narrow_cvt;

narrow_scalar! {
    /// An FP4 E2M1 value (1 sign, 2 exponent and 1 mantissa bit; exponent bias 1).
    ///
    /// The format has no infinities or NaNs; the representable magnitudes are 0, 0.5, 1, 1.5, 2,
    /// 3, 4 and 6, so the largest finite magnitude is 6.0.
    ///
    /// The value is stored in the low 4 bits of a byte; the four most significant bits are zero.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats>
    f4e2m1, width: 4,
    decode: |bits| minifloat_to_f32(bits, 2, 1, 1, Specials::None),
    consts: {
        /// Positive zero.
        ZERO = 0x0;
        /// Negative zero.
        NEG_ZERO = 0x8;
        /// One.
        ONE = 0x2;
        /// Negative one.
        NEG_ONE = 0xA;
        /// The largest finite value (6).
        MAX = 0x7;
        /// The smallest finite value (-6).
        MIN = 0xF;
        /// The smallest positive normal value (1).
        MIN_POSITIVE = 0x2;
        /// The smallest positive subnormal value (0.5).
        MIN_POSITIVE_SUBNORMAL = 0x1;
    }
}

vector_element! {
    elem: f4e2m1, pair: u8, packing: Nibbles,
    consts: [ZERO, NEG_ZERO, ONE, NEG_ONE, MAX, MIN, MIN_POSITIVE, MIN_POSITIVE_SUBNORMAL],
    note: "See [`f4e2m1`](crate::f4e2m1) for the format.",
    aliases: f4e2m1xN, f4e2m1x2, f4e2m1x4, f4e2m1x8, f4e2m1x16, f4e2m1x32,
}

narrow_cvt! {
    "FP4 E2M1",
    scalar: f4e2m1,
    x2: f4e2m1x2(u8), x4: f4e2m1x4(u16),
    basic: any(target_feature = "sm_100f", target_feature = "sm_101f", target_feature = "sm_110f", target_feature = "sm_120f"),
    from_f16: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
    rbits: "two 16-bit values are used; the lower 8 bits of both 16-bit halves are used for elements 0 and 1 and the upper 8 bits of both halves for elements 2 and 3 (the PTX ISA does not specify the assignment of the individual bytes any further).",
    core: {
        from_f32: cvt_e2m1x2_f32,
        from_f16x2: cvt_e2m1x2_f16x2,
        from_bf16x2: cvt_e2m1x2_bf16x2,
        to_f16x2: cvt_f16x2_e2m1x2,
        to_bf16x2_scaled: cvt_scale_bf16x2_e2m1x2,
        rs_x4: cvt_rs_e2m1x4_f32,
    },
}
