// --- LLM-generated --- //
//! The `s2f6` fixed-point format, as a scalar and packed vectors of 2, 4, 8, 16 and 32 elements.

// Only used by the conversions, which are gated on newer architectures.
#[allow(unused_imports)]
use super::narrow::s2f6_cvt;
use super::vector::{narrow_scalar, vector_element};

narrow_scalar! {
    /// An `s2f6` value.
    ///
    /// `s2f6` is an 8-bit signed two's complement fixed-point format with 2 sign/integer bits and 6
    /// fractional bits (`xx.xxxxxx`): the value is `bits as i8 * 2^-6`, ranging from `-2.0` to
    /// `127 * 2^-6 = 1.984375`.
    ///
    /// The format has no infinity or NaN. The conversions to and from [`bf16`](crate::bf16) and
    /// `f32` (optionally scaled by `ue8m0` factors) require `sm_100a`, `sm_103a`, `sm_110a`,
    /// `sm_120a` or `sm_121a`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#fixed-point-data-formats>
    s2f6, width: 8,
    decode: |bits| (bits as i8) as f32 / 64.0,
    consts: {
        /// Zero.
        ZERO = 0x00;
        /// One (`64 * 2^-6`).
        ONE = 0x40;
        /// Negative one.
        NEG_ONE = 0xC0;
        /// The largest value (`127 * 2^-6 = 1.984375`).
        MAX = 0x7F;
        /// The smallest value (`-2.0`).
        MIN = 0x80;
        /// The smallest positive value (`2^-6`).
        MIN_POSITIVE = 0x01;
    }
}

vector_element! {
    elem: s2f6, pair: u16, packing: Bytes,
    consts: [ZERO, ONE, NEG_ONE, MAX, MIN, MIN_POSITIVE],
    note: "See [`s2f6`](crate::s2f6) for the format.",
    aliases: s2f6xN, s2f6x2, s2f6x4, s2f6x8, s2f6x16, s2f6x32,
}

s2f6_cvt! {
    scalar: s2f6,
    x2: s2f6x2,
    cfg: any(
        target_feature = "sm_100a",
        target_feature = "sm_103a",
        target_feature = "sm_110a",
        target_feature = "sm_120a",
        target_feature = "sm_121a"
    ),
}
