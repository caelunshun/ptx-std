// --- LLM-generated --- //
use bytemuck::{Pod, Zeroable};

/// Two packed `s2f6` values.
///
/// `s2f6` is an 8-bit signed two's complement fixed-point format with 2 sign/integer bits and 6
/// fractional bits (`xx.xxxxxx`): the value of an element is `bits as i8 * 2^-6`, ranging from
/// `-2.0` to `127 * 2^-6 = 1.984375`. Element 0 occupies the low 8 bits.
///
/// The format has no infinity or NaN. Only the up-conversions to [`bf16x2`](crate::bf16x2) are
/// currently exposed; they require `sm_100a`, `sm_103a`, `sm_110a`, `sm_120a` or `sm_121a`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#fixed-point-data-formats>
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct s2f6x2(u16);

impl s2f6x2 {
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
    target_feature = "sm_100a",
    target_feature = "sm_103a",
    target_feature = "sm_110a",
    target_feature = "sm_120a",
    target_feature = "sm_121a"
))]
impl s2f6x2 {
    /// Converts the packed pair of `s2f6` values to a packed pair of `bf16` values.
    ///
    /// Element `i` of the result is converted from element `i` of `self`. The conversion is exact;
    /// `RELU` clamps negative results to zero.
    ///
    /// Implemented as [`Self::to_bf16x2_scaled`] with unit scale factors.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn to_bf16x2<const RELU: crate::Relu>(self) -> crate::bf16x2 {
        self.to_bf16x2_scaled::<RELU, { crate::Overflow::Infinity }>(crate::f8e8m0x2::ONES)
    }

    /// Converts the packed pair of `s2f6` values to a packed pair of `bf16` values, multiplying
    /// each element by its `ue8m0` scale factor.
    ///
    /// Element `i` of the result is converted from element `i` of `self` and multiplied by element
    /// `i` of `scale` (`.scaled::n2::ue8m0`). `RELU` clamps negative results to zero. With
    /// [`Overflow::SaturateFinite`](crate::Overflow::SaturateFinite), results outside the finite
    /// range are clamped to the largest finite `bf16` value of matching sign.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    #[inline]
    pub fn to_bf16x2_scaled<const RELU: crate::Relu, const OVERFLOW: crate::Overflow>(
        self,
        scale: crate::f8e8m0x2,
    ) -> crate::bf16x2 {
        const_bool!(R = matches!(RELU, crate::Relu::Enabled) => {
            const_bool!(S = matches!(OVERFLOW, crate::Overflow::SaturateFinite) => {
                crate::data_types::narrow::bf16x2_from_core(unsafe {
                    core::arch::nvptx::cvt_scale_bf16x2_s2f6x2::<R, S>(self.0, scale.to_bits())
                })
            })
        })
    }
}
