// --- LLM-generated --- //
//! Extensions for using PTX intrinsics operating on integer values
use core::{arch::nvptx as nv, marker::ConstParamTy};

use crate::Rounding;

use super::f32_ext::with_rounding;

/// Handling of out-of-range bit widths, positions and shift amounts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum ClampMode {
    /// Out-of-range values are clamped to the operand width (`.clamp`).
    Clamp,
    /// Out-of-range values wrap around: only the low 5 bits are used (`.wrap`).
    Wrap,
}

/// Selects which pair of bytes of the second operand is used by [`Dot2`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum WordHalf {
    /// The two lowest bytes (`.lo`).
    Low,
    /// The two highest bytes (`.hi`).
    High,
}

/// What [`WideIntExt::find_msb`] reports about the bit it found.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum MsbResult {
    /// The index of the bit, counted from the least significant bit.
    Index,
    /// The left shift that moves the bit to the most significant position (`.shiftamt`).
    ShiftAmount,
}

/// Byte selection mode of [`U32Ext::permute_bytes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum PermuteMode {
    /// Generic form: each 4-bit nibble of `selector[15:0]` selects one source byte out of the
    /// eight bytes of `{b, a}`, optionally replicating its sign bit across the result byte.
    Generic,
    /// Forward 4 extract (`.f4e`).
    ForwardExtract4,
    /// Backward 4 extract (`.b4e`).
    BackwardExtract4,
    /// Replicate 8: replicates a single byte of `a` into all four result bytes (`.rc8`).
    Replicate8,
    /// Edge clamp left (`.ecl`).
    EdgeClampLeft,
    /// Edge clamp right (`.ecr`).
    EdgeClampRight,
    /// Replicate 16: replicates one 16-bit half of `a` into both result halves (`.rc16`).
    Replicate16,
}

/// Extensions for 16-, 32- and 64-bit integers.
pub trait IntExt: Sized {
    /// Returns the high half of the double-width product `self * rhs` (e.g. the high 32 bits of
    /// the 64-bit product for 32-bit integers).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-mul>
    fn mul_hi(self, rhs: Self) -> Self;

    /// Computes `|self - b| + c` with wrapping arithmetic (sum of absolute differences).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-sad>
    fn abs_diff_add(self, b: Self, c: Self) -> Self;
}

/// Extensions for 32- and 64-bit integers.
pub trait WideIntExt: Sized {
    /// Finds the most significant non-sign bit.
    ///
    /// For unsigned integers and non-negative signed integers this is the most significant set
    /// bit; for negative signed integers it is the most significant clear bit. With
    /// [`MsbResult::Index`], returns the index of that bit counted from the least significant
    /// bit; with [`MsbResult::ShiftAmount`], returns the left shift that moves it to the most
    /// significant position. Returns `u32::MAX` if there is no such bit.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-bfind>
    fn find_msb<const RESULT: MsbResult>(self) -> u32;

    /// Converts to an `f32`, rounding with `RND`.
    ///
    /// `RND` must be [`Rounding::NearestEven`], [`Rounding::TowardZero`], [`Rounding::Down`] or
    /// [`Rounding::Up`].
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_f32<const RND: Rounding>(self) -> f32;

    /// Converts to an `f64`, rounding with `RND`.
    ///
    /// 32-bit integers are always represented exactly, in which case `RND` has no effect. `RND`
    /// must be [`Rounding::NearestEven`], [`Rounding::TowardZero`], [`Rounding::Down`] or
    /// [`Rounding::Up`].
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
    fn to_f64<const RND: Rounding>(self) -> f64;
}

/// Extensions for `u32`, which is also used for packed values and bit manipulation.
pub trait U32Ext: Sized {
    /// Returns the low 32 bits of the 48-bit product of the low 24 bits of `self` and `rhs`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-mul24>
    fn mul24_lo(self, rhs: u32) -> u32;

    /// Funnel shift left: shifts the 64-bit value `{self, lo}` (with `self` as the high word)
    /// left by `amount` and returns the high 32 bits.
    ///
    /// In [`ClampMode::Clamp`] mode the shift amount is clamped to `0..=32`; in
    /// [`ClampMode::Wrap`] mode it is taken modulo 32.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-shf>
    fn funnel_shift_left<const MODE: ClampMode>(self, lo: u32, amount: u32) -> u32;

    /// Funnel shift right: shifts the 64-bit value `{self, lo}` (with `self` as the high word)
    /// right by `amount` and returns the low 32 bits.
    ///
    /// In [`ClampMode::Clamp`] mode the shift amount is clamped to `0..=32`; in
    /// [`ClampMode::Wrap`] mode it is taken modulo 32.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#logic-and-shift-instructions-shf>
    fn funnel_shift_right<const MODE: ClampMode>(self, lo: u32, amount: u32) -> u32;

    /// Generates a mask of `len` set bits starting at bit position `start`.
    ///
    /// In [`ClampMode::Clamp`] mode, a `start` of 32 or more yields zero and the mask is
    /// truncated at bit 32; in [`ClampMode::Wrap`] mode `start` and `len` are taken modulo 32.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-bmsk>
    fn bit_mask<const MODE: ClampMode>(start: u32, len: u32) -> u32;

    /// Zero-extends the low `bits` bits of `self` to 32 bits, i.e. clears all higher bits.
    ///
    /// A `bits` of zero yields zero. If `bits >= 32`, [`ClampMode::Clamp`] returns `self`
    /// unchanged, while [`ClampMode::Wrap`] takes `bits` modulo 32.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-szext>
    fn zero_extend_from<const MODE: ClampMode>(self, bits: u32) -> u32;

    /// Finds the position of the `|offset|`-th set bit of `self`, used as a bit mask, searching
    /// from bit `base` (which is included in the search).
    ///
    /// The search proceeds towards the most significant bit if `offset` is positive and towards
    /// the least significant bit if it is negative. An `offset` of zero returns `base` if that
    /// bit is set. Returns `u32::MAX` if no such bit exists. `base` must be at most 31.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-fns>
    fn find_nth_set_bit(self, base: u32, offset: i32) -> u32;

    /// Picks four bytes from the 64-bit value `{b, self}` (with `self` as the low word) and
    /// assembles them into a 32-bit result.
    ///
    /// In [`PermuteMode::Generic`] mode, the selection is controlled by four 4-bit selectors in
    /// `selector[15:0]`. In all other modes, it is controlled by `selector[1:0]`. The modes
    /// [`PermuteMode::Replicate8`], [`PermuteMode::EdgeClampLeft`],
    /// [`PermuteMode::EdgeClampRight`] and [`PermuteMode::Replicate16`] only read bytes from
    /// `self`, and `b` is ignored.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt>
    fn permute_bytes<const MODE: PermuteMode>(self, b: u32, selector: u32) -> u32;
}

/// Extensions for `i32`.
pub trait I32Ext: Sized {
    /// Returns the low 32 bits of the 48-bit product of the sign-extended low 24 bits of `self`
    /// and `rhs`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-mul24>
    fn mul24_lo(self, rhs: i32) -> i32;

    /// Sign-extends the low `bits` bits of `self` to 32 bits.
    ///
    /// A `bits` of zero yields zero. If `bits >= 32`, [`ClampMode::Clamp`] returns `self`
    /// unchanged, while [`ClampMode::Wrap`] takes `bits` modulo 32.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-szext>
    fn sign_extend_from<const MODE: ClampMode>(self, bits: u32) -> i32;
}

/// Four-way byte dot product with accumulation (`dp4a`).
///
/// Implemented for every combination of `u32` and `i32`: the four bytes packed into `self` and
/// into `rhs` are interpreted as unsigned if the respective type is `u32` and as signed if it is
/// `i32`. The accumulator and result are signed if either operand is signed, and unsigned
/// otherwise.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-dp4a>
pub trait Dot4<Rhs> {
    /// The type of the accumulator and result.
    type Acc;

    /// Computes `acc + sum(self.byte[i] * rhs.byte[i])` for `i` in `0..4`, wrapping on overflow.
    fn dot4_acc(self, rhs: Rhs, acc: Self::Acc) -> Self::Acc;
}

/// Two-way dot product with accumulation (`dp2a`).
///
/// Implemented for every combination of `u32` and `i32`: the two 16-bit halves of `self` are
/// multiplied with the two selected bytes of `rhs`. Each operand is interpreted as unsigned if
/// its type is `u32` and as signed if it is `i32`. The accumulator and result are signed if
/// either operand is signed, and unsigned otherwise.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-dp2a>
pub trait Dot2<Rhs> {
    /// The type of the accumulator and result.
    type Acc;

    /// Computes `acc + self.half[0] * rhs.byte[k] + self.half[1] * rhs.byte[k + 1]`, where `k`
    /// is 0 for [`WordHalf::Low`] and 2 for [`WordHalf::High`], wrapping on overflow.
    fn dot2_acc<const HALF: WordHalf>(self, rhs: Rhs, acc: Self::Acc) -> Self::Acc;
}

macro_rules! impl_int_ext {
    ($($ty:ty: $mul_hi:ident, $sad:ident;)*) => {$(
        impl IntExt for $ty {
            #[inline]
            fn mul_hi(self, rhs: Self) -> Self {
                unsafe { nv::$mul_hi(self, rhs) }
            }

            #[inline]
            fn abs_diff_add(self, b: Self, c: Self) -> Self {
                unsafe { nv::$sad(self, b, c) }
            }
        }
    )*};
}

impl_int_ext! {
    i16: mul_hi_s16, sad_s16;
    u16: mul_hi_u16, sad_u16;
    i32: mul_hi_s32, sad_s32;
    u32: mul_hi_u32, sad_u32;
    i64: mul_hi_s64, sad_s64;
    u64: mul_hi_u64, sad_u64;
}

macro_rules! impl_wide_int_ext {
    ($($ty:ty: $bfind:ident, $to_f32:ident, $to_f64:ident;)*) => {$(
        impl WideIntExt for $ty {
            #[inline]
            fn find_msb<const RESULT: MsbResult>(self) -> u32 {
                const_bool!(SHIFT = matches!(RESULT, MsbResult::ShiftAmount) =>
                    unsafe { nv::$bfind::<SHIFT>(self) })
            }

            #[inline]
            fn to_f32<const RND: Rounding>(self) -> f32 {
                with_rounding!(RND => R in unsafe { nv::$to_f32::<R>(self) })
            }

            #[inline]
            fn to_f64<const RND: Rounding>(self) -> f64 {
                with_rounding!(RND => R in unsafe { nv::$to_f64::<R>(self) })
            }
        }
    )*};
}

impl_wide_int_ext! {
    i32: bfind_s32, cvt_f32_s32, cvt_f64_s32;
    u32: bfind_u32, cvt_f32_u32, cvt_f64_u32;
    i64: bfind_s64, cvt_f32_s64, cvt_f64_s64;
    u64: bfind_u64, cvt_f32_u64, cvt_f64_u64;
}

macro_rules! with_clamp_mode {
    ($mode:ident => $name:ident in $body:expr) => {
        const_map!($mode => $name: nv::ClampMode {
            ClampMode::Clamp => nv::ClampMode::Clamp,
            ClampMode::Wrap => nv::ClampMode::Wrap,
        } in $body)
    };
}

impl U32Ext for u32 {
    #[inline]
    fn mul24_lo(self, rhs: u32) -> u32 {
        unsafe { nv::mul24_lo_u32(self, rhs) }
    }

    #[inline]
    fn funnel_shift_left<const MODE: ClampMode>(self, lo: u32, amount: u32) -> u32 {
        with_clamp_mode!(MODE => M in unsafe { nv::shf_l::<M>(lo, self, amount) })
    }

    #[inline]
    fn funnel_shift_right<const MODE: ClampMode>(self, lo: u32, amount: u32) -> u32 {
        with_clamp_mode!(MODE => M in unsafe { nv::shf_r::<M>(lo, self, amount) })
    }

    #[inline]
    fn bit_mask<const MODE: ClampMode>(start: u32, len: u32) -> u32 {
        with_clamp_mode!(MODE => M in unsafe { nv::bmsk::<M>(start, len) })
    }

    #[inline]
    fn zero_extend_from<const MODE: ClampMode>(self, bits: u32) -> u32 {
        with_clamp_mode!(MODE => M in unsafe { nv::szext_u32::<M>(self, bits) })
    }

    #[inline]
    fn find_nth_set_bit(self, base: u32, offset: i32) -> u32 {
        unsafe { nv::fns(self, base, offset) }
    }

    #[inline]
    fn permute_bytes<const MODE: PermuteMode>(self, b: u32, selector: u32) -> u32 {
        const_map!(MODE => M: nv::PrmtMode {
            PermuteMode::Generic => nv::PrmtMode::Generic,
            PermuteMode::ForwardExtract4 => nv::PrmtMode::F4e,
            PermuteMode::BackwardExtract4 => nv::PrmtMode::B4e,
            PermuteMode::Replicate8 => nv::PrmtMode::Rc8,
            PermuteMode::EdgeClampLeft => nv::PrmtMode::Ecl,
            PermuteMode::EdgeClampRight => nv::PrmtMode::Ecr,
            PermuteMode::Replicate16 => nv::PrmtMode::Rc16,
        } in unsafe { nv::prmt::<M>(self, b, selector) })
    }
}

impl I32Ext for i32 {
    #[inline]
    fn mul24_lo(self, rhs: i32) -> i32 {
        unsafe { nv::mul24_lo_s32(self, rhs) }
    }

    #[inline]
    fn sign_extend_from<const MODE: ClampMode>(self, bits: u32) -> i32 {
        with_clamp_mode!(MODE => M in unsafe { nv::szext_s32::<M>(self, bits) })
    }
}

macro_rules! impl_dot {
    ($($a:ty, $b:ty => $acc:ty: $dp4a:ident, $dp2a:ident;)*) => {$(
        impl Dot4<$b> for $a {
            type Acc = $acc;

            #[inline]
            fn dot4_acc(self, rhs: $b, acc: $acc) -> $acc {
                unsafe { nv::$dp4a(self, rhs, acc) }
            }
        }

        impl Dot2<$b> for $a {
            type Acc = $acc;

            #[inline]
            fn dot2_acc<const HALF: WordHalf>(self, rhs: $b, acc: $acc) -> $acc {
                const_map!(HALF => H: nv::Dp2aMode {
                    WordHalf::Low => nv::Dp2aMode::Lo,
                    WordHalf::High => nv::Dp2aMode::Hi,
                } in unsafe { nv::$dp2a::<H>(self, rhs, acc) })
            }
        }
    )*};
}

impl_dot! {
    u32, u32 => u32: dp4a_u32_u32, dp2a_u32_u32;
    u32, i32 => i32: dp4a_u32_s32, dp2a_u32_s32;
    i32, u32 => i32: dp4a_s32_u32, dp2a_s32_u32;
    i32, i32 => i32: dp4a_s32_s32, dp2a_s32_s32;
}
