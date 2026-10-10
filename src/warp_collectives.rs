// --- LLM-generated --- //
//! Warp-level communication: lane masks, shuffles, votes, matches, reductions and warp barriers.
//!
//! All `.sync` collectives take a [`LaneMask`] naming the lanes of the warp that participate. They
//! are `unsafe` because they are *convergent* operations with a contract the compiler cannot
//! check:
//!
//! * the executing thread must be a member of the mask, and
//! * every (non-exited) thread named in the mask must execute the same operation with the same
//!   mask value, otherwise the behavior is undefined.
//!
//! Using [`LaneMask::FULL`] satisfies the first condition trivially, but the second still requires
//! that the whole warp reaches the operation together. Rust does not model convergence, so the
//! compiler may in principle move these operations across control flow; keep the operation and
//! the control flow that determines which lanes reach it straightforward.

#[cfg(target_feature = "sm_100f")]
use crate::modifiers::NanHandling;
use bytemuck::{Pod, Zeroable};
use core::{
    arch::nvptx as nv,
    marker::ConstParamTy,
    mem::size_of,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

/// A set of lanes of a warp, with bit `i` standing for lane `i`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct LaneMask(u32);

impl LaneMask {
    /// The empty set.
    pub const EMPTY: Self = Self(0);
    /// All 32 lanes of the warp.
    pub const FULL: Self = Self(u32::MAX);

    /// Creates a mask from its raw bits, where bit `i` stands for lane `i`.
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw bits of the mask, where bit `i` stands for lane `i`.
    #[inline(always)]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Creates a mask containing only `lane`. Only the low 5 bits of `lane` are used.
    #[inline(always)]
    pub const fn from_lane(lane: u32) -> Self {
        Self(1u32.wrapping_shl(lane))
    }

    /// Returns whether `lane` is in the mask. Only the low 5 bits of `lane` are used.
    #[inline(always)]
    pub const fn contains(self, lane: u32) -> bool {
        self.0 >> (lane & 31) & 1 != 0
    }

    /// Returns the number of lanes in the mask.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns whether the mask contains no lanes.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns whether the mask contains all 32 lanes.
    #[inline(always)]
    pub const fn is_full(self) -> bool {
        self.0 == u32::MAX
    }

    /// Returns the lowest lane in the mask, if any.
    #[inline(always)]
    pub const fn first_lane(self) -> Option<u32> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0.trailing_zeros())
        }
    }

    /// Returns the highest lane in the mask, if any.
    #[inline(always)]
    pub const fn last_lane(self) -> Option<u32> {
        if self.0 == 0 {
            None
        } else {
            Some(31 - self.0.leading_zeros())
        }
    }

    /// Returns the mask of lanes that are currently active, i.e. executing this instruction
    /// convergently with the calling thread (`activemask`).
    ///
    /// Exited threads and threads that took a different branch are not included. This only
    /// describes the threads that happen to be converged right now; it does not make it safe to
    /// use the result as the mask of a later collective, as that still requires all of its
    /// lanes to execute that collective.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-activemask>
    #[inline(always)]
    pub fn active() -> Self {
        Self(unsafe { nv::activemask() })
    }

    /// Returns the mask containing only the calling thread's lane (`%lanemask_eq`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-eq>
    #[inline(always)]
    pub fn current_lane() -> Self {
        Self(unsafe { nv::lanemask_eq() })
    }

    /// Returns the mask of lanes with an index less than the calling thread's lane
    /// (`%lanemask_lt`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-lt>
    #[inline(always)]
    pub fn lanes_below() -> Self {
        Self(unsafe { nv::lanemask_lt() })
    }

    /// Returns the mask of lanes with an index less than or equal to the calling thread's lane
    /// (`%lanemask_le`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-le>
    #[inline(always)]
    pub fn lanes_at_or_below() -> Self {
        Self(unsafe { nv::lanemask_le() })
    }

    /// Returns the mask of lanes with an index greater than the calling thread's lane
    /// (`%lanemask_gt`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-gt>
    #[inline(always)]
    pub fn lanes_above() -> Self {
        Self(unsafe { nv::lanemask_gt() })
    }

    /// Returns the mask of lanes with an index greater than or equal to the calling thread's lane
    /// (`%lanemask_ge`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-ge>
    #[inline(always)]
    pub fn lanes_at_or_above() -> Self {
        Self(unsafe { nv::lanemask_ge() })
    }

    /// Returns an iterator over the lanes in the mask, in increasing order.
    #[inline(always)]
    pub const fn lanes(self) -> Lanes {
        Lanes(self.0)
    }
}

/// Iterator over the lanes of a [`LaneMask`], in increasing order.
#[derive(Copy, Clone, Debug)]
pub struct Lanes(u32);

impl Iterator for Lanes {
    type Item = u32;

    #[inline(always)]
    fn next(&mut self) -> Option<u32> {
        if self.0 == 0 {
            None
        } else {
            let lane = self.0.trailing_zeros();
            self.0 &= self.0 - 1;
            Some(lane)
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for Lanes {}

impl IntoIterator for LaneMask {
    type Item = u32;
    type IntoIter = Lanes;

    #[inline(always)]
    fn into_iter(self) -> Lanes {
        self.lanes()
    }
}

impl From<u32> for LaneMask {
    #[inline(always)]
    fn from(bits: u32) -> Self {
        Self(bits)
    }
}

impl From<LaneMask> for u32 {
    #[inline(always)]
    fn from(mask: LaneMask) -> u32 {
        mask.0
    }
}

macro_rules! lane_mask_binop {
    ($trait:ident :: $method:ident, $assign_trait:ident :: $assign_method:ident, $op:tt) => {
        impl $trait for LaneMask {
            type Output = LaneMask;

            #[inline(always)]
            fn $method(self, rhs: LaneMask) -> LaneMask {
                LaneMask(self.0 $op rhs.0)
            }
        }

        impl $assign_trait for LaneMask {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: LaneMask) {
                self.0 = self.0 $op rhs.0;
            }
        }
    };
}

lane_mask_binop!(BitAnd::bitand, BitAndAssign::bitand_assign, &);
lane_mask_binop!(BitOr::bitor, BitOrAssign::bitor_assign, |);
lane_mask_binop!(BitXor::bitxor, BitXorAssign::bitxor_assign, ^);

impl Not for LaneMask {
    type Output = LaneMask;

    #[inline(always)]
    fn not(self) -> LaneMask {
        LaneMask(!self.0)
    }
}

/// Waits until all threads in `mask` have executed `bar.warp.sync` with the same mask. Also
/// guarantees memory ordering among the participating threads (`bar.warp.sync`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-warp-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same `sync_warp` with the same `mask`.
#[inline(always)]
pub unsafe fn sync_warp(mask: LaneMask) {
    unsafe { nv::bar_warp_sync(mask.0) }
}

/// How the source lane of a shuffle is computed from its operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum ShuffleMode {
    /// Reads from the lane with the index given by the operand (within the segment).
    Index,
    /// Reads from the lane `operand` below the executing lane (`laneid - operand`).
    Up,
    /// Reads from the lane `operand` above the executing lane (`laneid + operand`).
    Down,
    /// Reads from the lane `laneid ^ operand`.
    Xor,
}

/// Computes the packed `c` operand of `shfl.sync` for a segment width: the clamp value in bits
/// `[4:0]` and the segmentation mask in bits `[12:8]`.
const fn shuffle_clamp_operand(mode: ShuffleMode, width: u32) -> u32 {
    let segment_mask = (32 - width) << 8;
    match mode {
        ShuffleMode::Up => segment_mask,
        _ => segment_mask | 31,
    }
}

/// Applies `f` to every 32-bit word of `value`.
#[inline(always)]
fn map_words<T: Pod>(value: T, mut f: impl FnMut(u32) -> u32) -> T {
    const {
        assert!(
            size_of::<T>() != 0 && size_of::<T>().is_multiple_of(4),
            "shuffled types must have a size that is a non-zero multiple of 4 bytes"
        )
    };
    let mut out = T::zeroed();
    let src = (&value as *const T).cast::<u32>();
    let dst = (&mut out as *mut T).cast::<u32>();
    let mut i = 0;
    while i < size_of::<T>() / 4 {
        unsafe { dst.add(i).write_unaligned(f(src.add(i).read_unaligned())) };
        i += 1;
    }
    out
}

/// Shuffles one 32-bit word.
#[inline(always)]
unsafe fn shuffle_word<const MODE: ShuffleMode, const WIDTH: u32>(
    mask: LaneMask,
    word: u32,
    operand: u32,
) -> u32 {
    let c = const { shuffle_clamp_operand(MODE, WIDTH) };
    const_map!(MODE => M: nv::ShflMode {
        ShuffleMode::Index => nv::ShflMode::Idx,
        ShuffleMode::Up => nv::ShflMode::Up,
        ShuffleMode::Down => nv::ShflMode::Down,
        ShuffleMode::Xor => nv::ShflMode::Bfly,
    } in unsafe { nv::shfl_sync_b32::<M>(word, operand, c, mask.0) })
}

/// Shuffles one 32-bit word, also returning whether the source lane was in range.
#[inline(always)]
unsafe fn shuffle_word_checked<const MODE: ShuffleMode, const WIDTH: u32>(
    mask: LaneMask,
    word: u32,
    operand: u32,
) -> (u32, bool) {
    let c = const { shuffle_clamp_operand(MODE, WIDTH) };
    const_map!(MODE => M: nv::ShflMode {
        ShuffleMode::Index => nv::ShflMode::Idx,
        ShuffleMode::Up => nv::ShflMode::Up,
        ShuffleMode::Down => nv::ShflMode::Down,
        ShuffleMode::Xor => nv::ShflMode::Bfly,
    } in unsafe { nv::shfl_sync_pred_b32::<M>(word, operand, c, mask.0) })
}

/// Exchanges `value` between the lanes of `mask`, with the lanes of the warp divided into
/// independent segments of `WIDTH` consecutive lanes (`shfl.sync`).
///
/// Every lane of `mask` provides `value` and receives the value of its source lane, which is
/// selected according to `MODE` and `operand`, always within the lane's own segment:
///
/// * [`ShuffleMode::Index`]: the lane with index `operand` within the segment.
/// * [`ShuffleMode::Up`]: the lane `operand` below. If that falls before the start of the
///   segment, the lane receives its own `value`.
/// * [`ShuffleMode::Down`]: the lane `operand` above. If that falls after the end of the
///   segment, the lane receives its own `value`.
/// * [`ShuffleMode::Xor`]: the lane whose index is the executing lane's index XOR `operand`. If
///   that falls outside the segment, the lane receives its own `value`.
///
/// If the source lane is not in `mask` or has exited, the received value is undefined.
///
/// `T` may be any plain-data type whose size is a multiple of 4 bytes; larger values are shuffled
/// one 32-bit word at a time. `WIDTH` must be a power of two in `1..=32`. See
/// [`try_shuffle_segmented`] to distinguish the out-of-range case.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same shuffle with the same `mask`.
#[inline(always)]
pub unsafe fn shuffle_segmented<const MODE: ShuffleMode, const WIDTH: u32, T: Pod>(
    mask: LaneMask,
    value: T,
    operand: u32,
) -> T {
    const {
        assert!(
            WIDTH.is_power_of_two() && WIDTH <= 32,
            "WIDTH must be a power of two in 1..=32"
        )
    };
    map_words(value, |word| unsafe {
        shuffle_word::<MODE, WIDTH>(mask, word, operand)
    })
}

/// Like [`shuffle_segmented`], but returns `None` if the source lane lies outside the lane's
/// segment (in which case [`shuffle_segmented`] would return the lane's own value).
///
/// # Safety
///
/// Same as [`shuffle_segmented`].
#[inline(always)]
pub unsafe fn try_shuffle_segmented<const MODE: ShuffleMode, const WIDTH: u32, T: Pod>(
    mask: LaneMask,
    value: T,
    operand: u32,
) -> Option<T> {
    const {
        assert!(
            WIDTH.is_power_of_two() && WIDTH <= 32,
            "WIDTH must be a power of two in 1..=32"
        )
    };
    let mut in_range = true;
    let out = map_words(value, |word| {
        let (word, p) = unsafe { shuffle_word_checked::<MODE, WIDTH>(mask, word, operand) };
        in_range = p;
        word
    });
    if in_range { Some(out) } else { None }
}

/// Reads `value` from the lane `src_lane` of the warp (`shfl.sync.idx`). Only the low 5 bits of
/// `src_lane` are used.
///
/// If the source lane is not in `mask` or has exited, the received value is undefined. See
/// [`shuffle_segmented`] for the supported types and for segmented shuffles.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same shuffle with the same `mask`.
#[inline(always)]
pub unsafe fn shuffle<T: Pod>(mask: LaneMask, value: T, src_lane: u32) -> T {
    unsafe { shuffle_segmented::<{ ShuffleMode::Index }, 32, T>(mask, value, src_lane) }
}

/// Reads `value` from the lane `delta` below the executing lane (`shfl.sync.up`). Lanes with an
/// index below `delta` receive their own `value`.
///
/// If the source lane is not in `mask` or has exited, the received value is undefined. See
/// [`shuffle_segmented`] for the supported types and for segmented shuffles.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same shuffle with the same `mask`.
#[inline(always)]
pub unsafe fn shuffle_up<T: Pod>(mask: LaneMask, value: T, delta: u32) -> T {
    unsafe { shuffle_segmented::<{ ShuffleMode::Up }, 32, T>(mask, value, delta) }
}

/// Reads `value` from the lane `delta` above the executing lane (`shfl.sync.down`). Lanes whose
/// source would be past lane 31 receive their own `value`.
///
/// If the source lane is not in `mask` or has exited, the received value is undefined. See
/// [`shuffle_segmented`] for the supported types and for segmented shuffles.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same shuffle with the same `mask`.
#[inline(always)]
pub unsafe fn shuffle_down<T: Pod>(mask: LaneMask, value: T, delta: u32) -> T {
    unsafe { shuffle_segmented::<{ ShuffleMode::Down }, 32, T>(mask, value, delta) }
}

/// Reads `value` from the lane whose index is the executing lane's index XOR `lane_xor`
/// (`shfl.sync.bfly`), the "butterfly" exchange pattern. Only the low 5 bits of `lane_xor` are
/// used.
///
/// If the source lane is not in `mask` or has exited, the received value is undefined. See
/// [`shuffle_segmented`] for the supported types and for segmented shuffles.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same shuffle with the same `mask`.
#[inline(always)]
pub unsafe fn shuffle_xor<T: Pod>(mask: LaneMask, value: T, lane_xor: u32) -> T {
    unsafe { shuffle_segmented::<{ ShuffleMode::Xor }, 32, T>(mask, value, lane_xor) }
}

/// Like [`shuffle_up`], but returns `None` for lanes with an index below `delta`.
///
/// # Safety
///
/// Same as [`shuffle_up`].
#[inline(always)]
pub unsafe fn try_shuffle_up<T: Pod>(mask: LaneMask, value: T, delta: u32) -> Option<T> {
    unsafe { try_shuffle_segmented::<{ ShuffleMode::Up }, 32, T>(mask, value, delta) }
}

/// Like [`shuffle_down`], but returns `None` for lanes whose source would be past lane 31.
///
/// # Safety
///
/// Same as [`shuffle_down`].
#[inline(always)]
pub unsafe fn try_shuffle_down<T: Pod>(mask: LaneMask, value: T, delta: u32) -> Option<T> {
    unsafe { try_shuffle_segmented::<{ ShuffleMode::Down }, 32, T>(mask, value, delta) }
}

/// Returns whether `predicate` is true for all threads in `mask` (`vote.sync.all`). The result is
/// the same for all threads in `mask`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same vote with the same `mask`.
#[inline(always)]
pub unsafe fn vote_all(mask: LaneMask, predicate: bool) -> bool {
    unsafe { nv::vote_sync_pred::<{ nv::VoteMode::All }>(predicate, mask.0) }
}

/// Returns whether `predicate` is true for any thread in `mask` (`vote.sync.any`). The result is
/// the same for all threads in `mask`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same vote with the same `mask`.
#[inline(always)]
pub unsafe fn vote_any(mask: LaneMask, predicate: bool) -> bool {
    unsafe { nv::vote_sync_pred::<{ nv::VoteMode::Any }>(predicate, mask.0) }
}

/// Returns whether `predicate` has the same value (all true or all false) in all threads in `mask`
/// (`vote.sync.uni`). The result is the same for all threads in `mask`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same vote with the same `mask`.
#[inline(always)]
pub unsafe fn vote_uniform(mask: LaneMask, predicate: bool) -> bool {
    unsafe { nv::vote_sync_pred::<{ nv::VoteMode::Uni }>(predicate, mask.0) }
}

/// Returns the set of threads in `mask` for which `predicate` is true (`vote.sync.ballot`). The
/// result is the same for all threads in `mask`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same vote with the same `mask`.
#[inline(always)]
pub unsafe fn ballot(mask: LaneMask, predicate: bool) -> LaneMask {
    LaneMask(unsafe { nv::vote_sync_ballot_b32(predicate, mask.0) })
}

/// Returns the set of threads in `mask` whose `value` is bitwise equal to the executing thread's
/// (`match.any.sync`).
///
/// `T` must be 4 or 8 bytes in size; floating-point values are compared by bit pattern.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same match with the same `mask`.
#[inline(always)]
pub unsafe fn match_any<T: Pod>(mask: LaneMask, value: T) -> LaneMask {
    const {
        assert!(
            size_of::<T>() == 4 || size_of::<T>() == 8,
            "match.sync only supports 4-byte and 8-byte types"
        )
    };
    let ptr = &value as *const T;
    LaneMask(unsafe {
        if size_of::<T>() == 4 {
            nv::match_any_sync_b32(ptr.cast::<u32>().read_unaligned(), mask.0)
        } else {
            nv::match_any_sync_b64(ptr.cast::<u64>().read_unaligned(), mask.0)
        }
    })
}

/// Returns whether `value` is bitwise equal in all threads in `mask` (`match.all.sync`). The
/// result is the same for all threads in `mask`.
///
/// `T` must be 4 or 8 bytes in size; floating-point values are compared by bit pattern.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same match with the same `mask`.
#[inline(always)]
pub unsafe fn match_all<T: Pod>(mask: LaneMask, value: T) -> bool {
    const {
        assert!(
            size_of::<T>() == 4 || size_of::<T>() == 8,
            "match.sync only supports 4-byte and 8-byte types"
        )
    };
    let ptr = &value as *const T;
    unsafe {
        if size_of::<T>() == 4 {
            nv::match_all_sync_b32(ptr.cast::<u32>().read_unaligned(), mask.0).1
        } else {
            nv::match_all_sync_b64(ptr.cast::<u64>().read_unaligned(), mask.0).1
        }
    }
}

/// Reduction operation of `reduce` and `reduce_f32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum ReduxOp {
    /// Sum of all values (wrapping on overflow).
    Add,
    /// Minimum of all values.
    Min,
    /// Maximum of all values.
    Max,
    /// Bitwise AND of all values.
    And,
    /// Bitwise OR of all values.
    Or,
    /// Bitwise XOR of all values.
    Xor,
}

/// Whether the absolute values of the inputs are reduced by `reduce_f32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum Absolute {
    /// The inputs are reduced as they are.
    Disabled,
    /// The absolute values of the inputs are reduced (`.abs`).
    Enabled,
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
}

/// Integer types that can be reduced across a warp with [`reduce`]: `u32` and `i32`.
///
/// `Min` and `Max` compare according to the signedness of the type; `Add`, `And`, `Or` and `Xor`
/// operate on the bit patterns.
pub trait ReduxInt: Copy + sealed::Sealed {
    #[doc(hidden)]
    unsafe fn reduce<const OP: ReduxOp>(mask: LaneMask, value: Self) -> Self;
}

impl ReduxInt for u32 {
    #[inline(always)]
    unsafe fn reduce<const OP: ReduxOp>(mask: LaneMask, value: u32) -> u32 {
        unsafe {
            match OP {
                ReduxOp::Add => nv::redux_sync_u32::<{ nv::ReduxOp::Add }>(value, mask.0),
                ReduxOp::Min => nv::redux_sync_u32::<{ nv::ReduxOp::Min }>(value, mask.0),
                ReduxOp::Max => nv::redux_sync_u32::<{ nv::ReduxOp::Max }>(value, mask.0),
                ReduxOp::And => nv::redux_sync_b32::<{ nv::ReduxOp::And }>(value, mask.0),
                ReduxOp::Or => nv::redux_sync_b32::<{ nv::ReduxOp::Or }>(value, mask.0),
                ReduxOp::Xor => nv::redux_sync_b32::<{ nv::ReduxOp::Xor }>(value, mask.0),
            }
        }
    }
}

impl ReduxInt for i32 {
    #[inline(always)]
    unsafe fn reduce<const OP: ReduxOp>(mask: LaneMask, value: i32) -> i32 {
        unsafe {
            match OP {
                ReduxOp::Add => nv::redux_sync_s32::<{ nv::ReduxOp::Add }>(value, mask.0),
                ReduxOp::Min => nv::redux_sync_s32::<{ nv::ReduxOp::Min }>(value, mask.0),
                ReduxOp::Max => nv::redux_sync_s32::<{ nv::ReduxOp::Max }>(value, mask.0),
                ReduxOp::And => {
                    nv::redux_sync_b32::<{ nv::ReduxOp::And }>(value as u32, mask.0) as i32
                }
                ReduxOp::Or => {
                    nv::redux_sync_b32::<{ nv::ReduxOp::Or }>(value as u32, mask.0) as i32
                }
                ReduxOp::Xor => {
                    nv::redux_sync_b32::<{ nv::ReduxOp::Xor }>(value as u32, mask.0) as i32
                }
            }
        }
    }
}

/// Reduces `value` across the threads of `mask` with the operation `OP` and returns the result to
/// all of them (`redux.sync`). Works on `u32` and `i32`; all [`ReduxOp`]s are supported.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-redux-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same reduction with the same `mask`.
#[inline(always)]
pub unsafe fn reduce<const OP: ReduxOp, T: ReduxInt>(mask: LaneMask, value: T) -> T {
    unsafe { T::reduce::<OP>(mask, value) }
}

/// Reduces the `f32` `value` across the threads of `mask` with the operation `OP` and returns the
/// result to all of them (`redux.sync.{min,max}.f32`).
///
/// `OP` must be [`ReduxOp::Min`] or [`ReduxOp::Max`]. If `ABS` is [`Absolute::Enabled`], the
/// absolute values of the inputs are reduced. With [`NanHandling::Ignore`], NaN inputs are ignored
/// unless all inputs are NaN; with [`NanHandling::Propagate`], any NaN input makes the result
/// NaN.
///
/// Requires the `sm_100f` family or `sm_100a`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-redux-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same reduction with the same `mask`.
#[cfg(target_feature = "sm_100f")]
#[inline(always)]
pub unsafe fn reduce_f32<const OP: ReduxOp, const ABS: Absolute, const NAN: NanHandling>(
    mask: LaneMask,
    value: f32,
) -> f32 {
    const {
        assert!(
            matches!(OP, ReduxOp::Min | ReduxOp::Max),
            "floating-point reductions only support Min and Max"
        )
    };
    const_map!(OP => O: nv::ReduxOp {
        ReduxOp::Min => nv::ReduxOp::Min,
        ReduxOp::Max => nv::ReduxOp::Max,
    } in const_bool!(A = matches!(ABS, Absolute::Enabled) => const_bool!(
        N = matches!(NAN, NanHandling::Propagate) =>
        unsafe { nv::redux_sync_f32::<O, A, N>(value, mask.0) }
    )))
}

/// The result of `elect`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Election {
    /// The lane index of the elected thread.
    pub leader: u32,
    /// Whether the executing thread is the elected one.
    pub is_leader: bool,
}

/// Elects one thread among the threads of `mask` (`elect.sync`).
///
/// The elected thread is the same for all threads in `mask`.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-elect-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same `elect` with the same `mask`.
#[cfg(target_feature = "sm_90")]
#[inline(always)]
pub unsafe fn elect(mask: LaneMask) -> Election {
    let (leader, is_leader) = unsafe { nv::elect_sync(mask.0) };
    Election { leader, is_leader }
}
