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

use crate::data_types::array_from_fn;
#[cfg(target_feature = "sm_100f")]
use crate::{f32xN, modifiers::NanHandling};
use bytemuck::{Pod, Zeroable};
use core::{
    arch::nvptx as nv,
    marker::ConstParamTy,
    mem::size_of,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
    simd::Simd,
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
/// guarantees memory ordering among the participating threads (`bar.warp.sync`), so threads of
/// the warp can communicate through memory by storing, calling `sync_warp`, and then reading the
/// values stored by the other threads.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-warp-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all threads in `mask` must execute the
/// same `sync_warp` with the same `mask`.
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

/// Applies `f` to every 32-bit word of `value`. A final partial word is zero-extended before it is
/// passed to `f`, and only its low bytes are kept from the result.
#[inline(always)]
fn map_words<T: Pod>(value: T, mut f: impl FnMut(u32) -> u32) -> T {
    const { assert!(size_of::<T>() != 0, "zero-sized types cannot be exchanged") };
    let mut out = T::zeroed();
    let src = (&value as *const T).cast::<u8>();
    let dst = (&mut out as *mut T).cast::<u8>();
    let words = size_of::<T>() / 4;
    let mut i = 0;
    #[rustc_unroll(full)]
    while i < words {
        unsafe {
            let word = src.add(4 * i).cast::<u32>().read_unaligned();
            dst.add(4 * i).cast::<u32>().write_unaligned(f(word));
        }
        i += 1;
    }
    let rem = size_of::<T>() % 4;
    if rem != 0 {
        let word = f(unsafe { read_partial_word(src.add(4 * words), rem) });
        unsafe {
            core::ptr::copy_nonoverlapping(
                (&word as *const u32).cast::<u8>(),
                dst.add(4 * words),
                rem,
            )
        };
    }
    out
}

/// Reads `len < 4` bytes at `src` into the low bytes of a zeroed word.
#[inline(always)]
unsafe fn read_partial_word(src: *const u8, len: usize) -> u32 {
    let mut word = 0u32;
    unsafe { core::ptr::copy_nonoverlapping(src, (&mut word as *mut u32).cast::<u8>(), len) };
    word
}

/// A chunk of a value compared by `match.sync`.
enum MatchChunk {
    B32(u32),
    B64(u64),
}

/// Calls `f` on every 64-bit chunk of `value`, then on 32-bit chunks for the remaining bytes (a
/// final partial chunk is zero-extended).
#[inline(always)]
fn for_each_match_chunk<T: Pod>(value: &T, mut f: impl FnMut(MatchChunk)) {
    const { assert!(size_of::<T>() != 0, "zero-sized types cannot be matched") };
    let src = (value as *const T).cast::<u8>();
    let chunks = size_of::<T>() / 8;
    let mut i = 0;
    #[rustc_unroll(full)]
    while i < chunks {
        f(MatchChunk::B64(unsafe {
            src.add(8 * i).cast::<u64>().read_unaligned()
        }));
        i += 1;
    }
    let mut offset = 8 * chunks;
    if size_of::<T>() - offset >= 4 {
        f(MatchChunk::B32(unsafe {
            src.add(offset).cast::<u32>().read_unaligned()
        }));
        offset += 4;
    }
    if offset < size_of::<T>() {
        f(MatchChunk::B32(unsafe {
            read_partial_word(src.add(offset), size_of::<T>() - offset)
        }));
    }
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
///   that lies above the end of the segment, the lane receives its own `value`. Note that a
///   source lane in a *lower* segment is not considered out of range and is read normally.
///
/// Only the low 5 bits of `operand` are used. For [`ShuffleMode::Index`], the bits of `operand`
/// that select a segment (above `log2(WIDTH)`) are ignored, so the index wraps within the
/// segment, and the source lane is therefore always in range.
///
/// If the source lane is not in `mask` or has exited, the received value is undefined.
///
/// `T` may be any non-zero-sized plain-data type, such as the scalar and vector floating-point types
/// of this crate or `core::simd` vectors. It is shuffled one 32-bit word at a time; a final
/// partial word (e.g. of a 1- or 2-byte type) is zero-extended. `WIDTH` must be a power of two in
/// `1..=32`. See [`try_shuffle_segmented`] to distinguish the out-of-range case.
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

/// Like [`shuffle_segmented`], but returns `None` if the source lane is out of range (in which
/// case [`shuffle_segmented`] would return the lane's own value). This is the destination
/// predicate of `shfl.sync`. Never returns `None` for [`ShuffleMode::Index`].
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
/// index below `delta` receive their own `value`. Only the low 5 bits of `delta` are used.
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
/// source would be past lane 31 receive their own `value`. Only the low 5 bits of `delta` are
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

/// Returns whether `predicate` is true for all non-exited threads in `mask` (`vote.sync.all`).
/// The result is the same for all threads in `mask`.
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

/// Returns whether `predicate` is true for any non-exited thread in `mask` (`vote.sync.any`). The
/// result is the same for all threads in `mask`.
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

/// Returns whether `predicate` has the same value (all true or all false) in all non-exited
/// threads in `mask` (`vote.sync.uni`). The result is the same for all threads in `mask`.
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

/// Returns the set of threads in `mask` for which `predicate` is true (`vote.sync.ballot`). Bit
/// `i` of the result is the predicate of lane `i`; lanes not in `mask` contribute 0. The result
/// is the same for all threads in `mask`.
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

/// Returns the set of non-exited threads in `mask` whose `value` is bitwise equal to the executing
/// thread's (`match.any.sync`).
///
/// `T` may be any non-zero-sized plain-data type; floating-point values are compared by bit
/// pattern. Values are compared in 64-bit chunks with `match.any.sync.b64`, and the remaining bytes
/// in 32-bit chunks with `match.any.sync.b32` (a final partial chunk is zero-extended); the result is the intersection of
/// the lane sets of all chunks.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same match with the same `mask`.
#[inline(always)]
pub unsafe fn match_any<T: Pod>(mask: LaneMask, value: T) -> LaneMask {
    let mut lanes = LaneMask::FULL;
    for_each_match_chunk(&value, |chunk| {
        lanes &= LaneMask(unsafe {
            match chunk {
                MatchChunk::B32(x) => nv::match_any_sync_b32(x, mask.0),
                MatchChunk::B64(x) => nv::match_any_sync_b64(x, mask.0),
            }
        });
    });
    lanes
}

/// Returns whether `value` is bitwise equal in all non-exited threads in `mask`
/// (`match.all.sync`). The result is the same for all threads in `mask`.
///
/// `T` may be any non-zero-sized plain-data type; floating-point values are compared by bit
/// pattern. Values are compared in 64-bit chunks with `match.all.sync.b64`, and the remaining bytes
/// in 32-bit chunks with `match.all.sync.b32` (a final partial chunk is zero-extended); the result is true if all chunks
/// are equal.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same match with the same `mask`.
#[inline(always)]
pub unsafe fn match_all<T: Pod>(mask: LaneMask, value: T) -> bool {
    let mut equal = true;
    for_each_match_chunk(&value, |chunk| {
        equal &= unsafe {
            match chunk {
                MatchChunk::B32(x) => nv::match_all_sync_b32(x, mask.0).1,
                MatchChunk::B64(x) => nv::match_all_sync_b64(x, mask.0).1,
            }
        };
    });
    equal
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
    use crate::f32xN;
    use core::simd::Simd;

    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
    impl<const N: usize> Sealed for Simd<u32, N> {}
    impl<const N: usize> Sealed for Simd<i32, N> {}
    impl<const N: usize> Sealed for f32xN<N> {}
}

/// Integer types that can be reduced across a warp with [`reduce`]: `u32`, `i32`, and `Simd`
/// vectors of them, whose lanes are reduced independently (one `redux.sync` per lane).
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

macro_rules! redux_simd {
    ($($elem:ty),*) => {$(
        impl<const N: usize> ReduxInt for Simd<$elem, N> {
            #[inline(always)]
            unsafe fn reduce<const OP: ReduxOp>(mask: LaneMask, value: Self) -> Self {
                let value = value.to_array();
                Simd::from_array(array_from_fn(|i| unsafe {
                    <$elem as ReduxInt>::reduce::<OP>(mask, value[i])
                }))
            }
        }
    )*};
}

redux_simd!(u32, i32);

/// Reduces `value` across the non-exited threads of `mask` with the operation `OP` and returns the
/// result to all of them (`redux.sync`). Works on `u32` and `i32` and on `Simd` vectors of them,
/// whose lanes are reduced independently with one `redux.sync` each; all [`ReduxOp`]s are
/// supported. The result of [`ReduxOp::Add`] is truncated to 32 bits.
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

/// Floating-point types that can be reduced across a warp with [`reduce_f32`]: `f32` and the `f32`
/// vectors [`f32xN<N>`](crate::f32xN), whose elements are reduced independently (one
/// `redux.sync` per element).
#[cfg(target_feature = "sm_100f")]
pub trait ReduxF32: Copy + sealed::Sealed {
    #[doc(hidden)]
    unsafe fn reduce<const OP: ReduxOp, const ABS: Absolute, const NAN: NanHandling>(
        mask: LaneMask,
        value: Self,
    ) -> Self;
}

#[cfg(target_feature = "sm_100f")]
impl ReduxF32 for f32 {
    #[inline(always)]
    unsafe fn reduce<const OP: ReduxOp, const ABS: Absolute, const NAN: NanHandling>(
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
}

#[cfg(target_feature = "sm_100f")]
impl<const N: usize> ReduxF32 for f32xN<N> {
    #[inline(always)]
    unsafe fn reduce<const OP: ReduxOp, const ABS: Absolute, const NAN: NanHandling>(
        mask: LaneMask,
        value: Self,
    ) -> Self {
        let value = value.to_array();
        f32xN::from_array(array_from_fn(|i| unsafe {
            f32::reduce::<OP, ABS, NAN>(mask, value[i])
        }))
    }
}

/// Reduces the `f32` `value` (or each element of an `f32xN<N>` independently) across the
/// non-exited threads of `mask` with the operation `OP` and returns the result to all of them
/// (`redux.sync.{min,max}{.abs}{.NaN}.f32`, one per element).
///
/// `OP` must be [`ReduxOp::Min`] or [`ReduxOp::Max`]. If `ABS` is [`Absolute::Enabled`], the
/// absolute values of the inputs are reduced. With [`NanHandling::Ignore`], NaN inputs are ignored
/// unless all inputs are NaN; with [`NanHandling::Propagate`], any NaN input makes the result
/// NaN. In both cases a NaN result is the canonical NaN. +0.0 is considered greater than -0.0.
///
/// Requires `sm_100a` (PTX ISA 8.6) or the `sm_100f` family (PTX ISA 8.8).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-redux-sync>
///
/// # Safety
///
/// The executing thread must be a member of `mask`, and all non-exited threads in `mask` must
/// execute the same reduction with the same `mask`.
#[cfg(target_feature = "sm_100f")]
#[inline(always)]
pub unsafe fn reduce_f32<
    const OP: ReduxOp,
    const ABS: Absolute,
    const NAN: NanHandling,
    T: ReduxF32,
>(
    mask: LaneMask,
    value: T,
) -> T {
    unsafe { T::reduce::<OP, ABS, NAN>(mask, value) }
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
/// The elected thread is the same for all threads in `mask`, and the election is deterministic:
/// the same leader is elected for the same `mask` every time.
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
