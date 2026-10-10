// --- LLM-generated --- //
//! Ampere-style asynchronous global-to-shared copies (`cp.async`).
//!
//! Each thread issues copies of 4, 8 or 16 bytes with [`copy`] or [`copy_zfill`], groups the
//! copies issued so far with [`commit_group`], and later waits for groups to complete with
//! [`wait_group`] or [`wait_all`]. Completion can alternatively be tracked with an mbarrier.
//!
//! Note that the data is not visible to other threads of the CTA after [`wait_group`] or
//! [`wait_all`] alone: a CTA-wide barrier is still needed.
//!
//! Requires `sm_80`.

use core::{arch::nvptx as nv, marker::ConstParamTy, mem::size_of};

/// The cache operator of an asynchronous copy. Cache operators are performance hints only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum CacheOperator {
    /// Cache the data at all levels, including L1, as it is likely to be accessed again (`.ca`).
    ///
    /// Supported for copies of 4, 8 and 16 bytes.
    AllLevels,
    /// Cache the data only in L2, bypassing L1 (`.cg`).
    ///
    /// Only supported for copies of 16 bytes.
    L2Only,
}

#[inline(always)]
const fn assert_copy_supported<T, const CACHE: CacheOperator>() {
    assert!(
        matches!(size_of::<T>(), 4 | 8 | 16),
        "cp.async can only copy 4, 8 or 16 bytes"
    );
    assert!(
        !matches!(CACHE, CacheOperator::L2Only) || size_of::<T>() == 16,
        "CacheOperator::L2Only (.cg) can only be used for 16-byte copies"
    );
}

/// Asynchronously copies a `T` (4, 8 or 16 bytes) from global memory at `src` to shared memory
/// of the executing CTA at `dst`.
///
/// `CACHE` selects the cache operator; [`CacheOperator::L2Only`] requires a 16-byte `T`. The copy
/// is a weak memory operation, performed in the generic proxy, and is only guaranteed to have
/// completed after waiting for it with [`wait_group`], [`wait_all`] or an mbarrier. Two copies
/// within the same async group must not write to the same location.
///
/// # Safety
///
/// - `src` must point into global memory and `dst` into the shared memory of the executing CTA.
/// - Both must be valid for the access and aligned to `size_of::<T>()` bytes.
/// - The destination must not be accessed by other operations until the copy has completed.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async>
#[inline]
pub unsafe fn copy<T, const CACHE: CacheOperator>(dst: *mut T, src: *const T) {
    const { assert_copy_supported::<T, CACHE>() };
    let (dst, src) = (dst.cast(), src.cast());
    unsafe {
        match (CACHE, size_of::<T>()) {
            (CacheOperator::AllLevels, 4) => nv::cp_async_ca::<4>(dst, src),
            (CacheOperator::AllLevels, 8) => nv::cp_async_ca::<8>(dst, src),
            (CacheOperator::AllLevels, 16) => nv::cp_async_ca::<16>(dst, src),
            (CacheOperator::L2Only, 16) => nv::cp_async_cg(dst, src),
            _ => unreachable!(),
        }
    }
}

/// Like [`copy`], but only `src_bytes` bytes are read from `src`; the remaining bytes of the
/// destination are filled with zeros.
///
/// `src_bytes` is typically used for predication or to handle out-of-bounds tails: passing 0
/// zero-fills the destination without reading from `src`.
///
/// # Safety
///
/// The safety requirements of [`copy`] apply, except that only `src_bytes` bytes of `src` need
/// to be valid for reads. Additionally, `src_bytes` must not exceed `size_of::<T>()`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async>
#[inline]
pub unsafe fn copy_zfill<T, const CACHE: CacheOperator>(
    dst: *mut T,
    src: *const T,
    src_bytes: u32,
) {
    const { assert_copy_supported::<T, CACHE>() };
    let (dst, src) = (dst.cast(), src.cast());
    unsafe {
        match (CACHE, size_of::<T>()) {
            (CacheOperator::AllLevels, 4) => nv::cp_async_ca_src_size::<4>(dst, src, src_bytes),
            (CacheOperator::AllLevels, 8) => nv::cp_async_ca_src_size::<8>(dst, src, src_bytes),
            (CacheOperator::AllLevels, 16) => nv::cp_async_ca_src_size::<16>(dst, src, src_bytes),
            (CacheOperator::L2Only, 16) => nv::cp_async_cg_src_size(dst, src, src_bytes),
            _ => unreachable!(),
        }
    }
}

/// Commits all prior uncommitted `cp.async` operations of the executing thread into a new
/// async group.
///
/// If there are no uncommitted operations, an empty group is created.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group>
#[inline]
pub fn commit_group() {
    // SAFETY: only affects the executing thread's own async group bookkeeping.
    unsafe { nv::cp_async_commit_group() }
}

/// Waits until at most `PENDING` of the most recently committed `cp.async` groups of the
/// executing thread are still pending, i.e. all older groups have completed.
///
/// Use `wait_group::<0>()` to wait for all committed groups. Uncommitted `cp.async` operations
/// are not waited for. Once a group has completed, the writes of its copies are visible to the
/// executing thread only; this provides no ordering or visibility for other memory operations.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group>
#[inline]
pub fn wait_group<const PENDING: u32>() {
    // SAFETY: only waits for the executing thread's own async copies.
    unsafe { nv::cp_async_wait_group::<PENDING>() }
}

/// Waits until all prior `cp.async` operations of the executing thread have completed, whether
/// committed to a group or not (it is equivalent to [`commit_group`] followed by
/// `wait_group::<0>()`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group>
#[inline]
pub fn wait_all() {
    // SAFETY: only waits for the executing thread's own async copies.
    unsafe { nv::cp_async_wait_all() }
}
