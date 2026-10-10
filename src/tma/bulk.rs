// --- LLM-generated --- //
//! Non-tensor bulk asynchronous copies (`cp.async.bulk`).
//!
//! These copy a contiguous range of bytes between global and shared memory. Pointers to shared
//! memory are ordinary generic-address-space pointers. Source and destination addresses must be
//! 16-byte aligned, and sizes must be a multiple of 16 bytes; the source and destination ranges
//! must not overflow their memory spaces. Otherwise the behavior is undefined.
//!
//! Copies into shared memory signal completion on an [`MBarrier`]: the barrier must expect the
//! number of bytes copied (`expect_tx`) before the copy completes. Copies out of shared memory
//! are tracked with bulk async-groups ([`bulk_commit_group`], [`bulk_wait_group`]).
//!
//! Each thread that executes one of these operations issues its own copy; typically a single
//! elected thread does so. The copies are performed by the async proxy.

#[allow(unused_imports)]
use core::arch::nvptx as nv;
#[allow(unused_imports)]
use core::ffi::c_void;

#[allow(unused_imports)]
use super::CtaMask;
#[cfg(target_feature = "sm_90")]
#[allow(unused_imports)]
use super::{bulk_commit_group, bulk_wait_group};
#[allow(unused_imports)]
use crate::mbarrier::MBarrier;
#[allow(unused_imports)]
use crate::memory::CachePolicy;

/// Asynchronously copies `size` bytes from global memory at `src` to shared memory of the
/// executing CTA at `dst`, and performs a complete-transaction of `size` bytes on `mbar`
/// when the copy completes (`cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes`).
///
/// With `cache_policy`, the L2 eviction policy is applied to the read (`.L2::cache_hint`).
///
/// Requires `sm_90` and PTX ISA 8.6.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>
///
/// # Safety
///
/// - `dst` must point to `size` bytes of shared memory of the executing CTA, and `src` to `size`
///   bytes of global memory, both 16-byte aligned.
/// - `size` must be a multiple of 16.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through `mbar`.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_load(
    dst: *mut u8,
    src: *const u8,
    size: u32,
    mbar: &MBarrier,
    cache_policy: Option<CachePolicy>,
) {
    let mbar = mbar.as_ptr().cast();
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_global_to_shared_cta::<CH>(dst.cast(), mbar, src.cast(), size, policy)
    })
}

/// Asynchronously copies `size` bytes from global memory at `src` to shared memory of a CTA of the
/// executing cluster at `dst`, and performs a complete-transaction of `size` bytes on `mbar` when
/// the copy completes
/// (`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`).
///
/// If `multicast` is given, the data is copied to the same `dst` offset in the shared memory of
/// each CTA of the cluster selected by the mask, and the complete-transaction is performed on the
/// mbarrier at the same offset as `mbar` in each of them (`.multicast::cluster`). Otherwise the
/// data is copied to the CTA that `dst` points to, and `mbar` itself is signaled.
///
/// With `cache_policy`, the L2 eviction policy is applied to the read (`.L2::cache_hint`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>
///
/// # Safety
///
/// - `dst` must be a generic address of shared memory of a CTA in the executing cluster, valid for
///   `size` bytes, and `src` must point to `size` bytes of global memory, both 16-byte aligned.
/// - `size` must be a multiple of 16.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA (and, when
///   multicasting, the mbarrier at the same offset must be initialized in each destination CTA).
/// - Every CTA selected by `multicast` must be part of the executing cluster and still running.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_load_to_cluster(
    dst: *mut u8,
    src: *const u8,
    size: u32,
    mbar: &MBarrier,
    multicast: Option<CtaMask>,
    cache_policy: Option<CachePolicy>,
) {
    let mbar = mbar.as_ptr().cast();
    with_multicast!(multicast, MC, mask => with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_global_to_shared_cluster::<CH, MC>(
            dst.cast(),
            mbar,
            src.cast(),
            size,
            mask,
            policy,
        )
    }))
}

/// Asynchronously copies `size` bytes from shared memory of the executing CTA at `src` to shared
/// memory of a different CTA of the executing cluster at `dst`, and performs a
/// complete-transaction of `size` bytes on `mbar` when the copy completes
/// (`cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>
///
/// # Safety
///
/// - `src` must point to `size` bytes of shared memory of the executing CTA, and `dst` must be a
///   generic address of shared memory of a CTA in the executing cluster other than the executing
///   CTA, valid for `size` bytes, both 16-byte aligned.
/// - `size` must be a multiple of 16.
/// - `mbar` must be the generic address of an initialized mbarrier in shared memory of the
///   destination CTA, which expects `size` bytes.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_copy_to_cluster(dst: *mut u8, src: *const u8, size: u32, mbar: &MBarrier) {
    unsafe {
        nv::cp_async_bulk_shared_cta_to_cluster(dst.cast(), mbar.as_ptr().cast(), src.cast(), size)
    }
}

/// Asynchronously copies `size` bytes from shared memory of the executing CTA at `src` to global
/// memory at `dst` (`cp.async.bulk.global.shared::cta.bulk_group`).
///
/// The copy is tracked by the bulk async-group mechanism: call [`bulk_commit_group`] and then
/// [`bulk_wait_group`] to wait for it. With `cache_policy`, the L2 eviction policy is applied to
/// the write (`.L2::cache_hint`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>
///
/// # Safety
///
/// - `src` must point to `size` bytes of shared memory of the executing CTA, and `dst` to `size`
///   bytes of global memory, both 16-byte aligned.
/// - `size` must be a multiple of 16.
/// - Prior generic-proxy writes to `src` must have been made visible to the async proxy
///   (`fence.proxy.async`), and `src` must not be modified until the copy has finished reading
///   it (`bulk_wait_group::<_, { BulkWait::SourceRead }>`).
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_store(
    dst: *mut u8,
    src: *const u8,
    size: u32,
    cache_policy: Option<CachePolicy>,
) {
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_shared_cta_to_global::<CH>(dst.cast(), src.cast(), size, policy)
    })
}

/// Like [`bulk_store`], but only writes the bytes of `src` selected by `byte_mask`
/// (`cp.async.bulk.global.shared::cta.bulk_group.cp_mask`).
///
/// For each 16-byte chunk of the source, byte `i` is written only if bit `i` of `byte_mask` is
/// set.
///
/// Requires `sm_100` and PTX ISA 8.6.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>
///
/// # Safety
///
/// Same requirements as [`bulk_store`].
#[cfg(target_feature = "sm_100")]
#[inline]
pub unsafe fn bulk_store_masked(
    dst: *mut u8,
    src: *const u8,
    size: u32,
    byte_mask: u16,
    cache_policy: Option<CachePolicy>,
) {
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_shared_cta_to_global_bytemask::<CH>(
            dst.cast(),
            src.cast(),
            size,
            policy,
            byte_mask,
        )
    })
}

/// Prefetches `size` bytes of global memory starting at `src` into the L2 cache
/// (`cp.async.bulk.prefetch.L2.global`).
///
/// This is only a hint, which may be ignored; it is treated as a weak memory operation and is
/// not tracked by any completion mechanism. With `cache_policy`, the L2 eviction policy is
/// applied to the prefetched data (`.L2::cache_hint`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch>
///
/// # Safety
///
/// `src` must be a 16-byte aligned pointer into global memory, and `size` a multiple of 16.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_prefetch_l2(src: *const u8, size: u32, cache_policy: Option<CachePolicy>) {
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_prefetch_l2::<CH>(src.cast(), size, policy)
    })
}
