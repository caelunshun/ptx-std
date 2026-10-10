// --- LLM-generated --- //
//! Tensor Memory allocation and deallocation (`tcgen05.alloc`, `tcgen05.dealloc`,
//! `tcgen05.relinquish_alloc_permit`).

use core::arch::asm;

use super::{TmemAddr, assert_cta_group, cta_group_count};
use crate::modifiers::CtaGroup;

/// Allocates `columns` columns of Tensor Memory and writes the address of the allocation to
/// shared memory at `dst` (`tcgen05.alloc.cta_group::N.sync.aligned.b32`).
///
/// The allocation spans all 128 lanes of the columns; the written [`TmemAddr`] is the address of
/// lane 0 of its first column. `columns` must be a power of two in `32..=512`, and must not
/// increase between any two allocations in the execution order of the CTA.
///
/// This operation is *blocking*: if the requested amount of Tensor Memory is not available, it
/// blocks until it becomes available. The write to `dst` is a weak memory operation, so the
/// address must be made visible to other threads with the usual synchronization, e.g.
/// [`fence_before_thread_sync`](super::fence_before_thread_sync), a barrier and
/// [`fence_after_thread_sync`](super::fence_after_thread_sync) (see the allocation example in the
/// [module documentation](super#canonical-synchronization-patterns)).
///
/// With [`CtaGroup::One`], a single warp of the CTA performs the allocation. With
/// [`CtaGroup::Two`], one warp in each CTA of the CTA pair must collectively perform the
/// allocation, each passing a shared memory location `dst` of its own CTA, and the first warp to
/// execute the instruction may block until the warp of the peer CTA also executes it. The peer
/// CTA must be launched and its warp must eventually participate.
///
/// All Tensor Memory allocated by a kernel must be deallocated with [`dealloc`] before the kernel
/// exits. Once any thread of a CTA has executed [`relinquish_alloc_permit`], the CTA must not
/// allocate anymore.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit>
///
/// # Safety
///
/// - `dst` must point to a 4-byte aligned location in the shared memory of the executing CTA,
///   which is not accessed concurrently.
/// - `columns` must be a power of two in `32..=512`, and the same in all threads of the warp.
/// - The CTA has not executed [`relinquish_alloc_permit`].
/// - With [`CtaGroup::Two`], the peer CTA is launched and one of its warps executes the matching
///   `alloc`.
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn alloc<const CTA_GROUP: CtaGroup>(dst: *mut TmemAddr, columns: u32) {
    const { assert_cta_group(CTA_GROUP) };
    unsafe {
        asm!(
            "tcgen05.alloc.cta_group::{cg}.sync.aligned.b32 [{dst}], {columns};",
            cg = const cta_group_count(CTA_GROUP),
            dst = in(reg64) dst,
            columns = in(reg32) columns,
            options(nostack),
        )
    }
}

/// Deallocates the `columns` columns of Tensor Memory starting at `addr`
/// (`tcgen05.dealloc.cta_group::N.sync.aligned.b32`).
///
/// `addr` must be the address written by a previous [`alloc`] of the same CTA, and `columns` the
/// number of columns of that allocation. All Tensor Memory allocated by a kernel must be
/// deallocated before the kernel exits; the warps can exit immediately afterwards without further
/// synchronization.
///
/// With [`CtaGroup::Two`], one warp in each CTA of the CTA pair must collectively perform the
/// deallocation, and the instruction may block until the warp of the peer CTA also executes it.
/// Before deallocating, the issuing warp and the peer CTA warp must synchronize their Tensor
/// Memory accesses, i.e. ensure that there are no concurrent Tensor Memory accesses from the peer
/// CTA, for example with a cluster barrier. Because `dealloc` may or may not block, incorrect
/// synchronization leads to non-deterministic hangs. For example, the following hangs if the
/// `dealloc` of CTA 1 blocks until the warp of CTA 0 executes its `dealloc`, while CTA 0 waits at
/// the cluster barrier for CTA 1:
///
/// | CTA 0 warp | CTA 1 warp |
/// |---|---|
/// | `barrier.cluster.arrive` | `tcgen05.dealloc` |
/// | `barrier.cluster.wait` | `barrier.cluster.arrive` |
/// | `tcgen05.dealloc` | `barrier.cluster.wait` |
///
/// Instead, both CTAs should first synchronize with the cluster barrier, and then both warps
/// deallocate.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit>
///
/// # Safety
///
/// - `addr` and `columns` must describe an allocation made with [`alloc`] by the executing CTA
///   that has not been deallocated yet, and `columns` must be the same in all threads of the
///   warp.
/// - No outstanding or future operations may access the deallocated Tensor Memory.
/// - With [`CtaGroup::Two`], the peer CTA's accesses to the Tensor Memory have been synchronized
///   with the executing warp, and a warp of the peer CTA executes the matching `dealloc`.
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn dealloc<const CTA_GROUP: CtaGroup>(addr: TmemAddr, columns: u32) {
    const { assert_cta_group(CTA_GROUP) };
    unsafe {
        asm!(
            "tcgen05.dealloc.cta_group::{cg}.sync.aligned.b32 {addr}, {columns};",
            cg = const cta_group_count(CTA_GROUP),
            addr = in(reg32) addr.to_bits(),
            columns = in(reg32) columns,
            options(nostack),
        )
    }
}

/// Relinquishes the right of the executing CTA to allocate Tensor Memory
/// (`tcgen05.relinquish_alloc_permit.cta_group::N.sync.aligned`).
///
/// It is illegal for a CTA to [`alloc`] after any of its threads has executed this instruction.
/// It is typically executed by the allocating warp once all allocations are done, so that other
/// CTAs resident on the SM can allocate. Existing allocations stay valid and must still be
/// deallocated with [`dealloc`].
///
/// With [`CtaGroup::One`], a single warp of the CTA executes it. With [`CtaGroup::Two`], one warp
/// in each CTA of the CTA pair collectively executes it.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit>
///
/// # Safety
///
/// - The CTA does not [`alloc`] afterwards.
/// - With [`CtaGroup::Two`], the peer CTA is launched and one of its warps executes the matching
///   `relinquish_alloc_permit`.
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn relinquish_alloc_permit<const CTA_GROUP: CtaGroup>() {
    const { assert_cta_group(CTA_GROUP) };
    unsafe {
        asm!(
            "tcgen05.relinquish_alloc_permit.cta_group::{cg}.sync.aligned;",
            cg = const cta_group_count(CTA_GROUP),
            options(nostack),
        )
    }
}
