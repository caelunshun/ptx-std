// --- LLM-generated --- //
//! Specialized fences (`tcgen05.fence`) and asynchronous completion tracking (`tcgen05.commit`).

use core::arch::asm;

use super::{assert_cta_group, cta_group_count};
use crate::{MBarrier, modifiers::CtaGroup, tma::CtaMask};

/// Orders all prior asynchronous `tcgen05` operations of the executing thread before all
/// subsequent `tcgen05` operations and execution ordering operations (barriers, mbarrier
/// operations, morally strong memory operations, ...) of the thread
/// (`tcgen05.fence::before_thread_sync`).
///
/// Used by a thread before synchronizing with other threads that will access the Tensor Memory
/// accessed by the prior operations. It also prevents the compiler from moving prior `tcgen05`
/// operations below it.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence>
#[inline(always)]
pub fn fence_before_thread_sync() {
    // SAFETY: the fence has no preconditions.
    unsafe { asm!("tcgen05.fence::before_thread_sync;", options(nostack)) }
}

/// Orders all subsequent asynchronous `tcgen05` operations of the executing thread after all
/// prior `tcgen05` operations and execution ordering operations (barriers, mbarrier operations,
/// morally strong memory operations, ...) of the thread (`tcgen05.fence::after_thread_sync`).
///
/// Used by a thread after synchronizing with other threads whose prior Tensor Memory accesses must
/// be ordered before the subsequent operations. It also prevents the compiler from moving
/// subsequent `tcgen05` operations above it.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-special-sync-operations-fence>
#[inline(always)]
pub fn fence_after_thread_sync() {
    // SAFETY: the fence has no preconditions.
    unsafe { asm!("tcgen05.fence::after_thread_sync;", options(nostack)) }
}

/// Makes `mbar` track the completion of all prior asynchronous `tcgen05` operations
/// ([`mma`](super::mma) and its variants, [`copy`](super::copy) and
/// [`shift_down`](super::shift_down)) with the same `CTA_GROUP` initiated by the executing
/// thread (`tcgen05.commit.cta_group::N.mbarrier::arrive::one.b64`).
///
/// Once the tracked operations have completed, the system performs an arrive-on operation with
/// count 1 on `mbar`, at cluster scope. Waiting on the barrier thus observes their completion. The
/// commit is issued by a single thread and does not wait for the tracked operations; it
/// implicitly performs [`fence_before_thread_sync`], so that the prior `tcgen05` operations are
/// ordered before the arrive-on.
///
/// `mbar` is accessed through the generic proxy, and addressed with generic addressing. It can
/// therefore also be a barrier in the shared memory of another CTA of the cluster, e.g. one
/// obtained with [`MBarrier::map_to_cta_rank`]; see [`commit_multicast`] to signal several CTAs.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit>
///
/// # Safety
///
/// - `mbar` must be an initialized mbarrier in the shared memory of a CTA of the executing
///   cluster, which expects the arrive-on operation (see the
///   [safety contract](MBarrier#safety-contract-of-all-operations) of mbarrier operations).
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn commit<const CTA_GROUP: CtaGroup>(mbar: &MBarrier) {
    const { assert_cta_group(CTA_GROUP) };
    unsafe {
        asm!(
            "tcgen05.commit.cta_group::{cg}.mbarrier::arrive::one.b64 [{mbar}];",
            cg = const cta_group_count(CTA_GROUP),
            mbar = in(reg64) mbar.as_ptr(),
            options(nostack),
        )
    }
}

/// Like [`commit`], but the arrive-on operation is signaled on the mbarriers at the same offset
/// as `mbar` in the shared memory of each CTA selected by `ctas`
/// (`tcgen05.commit.cta_group::N.mbarrier::arrive::one.multicast::cluster.b64`).
///
/// Bit `i` of `ctas` selects the CTA with `%cluster_ctarank` `i`; the mask is 16 bits wide. `mbar`
/// only determines the offset of the signaled barriers: the barrier at that offset in the
/// executing CTA is signaled only if the executing CTA is selected.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen-async-sync-operations-commit>
///
/// # Safety
///
/// - `mbar` is in the shared memory of a CTA of the executing cluster.
/// - Every CTA selected by `ctas` is part of the executing cluster and still running, and has an
///   initialized mbarrier at the same offset as `mbar` that expects the arrive-on operation (see
///   the [safety contract](MBarrier#safety-contract-of-all-operations) of mbarrier operations).
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn commit_multicast<const CTA_GROUP: CtaGroup>(mbar: &MBarrier, ctas: CtaMask) {
    const { assert_cta_group(CTA_GROUP) };
    unsafe {
        asm!(
            "tcgen05.commit.cta_group::{cg}.mbarrier::arrive::one.multicast::cluster.b64 [{mbar}], {ctas};",
            cg = const cta_group_count(CTA_GROUP),
            mbar = in(reg64) mbar.as_ptr(),
            ctas = in(reg16) ctas.to_bits(),
            options(nostack),
        )
    }
}
