// --- LLM-generated --- //
//! Kernel and launch control: programmatic dependent launch, cluster launch control, exit, and register reallocation.

#[cfg(target_feature = "sm_100")]
use crate::{globals::Dim3, mbarrier::MBarrier};
use bytemuck::{Pod, Zeroable};
use core::{arch::nvptx as nv, marker::ConstParamTy};

/// Allows dependent grids, launched with programmatic dependent launch, to be scheduled once all
/// blocks of the current grid have issued this instruction or exited (`griddepcontrol.launch_dependents`).
///
/// This is only a scheduling hint for the current grid: the dependent grid may start before the
/// current grid completes, but is not guaranteed to. Repeated invocations by the threads of a
/// block have no additional effect beyond the first. A dependent grid that needs the results of
/// this grid must call [`wait_for_prerequisite_grids`].
///
/// A release fence preceding this instruction in program order synchronizes with the start of a
/// dependent grid if the fence has `gpu` scope and both grids are in the same memory
/// synchronization domain, or if the fence has `sys` scope.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol>
#[cfg(target_feature = "sm_90")]
#[inline]
pub fn launch_dependent_grids() {
    unsafe { nv::griddepcontrol::<{ nv::GriddepcontrolAction::LaunchDependents }>() }
}

/// Waits until all prerequisite grids of the current grid have completed and all of their memory
/// operations are performed and visible to the current grid (`griddepcontrol.wait`).
///
/// Must be used by a dependent grid whose prerequisite grid calls [`launch_dependent_grids`], for
/// correct functional execution. Does nothing if the grid was not launched with programmatic
/// dependent launch.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol>
#[cfg(target_feature = "sm_90")]
#[inline]
pub fn wait_for_prerequisite_grids() {
    unsafe { nv::griddepcontrol::<{ nv::GriddepcontrolAction::Wait }>() }
}

/// Terminates the executing thread (`exit`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#control-flow-instructions-exit>
#[inline]
pub fn exit() -> ! {
    unsafe { nv::exit() }
}

/// The 16-byte opaque response of a cluster launch cancellation request, written by
/// `try_cancel_launch` and decoded with `is_canceled` and
/// `first_cta_id`.
///
/// The response must live in 16-byte aligned shared memory of the block when passed to
/// `try_cancel_launch`, and must be read back from shared memory (after the completion of the
/// request has been observed on the mbarrier) before it is decoded. A zeroed response is never
/// reported as canceled.
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[repr(C, align(16))]
pub struct CancelResponse {
    bits: [u64; 2],
}

#[cfg(target_feature = "sm_100")]
impl CancelResponse {
    #[inline]
    fn as_u128(self) -> u128 {
        bytemuck::cast(self)
    }

    /// Returns whether a cluster launch was successfully canceled by the request that produced
    /// this response. If not, no cluster was left to cancel.
    ///
    /// Requires `sm_100`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel>
    #[inline]
    pub fn is_canceled(self) -> bool {
        unsafe { nv::clusterlaunchcontrol_query_cancel_is_canceled(self.as_u128()) }
    }

    /// Returns the block index, within the grid, of the first block of the canceled cluster, or
    /// `None` if the request did not cancel a cluster. No other successful response of the same
    /// grid contains the same block index.
    ///
    /// A kernel that cancels a pending cluster typically takes over the work of that cluster.
    ///
    /// Requires `sm_100`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-query-cancel>
    #[inline]
    pub fn first_cta_id(self) -> Option<Dim3> {
        if !self.is_canceled() {
            return None;
        }
        let response = self.as_u128();
        unsafe {
            Some(Dim3::new(
                nv::clusterlaunchcontrol_query_cancel_get_first_ctaid_x(response),
                nv::clusterlaunchcontrol_query_cancel_get_first_ctaid_y(response),
                nv::clusterlaunchcontrol_query_cancel_get_first_ctaid_z(response),
            ))
        }
    }
}

/// Which blocks receive the response of `try_cancel_launch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum CancelDelivery {
    /// Only the executing block receives the response.
    Block,
    /// The response is written to all blocks of the cluster (`.multicast::cluster::all`), each at
    /// the same offset within its shared memory as `response`, and the mbarrier of each block at
    /// the same offset as `barrier` is signaled. The writes are weak async-proxy writes.
    /// Undefined behavior if any block of the cluster has exited.
    ///
    /// Requires the `sm_100f`, `sm_101f`, `sm_110f` or `sm_120f` family.
    Cluster,
}

/// Asynchronously requests to cancel a cluster that has not launched yet
/// (`clusterlaunchcontrol.try_cancel.async`).
///
/// The 16-byte response is written to `response` through the async proxy, completing a 16-byte
/// transaction on `barrier` (at cluster scope). Once the corresponding phase of `barrier` has
/// completed, `response` can be read and decoded with [`CancelResponse::is_canceled`] and
/// [`CancelResponse::first_cta_id`]. The request atomically cancels a cluster that has not
/// started running yet, if there is one.
///
/// Requires `sm_100`; [`CancelDelivery::Cluster`] additionally requires the `sm_100f`,
/// `sm_101f`, `sm_110f` or `sm_120f` family.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel>
///
/// # Safety
///
/// * `response` must be 16-byte aligned, point to shared memory of the executing block, and must
///   not be accessed until the barrier phase completes.
/// * `barrier` must be an initialized mbarrier in shared memory of the executing block, expecting
///   a 16-byte transaction in its current phase (e.g. by an `arrive.expect_tx` of 16 bytes).
/// * With [`CancelDelivery::Cluster`], the same holds for the corresponding locations in every
///   block of the cluster, and no block of the cluster may have exited.
/// * Once the executing block has observed a request to be unsuccessful (see
///   [`CancelResponse::is_canceled`]), it must not issue another request.
/// * Before reusing `response` for a subsequent request, the generic-proxy reads of the previous
///   response must be ordered before the async-proxy write (e.g. with an async-proxy fence and a
///   barrier, as in the PTX documentation example).
#[cfg(target_feature = "sm_100")]
#[inline]
pub unsafe fn try_cancel_launch<const DELIVERY: CancelDelivery>(
    response: *mut CancelResponse,
    barrier: &MBarrier,
) {
    const {
        assert!(
            !matches!(DELIVERY, CancelDelivery::Cluster)
                || cfg!(any(
                    target_feature = "sm_100f",
                    target_feature = "sm_101f",
                    target_feature = "sm_110f",
                    target_feature = "sm_120f",
                )),
            "multicast cluster launch cancellation requires the sm_100f, sm_101f, sm_110f or sm_120f family"
        )
    };
    let response = response.cast::<u128>();
    if const { matches!(DELIVERY, CancelDelivery::Block) } {
        unsafe { nv::clusterlaunchcontrol_try_cancel_async(response, barrier.as_ptr()) }
    } else {
        unsafe { nv::clusterlaunchcontrol_try_cancel_async_multicast(response, barrier.as_ptr()) }
    }
}

/// Hints that the executing warp should own `COUNT` registers per thread from now on, requesting
/// additional registers from the pool released by other warps (`setmaxnreg.inc`).
///
/// `COUNT` must be a multiple of 8 in `24..=256`. The warp blocks until enough registers are
/// available in the CTA's register pool. The contents of the newly acquired registers are
/// undefined; they must be initialized before use. This is only a hint, and may have no effect
/// if the kernel was not launched with a valid maximum register count.
///
/// Requires `sm_90a`, or the `sm_100f`, `sm_101f`, `sm_110f` or `sm_120f` family.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg>
///
/// # Safety
///
/// All threads of the executing warp must execute this instruction together, and all warps of the
/// executing warpgroup must execute the same `setmaxnreg` with the same `COUNT`; the warps of
/// the warpgroup must synchronize explicitly before executing any further `setmaxnreg`. The
/// number of registers currently owned by the warp must be at most `COUNT`.
#[cfg(any(
    target_feature = "sm_90a",
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f",
    target_feature = "sm_120f",
))]
#[inline]
pub unsafe fn increase_max_registers<const COUNT: u32>() {
    const {
        assert!(
            COUNT >= 24 && COUNT <= 256 && COUNT.is_multiple_of(8),
            "COUNT must be a multiple of 8 in the range 24..=256"
        )
    };
    unsafe { nv::setmaxnreg::<{ nv::SetmaxnregAction::Inc }, COUNT>() }
}

/// Releases registers so that the executing warp owns only `COUNT` registers per thread from now
/// on, making the rest available to other warps (`setmaxnreg.dec`).
///
/// `COUNT` must be a multiple of 8 in `24..=256`. The warp's code after this instruction must not
/// use more than `COUNT` registers per thread; the compiler takes care of this when allocating
/// registers. The released registers are added to the CTA's register pool. This is only a hint,
/// and may have no effect if the kernel was not launched with a valid maximum register count.
///
/// Requires `sm_90a`, or the `sm_100f`, `sm_101f`, `sm_110f` or `sm_120f` family.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg>
///
/// # Safety
///
/// All threads of the executing warp must execute this instruction together, and all warps of the
/// executing warpgroup must execute the same `setmaxnreg` with the same `COUNT`; the warps of
/// the warpgroup must synchronize explicitly before executing any further `setmaxnreg`. The
/// number of registers currently owned by the warp must be at least `COUNT`.
#[cfg(any(
    target_feature = "sm_90a",
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f",
    target_feature = "sm_120f",
))]
#[inline]
pub unsafe fn decrease_max_registers<const COUNT: u32>() {
    const {
        assert!(
            COUNT >= 24 && COUNT <= 256 && COUNT.is_multiple_of(8),
            "COUNT must be a multiple of 8 in the range 24..=256"
        )
    };
    unsafe { nv::setmaxnreg::<{ nv::SetmaxnregAction::Dec }, COUNT>() }
}
