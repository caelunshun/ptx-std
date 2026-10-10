// --- LLM-generated --- //
//! Thread block cluster operations: cluster barriers and distributed shared memory access.
//!
//! The CTAs of a cluster are identified by their rank in `0..cluster_size`. The shared memory of
//! every CTA of the cluster can be accessed by all threads of the cluster ("distributed shared
//! memory") through generic pointers obtained with `map_to_cta`.
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster>

#[cfg(target_feature = "sm_90")]
use core::arch::nvptx as nv;

#[cfg(target_feature = "sm_90")]
use crate::{barrier::BarrierAlignment, mbarrier::MBarrier, modifiers::MemoryOrdering};

/// Signals the arrival of the executing thread at the cluster barrier, without waiting for the
/// other threads of the cluster. Must be followed by [`barrier_wait`].
///
/// `SEM` is [`MemoryOrdering::Release`] (`barrier.cluster.arrive`; orders the thread's prior
/// memory accesses before the barrier completes) or [`MemoryOrdering::Relaxed`]
/// (`barrier.cluster.arrive.relaxed`; no ordering or visibility guarantees for prior memory
/// accesses, which need an explicit fence if ordering is required). There are no ordering
/// guarantees for the memory accesses between the arrive and the following [`barrier_wait`].
///
/// When all non-exited threads of the cluster have arrived, the barrier completes and is
/// automatically reinitialized, so a thread may arrive again right after its wait returns.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster>
///
/// # Safety
///
/// All non-exited threads of the cluster must arrive at the barrier for it to complete, and a
/// thread may only arrive once per barrier phase, before it waits with [`barrier_wait`]. With
/// [`BarrierAlignment::Aligned`], all threads of the warp must execute the same barrier
/// instruction.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn barrier_arrive<const SEM: MemoryOrdering, const ALIGNMENT: BarrierAlignment>() {
    const {
        assert!(
            matches!(SEM, MemoryOrdering::Release | MemoryOrdering::Relaxed),
            "cluster barrier arrive only supports MemoryOrdering::Release and MemoryOrdering::Relaxed"
        )
    }
    const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => const_map!(SEM => CM: nv::Semantics {
        MemoryOrdering::Release => nv::Semantics::Release,
        MemoryOrdering::Relaxed => nv::Semantics::Relaxed,
    } in unsafe { nv::barrier_cluster_arrive::<CM, ALIGNED>() }))
}

/// Waits until all non-exited threads of the cluster have arrived at the cluster barrier
/// ([`barrier_arrive`]) for the current barrier phase. Also provides acquire semantics: the memory
/// accesses (except asynchronous operations) requested before the arrivals, in program order, are
/// complete and visible to the executing thread afterwards (for [`MemoryOrdering::Release`]
/// arrivals; see [`barrier_arrive`]).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster>
///
/// # Safety
///
/// The executing thread must have arrived at the barrier in the current phase with
/// [`barrier_arrive`]. With [`BarrierAlignment::Aligned`], all threads of the warp must execute
/// the same barrier instruction.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn barrier_wait<const ALIGNMENT: BarrierAlignment>() {
    const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => unsafe {
        nv::barrier_cluster_wait::<ALIGNED>()
    })
}

/// Synchronizes all threads of the cluster: arrives at the cluster barrier with release
/// semantics and waits for all other threads with acquire semantics (the equivalent of
/// `cluster.sync()` in CUDA C++).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster>
///
/// # Safety
///
/// All non-exited threads of the cluster must execute this function (or the equivalent
/// [`barrier_arrive`] and [`barrier_wait`]). With [`BarrierAlignment::Aligned`], all threads of
/// the warp must execute the same barrier instructions.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn barrier_sync<const ALIGNMENT: BarrierAlignment>() {
    unsafe {
        barrier_arrive::<{ MemoryOrdering::Release }, ALIGNMENT>();
        barrier_wait::<ALIGNMENT>();
    }
}

/// Maps the address `ptr` of a shared memory location of the executing CTA to the generic
/// address of the corresponding location in the shared memory of the CTA with rank `cta_rank` in
/// the cluster (`mapa`).
///
/// The returned pointer can be dereferenced by any thread of the cluster (while the target CTA is
/// running), and can be used for the remote mbarrier operations of [`MBarrier`] and for
/// [`st_async`]. Mapping to the rank of the executing CTA yields a pointer to the same location.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-mapa>
///
/// # Safety
///
/// `ptr` must be a generic address in the shared memory window, and `cta_rank` must be less than
/// the number of CTAs in the cluster. The resulting pointer is only valid for access as long as
/// the target CTA has not exited and the usual data race rules for the accessed memory hold.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn map_to_cta<T>(ptr: *mut T, cta_rank: u32) -> *mut T {
    unsafe { nv::mapa::<{ nv::StateSpace::Generic }, T>(ptr, cta_rank) }
}

/// Returns the rank in the cluster of the CTA whose shared memory contains the address `ptr`
/// (`getctarank`). Inverse of [`map_to_cta`].
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-getctarank>
///
/// # Safety
///
/// `ptr` must be a generic address that falls within the shared memory of a CTA of the executing
/// cluster (an address in the shared memory window of the executing CTA, or an address obtained
/// from [`map_to_cta`]).
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn owning_cta_rank<T>(ptr: *const T) -> u32 {
    unsafe { nv::getctarank::<{ nv::StateSpace::Generic }, T>(ptr) }
}

mod sealed {
    #[cfg(target_feature = "sm_90")]
    use core::{arch::nvptx as nv, ffi::c_void};

    pub trait Sealed: Copy {
        #[cfg(target_feature = "sm_90")]
        unsafe fn st_async(dst: *mut Self, value: Self, mbar: *mut c_void);
    }

    macro_rules! impl_sealed {
        ($($ty:ty => $core_ty:ty, $core_fn:ident, $v:ident => $to_bits:expr;)*) => {$(
            impl Sealed for $ty {
                #[cfg(target_feature = "sm_90")]
                #[inline]
                unsafe fn st_async(dst: *mut Self, $v: Self, mbar: *mut c_void) {
                    unsafe { nv::$core_fn(dst.cast::<$core_ty>(), $to_bits, mbar) }
                }
            }
        )*};
    }

    impl_sealed! {
        u32 => u32, st_async_u32, v => v;
        i32 => u32, st_async_u32, v => v as u32;
        f32 => u32, st_async_u32, v => v.to_bits();
        u64 => u64, st_async_u64, v => v;
        i64 => u64, st_async_u64, v => v as u64;
        f64 => u64, st_async_u64, v => v.to_bits();
    }
}

/// Types that can be stored with `st_async`: 32-bit and 64-bit integers and floats. This trait
/// is sealed.
pub trait ClusterStoreValue: sealed::Sealed {}

impl ClusterStoreValue for u32 {}
impl ClusterStoreValue for i32 {}
impl ClusterStoreValue for f32 {}
impl ClusterStoreValue for u64 {}
impl ClusterStoreValue for i64 {}
impl ClusterStoreValue for f64 {}

/// Asynchronously stores `value` to `dst` in the shared memory of a CTA of the cluster, and
/// signals the completion of the store on the mbarrier `mbar` in the same CTA by performing a
/// complete-tx operation with the size of the stored value in bytes (`st.async` with
/// `.mbarrier::complete_tx::bytes`).
///
/// The receiving CTA typically initializes `mbar` and registers the number of bytes it expects
/// with [`MBarrier::expect_tx`], then waits on the barrier to observe the stores.
///
/// The store is a weak memory operation performed in the generic proxy: its effects become
/// visible to other threads only when synchronization is established through `mbar`.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async>
///
/// # Safety
///
/// The cluster must consist of more than one CTA (`cluster_size() > 1`). `dst` and `mbar` must be
/// generic addresses in the shared memory of the same CTA of the executing cluster (e.g. obtained
/// from [`map_to_cta`], or in the executing CTA itself), `dst` must be aligned for `T`, and `mbar`
/// must point to an initialized mbarrier of that CTA (see the
/// [safety contract](MBarrier#safety-contract-of-all-operations)).
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn st_async<T: ClusterStoreValue>(dst: *mut T, value: T, mbar: *mut MBarrier) {
    unsafe { T::st_async(dst, value, mbar.cast()) }
}
