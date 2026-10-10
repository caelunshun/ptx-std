// --- LLM-generated --- //
//! Memory barrier objects (`mbarrier`): asynchronous, split arrive/wait barriers in shared memory
//! with support for transaction counts (used for tracking completion of bulk async copies).
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier>

use core::{arch::nvptx as nv, cell::UnsafeCell, mem::MaybeUninit, ops::Not};

use bytemuck::{Pod, Zeroable};

use crate::modifiers::{MemoryOrdering, Scope};

/// Forwards `$scope` and `$sem` (which must be `Release` or `Relaxed`, as for arrive-like
/// operations) to local consts of core's enum types, then evaluates `$body`.
#[cfg_attr(not(target_feature = "sm_90"), allow(unused_macros))]
macro_rules! release_variant {
    ($scope:ident, $sem:ident, |$cs:ident, $cm:ident| $body:expr) => {{
        const {
            assert!(
                matches!($scope, Scope::Cta | Scope::Cluster),
                "only Scope::Cta and Scope::Cluster are supported by mbarrier operations"
            );
            assert!(
                matches!($sem, MemoryOrdering::Release | MemoryOrdering::Relaxed),
                "only MemoryOrdering::Release and MemoryOrdering::Relaxed are supported by mbarrier arrive operations"
            );
        }
        const_map!($scope => $cs: nv::Scope {
            Scope::Cta => nv::Scope::Cta,
            Scope::Cluster => nv::Scope::Cluster,
        } in const_map!($sem => $cm: nv::Semantics {
            MemoryOrdering::Release => nv::Semantics::Release,
            MemoryOrdering::Relaxed => nv::Semantics::Relaxed,
        } in $body))
    }};
}

/// Like `release_variant!`, but for wait-like operations (`Acquire` or `Relaxed`).
macro_rules! acquire_variant {
    ($scope:ident, $sem:ident, |$cs:ident, $cm:ident| $body:expr) => {{
        const {
            assert!(
                matches!($scope, Scope::Cta | Scope::Cluster),
                "only Scope::Cta and Scope::Cluster are supported by mbarrier operations"
            );
            assert!(
                matches!($sem, MemoryOrdering::Acquire | MemoryOrdering::Relaxed),
                "only MemoryOrdering::Acquire and MemoryOrdering::Relaxed are supported by mbarrier wait operations"
            );
        }
        const_map!($scope => $cs: nv::Scope {
            Scope::Cta => nv::Scope::Cta,
            Scope::Cluster => nv::Scope::Cluster,
        } in const_map!($sem => $cm: nv::Semantics {
            MemoryOrdering::Acquire => nv::Semantics::Acquire,
            MemoryOrdering::Relaxed => nv::Semantics::Relaxed,
        } in $body))
    }};
}

/// A memory barrier object (`mbarrier`) residing in shared memory.
///
/// An mbarrier is an 8-byte, 8-byte-aligned object that tracks a sequence of *phases*. Each phase
/// completes once both
/// - its *pending arrival count* has reached zero (initially the *expected arrival count* passed
///   to [`init`](Self::init), decremented by every arrive-on operation), and
/// - its *transaction count* (*tx-count*) is zero (incremented by `expect_tx`,
///   decremented by `complete_tx` or by completion of an asynchronous
///   operation that is tracked by the mbarrier, e.g. a TMA copy).
///
/// When a phase completes, the barrier moves on to the next phase and the pending arrival count is
/// reset to the expected arrival count. Threads wait for a phase either through the
/// [`ArrivalToken`] returned by an arrive operation or through the [`PhaseParity`] of the phase;
/// see [`Phase`].
///
/// # Safety contract of all operations
///
/// Every operation on an mbarrier is `unsafe` because the PTX memory model specifies undefined
/// behavior unless the following hold:
/// - the `MBarrier` is located in the shared memory of the executing CTA (or, for the `*_remote`
///   operations, the CTA that the pointer was mapped to, see `map_to_cta_rank`),
/// - it was initialized with [`init`](Self::init) and not invalidated with [`inval`](Self::inval),
///   and the initialization has been made visible to the accessing threads (e.g. through a
///   barrier, and for access from other CTAs `fence_init` followed by a
///   cluster barrier),
/// - the pending arrival count and the tx-count stay within their ranges (the pending count must
///   not underflow, and the tx-count must stay within `[-(2^20 - 1), 2^20 - 1]`).
///
/// The `# Safety` sections of the individual operations only list requirements in addition to
/// these.
#[derive(Debug)]
#[repr(transparent)]
pub struct MBarrier(UnsafeCell<MaybeUninit<u64>>);

// SAFETY: mbarrier operations are performed atomically by the hardware.
unsafe impl Sync for MBarrier {}

/// Opaque state of an [`MBarrier`] captured by an arrive operation, identifying the phase that the
/// arrive-on operation contributed to.
///
/// Pass it to the wait operations of the barrier (e.g. [`MBarrier::wait`]) to wait for that phase
/// to complete. It is only valid for waiting during the phase it was returned for and the
/// immediately following phase.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct ArrivalToken(u64);

impl ArrivalToken {
    /// Returns the pending arrival count of the barrier at the time of the arrive-on operation
    /// that returned this token.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-pending-count>
    ///
    /// # Safety
    ///
    /// The token must have been returned by [`MBarrier::arrive_no_complete`] or
    /// [`MBarrier::arrive_drop_no_complete`].
    #[inline]
    pub unsafe fn pending_count(self) -> u32 {
        unsafe { nv::mbarrier_pending_count(self.0) }
    }
}

/// The parity of a barrier phase: even for phases 0, 2, 4, ..., odd for phases 1, 3, 5, ...
///
/// A freshly initialized barrier is in phase 0 ([`PhaseParity::Even`]). Waiting on a parity waits
/// for the completion of the current phase if it has that parity, or tests the completion of the
/// immediately preceding phase otherwise; the typical usage is to flip the parity (with `!`) after
/// each successful wait.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum PhaseParity {
    /// Even-numbered phases (0, 2, ...).
    Even = 0,
    /// Odd-numbered phases (1, 3, ...).
    Odd = 1,
}

impl PhaseParity {
    /// Returns the parity of the phase with the given number.
    #[inline]
    pub const fn of_phase(phase: u32) -> Self {
        if phase & 1 == 0 {
            PhaseParity::Even
        } else {
            PhaseParity::Odd
        }
    }

    /// Returns the opposite parity.
    #[inline]
    pub const fn flip(self) -> Self {
        match self {
            PhaseParity::Even => PhaseParity::Odd,
            PhaseParity::Odd => PhaseParity::Even,
        }
    }
}

impl Not for PhaseParity {
    type Output = Self;

    #[inline]
    fn not(self) -> Self {
        self.flip()
    }
}

mod sealed {
    use super::*;

    pub trait Sealed: Copy {
        /// `test.wait` with a runtime-selected phase on `mbar` (`.shared::cta`).
        unsafe fn test_wait<const SCOPE: Scope, const SEM: MemoryOrdering>(
            self,
            mbar: *mut u64,
        ) -> bool;

        #[cfg(target_feature = "sm_90")]
        unsafe fn try_wait<const SCOPE: Scope, const SEM: MemoryOrdering>(
            self,
            mbar: *mut u64,
        ) -> bool;

        #[cfg(target_feature = "sm_90")]
        unsafe fn try_wait_timed<const SCOPE: Scope, const SEM: MemoryOrdering>(
            self,
            mbar: *mut u64,
            suspend_time_hint_ns: u32,
        ) -> bool;
    }

    impl Sealed for ArrivalToken {
        #[inline]
        unsafe fn test_wait<const SCOPE: Scope, const SEM: MemoryOrdering>(
            self,
            mbar: *mut u64,
        ) -> bool {
            if const { matches!((SCOPE, SEM), (Scope::Cta, MemoryOrdering::Acquire)) } {
                unsafe { nv::mbarrier_test_wait::<{ nv::StateSpace::SharedCta }>(mbar, self.0) }
            } else {
                acquire_variant!(SCOPE, SEM, |CS, CM| unsafe {
                    nv::mbarrier_test_wait_scoped::<CS, CM>(mbar, self.0)
                })
            }
        }

        #[cfg(target_feature = "sm_90")]
        #[inline]
        unsafe fn try_wait<const SCOPE: Scope, const SEM: MemoryOrdering>(
            self,
            mbar: *mut u64,
        ) -> bool {
            acquire_variant!(SCOPE, SEM, |CS, CM| unsafe {
                nv::mbarrier_try_wait_scoped::<CS, CM>(mbar, self.0)
            })
        }

        #[cfg(target_feature = "sm_90")]
        #[inline]
        unsafe fn try_wait_timed<const SCOPE: Scope, const SEM: MemoryOrdering>(
            self,
            mbar: *mut u64,
            suspend_time_hint_ns: u32,
        ) -> bool {
            acquire_variant!(SCOPE, SEM, |CS, CM| unsafe {
                nv::mbarrier_try_wait_scoped_tl::<CS, CM>(mbar, self.0, suspend_time_hint_ns)
            })
        }
    }

    impl Sealed for PhaseParity {
        #[inline]
        unsafe fn test_wait<const SCOPE: Scope, const SEM: MemoryOrdering>(
            self,
            mbar: *mut u64,
        ) -> bool {
            acquire_variant!(SCOPE, SEM, |CS, CM| unsafe {
                nv::mbarrier_test_wait_parity::<CS, CM>(mbar, self as u32)
            })
        }

        #[cfg(target_feature = "sm_90")]
        #[inline]
        unsafe fn try_wait<const SCOPE: Scope, const SEM: MemoryOrdering>(
            self,
            mbar: *mut u64,
        ) -> bool {
            acquire_variant!(SCOPE, SEM, |CS, CM| unsafe {
                nv::mbarrier_try_wait_parity::<CS, CM>(mbar, self as u32)
            })
        }

        #[cfg(target_feature = "sm_90")]
        #[inline]
        unsafe fn try_wait_timed<const SCOPE: Scope, const SEM: MemoryOrdering>(
            self,
            mbar: *mut u64,
            suspend_time_hint_ns: u32,
        ) -> bool {
            acquire_variant!(SCOPE, SEM, |CS, CM| unsafe {
                nv::mbarrier_try_wait_parity_tl::<CS, CM>(mbar, self as u32, suspend_time_hint_ns)
            })
        }
    }
}

/// Identifies a barrier phase to wait for: either an [`ArrivalToken`] returned by an arrive
/// operation or a [`PhaseParity`]. This trait is sealed.
pub trait Phase: sealed::Sealed {}

impl Phase for ArrivalToken {}
impl Phase for PhaseParity {}

impl MBarrier {
    /// Creates an uninitialized barrier, e.g. for use as a shared memory `static`.
    ///
    /// The barrier must be initialized with [`init`](Self::init) before use.
    #[inline]
    pub const fn uninit() -> Self {
        MBarrier(UnsafeCell::new(MaybeUninit::uninit()))
    }

    /// Generic-address-space pointer to the barrier object, for passing to intrinsics.
    pub(crate) fn as_ptr(&self) -> *mut u64 {
        self.0.get().cast()
    }

    /// Initializes the barrier with the expected arrival count `expected_arrivals`, entering
    /// phase 0 with a zero tx-count.
    ///
    /// Typically executed by a single thread, followed by `fence_init` (if
    /// the barrier is accessed through the async proxy or by other CTAs of the cluster) and a
    /// barrier that makes the initialization visible to all threads that use the barrier.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init>
    ///
    /// # Safety
    ///
    /// The barrier must be located in shared memory, and must not currently be in use by
    /// other threads (it is either uninitialized or has been invalidated). `expected_arrivals`
    /// must be in `1..=2^20 - 1`.
    #[inline]
    pub unsafe fn init(&self, expected_arrivals: u32) {
        unsafe {
            nv::mbarrier_init::<{ nv::StateSpace::SharedCta }>(self.as_ptr(), expected_arrivals)
        }
    }

    /// Invalidates the barrier, after which its memory can be used for other purposes or the
    /// barrier can be re-initialized with [`init`](Self::init).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-inval>
    ///
    /// # Safety
    ///
    /// The barrier must be initialized and no thread may still use it, including pending
    /// asynchronous operations (e.g. bulk copies) that will signal it.
    #[inline]
    pub unsafe fn inval(&self) {
        unsafe { nv::mbarrier_inval::<{ nv::StateSpace::SharedCta }>(self.as_ptr()) }
    }

    /// Orders prior [`init`](Self::init) operations of this thread against subsequent operations
    /// of the whole cluster, making the initialized barriers visible to the async proxy and to
    /// other CTAs in the cluster (`fence.mbarrier_init.release.cluster`).
    ///
    /// Needed after initializing a barrier that is used by TMA operations or by threads of other
    /// CTAs, followed by a cluster barrier (see
    /// [`cluster::barrier_sync`](crate::cluster::barrier_sync)).
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub fn fence_init() {
        unsafe { nv::fence_mbarrier_init() }
    }

    /// Makes the barrier track the completion of all prior `cp.async` operations of the executing
    /// thread: once they complete, an arrive-on operation is triggered on the barrier.
    ///
    /// The pending arrival count of the barrier is incremented by one before the arrive-on
    /// operation is triggered, so the net effect on the pending count is zero. Use
    /// [`cp_async_arrive_no_inc`](Self::cp_async_arrive_no_inc) to account for the arrival in the
    /// expected arrival count instead.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-mbarrier-arrive>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[inline]
    pub unsafe fn cp_async_arrive(&self) {
        unsafe {
            nv::cp_async_mbarrier_arrive::<false, { nv::StateSpace::SharedCta }>(
                self.as_ptr().cast(),
            )
        }
    }

    /// Like [`cp_async_arrive`](Self::cp_async_arrive), but the pending arrival count is not
    /// incremented (`.noinc`): the triggered arrive-on operation decrements it, so the expected
    /// arrival count passed to [`init`](Self::init) must include the arrival of this operation.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-mbarrier-arrive>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[inline]
    pub unsafe fn cp_async_arrive_no_inc(&self) {
        unsafe {
            nv::cp_async_mbarrier_arrive::<true, { nv::StateSpace::SharedCta }>(
                self.as_ptr().cast(),
            )
        }
    }

    // ---- arrive ----

    /// Performs an arrive-on operation, decrementing the pending arrival count by one, with
    /// release semantics at CTA scope. Returns a token for the phase arrived at.
    ///
    /// If this causes the pending count to reach zero (and the tx-count is zero), the phase
    /// completes.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[inline]
    pub unsafe fn arrive(&self) -> ArrivalToken {
        ArrivalToken(unsafe { nv::mbarrier_arrive::<{ nv::StateSpace::SharedCta }>(self.as_ptr()) })
    }

    /// Performs an arrive-on operation that decrements the pending arrival count by `count`,
    /// which must not complete the current phase.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive>
    ///
    /// # Safety
    ///
    /// The operation must not cause the barrier to complete its current phase, i.e. `count` must
    /// be smaller than the pending arrival count (or the tx-count must remain non-zero). See also
    /// the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[inline]
    pub unsafe fn arrive_no_complete(&self, count: u32) -> ArrivalToken {
        ArrivalToken(unsafe {
            nv::mbarrier_arrive_nocomplete::<{ nv::StateSpace::SharedCta }>(self.as_ptr(), count)
        })
    }

    /// Decrements the expected arrival count of the current and all subsequent phases by one,
    /// and performs an arrive-on operation (release, CTA scope). Used by a thread that leaves
    /// the set of participating threads.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations). The expected arrival
    /// count must not drop below zero.
    #[inline]
    pub unsafe fn arrive_drop(&self) -> ArrivalToken {
        ArrivalToken(unsafe {
            nv::mbarrier_arrive_drop::<{ nv::StateSpace::SharedCta }>(self.as_ptr())
        })
    }

    /// Like [`arrive_drop`](Self::arrive_drop), but decrements the expected arrival count and
    /// performs the arrive-on operation with `count`, which must not complete the current phase.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop>
    ///
    /// # Safety
    ///
    /// The operation must not cause the barrier to complete its current phase. See also the
    /// [safety contract](MBarrier#safety-contract-of-all-operations).
    #[inline]
    pub unsafe fn arrive_drop_no_complete(&self, count: u32) -> ArrivalToken {
        ArrivalToken(unsafe {
            nv::mbarrier_arrive_drop_nocomplete::<{ nv::StateSpace::SharedCta }>(
                self.as_ptr(),
                count,
            )
        })
    }

    /// Performs an arrive-on operation that decrements the pending arrival count by `count`, at
    /// scope `SCOPE` with memory ordering `SEM`.
    ///
    /// `SCOPE` must be [`Scope::Cta`] or [`Scope::Cluster`], and `SEM` must be
    /// [`MemoryOrdering::Release`] or [`MemoryOrdering::Relaxed`].
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn arrive_scoped<const SCOPE: Scope, const SEM: MemoryOrdering>(
        &self,
        count: u32,
    ) -> ArrivalToken {
        let ptr = self.as_ptr();
        ArrivalToken(release_variant!(SCOPE, SEM, |CS, CM| unsafe {
            nv::mbarrier_arrive_scoped::<CS, CM>(ptr, count)
        }))
    }

    /// Like [`arrive_scoped`](Self::arrive_scoped), but additionally decrements the expected
    /// arrival count of the current and all subsequent phases by `count`.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations). The expected arrival
    /// count must not drop below zero.
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn arrive_drop_scoped<const SCOPE: Scope, const SEM: MemoryOrdering>(
        &self,
        count: u32,
    ) -> ArrivalToken {
        let ptr = self.as_ptr();
        ArrivalToken(release_variant!(SCOPE, SEM, |CS, CM| unsafe {
            nv::mbarrier_arrive_drop_scoped::<CS, CM>(ptr, count)
        }))
    }

    /// Performs an expect-tx operation with `tx_bytes` followed by an arrive-on operation
    /// (release semantics, CTA scope), as one atomic operation.
    ///
    /// This is the usual way for the thread that issues a bulk async copy to register the number
    /// of bytes the copy will deliver and to arrive on the barrier.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn arrive_expect_tx(&self, tx_bytes: u32) -> ArrivalToken {
        unsafe {
            self.arrive_expect_tx_scoped::<{ Scope::Cta }, { MemoryOrdering::Release }>(tx_bytes)
        }
    }

    /// Like [`arrive_expect_tx`](Self::arrive_expect_tx), at scope `SCOPE` with memory ordering
    /// `SEM`.
    ///
    /// `SCOPE` must be [`Scope::Cta`] or [`Scope::Cluster`], and `SEM` must be
    /// [`MemoryOrdering::Release`] or [`MemoryOrdering::Relaxed`].
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn arrive_expect_tx_scoped<const SCOPE: Scope, const SEM: MemoryOrdering>(
        &self,
        tx_bytes: u32,
    ) -> ArrivalToken {
        let ptr = self.as_ptr();
        ArrivalToken(release_variant!(SCOPE, SEM, |CS, CM| unsafe {
            nv::mbarrier_arrive_expect_tx::<CS, CM>(ptr, tx_bytes)
        }))
    }

    /// Like [`arrive_expect_tx_scoped`](Self::arrive_expect_tx_scoped), but additionally
    /// decrements the expected arrival count of the current and all subsequent phases by one.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations). The expected arrival
    /// count must not drop below zero.
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn arrive_drop_expect_tx_scoped<const SCOPE: Scope, const SEM: MemoryOrdering>(
        &self,
        tx_bytes: u32,
    ) -> ArrivalToken {
        let ptr = self.as_ptr();
        ArrivalToken(release_variant!(SCOPE, SEM, |CS, CM| unsafe {
            nv::mbarrier_arrive_drop_expect_tx::<CS, CM>(ptr, tx_bytes)
        }))
    }

    // ---- transaction count ----

    /// Increments the tx-count of the current phase by `tx_bytes`, so that the phase cannot
    /// complete before the expected number of bytes has been delivered by asynchronous
    /// operations tracked by this barrier (CTA scope).
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn expect_tx(&self, tx_bytes: u32) {
        unsafe { self.expect_tx_scoped::<{ Scope::Cta }>(tx_bytes) }
    }

    /// Like [`expect_tx`](Self::expect_tx), at scope `SCOPE` ([`Scope::Cta`] or
    /// [`Scope::Cluster`]).
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn expect_tx_scoped<const SCOPE: Scope>(&self, tx_bytes: u32) {
        let ptr = self.as_ptr();
        const {
            assert!(
                matches!(SCOPE, Scope::Cta | Scope::Cluster),
                "only Scope::Cta and Scope::Cluster are supported by mbarrier operations"
            )
        }
        const_map!(SCOPE => CS: nv::Scope {
            Scope::Cta => nv::Scope::Cta,
            Scope::Cluster => nv::Scope::Cluster,
        } in unsafe {
            nv::mbarrier_expect_tx::<CS, { nv::StateSpace::SharedCta }>(ptr, tx_bytes)
        })
    }

    /// Decrements the tx-count of the current phase by `tx_bytes` (CTA scope), as if
    /// an asynchronous operation delivering that many bytes had completed. If the pending
    /// arrival count and the tx-count both reach zero, the phase completes.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn complete_tx(&self, tx_bytes: u32) {
        unsafe { self.complete_tx_scoped::<{ Scope::Cta }>(tx_bytes) }
    }

    /// Like [`complete_tx`](Self::complete_tx), at scope `SCOPE` ([`Scope::Cta`] or
    /// [`Scope::Cluster`]).
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx>
    ///
    /// # Safety
    ///
    /// See the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn complete_tx_scoped<const SCOPE: Scope>(&self, tx_bytes: u32) {
        let ptr = self.as_ptr();
        const {
            assert!(
                matches!(SCOPE, Scope::Cta | Scope::Cluster),
                "only Scope::Cta and Scope::Cluster are supported by mbarrier operations"
            )
        }
        const_map!(SCOPE => CS: nv::Scope {
            Scope::Cta => nv::Scope::Cta,
            Scope::Cluster => nv::Scope::Cluster,
        } in unsafe {
            nv::mbarrier_complete_tx::<CS, { nv::StateSpace::SharedCta }>(ptr, tx_bytes)
        })
    }

    // ---- operations on barriers in other CTAs of the cluster ----

    /// Maps this barrier to the corresponding barrier in the shared memory of the CTA with rank
    /// `cta_rank` in the cluster, for use with the `*_remote` operations.
    ///
    /// See [`cluster::map_to_cta`](crate::cluster::map_to_cta).
    ///
    /// Requires `sm_90`.
    ///
    /// # Safety
    ///
    /// `cta_rank` must be a valid CTA rank in the cluster, and the barrier must be located in
    /// shared memory.
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn map_to_cta_rank(&self, cta_rank: u32) -> *mut MBarrier {
        unsafe { crate::cluster::map_to_cta(self as *const MBarrier as *mut MBarrier, cta_rank) }
    }

    /// Performs an arrive-on operation that decrements the pending arrival count by `count` on
    /// the barrier `barrier` in the shared memory of a CTA of the cluster (possibly the
    /// executing CTA), at scope `SCOPE` with memory ordering `SEM`. The barrier state is not
    /// returned.
    ///
    /// `SCOPE` must be [`Scope::Cta`] or [`Scope::Cluster`], and `SEM` must be
    /// [`MemoryOrdering::Release`] or [`MemoryOrdering::Relaxed`].
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive>
    ///
    /// # Safety
    ///
    /// `barrier` must point to an mbarrier in the shared memory of a CTA of the executing
    /// cluster, e.g. obtained from [`map_to_cta_rank`](Self::map_to_cta_rank). See also the
    /// [safety contract](MBarrier#safety-contract-of-all-operations).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn arrive_remote<const SCOPE: Scope, const SEM: MemoryOrdering>(
        barrier: *mut MBarrier,
        count: u32,
    ) {
        release_variant!(SCOPE, SEM, |CS, CM| unsafe {
            nv::mbarrier_arrive_scoped_cluster::<CS, CM>(barrier.cast(), count)
        })
    }

    /// Like [`arrive_remote`](Self::arrive_remote), but additionally decrements the expected
    /// arrival count of the current and all subsequent phases by `count`.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop>
    ///
    /// # Safety
    ///
    /// See [`arrive_remote`](Self::arrive_remote). The expected arrival count must not drop below
    /// zero.
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn arrive_drop_remote<const SCOPE: Scope, const SEM: MemoryOrdering>(
        barrier: *mut MBarrier,
        count: u32,
    ) {
        release_variant!(SCOPE, SEM, |CS, CM| unsafe {
            nv::mbarrier_arrive_drop_scoped_cluster::<CS, CM>(barrier.cast(), count)
        })
    }

    /// Performs an expect-tx operation with `tx_bytes` followed by an arrive-on operation on the
    /// barrier `barrier` in the shared memory of a CTA of the cluster.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive>
    ///
    /// # Safety
    ///
    /// See [`arrive_remote`](Self::arrive_remote).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn arrive_expect_tx_remote<const SCOPE: Scope, const SEM: MemoryOrdering>(
        barrier: *mut MBarrier,
        tx_bytes: u32,
    ) {
        release_variant!(SCOPE, SEM, |CS, CM| unsafe {
            nv::mbarrier_arrive_expect_tx_cluster::<CS, CM>(barrier.cast(), tx_bytes)
        })
    }

    /// Like [`arrive_expect_tx_remote`](Self::arrive_expect_tx_remote), but additionally
    /// decrements the expected arrival count of the current and all subsequent phases by one.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop>
    ///
    /// # Safety
    ///
    /// See [`arrive_remote`](Self::arrive_remote). The expected arrival count must not drop below
    /// zero.
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn arrive_drop_expect_tx_remote<const SCOPE: Scope, const SEM: MemoryOrdering>(
        barrier: *mut MBarrier,
        tx_bytes: u32,
    ) {
        release_variant!(SCOPE, SEM, |CS, CM| unsafe {
            nv::mbarrier_arrive_drop_expect_tx_cluster::<CS, CM>(barrier.cast(), tx_bytes)
        })
    }

    /// Performs an expect-tx operation with `tx_bytes` on the barrier `barrier` in the shared
    /// memory of a CTA of the cluster, at scope `SCOPE` ([`Scope::Cta`] or [`Scope::Cluster`]).
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx>
    ///
    /// # Safety
    ///
    /// See [`arrive_remote`](Self::arrive_remote).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn expect_tx_remote<const SCOPE: Scope>(barrier: *mut MBarrier, tx_bytes: u32) {
        const {
            assert!(
                matches!(SCOPE, Scope::Cta | Scope::Cluster),
                "only Scope::Cta and Scope::Cluster are supported by mbarrier operations"
            )
        }
        const_map!(SCOPE => CS: nv::Scope {
            Scope::Cta => nv::Scope::Cta,
            Scope::Cluster => nv::Scope::Cluster,
        } in unsafe {
            nv::mbarrier_expect_tx::<CS, { nv::StateSpace::SharedCluster }>(barrier.cast(), tx_bytes)
        })
    }

    /// Performs a complete-tx operation with `tx_bytes` on the barrier `barrier` in the shared
    /// memory of a CTA of the cluster, at scope `SCOPE` ([`Scope::Cta`] or [`Scope::Cluster`]).
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx>
    ///
    /// # Safety
    ///
    /// See [`arrive_remote`](Self::arrive_remote).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn complete_tx_remote<const SCOPE: Scope>(barrier: *mut MBarrier, tx_bytes: u32) {
        const {
            assert!(
                matches!(SCOPE, Scope::Cta | Scope::Cluster),
                "only Scope::Cta and Scope::Cluster are supported by mbarrier operations"
            )
        }
        const_map!(SCOPE => CS: nv::Scope {
            Scope::Cta => nv::Scope::Cta,
            Scope::Cluster => nv::Scope::Cluster,
        } in unsafe {
            nv::mbarrier_complete_tx::<CS, { nv::StateSpace::SharedCluster }>(barrier.cast(), tx_bytes)
        })
    }

    // ---- waiting ----

    /// Tests, without blocking, whether the phase identified by `phase` (an [`ArrivalToken`] or
    /// a [`PhaseParity`]) has completed, with acquire semantics at CTA scope.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait>
    ///
    /// # Safety
    ///
    /// `phase` must identify the current phase or the immediately preceding phase of the barrier.
    /// See also the [safety contract](MBarrier#safety-contract-of-all-operations).
    #[inline]
    pub unsafe fn test_wait(&self, phase: impl Phase) -> bool {
        unsafe { phase.test_wait::<{ Scope::Cta }, { MemoryOrdering::Acquire }>(self.as_ptr()) }
    }

    /// Like [`test_wait`](Self::test_wait), at scope `SCOPE` with memory ordering `SEM`.
    ///
    /// `SCOPE` must be [`Scope::Cta`] or [`Scope::Cluster`], and `SEM` must be
    /// [`MemoryOrdering::Acquire`] or [`MemoryOrdering::Relaxed`]. Everything except
    /// CTA-scope acquire requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait>
    ///
    /// # Safety
    ///
    /// See [`test_wait`](Self::test_wait).
    #[inline]
    pub unsafe fn test_wait_scoped<const SCOPE: Scope, const SEM: MemoryOrdering>(
        &self,
        phase: impl Phase,
    ) -> bool {
        const {
            assert!(
                matches!((SCOPE, SEM), (Scope::Cta, MemoryOrdering::Acquire))
                    || cfg!(target_feature = "sm_90"),
                "test_wait with a scope other than Scope::Cta or ordering other than MemoryOrdering::Acquire requires sm_90"
            )
        }
        unsafe { phase.test_wait::<SCOPE, SEM>(self.as_ptr()) }
    }

    /// Waits for the phase identified by `phase` (an [`ArrivalToken`] or a [`PhaseParity`]) to
    /// complete, with acquire semantics at CTA scope. Returns `true` if the phase completed, and
    /// `false` if the wait timed out (after a system-dependent time limit), in which case it can
    /// simply be retried.
    ///
    /// The thread may be suspended while waiting. See [`wait`](Self::wait) to wait in a loop.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait>
    ///
    /// # Safety
    ///
    /// See [`test_wait`](Self::test_wait).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn try_wait(&self, phase: impl Phase) -> bool {
        unsafe { phase.try_wait::<{ Scope::Cta }, { MemoryOrdering::Acquire }>(self.as_ptr()) }
    }

    /// Like [`try_wait`](Self::try_wait), at scope `SCOPE` with memory ordering `SEM`.
    ///
    /// `SCOPE` must be [`Scope::Cta`] or [`Scope::Cluster`], and `SEM` must be
    /// [`MemoryOrdering::Acquire`] or [`MemoryOrdering::Relaxed`].
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait>
    ///
    /// # Safety
    ///
    /// See [`test_wait`](Self::test_wait).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn try_wait_scoped<const SCOPE: Scope, const SEM: MemoryOrdering>(
        &self,
        phase: impl Phase,
    ) -> bool {
        unsafe { phase.try_wait::<SCOPE, SEM>(self.as_ptr()) }
    }

    /// Like [`try_wait`](Self::try_wait), but the thread is suspended for at most
    /// `suspend_time_hint_ns` nanoseconds (a hint) instead of a system-dependent time limit.
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait>
    ///
    /// # Safety
    ///
    /// See [`test_wait`](Self::test_wait).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn try_wait_timed(&self, phase: impl Phase, suspend_time_hint_ns: u32) -> bool {
        unsafe {
            phase.try_wait_timed::<{ Scope::Cta }, { MemoryOrdering::Acquire }>(
                self.as_ptr(),
                suspend_time_hint_ns,
            )
        }
    }

    /// Like [`try_wait_timed`](Self::try_wait_timed), at scope `SCOPE` with memory ordering
    /// `SEM`.
    ///
    /// `SCOPE` must be [`Scope::Cta`] or [`Scope::Cluster`], and `SEM` must be
    /// [`MemoryOrdering::Acquire`] or [`MemoryOrdering::Relaxed`].
    ///
    /// Requires `sm_90`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait>
    ///
    /// # Safety
    ///
    /// See [`test_wait`](Self::test_wait).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn try_wait_timed_scoped<const SCOPE: Scope, const SEM: MemoryOrdering>(
        &self,
        phase: impl Phase,
        suspend_time_hint_ns: u32,
    ) -> bool {
        unsafe { phase.try_wait_timed::<SCOPE, SEM>(self.as_ptr(), suspend_time_hint_ns) }
    }

    /// Blocks until the phase identified by `phase` (an [`ArrivalToken`] or a [`PhaseParity`])
    /// has completed, with acquire semantics at CTA scope.
    ///
    /// Loops on `try_wait` (or on [`test_wait`](Self::test_wait) before
    /// `sm_90`).
    ///
    /// # Safety
    ///
    /// `phase` must identify the current phase or the immediately preceding phase of the barrier,
    /// and the phase must eventually complete. See also the
    /// [safety contract](MBarrier#safety-contract-of-all-operations).
    #[inline]
    pub unsafe fn wait(&self, phase: impl Phase) {
        #[cfg(target_feature = "sm_90")]
        while !unsafe { self.try_wait(phase) } {}
        #[cfg(not(target_feature = "sm_90"))]
        while !unsafe { self.test_wait(phase) } {}
    }

    /// Like [`wait`](Self::wait), at scope `SCOPE` with memory ordering `SEM`.
    ///
    /// `SCOPE` must be [`Scope::Cta`] or [`Scope::Cluster`], and `SEM` must be
    /// [`MemoryOrdering::Acquire`] or [`MemoryOrdering::Relaxed`].
    ///
    /// Requires `sm_90`.
    ///
    /// # Safety
    ///
    /// See [`wait`](Self::wait).
    #[cfg(target_feature = "sm_90")]
    #[inline]
    pub unsafe fn wait_scoped<const SCOPE: Scope, const SEM: MemoryOrdering>(
        &self,
        phase: impl Phase,
    ) {
        while !unsafe { self.try_wait_scoped::<SCOPE, SEM>(phase) } {}
    }
}
