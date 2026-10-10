// --- LLM-generated --- //
//! CTA-scoped hardware barriers (non-mbarrier): the 16 named barriers of a CTA, including
//! barriers with a subset of the threads and barriers that reduce a predicate over the threads.
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
//!
//! All operations are `unsafe`: a barrier deadlocks (or has undefined behavior) unless every
//! thread that the barrier expects reaches it, so it must not be executed in diverged control
//! flow that some of those threads do not follow. The exact requirements are given in the
//! `# Safety` sections.

use core::{arch::nvptx as nv, marker::ConstParamTy};

/// Whether all threads of a warp are guaranteed to execute the same barrier instruction (the
/// `.aligned` qualifier of a barrier instruction).
///
/// The aligned form can be cheaper, but is undefined behavior if the threads of a warp reach the
/// barrier through different control flow paths: in conditionally executed code, it must only be
/// used if all threads evaluate the condition identically. (The PTX specification words this as
/// all threads in the CTA executing the same barrier instruction; at the very least, all threads
/// of a warp must.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum BarrierAlignment {
    /// The threads of a warp may reach the barrier from different, diverged code paths.
    Unaligned,
    /// All non-exited threads of a warp execute the same barrier instruction (`.aligned`).
    Aligned,
}

/// Number of named barriers per CTA.
pub const NAMED_BARRIER_COUNT: u32 = 16;

/// One of the 16 named barriers of a CTA, identified by an id in `0..16`.
///
/// Barrier 0 is the one used by [`sync_threads`] (CUDA's `__syncthreads()`).
///
/// A barrier can be used for different groups of threads at different times, but all threads
/// taking part in a barrier phase must agree on the number of participating threads. Using
/// different barrier ids for independent groups of warps (e.g. in warp-specialized kernels)
/// allows them to synchronize independently.
///
/// Memory ordering: all operations on a barrier (`sync*`, `arrive` and the reductions) guarantee
/// that, when the barrier completes, the memory accesses requested by the executing thread prior
/// to the barrier operation are performed relative to all threads participating in the barrier.
/// The `sync*` and reduction operations further guarantee that the executing thread requests no
/// new memory access before the barrier completes; [`arrive`](Self::arrive) does not wait, so it
/// gives no such guarantee.
///
/// When a barrier completes, the waiting threads are restarted and the barrier is reset so it
/// can be reused immediately. Care must be taken that a warp does not execute more barrier
/// operations than intended on a barrier before it is reset (an `arrive` followed by any other
/// operation on the same barrier). The reduction operations (`sync_and_*`) must not be mixed
/// with `sync` or `arrive` operations on the same active barrier: execution is unpredictable.#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct NamedBarrier(u32);

impl NamedBarrier {
    /// Barrier 0, used by [`sync_threads`].
    pub const DEFAULT: NamedBarrier = NamedBarrier(0);

    /// Creates a handle to the barrier with id `id`.
    ///
    /// # Panics
    ///
    /// Panics (at compile time in const contexts) if `id >= 16`.
    #[inline]
    pub const fn new(id: u32) -> Self {
        assert!(
            id < NAMED_BARRIER_COUNT,
            "named barrier id must be less than 16"
        );
        NamedBarrier(id)
    }

    /// Returns the id of the barrier, in `0..16`.
    #[inline]
    pub const fn id(self) -> u32 {
        self.0
    }

    /// Waits at the barrier until all non-exited threads of the CTA have arrived (or arrived
    /// with [`arrive`](Self::arrive)) (`barrier.sync`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
    ///
    /// # Safety
    ///
    /// All non-exited threads of the CTA must eventually execute a matching barrier operation on
    /// this barrier. With [`BarrierAlignment::Aligned`], all threads of a warp must execute the
    /// same barrier instruction.
    #[inline]
    pub unsafe fn sync<const ALIGNMENT: BarrierAlignment>(self) {
        const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => unsafe {
            nv::barrier_sync::<ALIGNED>(self.0)
        })
    }

    /// Waits at the barrier until `num_threads` threads have arrived (or arrived with
    /// [`arrive`](Self::arrive)) (`barrier.sync` with a thread count).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
    ///
    /// # Safety
    ///
    /// `num_threads` must be a multiple of the warp size (32) and the same for all threads
    /// using the barrier in this phase, and exactly that many threads must execute a matching
    /// barrier operation on this barrier. With [`BarrierAlignment::Aligned`], all threads of a
    /// warp must execute the same barrier instruction.
    #[inline]
    pub unsafe fn sync_among<const ALIGNMENT: BarrierAlignment>(self, num_threads: u32) {
        const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => unsafe {
            nv::barrier_sync_count::<ALIGNED>(self.0, num_threads)
        })
    }

    /// Signals the arrival of the executing threads at the barrier, which expects `num_threads`
    /// threads in total, without waiting for the other threads (`barrier.arrive`).
    ///
    /// Typically used by producer warps to notify consumer warps that wait with
    /// [`sync_among`](Self::sync_among).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
    ///
    /// # Safety
    ///
    /// `num_threads` must be a non-zero multiple of the warp size (32) and the same for all
    /// threads using the barrier in this phase. With [`BarrierAlignment::Aligned`], all threads of
    /// a warp must execute the same barrier instruction.
    #[inline]
    pub unsafe fn arrive<const ALIGNMENT: BarrierAlignment>(self, num_threads: u32) {
        const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => unsafe {
            nv::barrier_arrive::<ALIGNED>(self.0, num_threads)
        })
    }

    /// Waits at the barrier until all non-exited threads of the CTA have arrived, and returns the
    /// number of threads for which `predicate` is true (`barrier.red.popc`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
    ///
    /// # Safety
    ///
    /// See [`sync`](Self::sync).
    #[inline]
    pub unsafe fn sync_and_count<const ALIGNMENT: BarrierAlignment>(self, predicate: bool) -> u32 {
        const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => unsafe {
            nv::barrier_red_popc::<ALIGNED>(self.0, predicate)
        })
    }

    /// Waits at the barrier until `num_threads` threads have arrived, and returns the number of
    /// those threads for which `predicate` is true (`barrier.red.popc` with a thread count).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
    ///
    /// # Safety
    ///
    /// See [`sync_among`](Self::sync_among).
    #[inline]
    pub unsafe fn sync_and_count_among<const ALIGNMENT: BarrierAlignment>(
        self,
        predicate: bool,
        num_threads: u32,
    ) -> u32 {
        const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => unsafe {
            nv::barrier_red_popc_count::<ALIGNED>(self.0, num_threads, predicate)
        })
    }

    /// Waits at the barrier until all non-exited threads of the CTA have arrived, and returns
    /// whether `predicate` is true for all of them (`barrier.red.and`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
    ///
    /// # Safety
    ///
    /// See [`sync`](Self::sync).
    #[inline]
    pub unsafe fn sync_and_all<const ALIGNMENT: BarrierAlignment>(self, predicate: bool) -> bool {
        const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => unsafe {
            nv::barrier_red_pred::<{ nv::BarrierRedOp::And }, ALIGNED>(self.0, predicate)
        })
    }

    /// Waits at the barrier until `num_threads` threads have arrived, and returns whether
    /// `predicate` is true for all of them (`barrier.red.and` with a thread count).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
    ///
    /// # Safety
    ///
    /// See [`sync_among`](Self::sync_among).
    #[inline]
    pub unsafe fn sync_and_all_among<const ALIGNMENT: BarrierAlignment>(
        self,
        predicate: bool,
        num_threads: u32,
    ) -> bool {
        const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => unsafe {
            nv::barrier_red_pred_count::<{ nv::BarrierRedOp::And }, ALIGNED>(
                self.0,
                num_threads,
                predicate,
            )
        })
    }

    /// Waits at the barrier until all non-exited threads of the CTA have arrived, and returns
    /// whether `predicate` is true for any of them (`barrier.red.or`).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
    ///
    /// # Safety
    ///
    /// See [`sync`](Self::sync).
    #[inline]
    pub unsafe fn sync_and_any<const ALIGNMENT: BarrierAlignment>(self, predicate: bool) -> bool {
        const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => unsafe {
            nv::barrier_red_pred::<{ nv::BarrierRedOp::Or }, ALIGNED>(self.0, predicate)
        })
    }

    /// Waits at the barrier until `num_threads` threads have arrived, and returns whether
    /// `predicate` is true for any of them (`barrier.red.or` with a thread count).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
    ///
    /// # Safety
    ///
    /// See [`sync_among`](Self::sync_among).
    #[inline]
    pub unsafe fn sync_and_any_among<const ALIGNMENT: BarrierAlignment>(
        self,
        predicate: bool,
        num_threads: u32,
    ) -> bool {
        const_bool!(ALIGNED = matches!(ALIGNMENT, BarrierAlignment::Aligned) => unsafe {
            nv::barrier_red_pred_count::<{ nv::BarrierRedOp::Or }, ALIGNED>(
                self.0,
                num_threads,
                predicate,
            )
        })
    }
}

/// Waits until all non-exited threads of the CTA have reached this point and all their prior
/// memory accesses are visible to the threads of the CTA. Equivalent to CUDA's
/// `__syncthreads()` (aligned barrier 0).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
///
/// # Safety
///
/// All non-exited threads of the CTA must eventually reach a `sync_threads` or an equivalent
/// operation on [`NamedBarrier::DEFAULT`], and all threads of a warp must execute the same
/// instruction: in conditionally executed code, `sync_threads` must only be called if all threads
/// of the warp evaluate the condition identically.
#[inline]
pub unsafe fn sync_threads() {
    unsafe { NamedBarrier::DEFAULT.sync::<{ BarrierAlignment::Aligned }>() }
}
