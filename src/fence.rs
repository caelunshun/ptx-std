// --- LLM-generated --- //
//! Memory fences and proxy fences.
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>

use core::{arch::nvptx as nv, marker::ConstParamTy};

#[cfg(target_feature = "sm_90")]
use crate::modifiers::MemoryOrdering;
use crate::modifiers::Scope;

/// A proxy through which memory can be accessed, for [`fence_proxy`].
///
/// Memory accesses through the generic proxy (ordinary loads and stores) are not automatically
/// ordered with accesses through other proxies, such as the async proxy used by bulk async copies
/// (TMA). The async kinds can be restricted to a single state space; without a restriction
/// ([`ProxyKind::Async`]) the ordering applies to all state spaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum ProxyKind {
    /// Accesses performed through virtually aliased addresses to the same memory location
    /// (`.alias`).
    Alias,
    /// The async proxy, for all state spaces (`.async`).
    Async,
    /// The async proxy, restricted to the global state space (`.async.global`).
    AsyncGlobal,
    /// The async proxy, restricted to CTA-local shared memory (`.async.shared::cta`).
    AsyncSharedCta,
    /// The async proxy, restricted to cluster-wide shared memory (`.async.shared::cluster`).
    AsyncSharedCluster,
}

/// Establishes a sequentially consistent memory fence at scope `SCOPE` (`fence.sc`; also known
/// as `membar`).
///
/// Orders all prior memory accesses of the executing thread against all subsequent ones, with
/// respect to the threads in `SCOPE`. Instances of `fence.sc` with sufficient scope always
/// synchronize by forming a total order per scope, determined at runtime. This is slower than a
/// release/acquire fence (`fence.acq_rel`), which is sufficient for most programs.
///
/// [`Scope::Cta`], [`Scope::Gpu`] and [`Scope::System`] are emitted as `membar.cta`, `membar.gl`
/// and `membar.sys`, which on `sm_70` and higher are synonyms of `fence.sc.cta`, `fence.sc.gpu`
/// and `fence.sc.sys`. [`Scope::Cluster`] emits `fence.sc.cluster` and requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>
#[inline]
pub fn fence_seq_cst<const SCOPE: Scope>() {
    const {
        assert!(
            !matches!(SCOPE, Scope::Cluster) || cfg!(target_feature = "sm_90"),
            "Scope::Cluster fences require sm_90"
        )
    }
    const_map!(SCOPE => CS: nv::Scope {
        Scope::Cta => nv::Scope::Cta,
        Scope::Cluster => nv::Scope::Cluster,
        Scope::Gpu => nv::Scope::Gpu,
        Scope::System => nv::Scope::Sys,
    } in unsafe { nv::fence_sc::<CS>() })
}

/// Establishes a memory fence for synchronization through shared memory within the cluster,
/// with acquire or release semantics.
///
/// `SEM` must be [`MemoryOrdering::Acquire`] (`fence.acquire.sync_restrict::shared::cluster.cluster`)
/// or [`MemoryOrdering::Release`] (`fence.release.sync_restrict::shared::cta.cluster`). The
/// `.sync_restrict` qualifier restricts the memory operations that the fence orders: an acquire
/// fence only applies to operations on objects in `.shared::cluster`, a release fence only to
/// operations on objects in `.shared::cta`. The scope is always the cluster. They are intended
/// to be used together with mbarrier-based synchronization, e.g. an acquire fence after a
/// successful mbarrier wait and a release fence before an mbarrier arrive.
///
/// Requires `sm_90` and PTX ISA 8.6.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>
#[cfg(target_feature = "sm_90")]
#[inline]
pub fn fence_sync_restrict<const SEM: MemoryOrdering>() {
    const {
        assert!(
            matches!(SEM, MemoryOrdering::Acquire | MemoryOrdering::Release),
            "only MemoryOrdering::Acquire and MemoryOrdering::Release are supported"
        )
    }
    const_map!(SEM => CM: nv::Semantics {
        MemoryOrdering::Acquire => nv::Semantics::Acquire,
        MemoryOrdering::Release => nv::Semantics::Release,
    } in unsafe { nv::fence_sync_restrict::<CM>() })
}

/// Establishes a bi-directional memory ordering between the generic proxy and the proxy `KIND`,
/// for the memory accessed by the executing thread (`fence.proxy`).
///
/// Needed, for example, after writing shared memory with ordinary stores and before a bulk async
/// copy reads that memory ([`ProxyKind::AsyncSharedCta`]).
///
/// A bi-directional proxy fence takes effect within the executing thread and has no scope: it
/// does not synchronize with other fences directly, but its cross-proxy effect composes with
/// other forms of synchronization (e.g. barriers or mbarriers) according to the memory
/// consistency model.
///
/// [`ProxyKind::Alias`] needs `sm_70` and PTX ISA 7.5; the async kinds require `sm_90` and
/// PTX ISA 8.0.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>
#[inline]
pub fn fence_proxy<const KIND: ProxyKind>() {
    const {
        assert!(
            matches!(KIND, ProxyKind::Alias) || cfg!(target_feature = "sm_90"),
            "async proxy fences require sm_90"
        )
    }
    const_map!(KIND => CK: nv::ProxyKind {
        ProxyKind::Alias => nv::ProxyKind::Alias,
        ProxyKind::Async => nv::ProxyKind::Async,
        ProxyKind::AsyncGlobal => nv::ProxyKind::AsyncGlobal,
        ProxyKind::AsyncSharedCta => nv::ProxyKind::AsyncSharedCta,
        ProxyKind::AsyncSharedCluster => nv::ProxyKind::AsyncSharedCluster,
    } in unsafe { nv::fence_proxy::<CK>() })
}

/// Establishes a uni-directional proxy ordering from the generic proxy to the async proxy, for
/// synchronization through shared memory within the cluster, with acquire or release semantics
/// (`fence.proxy.async::generic`).
///
/// Orders prior generic-proxy memory accesses against subsequent async-proxy accesses. `SEM`
/// must be [`MemoryOrdering::Acquire`]
/// (`fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster`) or
/// [`MemoryOrdering::Release`]
/// (`fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster`). As for
/// [`fence_sync_restrict`], an acquire fence only applies to operations on `.shared::cluster`
/// objects, a release fence only to `.shared::cta` objects, and the scope is the cluster.
///
/// Requires `sm_90` and PTX ISA 8.6.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>
#[cfg(target_feature = "sm_90")]
#[inline]
pub fn fence_proxy_async_generic_sync_restrict<const SEM: MemoryOrdering>() {
    const {
        assert!(
            matches!(SEM, MemoryOrdering::Acquire | MemoryOrdering::Release),
            "only MemoryOrdering::Acquire and MemoryOrdering::Release are supported"
        )
    }
    const_map!(SEM => CM: nv::Semantics {
        MemoryOrdering::Acquire => nv::Semantics::Acquire,
        MemoryOrdering::Release => nv::Semantics::Release,
    } in unsafe { nv::fence_proxy_async_generic_sync_restrict::<CM>() })
}
