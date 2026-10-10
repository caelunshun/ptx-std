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
/// (TMA).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum ProxyKind {
    /// Accesses performed through virtually aliased addresses (`.alias`).
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
/// respect to the threads in `SCOPE`, and participates in the total order of all `fence.sc`
/// operations.
///
/// [`Scope::Cluster`] requires `sm_90`.
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
/// or [`MemoryOrdering::Release`] (`fence.release.sync_restrict::shared::cta.cluster`). These
/// fences only order accesses to shared memory (an acquire fence orders accesses to
/// `.shared::cluster`, a release fence accesses to `.shared::cta`) and can be cheaper than a
/// general cluster-scope fence. They are intended to be used together with mbarrier-based
/// synchronization.
///
/// Requires `sm_90`.
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
/// [`ProxyKind::Alias`] needs `sm_70`; the async kinds require `sm_90`.
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

/// Establishes a memory fence between the async proxy and the generic proxy, for
/// synchronization through shared memory within the cluster, with acquire or release semantics
/// (`fence.proxy.async::generic`).
///
/// `SEM` must be [`MemoryOrdering::Acquire`]
/// (`fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster`) or
/// [`MemoryOrdering::Release`]
/// (`fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster`).
///
/// Requires `sm_90`.
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
