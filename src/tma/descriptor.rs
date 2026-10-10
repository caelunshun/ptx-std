// --- LLM-generated --- //
//! Tensor maps ([`TmaDescriptor`]) and the proxy fences that publish tensor maps written to global
//! memory to the TMA (the tensormap proxy).

#[allow(unused_imports)]
use core::arch::nvptx as nv;

use bytemuck::{Pod, Zeroable};

#[allow(unused_imports)]
use crate::modifiers::Scope;

/// A tensor map (`CUtensorMap`): a 128-byte, 64-byte-aligned opaque object describing the layout
/// of a tensor in global memory and of the boxes copied by the TMA.
///
/// Tensor maps are created on the host (for example with `cuTensorMapEncodeTiled`), and the
/// TMA instructions take a reference to one. The tensor map must reside in one of:
///
/// - **Kernel parameter space.** This is the usual way: pass the tensor map by value as a kernel
///   argument on the host side, and declare it in the kernel as a grid-constant reference,
///   `#[nvptx(grid_constant)] map: &TmaDescriptor`. The reference then points into parameter
///   space (through a generic address), and the tensor map can be used directly by TMA
///   instructions.
/// - **Constant memory.**
/// - **Global memory.** After the tensor map has been written by the generic proxy (for example
///   by a copy), it must be released with `tensormap_fence_release` by the writing thread and
///   acquired with `tensormap_fence_acquire` by each thread that is going to use it (after
///   synchronizing with the writing thread), before it is used by a TMA instruction. Tensor maps
///   are accessed through the *tensormap proxy*, which is not implicitly synchronized with the
///   generic proxy at kernel start.
///
/// The contents are opaque (their layout is undocumented); this type is plain old data so that it
/// can be passed through kernel parameters and copied around.
///
/// In-kernel modification of tensor maps (`tensormap.replace`) is intentionally not exposed: its
/// interaction with the hardware tensor map cache has poorly documented memory semantics.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tensormap>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(C, align(64))]
pub struct TmaDescriptor([u64; 16]);

impl TmaDescriptor {
    /// Size in bytes of a tensor map.
    pub const SIZE: usize = 128;

    /// Alignment in bytes of a tensor map.
    pub const ALIGN: usize = 64;
}

/// Releases prior generic-proxy writes to tensor maps (such as an ordinary copy of a
/// [`TmaDescriptor`] into global memory) so that they are visible to the tensormap proxy
/// at scope `SCOPE`, i.e. to the threads of that scope
/// (`fence.proxy.tensormap::generic.release.<scope>`).
///
/// Threads that are going to use the modified tensor map must then execute
/// [`tensormap_fence_acquire`].
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>
#[cfg(target_feature = "sm_90")]
#[inline]
pub fn tensormap_fence_release<const SCOPE: Scope>() {
    const_map!(SCOPE => S: nv::Scope {
        Scope::Cta => nv::Scope::Cta,
        Scope::Cluster => nv::Scope::Cluster,
        Scope::Gpu => nv::Scope::Gpu,
        Scope::System => nv::Scope::Sys,
    } in unsafe { nv::fence_proxy_tensormap_generic_release::<S>() })
}

/// Acquires the tensor map `tmap` (all of its 128 bytes) for the tensormap proxy at scope
/// `SCOPE`, so that modifications released by [`tensormap_fence_release`] are observed by
/// subsequent TMA operations of the executing thread
/// (`fence.proxy.tensormap::generic.acquire.<scope>`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar>
///
/// # Safety
///
/// `tmap` must reside in global memory; otherwise the behavior is undefined.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tensormap_fence_acquire<const SCOPE: Scope>(tmap: &TmaDescriptor) {
    const_map!(SCOPE => S: nv::Scope {
        Scope::Cta => nv::Scope::Cta,
        Scope::Cluster => nv::Scope::Cluster,
        Scope::Gpu => nv::Scope::Gpu,
        Scope::System => nv::Scope::Sys,
    } in unsafe {
        nv::fence_proxy_tensormap_generic_acquire::<S>(core::ptr::from_ref(tmap).cast())
    })
}
