// --- LLM-generated --- //
//! Tensor Memory Accelerator (TMA): tensor maps, bulk copies, and bulk tensor copies.
//!
//! The TMA is a hardware unit that copies data asynchronously between global and shared memory.
//! Two families of operations are provided:
//!
//! - Bulk copies of a contiguous range of bytes (`cp.async.bulk`, see `bulk_load`,
//!   `bulk_store`, ...).
//! - Tensor copies (`cp.async.bulk.tensor`, see `tma_load_tile`, `tma_store_tile`, ...) which
//!   move multi-dimensional boxes of a tensor described by a [`TmaDescriptor`].
//!
//! # Completion mechanisms
//!
//! Copies into shared memory complete asynchronously and signal their completion on an
//! [`MBarrier`](crate::MBarrier): the barrier must be told beforehand how many bytes to expect
//! (via its `expect_tx` operation), and the copy performs a complete-transaction on it once the
//! data has landed in shared memory. The complete-transaction has release semantics at cluster
//! scope, so the data is visible to threads that observe the completion of the barrier phase.
//!
//! Copies out of shared memory (stores, reductions) are tracked using *bulk async-groups*
//! instead: `bulk_commit_group` closes the group of copies issued so far by the executing
//! thread, and `bulk_wait_group` waits for committed groups to complete. Bulk async-groups
//! cover both the non-tensor and tensor shared-to-global copies. Prefetches have no completion
//! mechanism.
//!
//! Each thread that executes a copy operation issues its own copy: to perform a single copy,
//! execute the operation in one (elected) thread only.
//!
//! # Proxies
//!
//! TMA operations are performed by the *async proxy*. Writes to shared memory through the generic
//! proxy (ordinary stores) must be made visible to it with a proxy fence (`fence.proxy.async`)
//! before a TMA store reads them, and likewise for global memory written by ordinary stores and
//! later read by a TMA load. The completion of a copy is followed by an implicit generic-async
//! proxy fence, so its results are visible to the generic proxy as soon as the completion has been
//! observed.
//!
//! Tensor maps are read through yet another proxy, the *tensormap proxy*; see [`TmaDescriptor`].

#[allow(unused_macros)]
macro_rules! with_cache_policy {
    ($policy:expr, $hint:ident, $bits:ident => $body:expr) => {
        match $policy {
            Some(policy) => {
                const $hint: bool = core::gca!(true);
                let $bits = policy.to_bits();
                $body
            }
            None => {
                const $hint: bool = core::gca!(false);
                let $bits = 0u64;
                $body
            }
        }
    };
}

#[allow(unused_macros)]
macro_rules! with_multicast {
    ($mask:expr, $flag:ident, $bits:ident => $body:expr) => {
        match $mask {
            Some(mask) => {
                const $flag: bool = core::gca!(true);
                let $bits = mask.to_bits();
                $body
            }
            None => {
                const $flag: bool = core::gca!(false);
                let $bits = 0u16;
                $body
            }
        }
    };
}

mod bulk;
mod descriptor;
mod tensor;

use core::marker::ConstParamTy;

use bytemuck::{Pod, Zeroable};
#[allow(unused_imports)]
use core::arch::nvptx as nv;

#[allow(unused_imports)]
pub use bulk::*;
pub use descriptor::*;
pub use tensor::*;

/// Whether the target supports the Blackwell additions to `cp.async.bulk.tensor`: `.cta_group`,
/// the `gather4`/`scatter4` modes, and the `im2col::w` modes for `.shared::cluster` copies.
#[allow(dead_code)]
pub(crate) const BLACKWELL_TMA: bool = cfg!(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f"
));

/// A set of CTAs within the executing cluster, identified by their rank in the cluster
/// (`%cluster_ctarank`). Bit `i` of the mask selects the CTA of rank `i`.
///
/// Used to select the destination CTAs of a multicast copy.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct CtaMask(u16);

impl CtaMask {
    /// The empty set.
    pub const EMPTY: Self = Self(0);

    /// Creates a mask from its raw bits, where bit `i` selects the CTA of rank `i`.
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bits of the mask.
    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// The set containing only the CTA of rank `rank`. `rank` must be less than 16.
    #[inline]
    pub const fn single(rank: u32) -> Self {
        assert!(rank < 16, "cluster rank out of range");
        Self(1 << rank)
    }

    /// The set containing the first `count` CTAs of the cluster. `count` must be at most 16.
    #[inline]
    pub const fn first(count: u32) -> Self {
        assert!(count <= 16, "cluster size out of range");
        Self(((1u32 << count) - 1) as u16)
    }

    /// Returns whether the CTA of rank `rank` is in the set.
    #[inline]
    pub const fn contains(self, rank: u32) -> bool {
        rank < 16 && self.0 & (1 << rank) != 0
    }

    /// Returns the set with the CTA of rank `rank` added. `rank` must be less than 16.
    #[inline]
    pub const fn with(self, rank: u32) -> Self {
        Self(self.0 | Self::single(rank).0)
    }
}

impl core::ops::BitOr for CtaMask {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for CtaMask {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl core::ops::BitAnd for CtaMask {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

/// What `bulk_wait_group` waits for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum BulkWait {
    /// Wait until the groups have fully completed: their writes are performed and visible to the
    /// executing thread.
    Completion,
    /// Wait only until the groups have finished *reading* their source memory and, for tensor
    /// copies, the tensor map (`.read`).
    ///
    /// The source shared memory may be reused afterwards, but the writes to the destination are
    /// not necessarily complete or visible.
    SourceRead,
}

/// Commits all prior uncommitted bulk async operations of the executing thread into a new bulk
/// async-group (`cp.async.bulk.commit_group`).
///
/// Bulk async-groups track the shared-to-global bulk copies ([`bulk_store`],
/// [`bulk_store_masked`]) and the tensor stores and reductions ([`tma_store_tile`],
/// [`tma_reduce_tile`], ...). Loads into shared memory are tracked by mbarriers instead.
/// If there are no uncommitted operations, an empty group is created. The groups are per-thread,
/// and there is no ordering between the operations of a group.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group>
#[cfg(target_feature = "sm_90")]
#[inline]
pub fn bulk_commit_group() {
    unsafe { nv::cp_async_bulk_commit_group() }
}

/// Waits until at most `N` of the most recently committed bulk async-groups of the executing
/// thread are pending (`cp.async.bulk.wait_group`).
///
/// All committed groups older than the `N` most recent ones are complete when the function
/// returns, so `N = 0` waits for all committed groups. With [`BulkWait::SourceRead`], only waits
/// until the groups have finished reading their source shared memory and tensor map (`.read`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group>
#[cfg(target_feature = "sm_90")]
#[inline]
pub fn bulk_wait_group<const N: u32, const WAIT: BulkWait>() {
    const_bool!(READ = matches!(WAIT, BulkWait::SourceRead) => unsafe {
        nv::cp_async_bulk_wait_group::<N, READ>()
    })
}

/// Prefetches the cache line containing the tensor map `tmap` (`prefetch.tensormap`), so that
/// later TMA operations using it start faster.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prefetch-prefetchu>
///
/// # Safety
///
/// `tmap` must reside in constant or kernel parameter memory: the PTX ISA only specifies the
/// prefetch for tensor maps in these state spaces.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn prefetch_tensormap(tmap: &TmaDescriptor) {
    unsafe { nv::prefetch_tensormap(core::ptr::from_ref(tmap).cast()) }
}
