// --- LLM-generated --- //
//! Bulk asynchronous tensor copies (`cp.async.bulk.tensor`, `cp.reduce.async.bulk.tensor`,
//! `cp.async.bulk.prefetch.tensor`).
//!
//! All of these operate on a tensor described by a [`TmaDescriptor`] and move a *box* of the
//! tensor between global and shared memory, starting at the tensor coordinates `coords`.
//!
//! # Conventions
//!
//! - `N` is the rank of the tensor (and the number of coordinates), from 1 to 5. Tile copies
//!   support all ranks, im2col copies ranks 3 to 5. `coords[0]` is the coordinate of the
//!   innermost (contiguous) dimension of the tensor. Coordinates are signed and may be out of
//!   bounds: out-of-bounds elements read as the fill value of the tensor map and are not written.
//! - Shared memory pointers are generic addresses. The shared memory box must be aligned to
//!   128 bytes (or more, depending on the swizzle mode of the tensor map).
//! - Loads into shared memory signal completion on an [`MBarrier`]: the barrier must expect the
//!   number of bytes of the box (`expect_tx`) before the copy completes. Stores and reductions
//!   are tracked using bulk async-groups ([`bulk_commit_group`], [`bulk_wait_group`]).
//! - `cache_policy` optionally applies an L2 eviction policy to the copy (`.L2::cache_hint`).
//! - `tmap` must reside in kernel parameter, constant, or global memory; see [`TmaDescriptor`].
//!
//! # Modes
//!
//! - **tile**: copies a dense box of the tensor.
//! - **gather4**: copies four rows of a 2D tensor, selected by arbitrary row indices, starting at
//!   a column offset. Blackwell only.
//! - **scatter4**: the store counterpart of gather4. Blackwell only.
//! - **im2col**: copies the box of an image-to-column transformed tensor, with the filter
//!   position given by per-spatial-dimension offsets (`N - 2` of them).
//! - **im2col::w** and **im2col::w::128**: im2col variants that traverse along the W dimension
//!   with a halo, given by a [`WTraversal`]. Blackwell only.
//!
//! # Destinations
//!
//! Loads come in two flavors. The plain `tma_load_*` functions write to shared memory of the
//! executing CTA. The `tma_load_*_to_cluster` functions write to `.shared::cluster` memory, which
//! allows targeting another CTA of the cluster, multicasting to several CTAs, and (on Blackwell)
//! the CTA-pair mode selected by [`CtaGroup`].
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>

#[allow(unused_imports)]
use core::arch::nvptx as nv;
use core::ffi::c_void;
use core::marker::ConstParamTy;

#[allow(unused_imports)]
use super::{BLACKWELL_TMA, CtaMask, TmaDescriptor};
#[cfg(target_feature = "sm_90")]
#[allow(unused_imports)]
use super::{bulk_commit_group, bulk_wait_group};
#[allow(unused_imports)]
use crate::mbarrier::MBarrier;
#[allow(unused_imports)]
use crate::memory::CachePolicy;
#[allow(unused_imports)]
use crate::modifiers::CtaGroup;

/// Converts an array to an array of a different length, for use in arms of a `match` on the
/// array length, where the lengths are known to be equal.
///
/// # Safety
///
/// `N` must be equal to `K` whenever the call is executed.
#[inline(always)]
#[allow(dead_code)]
unsafe fn array_as<T: Copy, const N: usize, const K: usize>(array: [T; N]) -> [T; K] {
    unsafe { core::mem::transmute_copy(&array) }
}

#[allow(dead_code)]
#[inline(always)]
fn cast_args(
    ptr: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
) -> (*mut c_void, *mut c_void, *const c_void) {
    (
        ptr.cast(),
        mbar.as_ptr().cast(),
        core::ptr::from_ref(tmap).cast(),
    )
}

#[allow(unused_macros)]
macro_rules! by_rank {
    ($n:expr, $coords:expr, { $($k:literal => |$c:ident| $body:expr),+ $(,)? }) => {
        $(if const { $n == $k } {
            let $c: [i32; $k] = unsafe { array_as($coords) };
            $body
        } else)+ {
            unsafe { core::hint::unreachable_unchecked() }
        }
    };
}

#[allow(unused_macros)]
macro_rules! by_rank_offsets {
    (
        $n:expr, $coords:expr, $offsets:expr,
        { $($k:literal, $m:literal => |$c:ident, $o:ident| $body:expr),+ $(,)? }
    ) => {
        $(if const { $n == $k } {
            let $c: [i32; $k] = unsafe { array_as($coords) };
            let $o: [u16; $m] = unsafe { array_as($offsets) };
            $body
        } else)+ {
            unsafe { core::hint::unreachable_unchecked() }
        }
    };
}

#[allow(unused_macros)]
macro_rules! with_cluster_options {
    (
        $multicast:expr, $policy:expr, $group:expr,
        |$mc:ident, $mask:ident, $ch:ident, $bits:ident, $cg:ident| $body:expr
    ) => {
        with_multicast!($multicast, $mc, $mask => with_cache_policy!($policy, $ch, $bits =>
            const_map!($group => $cg: i32 {
                CtaGroup::Unspecified => 0,
                CtaGroup::One => 1,
                CtaGroup::Two => 2,
            } in $body)
        ))
    };
}

/// Halo and offset of the traversal along the W dimension of the im2col::w and im2col::w::128
/// modes (`wHalo` and `wOffset`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WTraversal {
    /// Number of halo elements (`wHalo`).
    pub halo: u16,
    /// Offset of the traversal (`wOffset`).
    pub offset: u16,
}

/// Reduction operation applied by `tma_reduce_tile` and `tma_reduce_im2col` (`.redOp`).
///
/// The element type is given by the tensor map. Not every operation is supported for every
/// element type; see the PTX ISA.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum TmaReduceOp {
    /// Addition (`.add`).
    Add,
    /// Minimum (`.min`).
    Min,
    /// Maximum (`.max`).
    Max,
    /// Wrapping increment (`.inc`): `if old >= v { 0 } else { old + 1 }`.
    WrapIncrement,
    /// Wrapping decrement (`.dec`): `if old == 0 || old > v { v } else { old - 1 }`.
    WrapDecrement,
    /// Bitwise and (`.and`).
    And,
    /// Bitwise or (`.or`).
    Or,
    /// Bitwise xor (`.xor`).
    Xor,
}

/// Asynchronously copies the tile of the tensor described by `tmap` at `coords` from
/// global memory to shared memory of the executing CTA at `dst`, and performs a
/// complete-transaction on `mbar` when the copy completes
/// (`cp.async.bulk.tensor.Nd.shared::cta.global.tile.mbarrier::complete_tx::bytes`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `dst` must point to the shared memory of the executing CTA, aligned and large enough for the
///   box.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through `mbar`.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tma_load_tile<const N: usize>(
    dst: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    cache_policy: Option<CachePolicy>,
) {
    const { assert!(N >= 1 && N <= 5, "tensor rank must be between 1 and 5") };
    let (dst, mbar, tmap) = cast_args(dst, mbar, tmap);
    with_cache_policy!(cache_policy, CH, policy => by_rank!(N, coords, {
        1 => |c| unsafe { nv::cp_async_bulk_tensor_1d_g2s_cta_tile::<CH>(dst, mbar, tmap, c, policy) },
        2 => |c| unsafe { nv::cp_async_bulk_tensor_2d_g2s_cta_tile::<CH>(dst, mbar, tmap, c, policy) },
        3 => |c| unsafe { nv::cp_async_bulk_tensor_3d_g2s_cta_tile::<CH>(dst, mbar, tmap, c, policy) },
        4 => |c| unsafe { nv::cp_async_bulk_tensor_4d_g2s_cta_tile::<CH>(dst, mbar, tmap, c, policy) },
        5 => |c| unsafe { nv::cp_async_bulk_tensor_5d_g2s_cta_tile::<CH>(dst, mbar, tmap, c, policy) },
    }))
}

/// Asynchronously copies four rows of the 2D tensor described by `tmap` from global memory to
/// shared memory of the executing CTA at `dst`, and performs a complete-transaction on `mbar`
/// when the copy completes
/// (`cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes`).
///
/// The rows `rows` are copied from the column `column` onwards, to consecutive rows of the box
/// in shared memory.
///
/// Requires `sm_100`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `dst` must point to the shared memory of the executing CTA, aligned and large enough for the
///   box.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through `mbar`.
#[cfg(target_feature = "sm_100")]
#[inline]
pub unsafe fn tma_load_tile_gather4(
    dst: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
    column: i32,
    rows: [i32; 4],
    cache_policy: Option<CachePolicy>,
) {
    let (dst, mbar, tmap) = cast_args(dst, mbar, tmap);
    let coords = [column, rows[0], rows[1], rows[2], rows[3]];
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_tensor_2d_g2s_cta_tile_gather4::<CH>(dst, mbar, tmap, coords, policy)
    })
}

/// Asynchronously copies the im2col-layout box of the tensor described by `tmap` at `coords` from
/// global memory to shared memory of the executing CTA at `dst`, and performs a
/// complete-transaction on `mbar` when the copy completes
/// (`cp.async.bulk.tensor.Nd.shared::cta.global.im2col.mbarrier::complete_tx::bytes`).
///
/// `offsets` are the `N - 2` im2col offsets of the filter position, one per spatial dimension.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `dst` must point to the shared memory of the executing CTA, aligned and large enough for the
///   box.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through `mbar`.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tma_load_im2col<const N: usize, const M: usize>(
    dst: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    offsets: [u16; M],
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    const { assert!(M + 2 == N, "im2col copies require N - 2 offsets") };
    let (dst, mbar, tmap) = cast_args(dst, mbar, tmap);
    with_cache_policy!(cache_policy, CH, policy => by_rank_offsets!(N, coords, offsets, {
        3, 1 => |c, o| unsafe { nv::cp_async_bulk_tensor_3d_g2s_cta_im2col::<CH>(dst, mbar, tmap, c, o, policy) },
        4, 2 => |c, o| unsafe { nv::cp_async_bulk_tensor_4d_g2s_cta_im2col::<CH>(dst, mbar, tmap, c, o, policy) },
        5, 3 => |c, o| unsafe { nv::cp_async_bulk_tensor_5d_g2s_cta_im2col::<CH>(dst, mbar, tmap, c, o, policy) },
    }))
}

/// Asynchronously copies the im2col::w-layout box of the tensor described by `tmap` at `coords` from
/// global memory to shared memory of the executing CTA at `dst`, and performs a
/// complete-transaction on `mbar` when the copy completes
/// (`cp.async.bulk.tensor.Nd.shared::cta.global.im2col::w.mbarrier::complete_tx::bytes`).
///
/// `w` gives the halo and offset of the traversal along the W dimension.
///
/// Requires `sm_100`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `dst` must point to the shared memory of the executing CTA, aligned and large enough for the
///   box.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through `mbar`.
#[cfg(target_feature = "sm_100")]
#[inline]
pub unsafe fn tma_load_im2col_w<const N: usize>(
    dst: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    w: WTraversal,
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    let (dst, mbar, tmap) = cast_args(dst, mbar, tmap);
    with_cache_policy!(cache_policy, CH, policy => by_rank!(N, coords, {
        3 => |c| unsafe { nv::cp_async_bulk_tensor_3d_g2s_cta_im2col_w::<CH>(dst, mbar, tmap, c, w.halo, w.offset, policy) },
        4 => |c| unsafe { nv::cp_async_bulk_tensor_4d_g2s_cta_im2col_w::<CH>(dst, mbar, tmap, c, w.halo, w.offset, policy) },
        5 => |c| unsafe { nv::cp_async_bulk_tensor_5d_g2s_cta_im2col_w::<CH>(dst, mbar, tmap, c, w.halo, w.offset, policy) },
    }))
}

/// Asynchronously copies the im2col::w::128-layout box of the tensor described by `tmap` at `coords` from
/// global memory to shared memory of the executing CTA at `dst`, and performs a
/// complete-transaction on `mbar` when the copy completes
/// (`cp.async.bulk.tensor.Nd.shared::cta.global.im2col::w::128.mbarrier::complete_tx::bytes`).
///
/// `w` gives the halo and offset of the traversal along the W dimension.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f` (or the corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `dst` must point to the shared memory of the executing CTA, aligned and large enough for the
///   box.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through `mbar`.
#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f"
))]
#[inline]
pub unsafe fn tma_load_im2col_w_128<const N: usize>(
    dst: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    w: WTraversal,
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    let (dst, mbar, tmap) = cast_args(dst, mbar, tmap);
    with_cache_policy!(cache_policy, CH, policy => by_rank!(N, coords, {
        3 => |c| unsafe { nv::cp_async_bulk_tensor_3d_g2s_cta_im2col_w_128::<CH>(dst, mbar, tmap, c, w.halo, w.offset, policy) },
        4 => |c| unsafe { nv::cp_async_bulk_tensor_4d_g2s_cta_im2col_w_128::<CH>(dst, mbar, tmap, c, w.halo, w.offset, policy) },
        5 => |c| unsafe { nv::cp_async_bulk_tensor_5d_g2s_cta_im2col_w_128::<CH>(dst, mbar, tmap, c, w.halo, w.offset, policy) },
    }))
}

/// Asynchronously copies the tile of the tensor described by `tmap` at `coords` from
/// global memory to `.shared::cluster` memory at `dst`, and performs a complete-transaction on
/// the mbarrier at the same offset as `mbar` in the destination CTA when the copy completes
/// (`cp.async.bulk.tensor.Nd.shared::cluster.global.tile`).
///
/// If `multicast` is given, the data is copied to the same offset in the shared memory of each CTA
/// of the cluster selected by the mask, and each of them signals the mbarrier at the offset of
/// `mbar` (`.multicast::cluster`). `CTA_GROUP` selects the `.cta_group` qualifier.
///
/// Requires `sm_90`; a `CTA_GROUP` other than [`CtaGroup::Unspecified`] requires `sm_100f`, `sm_101f` or `sm_110f` (or the corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `dst` must be a generic address of shared memory of a CTA in the executing cluster, aligned
///   and large enough for the box.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA; when copying
///   to another CTA or multicasting, the mbarrier at the same offset must be initialized in each
///   destination CTA.
/// - Every CTA selected by `multicast` must be part of the executing cluster and still running.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through the mbarrier.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tma_load_tile_to_cluster<const N: usize, const CTA_GROUP: CtaGroup>(
    dst: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    multicast: Option<CtaMask>,
    cache_policy: Option<CachePolicy>,
) {
    const { assert!(N >= 1 && N <= 5, "tensor rank must be between 1 and 5") };
    const {
        assert!(
            matches!(CTA_GROUP, CtaGroup::Unspecified) || BLACKWELL_TMA,
            "`.cta_group` requires sm_100f, sm_101f or sm_110f"
        )
    };
    let (dst, mbar, tmap) = cast_args(dst, mbar, tmap);
    with_cluster_options!(
        multicast,
        cache_policy,
        CTA_GROUP,
        |MC, mask, CH, policy, CG| by_rank!(N, coords, {
            1 => |c| unsafe { nv::cp_async_bulk_tensor_1d_g2s_tile::<CG, MC, CH>(dst, mbar, tmap, c, mask, policy) },
            2 => |c| unsafe { nv::cp_async_bulk_tensor_2d_g2s_tile::<CG, MC, CH>(dst, mbar, tmap, c, mask, policy) },
            3 => |c| unsafe { nv::cp_async_bulk_tensor_3d_g2s_tile::<CG, MC, CH>(dst, mbar, tmap, c, mask, policy) },
            4 => |c| unsafe { nv::cp_async_bulk_tensor_4d_g2s_tile::<CG, MC, CH>(dst, mbar, tmap, c, mask, policy) },
            5 => |c| unsafe { nv::cp_async_bulk_tensor_5d_g2s_tile::<CG, MC, CH>(dst, mbar, tmap, c, mask, policy) },
        })
    )
}

/// Asynchronously copies four rows of the 2D tensor described by `tmap` from global memory to
/// `.shared::cluster` memory at `dst`, and performs a complete-transaction on the mbarrier at the
/// same offset as `mbar` in the destination CTA when the copy completes
/// (`cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4`).
///
/// The rows `rows` are copied from the column `column` onwards, to consecutive rows of the box
/// in shared memory. See [`tma_load_tile_to_cluster`] for `multicast` and `CTA_GROUP`.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f` (or the corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `dst` must be a generic address of shared memory of a CTA in the executing cluster, aligned
///   and large enough for the box.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA; when copying
///   to another CTA or multicasting, the mbarrier at the same offset must be initialized in each
///   destination CTA.
/// - Every CTA selected by `multicast` must be part of the executing cluster and still running.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through the mbarrier.
#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f"
))]
#[inline]
pub unsafe fn tma_load_tile_gather4_to_cluster<const CTA_GROUP: CtaGroup>(
    dst: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
    column: i32,
    rows: [i32; 4],
    multicast: Option<CtaMask>,
    cache_policy: Option<CachePolicy>,
) {
    let (dst, mbar, tmap) = cast_args(dst, mbar, tmap);
    let coords = [column, rows[0], rows[1], rows[2], rows[3]];
    with_cluster_options!(
        multicast,
        cache_policy,
        CTA_GROUP,
        |MC, mask, CH, policy, CG| unsafe {
            nv::cp_async_bulk_tensor_2d_g2s_tile_gather4::<CG, MC, CH>(
                dst, mbar, tmap, coords, mask, policy,
            )
        }
    )
}

/// Asynchronously copies the im2col-layout box of the tensor described by `tmap` at `coords` from
/// global memory to `.shared::cluster` memory at `dst`, and performs a complete-transaction on
/// the mbarrier at the same offset as `mbar` in the destination CTA when the copy completes
/// (`cp.async.bulk.tensor.Nd.shared::cluster.global.im2col`).
///
/// If `multicast` is given, the data is copied to the same offset in the shared memory of each CTA
/// of the cluster selected by the mask, and each of them signals the mbarrier at the offset of
/// `mbar` (`.multicast::cluster`). `CTA_GROUP` selects the `.cta_group` qualifier.
///
/// `offsets` are the `N - 2` im2col offsets of the filter position, one per spatial dimension.
///
/// Requires `sm_90`; a `CTA_GROUP` other than [`CtaGroup::Unspecified`] requires `sm_100f`, `sm_101f` or `sm_110f` (or the corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `dst` must be a generic address of shared memory of a CTA in the executing cluster, aligned
///   and large enough for the box.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA; when copying
///   to another CTA or multicasting, the mbarrier at the same offset must be initialized in each
///   destination CTA.
/// - Every CTA selected by `multicast` must be part of the executing cluster and still running.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through the mbarrier.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tma_load_im2col_to_cluster<
    const N: usize,
    const M: usize,
    const CTA_GROUP: CtaGroup,
>(
    dst: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    offsets: [u16; M],
    multicast: Option<CtaMask>,
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    const { assert!(M + 2 == N, "im2col copies require N - 2 offsets") };
    const {
        assert!(
            matches!(CTA_GROUP, CtaGroup::Unspecified) || BLACKWELL_TMA,
            "`.cta_group` requires sm_100f, sm_101f or sm_110f"
        )
    };
    let (dst, mbar, tmap) = cast_args(dst, mbar, tmap);
    with_cluster_options!(
        multicast,
        cache_policy,
        CTA_GROUP,
        |MC, mask, CH, policy, CG| by_rank_offsets!(N, coords, offsets, {
            3, 1 => |c, o| unsafe { nv::cp_async_bulk_tensor_3d_g2s_im2col::<CG, MC, CH>(dst, mbar, tmap, c, o, mask, policy) },
            4, 2 => |c, o| unsafe { nv::cp_async_bulk_tensor_4d_g2s_im2col::<CG, MC, CH>(dst, mbar, tmap, c, o, mask, policy) },
            5, 3 => |c, o| unsafe { nv::cp_async_bulk_tensor_5d_g2s_im2col::<CG, MC, CH>(dst, mbar, tmap, c, o, mask, policy) },
        })
    )
}

/// Asynchronously copies the im2col::w-layout box of the tensor described by `tmap` at `coords` from
/// global memory to `.shared::cluster` memory at `dst`, and performs a complete-transaction on
/// the mbarrier at the same offset as `mbar` in the destination CTA when the copy completes
/// (`cp.async.bulk.tensor.Nd.shared::cluster.global.im2col::w`).
///
/// If `multicast` is given, the data is copied to the same offset in the shared memory of each CTA
/// of the cluster selected by the mask, and each of them signals the mbarrier at the offset of
/// `mbar` (`.multicast::cluster`). `CTA_GROUP` selects the `.cta_group` qualifier.
///
/// `w` gives the halo and offset of the traversal along the W dimension.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f` (or the corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `dst` must be a generic address of shared memory of a CTA in the executing cluster, aligned
///   and large enough for the box.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA; when copying
///   to another CTA or multicasting, the mbarrier at the same offset must be initialized in each
///   destination CTA.
/// - Every CTA selected by `multicast` must be part of the executing cluster and still running.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through the mbarrier.
#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f"
))]
#[inline]
pub unsafe fn tma_load_im2col_w_to_cluster<const N: usize, const CTA_GROUP: CtaGroup>(
    dst: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    w: WTraversal,
    multicast: Option<CtaMask>,
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    const {
        assert!(
            matches!(CTA_GROUP, CtaGroup::Unspecified) || BLACKWELL_TMA,
            "`.cta_group` requires sm_100f, sm_101f or sm_110f"
        )
    };
    let (dst, mbar, tmap) = cast_args(dst, mbar, tmap);
    with_cluster_options!(
        multicast,
        cache_policy,
        CTA_GROUP,
        |MC, mask, CH, policy, CG| by_rank!(N, coords, {
            3 => |c| unsafe { nv::cp_async_bulk_tensor_3d_g2s_im2col_w::<CG, MC, CH>(dst, mbar, tmap, c, w.halo, w.offset, mask, policy) },
            4 => |c| unsafe { nv::cp_async_bulk_tensor_4d_g2s_im2col_w::<CG, MC, CH>(dst, mbar, tmap, c, w.halo, w.offset, mask, policy) },
            5 => |c| unsafe { nv::cp_async_bulk_tensor_5d_g2s_im2col_w::<CG, MC, CH>(dst, mbar, tmap, c, w.halo, w.offset, mask, policy) },
        })
    )
}

/// Asynchronously copies the im2col::w::128-layout box of the tensor described by `tmap` at `coords` from
/// global memory to `.shared::cluster` memory at `dst`, and performs a complete-transaction on
/// the mbarrier at the same offset as `mbar` in the destination CTA when the copy completes
/// (`cp.async.bulk.tensor.Nd.shared::cluster.global.im2col::w::128`).
///
/// If `multicast` is given, the data is copied to the same offset in the shared memory of each CTA
/// of the cluster selected by the mask, and each of them signals the mbarrier at the offset of
/// `mbar` (`.multicast::cluster`). `CTA_GROUP` selects the `.cta_group` qualifier.
///
/// `w` gives the halo and offset of the traversal along the W dimension.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f` (or the corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `dst` must be a generic address of shared memory of a CTA in the executing cluster, aligned
///   and large enough for the box.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA; when copying
///   to another CTA or multicasting, the mbarrier at the same offset must be initialized in each
///   destination CTA.
/// - Every CTA selected by `multicast` must be part of the executing cluster and still running.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through the mbarrier.
#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f"
))]
#[inline]
pub unsafe fn tma_load_im2col_w_128_to_cluster<const N: usize, const CTA_GROUP: CtaGroup>(
    dst: *mut u8,
    mbar: &MBarrier,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    w: WTraversal,
    multicast: Option<CtaMask>,
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    const {
        assert!(
            matches!(CTA_GROUP, CtaGroup::Unspecified) || BLACKWELL_TMA,
            "`.cta_group` requires sm_100f, sm_101f or sm_110f"
        )
    };
    let (dst, mbar, tmap) = cast_args(dst, mbar, tmap);
    with_cluster_options!(
        multicast,
        cache_policy,
        CTA_GROUP,
        |MC, mask, CH, policy, CG| by_rank!(N, coords, {
            3 => |c| unsafe { nv::cp_async_bulk_tensor_3d_g2s_im2col_w_128::<CG, MC, CH>(dst, mbar, tmap, c, w.halo, w.offset, mask, policy) },
            4 => |c| unsafe { nv::cp_async_bulk_tensor_4d_g2s_im2col_w_128::<CG, MC, CH>(dst, mbar, tmap, c, w.halo, w.offset, mask, policy) },
            5 => |c| unsafe { nv::cp_async_bulk_tensor_5d_g2s_im2col_w_128::<CG, MC, CH>(dst, mbar, tmap, c, w.halo, w.offset, mask, policy) },
        })
    )
}

/// Asynchronously copies the tile from shared memory of the executing CTA at `src` to the tensor
/// described by `tmap` in global memory, at `coords`
/// (`cp.async.bulk.tensor.Nd.global.shared::cta.tile.bulk_group`).
///
/// The copy is tracked by the bulk async-group mechanism: call [`bulk_commit_group`] and then
/// [`bulk_wait_group`] to wait for it.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `src` must point to the shared memory of the executing CTA, aligned and large enough for the
///   box.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - Prior generic-proxy writes to `src` must have been made visible to the async proxy, and
///   `src` must not be modified until the copy has finished reading it
///   (`bulk_wait_group::<_, { BulkWait::SourceRead }>`).
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tma_store_tile<const N: usize>(
    src: *const u8,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    cache_policy: Option<CachePolicy>,
) {
    const { assert!(N >= 1 && N <= 5, "tensor rank must be between 1 and 5") };
    let (src, tmap): (*const c_void, *const c_void) =
        (src.cast(), core::ptr::from_ref(tmap).cast());
    with_cache_policy!(cache_policy, CH, policy => by_rank!(N, coords, {
        1 => |c| unsafe { nv::cp_async_bulk_tensor_1d_s2g_tile::<CH>(src, tmap, c, policy) },
        2 => |c| unsafe { nv::cp_async_bulk_tensor_2d_s2g_tile::<CH>(src, tmap, c, policy) },
        3 => |c| unsafe { nv::cp_async_bulk_tensor_3d_s2g_tile::<CH>(src, tmap, c, policy) },
        4 => |c| unsafe { nv::cp_async_bulk_tensor_4d_s2g_tile::<CH>(src, tmap, c, policy) },
        5 => |c| unsafe { nv::cp_async_bulk_tensor_5d_s2g_tile::<CH>(src, tmap, c, policy) },
    }))
}

/// Asynchronously copies the im2col-layout box from shared memory of the executing CTA at `src` to the tensor
/// described by `tmap` in global memory, at `coords`
/// (`cp.async.bulk.tensor.Nd.global.shared::cta.im2col.bulk_group`).
///
/// The copy is tracked by the bulk async-group mechanism: call [`bulk_commit_group`] and then
/// [`bulk_wait_group`] to wait for it.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `src` must point to the shared memory of the executing CTA, aligned and large enough for the
///   box.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - Prior generic-proxy writes to `src` must have been made visible to the async proxy, and
///   `src` must not be modified until the copy has finished reading it
///   (`bulk_wait_group::<_, { BulkWait::SourceRead }>`).
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tma_store_im2col<const N: usize>(
    src: *const u8,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    let (src, tmap): (*const c_void, *const c_void) =
        (src.cast(), core::ptr::from_ref(tmap).cast());
    with_cache_policy!(cache_policy, CH, policy => by_rank!(N, coords, {
        3 => |c| unsafe { nv::cp_async_bulk_tensor_3d_s2g_im2col::<CH>(src, tmap, c, policy) },
        4 => |c| unsafe { nv::cp_async_bulk_tensor_4d_s2g_im2col::<CH>(src, tmap, c, policy) },
        5 => |c| unsafe { nv::cp_async_bulk_tensor_5d_s2g_im2col::<CH>(src, tmap, c, policy) },
    }))
}

/// Asynchronously copies four rows from shared memory of the executing CTA at `src` to the 2D
/// tensor described by `tmap` in global memory, at the column `column` and the rows `rows`
/// (`cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group`).
///
/// The copy is tracked by the bulk async-group mechanism: call [`bulk_commit_group`] and then
/// [`bulk_wait_group`] to wait for it.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f` (or the corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
///
/// # Safety
///
/// - `src` must point to the shared memory of the executing CTA, aligned and large enough for the
///   box.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - Prior generic-proxy writes to `src` must have been made visible to the async proxy, and
///   `src` must not be modified until the copy has finished reading it
///   (`bulk_wait_group::<_, { BulkWait::SourceRead }>`).
#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f"
))]
#[inline]
pub unsafe fn tma_store_tile_scatter4(
    src: *const u8,
    tmap: &TmaDescriptor,
    column: i32,
    rows: [i32; 4],
    cache_policy: Option<CachePolicy>,
) {
    let (src, tmap): (*const c_void, *const c_void) =
        (src.cast(), core::ptr::from_ref(tmap).cast());
    let coords = [column, rows[0], rows[1], rows[2], rows[3]];
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_tensor_2d_s2g_tile_scatter4::<CH>(src, tmap, coords, policy)
    })
}

/// Asynchronously reduces the tile from shared memory of the executing CTA at `src` into the
/// tensor described by `tmap` in global memory, at `coords`, using the operation `OP`
/// (`cp.reduce.async.bulk.tensor.Nd.global.shared::cta.<op>.tile.bulk_group`).
///
/// The element type is given by the tensor map. The copy is tracked by the bulk async-group
/// mechanism: call [`bulk_commit_group`] and then [`bulk_wait_group`] to wait for it.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor>
///
/// # Safety
///
/// - `src` must point to the shared memory of the executing CTA, aligned and large enough for the
///   box.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - Prior generic-proxy writes to `src` must have been made visible to the async proxy, and
///   `src` must not be modified until the copy has finished reading it
///   (`bulk_wait_group::<_, { BulkWait::SourceRead }>`).
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tma_reduce_tile<const N: usize, const OP: TmaReduceOp>(
    src: *const u8,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    cache_policy: Option<CachePolicy>,
) {
    const { assert!(N >= 1 && N <= 5, "tensor rank must be between 1 and 5") };
    let (src, tmap): (*const c_void, *const c_void) =
        (src.cast(), core::ptr::from_ref(tmap).cast());
    with_cache_policy!(cache_policy, CH, policy => const_map!(OP => RO: nv::TensorReduceOp {
        TmaReduceOp::Add => nv::TensorReduceOp::Add,
        TmaReduceOp::Min => nv::TensorReduceOp::Min,
        TmaReduceOp::Max => nv::TensorReduceOp::Max,
        TmaReduceOp::WrapIncrement => nv::TensorReduceOp::Inc,
        TmaReduceOp::WrapDecrement => nv::TensorReduceOp::Dec,
        TmaReduceOp::And => nv::TensorReduceOp::And,
        TmaReduceOp::Or => nv::TensorReduceOp::Or,
        TmaReduceOp::Xor => nv::TensorReduceOp::Xor,
    } in by_rank!(N, coords, {
        1 => |c| unsafe { nv::cp_async_bulk_tensor_1d_reduce_tile::<RO, CH>(src, tmap, c, policy) },
        2 => |c| unsafe { nv::cp_async_bulk_tensor_2d_reduce_tile::<RO, CH>(src, tmap, c, policy) },
        3 => |c| unsafe { nv::cp_async_bulk_tensor_3d_reduce_tile::<RO, CH>(src, tmap, c, policy) },
        4 => |c| unsafe { nv::cp_async_bulk_tensor_4d_reduce_tile::<RO, CH>(src, tmap, c, policy) },
        5 => |c| unsafe { nv::cp_async_bulk_tensor_5d_reduce_tile::<RO, CH>(src, tmap, c, policy) },
    })))
}

/// Asynchronously reduces the im2col-layout box from shared memory of the executing CTA at `src` into the
/// tensor described by `tmap` in global memory, at `coords`, using the operation `OP`
/// (`cp.reduce.async.bulk.tensor.Nd.global.shared::cta.<op>.im2col.bulk_group`).
///
/// The element type is given by the tensor map. The copy is tracked by the bulk async-group
/// mechanism: call [`bulk_commit_group`] and then [`bulk_wait_group`] to wait for it.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk-tensor>
///
/// # Safety
///
/// - `src` must point to the shared memory of the executing CTA, aligned and large enough for the
///   box.
/// - `tmap` must be a valid tensor map in a state space allowed for TMA operations.
/// - Prior generic-proxy writes to `src` must have been made visible to the async proxy, and
///   `src` must not be modified until the copy has finished reading it
///   (`bulk_wait_group::<_, { BulkWait::SourceRead }>`).
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tma_reduce_im2col<const N: usize, const OP: TmaReduceOp>(
    src: *const u8,
    tmap: &TmaDescriptor,
    coords: [i32; N],
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    let (src, tmap): (*const c_void, *const c_void) =
        (src.cast(), core::ptr::from_ref(tmap).cast());
    with_cache_policy!(cache_policy, CH, policy => const_map!(OP => RO: nv::TensorReduceOp {
        TmaReduceOp::Add => nv::TensorReduceOp::Add,
        TmaReduceOp::Min => nv::TensorReduceOp::Min,
        TmaReduceOp::Max => nv::TensorReduceOp::Max,
        TmaReduceOp::WrapIncrement => nv::TensorReduceOp::Inc,
        TmaReduceOp::WrapDecrement => nv::TensorReduceOp::Dec,
        TmaReduceOp::And => nv::TensorReduceOp::And,
        TmaReduceOp::Or => nv::TensorReduceOp::Or,
        TmaReduceOp::Xor => nv::TensorReduceOp::Xor,
    } in by_rank!(N, coords, {
        3 => |c| unsafe { nv::cp_async_bulk_tensor_3d_reduce_im2col::<RO, CH>(src, tmap, c, policy) },
        4 => |c| unsafe { nv::cp_async_bulk_tensor_4d_reduce_im2col::<RO, CH>(src, tmap, c, policy) },
        5 => |c| unsafe { nv::cp_async_bulk_tensor_5d_reduce_im2col::<RO, CH>(src, tmap, c, policy) },
    })))
}

/// Prefetches the tile of the tensor described by `tmap` at `coords` into the L2
/// cache
/// (`cp.async.bulk.prefetch.tensor.Nd.L2.global.tile`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor>
///
/// # Safety
///
/// `tmap` must be a valid tensor map in a state space allowed for TMA operations.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tma_prefetch_tile<const N: usize>(
    tmap: &TmaDescriptor,
    coords: [i32; N],
    cache_policy: Option<CachePolicy>,
) {
    const { assert!(N >= 1 && N <= 5, "tensor rank must be between 1 and 5") };
    let tmap: *const c_void = core::ptr::from_ref(tmap).cast();
    with_cache_policy!(cache_policy, CH, policy => by_rank!(N, coords, {
        1 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_1d_tile::<CH>(tmap, c, policy) },
        2 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_2d_tile::<CH>(tmap, c, policy) },
        3 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_3d_tile::<CH>(tmap, c, policy) },
        4 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_4d_tile::<CH>(tmap, c, policy) },
        5 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_5d_tile::<CH>(tmap, c, policy) },
    }))
}

/// Prefetches four rows of the 2D tensor described by `tmap`, at the column `column` and the
/// rows `rows`, into the L2 cache (`cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4`).
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f` (or the corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor>
///
/// # Safety
///
/// `tmap` must be a valid tensor map in a state space allowed for TMA operations.
#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f"
))]
#[inline]
pub unsafe fn tma_prefetch_tile_gather4(
    tmap: &TmaDescriptor,
    column: i32,
    rows: [i32; 4],
    cache_policy: Option<CachePolicy>,
) {
    let tmap: *const c_void = core::ptr::from_ref(tmap).cast();
    let coords = [column, rows[0], rows[1], rows[2], rows[3]];
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_prefetch_tensor_2d_tile_gather4::<CH>(tmap, coords, policy)
    })
}

/// Prefetches the im2col-layout box of the tensor described by `tmap` at `coords` into the L2
/// cache
/// (`cp.async.bulk.prefetch.tensor.Nd.L2.global.im2col`).
///
/// `offsets` are the `N - 2` im2col offsets of the filter position, one per spatial dimension.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor>
///
/// # Safety
///
/// `tmap` must be a valid tensor map in a state space allowed for TMA operations.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn tma_prefetch_im2col<const N: usize, const M: usize>(
    tmap: &TmaDescriptor,
    coords: [i32; N],
    offsets: [u16; M],
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    const { assert!(M + 2 == N, "im2col copies require N - 2 offsets") };
    let tmap: *const c_void = core::ptr::from_ref(tmap).cast();
    with_cache_policy!(cache_policy, CH, policy => by_rank_offsets!(N, coords, offsets, {
        3, 1 => |c, o| unsafe { nv::cp_async_bulk_prefetch_tensor_3d_im2col::<CH>(tmap, c, o, policy) },
        4, 2 => |c, o| unsafe { nv::cp_async_bulk_prefetch_tensor_4d_im2col::<CH>(tmap, c, o, policy) },
        5, 3 => |c, o| unsafe { nv::cp_async_bulk_prefetch_tensor_5d_im2col::<CH>(tmap, c, o, policy) },
    }))
}

/// Prefetches the im2col::w-layout box of the tensor described by `tmap` at `coords` into the L2
/// cache
/// (`cp.async.bulk.prefetch.tensor.Nd.L2.global.im2col::w`).
///
/// `w` gives the halo and offset of the traversal along the W dimension.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f` (or the corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor>
///
/// # Safety
///
/// `tmap` must be a valid tensor map in a state space allowed for TMA operations.
#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f"
))]
#[inline]
pub unsafe fn tma_prefetch_im2col_w<const N: usize>(
    tmap: &TmaDescriptor,
    coords: [i32; N],
    w: WTraversal,
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    let tmap: *const c_void = core::ptr::from_ref(tmap).cast();
    with_cache_policy!(cache_policy, CH, policy => by_rank!(N, coords, {
        3 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_3d_im2col_w::<CH>(tmap, c, w.halo, w.offset, policy) },
        4 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_4d_im2col_w::<CH>(tmap, c, w.halo, w.offset, policy) },
        5 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_5d_im2col_w::<CH>(tmap, c, w.halo, w.offset, policy) },
    }))
}

/// Prefetches the im2col::w::128-layout box of the tensor described by `tmap` at `coords` into the L2
/// cache
/// (`cp.async.bulk.prefetch.tensor.Nd.L2.global.im2col::w::128`).
///
/// `w` gives the halo and offset of the traversal along the W dimension.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f` (or the corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor>
///
/// # Safety
///
/// `tmap` must be a valid tensor map in a state space allowed for TMA operations.
#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f"
))]
#[inline]
pub unsafe fn tma_prefetch_im2col_w_128<const N: usize>(
    tmap: &TmaDescriptor,
    coords: [i32; N],
    w: WTraversal,
    cache_policy: Option<CachePolicy>,
) {
    const {
        assert!(
            N >= 3 && N <= 5,
            "im2col copies require a tensor rank between 3 and 5"
        )
    };
    let tmap: *const c_void = core::ptr::from_ref(tmap).cast();
    with_cache_policy!(cache_policy, CH, policy => by_rank!(N, coords, {
        3 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_3d_im2col_w_128::<CH>(tmap, c, w.halo, w.offset, policy) },
        4 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_4d_im2col_w_128::<CH>(tmap, c, w.halo, w.offset, policy) },
        5 => |c| unsafe { nv::cp_async_bulk_prefetch_tensor_5d_im2col_w_128::<CH>(tmap, c, w.halo, w.offset, policy) },
    }))
}
