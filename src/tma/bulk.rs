// --- LLM-generated --- //
//! Non-tensor bulk asynchronous copies (`cp.async.bulk`).
//!
//! These copy a contiguous range of bytes between global and shared memory. Pointers to shared
//! memory are ordinary generic-address-space pointers. Source and destination addresses must be
//! 16-byte aligned, and sizes must be a multiple of 16 bytes; the source and destination ranges
//! must not overflow their memory spaces. Otherwise the behavior is undefined.
//!
//! Copies into shared memory signal completion on an [`MBarrier`]: the barrier must expect the
//! number of bytes copied (`expect_tx`) before the copy completes. Copies out of shared memory
//! are tracked with bulk async-groups ([`bulk_commit_group`], [`bulk_wait_group`]).
//!
//! Each thread that executes one of these operations issues its own copy; typically a single
//! elected thread does so. The copies are performed by the async proxy.

#[allow(unused_imports)]
use core::arch::nvptx as nv;
#[allow(unused_imports)]
use core::ffi::c_void;

#[allow(unused_imports)]
use super::CtaMask;
#[cfg(target_feature = "sm_90")]
use super::TmaReduceOp;
#[cfg(target_feature = "sm_90")]
#[allow(unused_imports)]
use super::{bulk_commit_group, bulk_wait_group};
#[allow(unused_imports)]
use crate::mbarrier::MBarrier;
#[allow(unused_imports)]
use crate::memory::CachePolicy;
#[cfg(target_feature = "sm_90")]
use crate::{
    Scope,
    atomic::{AtomicSpace, space_address},
};
#[cfg(target_feature = "sm_90")]
use core::arch::asm;

/// Asynchronously copies `size` bytes from global memory at `src` to shared memory of the
/// executing CTA at `dst`, and performs a complete-transaction of `size` bytes on `mbar`
/// when the copy completes (`cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes`).
///
/// With `cache_policy`, the L2 eviction policy is applied to the read (`.L2::cache_hint`).
///
/// Requires `sm_90` and PTX ISA 8.6.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>
///
/// # Safety
///
/// - `dst` must point to `size` bytes of shared memory of the executing CTA, and `src` to `size`
///   bytes of global memory, both 16-byte aligned.
/// - `size` must be a multiple of 16.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA.
/// - The destination must not be accessed until the completion of the copy has been observed
///   through `mbar`.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_load(
    dst: *mut u8,
    src: *const u8,
    size: u32,
    mbar: &MBarrier,
    cache_policy: Option<CachePolicy>,
) {
    let mbar = mbar.as_ptr().cast();
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_global_to_shared_cta::<CH>(dst.cast(), mbar, src.cast(), size, policy)
    })
}

/// Asynchronously copies `size` bytes from global memory at `src` to shared memory of a CTA of the
/// executing cluster at `dst`, and performs a complete-transaction of `size` bytes on `mbar` when
/// the copy completes
/// (`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`).
///
/// If `multicast` is given, the data is copied to the same `dst` offset in the shared memory of
/// each CTA of the cluster selected by the mask, and the complete-transaction is performed on the
/// mbarrier at the same offset as `mbar` in each of them (`.multicast::cluster`). Otherwise the
/// data is copied to the CTA that `dst` points to, and `mbar` itself is signaled.
///
/// With `cache_policy`, the L2 eviction policy is applied to the read (`.L2::cache_hint`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>
///
/// # Safety
///
/// - `dst` must be a generic address of shared memory of a CTA in the executing cluster, valid for
///   `size` bytes, and `src` must point to `size` bytes of global memory, both 16-byte aligned.
/// - `size` must be a multiple of 16.
/// - `mbar` must be an initialized mbarrier in shared memory of the executing CTA (and, when
///   multicasting, the mbarrier at the same offset must be initialized in each destination CTA).
/// - Every CTA selected by `multicast` must be part of the executing cluster and still running.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_load_to_cluster(
    dst: *mut u8,
    src: *const u8,
    size: u32,
    mbar: &MBarrier,
    multicast: Option<CtaMask>,
    cache_policy: Option<CachePolicy>,
) {
    let mbar = mbar.as_ptr().cast();
    with_multicast!(multicast, MC, mask => with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_global_to_shared_cluster::<CH, MC>(
            dst.cast(),
            mbar,
            src.cast(),
            size,
            mask,
            policy,
        )
    }))
}

/// Asynchronously copies `size` bytes from shared memory of the executing CTA at `src` to shared
/// memory of a different CTA of the executing cluster at `dst`, and performs a
/// complete-transaction of `size` bytes on `mbar` when the copy completes
/// (`cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>
///
/// # Safety
///
/// - `src` must point to `size` bytes of shared memory of the executing CTA, and `dst` must be a
///   generic address of shared memory of a CTA in the executing cluster other than the executing
///   CTA, valid for `size` bytes, both 16-byte aligned.
/// - `size` must be a multiple of 16.
/// - `mbar` must be the generic address of an initialized mbarrier in shared memory of the
///   destination CTA, which expects `size` bytes.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_copy_to_cluster(dst: *mut u8, src: *const u8, size: u32, mbar: &MBarrier) {
    unsafe {
        nv::cp_async_bulk_shared_cta_to_cluster(dst.cast(), mbar.as_ptr().cast(), src.cast(), size)
    }
}

/// Asynchronously copies `size` bytes from shared memory of the executing CTA at `src` to global
/// memory at `dst` (`cp.async.bulk.global.shared::cta.bulk_group`).
///
/// The copy is tracked by the bulk async-group mechanism: call [`bulk_commit_group`] and then
/// [`bulk_wait_group`] to wait for it. With `cache_policy`, the L2 eviction policy is applied to
/// the write (`.L2::cache_hint`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>
///
/// # Safety
///
/// - `src` must point to `size` bytes of shared memory of the executing CTA, and `dst` to `size`
///   bytes of global memory, both 16-byte aligned.
/// - `size` must be a multiple of 16.
/// - Prior generic-proxy writes to `src` must have been made visible to the async proxy
///   (`fence.proxy.async`), and `src` must not be modified until the copy has finished reading
///   it (`bulk_wait_group::<_, { BulkWait::SourceRead }>`).
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_store(
    dst: *mut u8,
    src: *const u8,
    size: u32,
    cache_policy: Option<CachePolicy>,
) {
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_shared_cta_to_global::<CH>(dst.cast(), src.cast(), size, policy)
    })
}

/// Like [`bulk_store`], but only writes the bytes of `src` selected by `byte_mask`
/// (`cp.async.bulk.global.shared::cta.bulk_group.cp_mask`).
///
/// For each 16-byte chunk of the source, byte `i` is written only if bit `i` of `byte_mask` is
/// set.
///
/// Requires `sm_100` and PTX ISA 8.6.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk>
///
/// # Safety
///
/// Same requirements as [`bulk_store`].
#[cfg(target_feature = "sm_100")]
#[inline]
pub unsafe fn bulk_store_masked(
    dst: *mut u8,
    src: *const u8,
    size: u32,
    byte_mask: u16,
    cache_policy: Option<CachePolicy>,
) {
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_shared_cta_to_global_bytemask::<CH>(
            dst.cast(),
            src.cast(),
            size,
            policy,
            byte_mask,
        )
    })
}

/// Prefetches `size` bytes of global memory starting at `src` into the L2 cache
/// (`cp.async.bulk.prefetch.L2.global`).
///
/// This is only a hint, which may be ignored; it is treated as a weak memory operation and is
/// not tracked by any completion mechanism. With `cache_policy`, the L2 eviction policy is
/// applied to the prefetched data (`.L2::cache_hint`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch>
///
/// # Safety
///
/// `src` must be a 16-byte aligned pointer into global memory, and `size` a multiple of 16.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_prefetch_l2(src: *const u8, size: u32, cache_policy: Option<CachePolicy>) {
    with_cache_policy!(cache_policy, CH, policy => unsafe {
        nv::cp_async_bulk_prefetch_l2::<CH>(src.cast(), size, policy)
    })
}

#[cfg(target_feature = "sm_90")]
mod sealed {
    /// Element type of a bulk reduction, which selects the `.type` qualifier.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Element {
        U32,
        S32,
        U64,
        S64,
        F32,
        F64,
        F16,
        Bf16,
    }

    pub trait BulkReduce: Copy {
        const ELEMENT: Element;
    }
}

/// Element types of [`bulk_reduce`] and [`bulk_reduce_to_cluster`].
///
/// This trait is sealed. Not every operation is supported for every element type and destination:
///
/// | Operation | [`bulk_reduce`] (global) | [`bulk_reduce_to_cluster`] (shared memory) |
/// |---|---|---|
/// | `Add` | `u32`, `i32`, `u64`, `i64`, `f32`, `f64`, [`f16`](struct@crate::f16), [`bf16`](crate::bf16) | `u32`, `i32`, `u64`, `i64` |
/// | `Min`, `Max` | `u32`, `i32`, `u64`, `i64`, [`f16`](struct@crate::f16), [`bf16`](crate::bf16) | `u32`, `i32` |
/// | `WrapIncrement`, `WrapDecrement` | `u32` | `u32` |
/// | `And`, `Or`, `Xor` | `u32`, `i32`, `u64`, `i64` | `u32`, `i32` |
///
/// Unsupported combinations are rejected at compile time. `i64` is added with the `.u64` form, as
/// PTX has no `.add.s64`; the result is identical in two's complement. The bitwise operations use
/// the `.b32` and `.b64` forms. Floating-point addition rounds to nearest even and preserves
/// subnormal inputs and results (`.noftz` for `f16` and `bf16`).
#[cfg(target_feature = "sm_90")]
pub trait BulkReduceElement: sealed::BulkReduce {}

#[cfg(target_feature = "sm_90")]
macro_rules! impl_bulk_reduce {
    ($($ty:ty => $element:ident),+ $(,)?) => {$(
        impl sealed::BulkReduce for $ty {
            const ELEMENT: sealed::Element = sealed::Element::$element;
        }
        impl BulkReduceElement for $ty {}
    )+};
}

#[cfg(target_feature = "sm_90")]
impl_bulk_reduce! {
    u32 => U32,
    i32 => S32,
    u64 => U64,
    i64 => S64,
    f32 => F32,
    f64 => F64,
    crate::f16 => F16,
    crate::bf16 => Bf16,
}

#[cfg(target_feature = "sm_90")]
const fn assert_bulk_reduce_supported(
    op: TmaReduceOp,
    element: sealed::Element,
    scope: Scope,
    to_cluster: bool,
) {
    use TmaReduceOp as Op;
    use sealed::Element as E;
    let supported = if to_cluster {
        matches!(
            (op, element),
            (Op::Add, E::U32 | E::S32 | E::U64 | E::S64)
                | (Op::Min | Op::Max, E::U32 | E::S32)
                | (Op::WrapIncrement | Op::WrapDecrement, E::U32)
                | (Op::And | Op::Or | Op::Xor, E::U32 | E::S32)
        )
    } else {
        matches!(
            (op, element),
            (Op::Add, _)
                | (
                    Op::Min | Op::Max,
                    E::U32 | E::S32 | E::U64 | E::S64 | E::F16 | E::Bf16
                )
                | (Op::WrapIncrement | Op::WrapDecrement, E::U32)
                | (
                    Op::And | Op::Or | Op::Xor,
                    E::U32 | E::S32 | E::U64 | E::S64
                )
        )
    };
    assert!(
        supported,
        "unsupported combination of reduction operation and element type, see BulkReduceElement"
    );
    assert!(
        !to_cluster || matches!(scope, Scope::Cta | Scope::Cluster),
        "bulk_reduce_to_cluster only supports Scope::Cta and Scope::Cluster"
    );
}

/// Invokes `asm_variants!` with the `.scope` and `.redOp.type` qualifiers of a bulk reduction.
#[cfg(target_feature = "sm_90")]
macro_rules! bulk_reduce_variants {
    ($cb:ident!($($pre:tt)*), $scope:expr, $op:expr, $element:expr) => {{
        use sealed::Element as E;
        use TmaReduceOp as Op;
        asm_variants!($cb!($($pre)*);
            $scope => {
                Scope::Cta => ".cta",
                Scope::Cluster => ".cluster",
                Scope::Gpu => ".gpu",
                Scope::System => ".sys",
            },
            ($op, $element) => {
                (Op::Add, E::U32) => ".add.u32",
                (Op::Add, E::S32) => ".add.s32",
                (Op::Add, E::U64 | E::S64) => ".add.u64",
                (Op::Add, E::F32) => ".add.f32",
                (Op::Add, E::F64) => ".add.f64",
                (Op::Add, E::F16) => ".add.noftz.f16",
                (Op::Add, E::Bf16) => ".add.noftz.bf16",
                (Op::Min, E::U32) => ".min.u32",
                (Op::Min, E::S32) => ".min.s32",
                (Op::Min, E::U64) => ".min.u64",
                (Op::Min, E::S64) => ".min.s64",
                (Op::Min, E::F16) => ".min.f16",
                (Op::Min, E::Bf16) => ".min.bf16",
                (Op::Max, E::U32) => ".max.u32",
                (Op::Max, E::S32) => ".max.s32",
                (Op::Max, E::U64) => ".max.u64",
                (Op::Max, E::S64) => ".max.s64",
                (Op::Max, E::F16) => ".max.f16",
                (Op::Max, E::Bf16) => ".max.bf16",
                (Op::WrapIncrement, E::U32) => ".inc.u32",
                (Op::WrapDecrement, E::U32) => ".dec.u32",
                (Op::And, E::U32 | E::S32) => ".and.b32",
                (Op::And, E::U64 | E::S64) => ".and.b64",
                (Op::Or, E::U32 | E::S32) => ".or.b32",
                (Op::Or, E::U64 | E::S64) => ".or.b64",
                (Op::Xor, E::U32 | E::S32) => ".xor.b32",
                (Op::Xor, E::U64 | E::S64) => ".xor.b64",
            },
        )
    }};
}

/// Asynchronously reduces `size` bytes of global memory at `dst` with the elements of shared
/// memory of the executing CTA at `src`, replacing each element of `dst` with the result of the
/// operation `OP` applied to it and the corresponding element of `src`
/// (`cp.reduce.async.bulk.relaxed.scope.global.shared::cta.bulk_group.redOp.type`).
///
/// Each element of `dst` is reduced atomically, with `.relaxed` semantics at scope `SCOPE`; there
/// is no ordering between different elements. The reads of `src` are weak memory operations. See
/// [`BulkReduceElement`] for the supported combinations of `OP` and `T`.
///
/// The reduction is tracked by the bulk async-group mechanism: call [`bulk_commit_group`] and then
/// [`bulk_wait_group`] to wait for it. With `cache_policy`, the L2 eviction policy is applied to
/// the access to `dst` (`.L2::cache_hint`). The memory operations are performed in the async
/// proxy.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk>
///
/// # Safety
///
/// - `src` must point to `size` bytes of shared memory of the executing CTA, and `dst` to `size`
///   bytes of global memory, both 16-byte aligned.
/// - `size` must be a multiple of 16.
/// - The reduction must not cause a data race on `dst`: conflicting concurrent accesses must be
///   atomic operations whose scope includes the executing thread, and vice versa.
/// - Prior generic-proxy writes to `src` must have been made visible to the async proxy
///   (`fence.proxy.async`), and `src` must not be modified until the reduction has finished
///   reading it (`bulk_wait_group::<_, { BulkWait::SourceRead }>`).
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_reduce<T: BulkReduceElement, const OP: TmaReduceOp, const SCOPE: Scope>(
    dst: *mut T,
    src: *const T,
    size: u32,
    cache_policy: Option<CachePolicy>,
) {
    const { assert_bulk_reduce_supported(OP, T::ELEMENT, SCOPE, false) };
    let dst = space_address::<{ AtomicSpace::Global }, _>(dst);
    let src = space_address::<{ AtomicSpace::SharedCta }, _>(src.cast_mut());
    macro_rules! emit {
        (hint $policy:ident $scope:literal, $op:literal) => {
            asm!(
                concat!(
                    "cp.reduce.async.bulk.relaxed", $scope,
                    ".global.shared::cta.bulk_group.L2::cache_hint", $op,
                    " [{dst}], [{src}], {size}, {policy};"
                ),
                dst = in(reg64) dst,
                src = in(reg64) src,
                size = in(reg32) size,
                policy = in(reg64) $policy,
                options(nostack),
            )
        };
        ($scope:literal, $op:literal) => {
            asm!(
                concat!(
                    "cp.reduce.async.bulk.relaxed", $scope, ".global.shared::cta.bulk_group", $op,
                    " [{dst}], [{src}], {size};"
                ),
                dst = in(reg64) dst,
                src = in(reg64) src,
                size = in(reg32) size,
                options(nostack),
            )
        };
    }
    match cache_policy {
        Some(policy) => {
            let policy = policy.to_bits();
            unsafe { bulk_reduce_variants!(emit!(hint policy), SCOPE, OP, T::ELEMENT) }
        }
        None => unsafe { bulk_reduce_variants!(emit!(), SCOPE, OP, T::ELEMENT) },
    }
}

/// Asynchronously reduces `size` bytes of shared memory of a CTA of the executing cluster at
/// `dst` with the elements of shared memory of the executing CTA at `src`, replacing each element
/// of `dst` with the result of the operation `OP` applied to it and the corresponding element of
/// `src`, and performs a complete-transaction of `size` bytes on `mbar` when the reduction
/// completes
/// (`cp.reduce.async.bulk.relaxed.scope.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.redOp.type`).
///
/// Each element of `dst` is reduced atomically, with `.relaxed` semantics at scope `SCOPE`, which
/// must be [`Scope::Cta`] or [`Scope::Cluster`]; there is no ordering between different elements.
/// Since the destination is usually in another CTA, [`Scope::Cluster`] is typically needed for the
/// reduction to be atomic with respect to the threads of the destination CTA. The reads of `src`
/// are weak memory operations. See [`BulkReduceElement`] for the supported combinations of `OP`
/// and `T`.
///
/// The complete-transaction on `mbar` has release semantics at cluster scope, so the results are
/// visible to the threads that observe the completion of the barrier phase. The memory operations
/// are performed in the async proxy; `mbar` is accessed through the generic proxy.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk>
///
/// # Safety
///
/// - `src` must point to `size` bytes of shared memory of the executing CTA, and `dst` must be a
///   generic address of shared memory of a CTA in the executing cluster (e.g. obtained from
///   [`map_to_cta`](crate::cluster::map_to_cta)), valid for `size` bytes, both 16-byte aligned.
/// - `size` must be a multiple of 16.
/// - `mbar` must be the generic address of an initialized mbarrier in shared memory of the
///   destination CTA, which expects `size` bytes.
/// - The reduction must not cause a data race on `dst`: conflicting concurrent accesses must be
///   atomic operations whose scope includes the executing thread, and vice versa.
/// - Prior generic-proxy writes to `src` and `dst` must have been made visible to the async proxy
///   (`fence.proxy.async`), and `src` must not be modified until the completion of the reduction
///   has been observed through `mbar`.
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn bulk_reduce_to_cluster<
    T: BulkReduceElement,
    const OP: TmaReduceOp,
    const SCOPE: Scope,
>(
    dst: *mut T,
    src: *const T,
    size: u32,
    mbar: &MBarrier,
) {
    const { assert_bulk_reduce_supported(OP, T::ELEMENT, SCOPE, true) };
    let dst = space_address::<{ AtomicSpace::SharedCluster }, _>(dst);
    let src = space_address::<{ AtomicSpace::SharedCta }, _>(src.cast_mut());
    let mbar = space_address::<{ AtomicSpace::SharedCluster }, _>(mbar.as_ptr());
    macro_rules! emit {
        ($scope:literal, $op:literal) => {
            asm!(
                concat!(
                    "cp.reduce.async.bulk.relaxed", $scope,
                    ".shared::cluster.shared::cta.mbarrier::complete_tx::bytes", $op,
                    " [{dst}], [{src}], {size}, [{mbar}];"
                ),
                dst = in(reg64) dst,
                src = in(reg64) src,
                size = in(reg32) size,
                mbar = in(reg64) mbar,
                options(nostack),
            )
        };
    }
    unsafe { bulk_reduce_variants!(emit!(), SCOPE, OP, T::ELEMENT) }
}
