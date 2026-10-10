// --- LLM-generated --- //
//! 5th generation TensorCore instructions (`tcgen05`): Tensor Memory allocation and access, copies
//! from shared memory to Tensor Memory, matrix multiply-accumulate, and the associated
//! synchronization.
//!
//! These are implemented with inline PTX, because the corresponding LLVM intrinsics take pointers
//! into the Tensor Memory address space, which cannot be expressed in Rust.
//!
//! This module requires the Blackwell family of targets (`sm_100f`, `sm_101f` or `sm_110f`, or the
//! corresponding `a` targets). Some operations and variants additionally require
//! architecture-specific (`a`) targets; this is noted in their documentation and checked at
//! compile time.
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions>
//!
//! # Tensor Memory
//!
//! Tensor Memory is dedicated on-chip memory of the TensorCore. Per CTA, it is a two-dimensional
//! matrix of 128 rows, called *lanes*, and 512 *columns* of 32-bit cells. It is addressed by
//! [`TmemAddr`], which holds a lane index and a column index.
//!
//! Tensor Memory is allocated dynamically in units of columns (each allocated column spans all 128
//! lanes) by a single warp of the CTA with [`alloc`], which writes the address of the allocation to
//! shared memory. The number of columns must be a power of two in `32..=512`. All Tensor Memory
//! allocated by a kernel must be deallocated with [`dealloc`] before the kernel exits, and a CTA
//! that does not need to allocate anymore should call [`relinquish_alloc_permit`] so that other
//! CTAs resident on the SM can allocate.
//!
//! Threads move data between registers and Tensor Memory with [`load`] and [`store`] (and their
//! variants). A warp can only access a quarter of the lanes: the warp with index `w` in the CTA
//! accesses lanes `32 * (w % 4)..32 * (w % 4) + 32`, so a warpgroup is needed to access all
//! lanes. All columns are accessible to every warp.
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory>
//!
//! # CTA groups and issue granularity
//!
//! Most `tcgen05` instructions take a [`CtaGroup`] parameter: with [`CtaGroup::One`], the
//! operation is performed on the Tensor Memory of the executing CTA; with [`CtaGroup::Two`], on the
//! Tensor Memory of both CTAs of the *CTA pair* (the two CTAs of the cluster whose
//! `%cluster_ctarank` differs only in the lowest bit, each of which is the *peer CTA* of the
//! other). **All `tcgen05` instructions of a kernel must use the same `CtaGroup`**, and
//! [`CtaGroup::Unspecified`] is rejected at compile time.
//!
//! | Operations | Issued by |
//! |---|---|
//! | [`mma`] (and variants), [`copy`], [`shift_down`], [`commit`] | A single thread. With [`CtaGroup::Two`], a single thread of the CTA pair issues the operation for both CTAs, and the peer CTA must not have exited. |
//! | [`alloc`], [`dealloc`], [`relinquish_alloc_permit`] | A single warp of the CTA. With [`CtaGroup::Two`], one warp in each CTA of the pair, collectively (the first warp may block until the peer warp performs the operation). |
//! | [`load`], [`store`], [`wait_load`], [`wait_store`] (and variants) | A warp, accessing a quarter of the lanes of the executing CTA's Tensor Memory. |
//! | [`fence_before_thread_sync`], [`fence_after_thread_sync`] | Each thread that needs to order its own Tensor Memory accesses. |
//!
//! Single-thread operations are typically issued by one thread selected with
//! [`elect`](crate::warp_collectives::elect).
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-issue-granularity>
//!
//! # Memory consistency
//!
//! [`alloc`], [`dealloc`], [`relinquish_alloc_permit`], the fences, the waits and [`commit`] are
//! *synchronous*: they are ordered with respect to the other `tcgen05` operations of the thread in
//! program order. [`mma`], [`copy`], [`shift_down`], [`load`] and [`store`] (and their variants)
//! are *asynchronous*: they may execute and complete in a different order than they were issued,
//! except for the following *pipelined* pairs issued by the same thread (or the same warp, if
//! synchronized), which access their Tensor Memory operands in issue order:
//!
//! - mma (A and sparsity metadata reads) → mma (D writes), copy (writes), shift (writes)
//! - mma (D writes / C reads) → mma (C reads / D writes), with the same accumulator address and
//!   shape
//! - copy (writes) → mma (A and sparsity metadata reads), with the same address (any address for
//!   the `4x256b` shape)
//! - shift (writes) → mma (A and sparsity metadata reads)
//! - shift (reads and writes) → copy with the `4x256b` shape (writes)
//! - store → load, store → store, load → store
//!
//! All operations in a pair must use the same `CtaGroup`. Instructions issued by different warps
//! are never pipelined.
//!
//! Completion of asynchronous operations is observed as follows:
//!
//! - [`mma`], [`copy`] and [`shift_down`]: [`commit`] makes an [`MBarrier`](crate::MBarrier)
//!   track the completion of all prior such operations of the executing thread, and performs an
//!   arrive-on operation on it once they have completed. Waiting on the barrier then observes
//!   their completion.
//! - [`load`] and [`store`]: [`wait_load`] and [`wait_store`] block until all prior loads or
//!   stores of the executing thread have completed. Results of a load may be used directly by
//!   later instructions of the same thread (register dependencies are always respected), but
//!   [`wait_load`] is needed before the loaded Tensor Memory is overwritten by another operation.
//!
//! To order `tcgen05` operations across threads, the specialized fences compose with the usual
//! synchronization (barriers, mbarriers, morally strong memory operations):
//! [`fence_before_thread_sync`] orders all prior asynchronous `tcgen05` operations of the thread
//! before subsequent synchronization, and [`fence_after_thread_sync`] orders all subsequent
//! asynchronous `tcgen05` operations after prior synchronization. [`commit`] implicitly performs
//! [`fence_before_thread_sync`].
//!
//! The shared memory accesses of [`mma`] and [`copy`] are performed through the async proxy. Shared
//! memory written with ordinary stores must be made visible to them with
//! [`fence_proxy`](crate::fence::fence_proxy)`::<{ ProxyKind::AsyncSharedCta }>()` (this is not
//! needed for memory written by TMA, which also uses the async proxy).
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model>
//!
//! ## Canonical synchronization patterns
//!
//! These examples omit the surrounding `unsafe` blocks. `CG` is the kernel's [`CtaGroup`].
//!
//! An MMA issued by one thread, whose result is read by an epilogue warp after the MMA completed:
//!
//! ```ignore
//! // MMA thread:
//! tcgen05::mma::<CG, { MmaModifiers::DEFAULT }, _>(d, a_desc, b_desc, IDESC, None, accumulate);
//! tcgen05::commit::<CG>(&mma_done);
//!
//! // Epilogue warp (accessing its quarter of the lanes of `d`):
//! mma_done.wait(phase);
//! tcgen05::fence_after_thread_sync();
//! let acc: [f32; 32] = tcgen05::load::<{ LdStShape::S32x32b }, { Packing::None }, _, _>(d);
//! tcgen05::wait_load();
//! ```
//!
//! A load by one warp whose Tensor Memory is subsequently overwritten by an MMA issued by another
//! thread:
//!
//! ```ignore
//! // Epilogue warp:
//! let acc: [f32; 32] = tcgen05::load::<{ LdStShape::S32x32b }, { Packing::None }, _, _>(d);
//! tcgen05::wait_load();
//! tcgen05::fence_before_thread_sync();
//! acc_free.arrive();
//!
//! // MMA thread:
//! acc_free.wait(phase);
//! tcgen05::fence_after_thread_sync();
//! tcgen05::mma::<CG, { MmaModifiers::DEFAULT }, _>(d, a_desc, b_desc, IDESC, None, false);
//! ```
//!
//! Allocation, where warp 0 allocates and all threads of the CTA use the allocation, and
//! deallocation once all threads are done with it:
//!
//! ```ignore
//! if warp_id() == 0 {
//!     tcgen05::alloc::<CG>(tmem_addr_slot.get(), 256);
//!     tcgen05::relinquish_alloc_permit::<CG>();
//! }
//! tcgen05::fence_before_thread_sync();
//! sync_threads();
//! tcgen05::fence_after_thread_sync();
//! let tmem = tmem_addr_slot.get().read();
//!
//! // ... use `tmem` ...
//!
//! tcgen05::fence_before_thread_sync();
//! sync_threads(); // (a cluster barrier with `CtaGroup::Two`)
//! tcgen05::fence_after_thread_sync();
//! if warp_id() == 0 {
//!     tcgen05::dealloc::<CG>(tmem, 256);
//! }
//! ```
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns>
//!
//! # Safety contract
//!
//! The `# Safety` sections of the individual operations list their specific requirements in
//! addition to the following, which apply to all `tcgen05` operations:
//!
//! - All `tcgen05` instructions of a kernel use the same [`CtaGroup`]. With [`CtaGroup::Two`], the
//!   kernel is launched with clusters of an even number of CTAs, and the peer CTA has not exited.
//! - Tensor Memory operands lie within a live allocation of the CTA (of both CTAs of the pair
//!   with [`CtaGroup::Two`]), and are not accessed concurrently by unordered operations where at
//!   least one of them writes (see [Memory consistency](#memory-consistency)).
//! - Warp-collective operations (those with `.sync.aligned` in their PTX, e.g. [`load`],
//!   [`store`], [`alloc`]) are executed by all threads of the warp, convergently, with the same
//!   operand values, and no thread of the warp has exited.
//! - [`load`] and [`store`] only access the lanes of the executing warp's quarter of Tensor Memory.
//! - Shared memory read by asynchronous operations is not modified until the operations have
//!   completed, and is made visible to the async proxy before they are issued.
//!
//! # Unsupported features
//!
//! Features introduced in PTX ISA 9.4 are not supported, because the toolchain is limited to PTX
//! ISA 9.3: exclusive allocations (`.exclusive`), `tcgen05.ld.spcompress`, `.kind::ti16`,
//! `.decompress::lut::b`, `.collector::b` usage of non-weight-stationary MMAs, the
//! `.multicast::cluster::{16b,32b}` and `.sync_restrict` forms of `tcgen05.commit`, and the
//! `sm_107`-specific instruction descriptor fields.

mod alloc;
mod copy;
mod descriptor;
mod ld_st;
mod mma;
mod sync;

use core::marker::ConstParamTy;

use bytemuck::{Pod, Zeroable};

#[allow(unused_imports)]
pub use crate::modifiers::CtaGroup;
pub use crate::tma::CtaMask;
#[allow(unused_imports)]
pub use alloc::*;
#[allow(unused_imports)]
pub use copy::*;
#[allow(unused_imports)]
pub use descriptor::*;
#[allow(unused_imports)]
pub use ld_st::*;
#[allow(unused_imports)]
pub use mma::*;
#[allow(unused_imports)]
pub use sync::*;

/// Whether the target is architecture-specific (`sm_100a`, `sm_101a`, `sm_103a` or `sm_110a`).
/// Some `tcgen05` instructions and variants are not supported by the family-specific (`f`)
/// targets.
#[allow(dead_code)]
pub(crate) const ARCH_SPECIFIC: bool = cfg!(any(
    target_feature = "sm_100a",
    target_feature = "sm_101a",
    target_feature = "sm_103a",
    target_feature = "sm_110a"
));

/// Rejects [`CtaGroup::Unspecified`], which is invalid for `tcgen05` instructions. Call this in an
/// inline `const` block at the top of every function taking a `CtaGroup`.
#[allow(dead_code)]
pub(crate) const fn assert_cta_group(group: CtaGroup) {
    assert!(
        !matches!(group, CtaGroup::Unspecified),
        "tcgen05 instructions require CtaGroup::One or CtaGroup::Two"
    );
}

/// Returns the number of CTAs of the `.cta_group` qualifier `group`, for substitution into
/// `.cta_group::{n}` with an `asm!` `const` operand.
///
/// [`CtaGroup::Unspecified`] maps to 0 rather than failing const evaluation, as the operands of all
/// `asm_variants!` branches are evaluated, which would repeat the error for every branch; it is
/// rejected once per function by [`assert_cta_group`] instead.
#[allow(dead_code)]
pub(crate) const fn cta_group_count(group: CtaGroup) -> u32 {
    match group {
        CtaGroup::One => 1,
        CtaGroup::Two => 2,
        CtaGroup::Unspecified => 0,
    }
}

/// An address in Tensor Memory: a lane (row) index in bits 16..32 and a column index in bits 0..16.
///
/// Tensor Memory has 128 lanes and 512 columns of 32-bit cells per CTA. Addresses are obtained
/// from [`alloc`] (which returns the address of lane 0 of the first allocated column), and offset
/// from there with [`add_lanes`](Self::add_lanes) and [`add_columns`](Self::add_columns).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory-addressing>
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct TmemAddr(u32);

impl TmemAddr {
    /// Creates the address of `lane` (less than 128) and `column` (less than 512).
    #[inline(always)]
    pub const fn new(lane: u32, column: u32) -> Self {
        Self((lane << 16) | (column & 0xffff))
    }

    /// Wraps the raw 32-bit address.
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw 32-bit address.
    #[inline(always)]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// Returns the lane index.
    #[inline(always)]
    pub const fn lane(self) -> u32 {
        self.0 >> 16
    }

    /// Returns the column index.
    #[inline(always)]
    pub const fn column(self) -> u32 {
        self.0 & 0xffff
    }

    /// Returns the address `lanes` lanes further down. The resulting lane must be less than 128.
    ///
    /// For example, `addr.add_lanes(32 * (warp_id() % 4))` selects the quarter of Tensor Memory
    /// accessible to the executing warp with [`load`] and [`store`].
    #[inline(always)]
    pub const fn add_lanes(self, lanes: u32) -> Self {
        Self(self.0.wrapping_add(lanes << 16))
    }

    /// Returns the address `columns` columns further right. The resulting column must be less
    /// than 512.
    #[inline(always)]
    pub const fn add_columns(self, columns: u32) -> Self {
        Self(self.0.wrapping_add(columns))
    }
}

/// The kind of a matrix multiply-accumulate operation (`.kind`), which determines the possible
/// element types of the matrices and the layout of its [`InstrDescriptor`].
///
/// The exact element types are given by the instruction descriptor.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-shape>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum MmaKind {
    /// `.kind::f16`: A and B are `f16` or `bf16`, D is `f16` or `f32`.
    F16,
    /// `.kind::tf32`: A and B are `tf32`, D is `f32`.
    Tf32,
    /// `.kind::f8f6f4`: A and B are any of the FP8 (`e4m3`, `e5m2`), FP6 (`e2m3`, `e3m2`) and FP4
    /// (`e2m1`) types, D is `f16` or `f32`.
    F8f6f4,
    /// `.kind::i8`: A and B are `s8` or `u8`, D is `s32`.
    I8,
    /// `.kind::mxf8f6f4`: block-scaled variant of [`F8f6f4`](Self::F8f6f4) with `ue8m0` scale
    /// factors, D is `f32`.
    Mxf8f6f4,
    /// `.kind::mxf4`: block-scaled `e2m1` A and B with `ue8m0` scale factors, D is `f32`.
    Mxf4,
    /// `.kind::mxf4nvf4`: block-scaled `e2m1` A and B with `ue8m0`, `ue4m3` or `ue5m3` scale
    /// factors, D is `f32`.
    Mxf4nvf4,
}

impl MmaKind {
    /// Whether this is a block-scaled kind (`.kind::mxf8f6f4`, `.kind::mxf4`, `.kind::mxf4nvf4`).
    #[inline(always)]
    pub const fn is_block_scaled(self) -> bool {
        matches!(self, MmaKind::Mxf8f6f4 | MmaKind::Mxf4 | MmaKind::Mxf4nvf4)
    }
}
