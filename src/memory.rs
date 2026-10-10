// --- LLM-generated --- //
//! Various memory operations like prefetching, L2 discard, cache priorities, ...

use core::arch::asm;
use core::arch::nvptx as nv;
use core::marker::ConstParamTy;

use bytemuck::{Pod, Zeroable};

#[cfg(target_feature = "sm_100")]
use crate::modifiers::Scope;
use crate::{bf16, f16};

mod sealed {
    pub trait UniformLoad {}

    #[cfg(target_feature = "sm_100")]
    pub trait AsyncStore {}
}

/// A cache level to prefetch data into (`.level`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum CacheLevel {
    /// The L1 data cache (`.L1`).
    L1,
    /// The L2 cache (`.L2`).
    L2,
}

/// The eviction priority applied to cache lines in the L2 cache
/// (`.level::eviction_priority`).
///
/// Not every operation supports every priority; see the documentation of each operation. Eviction
/// priorities are performance hints only.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-eviction-priority-hints>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum L2EvictionPriority {
    /// Cache the data with normal eviction priority, the default (`.L2::evict_normal`).
    Normal,
    /// Evict the cache line last, i.e. only after other data with normal or first eviction
    /// priority has been evicted. Suitable for data that should remain persistent in the cache
    /// (`.L2::evict_last`).
    Last,
    /// Evict the cache line first, i.e. it will likely be evicted when eviction is required.
    /// Suitable for streaming data (`.L2::evict_first`).
    First,
    /// Do not change the eviction priority of the cache line (`.L2::evict_unchanged`).
    Unchanged,
}

/// An L2 cache eviction policy, used as a hint by memory operations with the
/// `.L2::cache_hint` qualifier (TMA copies, atomics, ...).
///
/// A policy is a 64-bit opaque value produced by the `createpolicy` PTX instruction; see
/// [`CachePolicy::fractional`], [`CachePolicy::range`] and
/// [`CachePolicy::from_access_property`]. Passing `None` instead of a policy to an operation
/// omits the `.L2::cache_hint` qualifier.
///
/// The policy is only a performance hint and does not affect the memory model. It is only
/// applied to accesses to global memory. `createpolicy` requires `sm_80`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-eviction-priority-hints>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct CachePolicy(u64);

const fn assert_secondary_priority_supported(secondary: L2EvictionPriority) {
    assert!(
        matches!(
            secondary,
            L2EvictionPriority::First | L2EvictionPriority::Unchanged
        ),
        "the secondary priority of a cache policy must be First or Unchanged"
    );
}

macro_rules! create_policy {
    ($primary:ident, $secondary:ident; $emit:ident) => {
        asm_variants!($emit;
            $primary => {
                L2EvictionPriority::Normal => ".L2::evict_normal",
                L2EvictionPriority::Last => ".L2::evict_last",
                L2EvictionPriority::First => ".L2::evict_first",
                L2EvictionPriority::Unchanged => ".L2::evict_unchanged",
            },
            $secondary => {
                L2EvictionPriority::First => ".L2::evict_first",
                L2EvictionPriority::Unchanged => ".L2::evict_unchanged",
            },
        )
    };
}

impl CachePolicy {
    /// Wraps the raw policy value produced by `createpolicy`. The value is opaque and should only
    /// originate from `createpolicy`.
    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// Returns the raw policy value.
    #[inline]
    pub const fn to_bits(self) -> u64 {
        self.0
    }

    /// Creates a policy that applies the eviction priority `PRIORITY` to all cache accesses
    /// (`createpolicy.fractional` with the default fraction of 1.0).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-createpolicy>
    #[inline]
    pub fn new<const PRIORITY: L2EvictionPriority>() -> Self {
        let policy: u64;
        macro_rules! emit {
            ($primary:literal, $secondary:literal) => {
                // SAFETY: `createpolicy` only writes a register.
                unsafe {
                    asm!(
                        concat!("createpolicy.fractional", $primary, $secondary, ".b64 {p};"),
                        p = out(reg64) policy,
                        options(pure, nomem, nostack),
                    )
                }
            };
        }
        const SECONDARY: L2EvictionPriority = L2EvictionPriority::Unchanged;
        create_policy!(PRIORITY, SECONDARY; emit);
        Self(policy)
    }

    /// Creates a fraction-based policy (`createpolicy.fractional`): a memory access using the
    /// policy gets the eviction priority `PRIMARY` with probability `fraction`, and the eviction
    /// priority `SECONDARY` otherwise.
    ///
    /// `SECONDARY` must be [`L2EvictionPriority::First`] or [`L2EvictionPriority::Unchanged`].
    ///
    /// # Safety
    ///
    /// `fraction` must be in the range `(0.0, 1.0]`.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-createpolicy>
    #[inline]
    pub unsafe fn fractional<
        const PRIMARY: L2EvictionPriority,
        const SECONDARY: L2EvictionPriority,
    >(
        fraction: f32,
    ) -> Self {
        const { assert_secondary_priority_supported(SECONDARY) };
        let policy: u64;
        macro_rules! emit {
            ($primary:literal, $secondary:literal) => {
                unsafe {
                    asm!(
                        concat!("createpolicy.fractional", $primary, $secondary, ".b64 {p}, {f};"),
                        p = out(reg64) policy,
                        f = in(reg32) fraction,
                        options(pure, nomem, nostack),
                    )
                }
            };
        }
        create_policy!(PRIMARY, SECONDARY; emit);
        Self(policy)
    }

    /// Creates a range-based policy (`createpolicy.range`) for the address range around `base`:
    ///
    /// - The eviction priority `PRIMARY` applies to the primary range
    ///   `[base, base + primary_size)`.
    /// - The eviction priority `SECONDARY` applies to the trailing secondary range
    ///   `[base + primary_size, base + total_size)` and to the preceding secondary range
    ///   `[base - (total_size - primary_size), base)`.
    /// - The eviction priority applied to addresses outside of these ranges is unspecified.
    ///
    /// Sizes are in bytes. `SECONDARY` must be [`L2EvictionPriority::First`] or
    /// [`L2EvictionPriority::Unchanged`].
    ///
    /// # Safety
    ///
    /// - `base` must be a generic address within the window of the global state space (see
    ///   [`is_in_address_space`]); otherwise the behavior is undefined. It does not need to be
    ///   valid for reads or writes.
    /// - `primary_size` must be less than or equal to `total_size`. (The maximum `total_size` of
    ///   4 GiB is enforced by the `u32` type.)
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-createpolicy>
    #[inline]
    pub unsafe fn range<
        const PRIMARY: L2EvictionPriority,
        const SECONDARY: L2EvictionPriority,
        T,
    >(
        base: *const T,
        primary_size: u32,
        total_size: u32,
    ) -> Self {
        const { assert_secondary_priority_supported(SECONDARY) };
        let policy: u64;
        macro_rules! emit {
            ($primary:literal, $secondary:literal) => {
                unsafe {
                    asm!(
                        concat!(
                            "createpolicy.range", $primary, $secondary,
                            ".b64 {p}, [{a}], {primary}, {total};"
                        ),
                        p = out(reg64) policy,
                        a = in(reg64) base.expose_provenance(),
                        primary = in(reg32) primary_size,
                        total = in(reg32) total_size,
                        options(pure, nomem, nostack),
                    )
                }
            };
        }
        create_policy!(PRIMARY, SECONDARY; emit);
        Self(policy)
    }

    /// Converts an access property created by the CUDA APIs (e.g. `cuda::access_property`) into
    /// a cache eviction policy (`createpolicy.cvt.L2`). Refer to the CUDA programming guide for
    /// details on access properties.
    ///
    /// # Safety
    ///
    /// `access_property` must be the raw 64-bit value of an access property created by the CUDA
    /// APIs.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-createpolicy>
    #[inline]
    pub unsafe fn from_access_property(access_property: u64) -> Self {
        let policy: u64;
        unsafe {
            asm!(
                "createpolicy.cvt.L2.b64 {p}, {a};",
                p = out(reg64) policy,
                a = in(reg64) access_property,
                options(pure, nomem, nostack),
            )
        }
        Self(policy)
    }
}

/// A PTX state space, used to query which window of the generic address space a pointer lies in
/// with [`is_in_address_space`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum AddressSpace {
    /// Constant memory (`.const`).
    Constant,
    /// Global memory (`.global`). This window also contains the kernel parameter space.
    Global,
    /// Thread-local memory (`.local`).
    Local,
    /// Shared memory of the executing CTA (`.shared::cta`).
    Shared,
    /// Shared memory of any CTA in the executing cluster (`.shared::cluster`). Contains the
    /// [`AddressSpace::Shared`] window.
    ///
    /// Requires `sm_90`.
    SharedCluster,
}

/// Prefetches the cache line containing `ptr`, in global or local memory, into the cache level
/// `LEVEL`.
///
/// This is only a performance hint and does not affect the memory model, so `ptr` does not need
/// to be valid. Prefetching a shared memory address is a no-op.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prefetch-prefetchu>
#[cfg(target_feature = "sm_90")]
#[inline]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn prefetch<const LEVEL: CacheLevel, T>(ptr: *const T) {
    // SAFETY: a prefetch through a generic address is only a hint and cannot fault or write.
    unsafe {
        match LEVEL {
            CacheLevel::L1 => nv::prefetch_l1::<{ nv::StateSpace::Generic }, T>(ptr),
            CacheLevel::L2 => nv::prefetch_l2::<{ nv::StateSpace::Generic }, T>(ptr),
        }
    }
}

/// Prefetches the cache line containing `ptr` into the L2 cache, with the eviction priority
/// `PRIORITY` applied to the prefetched cache line. `PRIORITY` must be
/// [`L2EvictionPriority::Normal`] or [`L2EvictionPriority::Last`].
///
/// # Safety
///
/// `ptr` must point into global memory; otherwise the behavior is undefined.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prefetch-prefetchu>
#[cfg(target_feature = "sm_90")]
#[inline]
pub unsafe fn prefetch_l2_with_priority<const PRIORITY: L2EvictionPriority, T>(ptr: *const T) {
    const {
        assert!(
            matches!(
                PRIORITY,
                L2EvictionPriority::Normal | L2EvictionPriority::Last
            ),
            "prefetch only supports the Normal and Last eviction priorities"
        )
    };
    const_map!(PRIORITY => PRIO: nv::L2EvictionPriority {
        L2EvictionPriority::Normal => nv::L2EvictionPriority::EvictNormal,
        L2EvictionPriority::Last => nv::L2EvictionPriority::EvictLast,
    } in unsafe { nv::prefetch_l2_evict::<{ nv::StateSpace::Global }, PRIO, T>(ptr) })
}

/// Prefetches the cache line containing the generic address `ptr` into the uniform (constant)
/// cache, for subsequent warp-uniform read-only accesses.
///
/// This is only a performance hint, so `ptr` does not need to be valid. No operation occurs if
/// the address maps to constant, local or shared memory.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prefetch-prefetchu>
#[cfg(target_feature = "sm_90")]
#[inline]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn prefetch_uniform<T>(ptr: *const T) {
    // SAFETY: a prefetch is only a hint and cannot fault or write.
    unsafe { nv::prefetchu_l1(ptr) }
}

/// Resets the L2 eviction priority of the 128 bytes `[ptr, ptr + 128)` to `evict_normal`
/// (`applypriority.global.L2::evict_normal`). 128 is the only size supported by PTX.
///
/// # Safety
///
/// `ptr` must point into global memory (otherwise the behavior is undefined) and be aligned to
/// 128 bytes.
///
/// Requires `sm_80`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-applypriority>
#[inline]
pub unsafe fn apply_l2_evict_normal_priority(ptr: *const u8) {
    unsafe { nv::applypriority_l2_evict_normal::<{ nv::StateSpace::Global }, u8>(ptr) }
}

/// Hints that the 128 bytes starting at `ptr` can be destructively discarded from the L2 cache
/// without being written back to memory.
///
/// Semantically this is a weak write of an unstable indeterminate value: until the memory is
/// overwritten, reads may return a different bit pattern each time. 128 is the only size
/// supported by PTX.
///
/// # Safety
///
/// `ptr` must point into global memory (otherwise the behavior is undefined) and be aligned to
/// 128 bytes. The 128 bytes must be valid for writes and must not be read before being
/// overwritten, unless the caller can accept arbitrary values.
///
/// Requires `sm_80`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-discard>
#[inline]
pub unsafe fn discard_l2(ptr: *mut u8) {
    unsafe { nv::discard_l2::<{ nv::StateSpace::Global }, u8>(ptr) }
}

/// Types that can be loaded with [`load_uniform`].
///
/// This trait is sealed: it is implemented for the primitive integers and floats, `f16`,
/// [`bf16`] and raw pointers.
pub trait UniformLoad: Copy + sealed::UniformLoad {
    #[doc(hidden)]
    unsafe fn load_uniform_impl(ptr: *const Self) -> Self;
}

macro_rules! impl_uniform_load {
    ($($ty:ty => |$p:ident| $load:expr),+ $(,)?) => {$(
        impl sealed::UniformLoad for $ty {}
        impl UniformLoad for $ty {
            #[inline(always)]
            unsafe fn load_uniform_impl($p: *const Self) -> Self {
                unsafe { $load }
            }
        }
    )+};
}

impl_uniform_load! {
    u8 => |p| nv::ldu_u8(p),
    i8 => |p| nv::ldu_u8(p.cast()) as i8,
    u16 => |p| nv::ldu_u16(p),
    i16 => |p| nv::ldu_u16(p.cast()) as i16,
    u32 => |p| nv::ldu_u32(p),
    i32 => |p| nv::ldu_u32(p.cast()) as i32,
    u64 => |p| nv::ldu_u64(p),
    i64 => |p| nv::ldu_u64(p.cast()) as i64,
    usize => |p| nv::ldu_u64(p.cast()) as usize,
    isize => |p| nv::ldu_u64(p.cast()) as isize,
    f32 => |p| nv::ldu_f32(p),
    f64 => |p| nv::ldu_f64(p),
    f16 => |p| f16::from_bits(nv::ldu_u16(p.cast())),
    bf16 => |p| bf16::from_bits(nv::ldu_u16(p.cast())),
}

impl<T> sealed::UniformLoad for *const T {}
impl<T> UniformLoad for *const T {
    #[inline(always)]
    unsafe fn load_uniform_impl(ptr: *const Self) -> Self {
        unsafe { nv::ldu_ptr(ptr) }
    }
}

impl<T> sealed::UniformLoad for *mut T {}
impl<T> UniformLoad for *mut T {
    #[inline(always)]
    unsafe fn load_uniform_impl(ptr: *const Self) -> Self {
        unsafe { nv::ldu_ptr(ptr.cast::<*const T>()).cast_mut() }
    }
}

/// Loads a `T` from `ptr` through the read-only, warp-uniform data path (`ldu`).
///
/// # Safety
///
/// - `ptr` must be valid for reads, aligned, and point into global memory.
/// - The address must be the same across all threads in the warp.
/// - The memory must not be written to by any thread for the duration of the kernel.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ldu>
#[inline]
pub unsafe fn load_uniform<T: UniformLoad>(ptr: *const T) -> T {
    unsafe { T::load_uniform_impl(ptr) }
}

// `const fn`s are needed for the compile-time checks below, as `generic_const_exprs` rejects
// control flow in inline const blocks that depend on generic parameters.
const fn assert_address_space_supported(space: AddressSpace) {
    assert!(
        !matches!(space, AddressSpace::SharedCluster) || cfg!(target_feature = "sm_90"),
        "AddressSpace::SharedCluster requires sm_90"
    );
}

/// Returns whether the generic address `ptr` falls within the window of the address space
/// `SPACE` (`isspacep`).
///
/// The address does not need to be valid; this only inspects the address value.
/// [`AddressSpace::SharedCluster`] is true for every shared memory address accessible to the
/// threads of the cluster, whereas [`AddressSpace::Shared`] is true only for addresses in the
/// executing CTA.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-isspacep>
#[inline]
pub fn is_in_address_space<const SPACE: AddressSpace, T>(ptr: *const T) -> bool {
    const { assert_address_space_supported(SPACE) };
    const_map!(SPACE => S: nv::StateSpace {
        AddressSpace::Constant => nv::StateSpace::Const,
        AddressSpace::Global => nv::StateSpace::Global,
        AddressSpace::Local => nv::StateSpace::Local,
        AddressSpace::Shared => nv::StateSpace::SharedCta,
        AddressSpace::SharedCluster => nv::StateSpace::SharedCluster,
    } in unsafe { nv::isspacep::<S>(ptr.cast()) })
}

/// Converts the generic address `ptr` of a location in the shared memory of the executing CTA to
/// its 32-bit address in the `.shared::cta` state space (`cvta.to.shared`).
///
/// Shared memory addresses are needed, for example, to build the matrix descriptors of the
/// `tcgen05` instructions. They are also valid `.shared::cluster` addresses referring to the
/// executing CTA.
///
/// The result is undefined if `ptr` does not point into the shared memory window of the
/// executing CTA (see [`is_in_address_space`]).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvta>
#[inline(always)]
pub fn to_shared_address<T>(ptr: *const T) -> u32 {
    let address: u64;
    // SAFETY: `cvta` only reads and writes registers.
    unsafe {
        asm!(
            "cvta.to.shared.u64 {address}, {ptr};",
            address = out(reg64) address,
            ptr = in(reg64) ptr.expose_provenance(),
            options(pure, nomem, nostack),
        );
    }
    // Shared memory addresses fit in 32 bits.
    address as u32
}

/// Zero-fills `size` bytes of shared memory of the executing CTA starting at `ptr`, using a
/// bulk store (`st.bulk`).
///
/// The store is a weak memory operation. PTX only supports initializing to zero.
///
/// # Safety
///
/// - `ptr` must point into the shared memory of the executing CTA (otherwise the behavior is
///   undefined), and the `size` bytes must be valid for writes.
/// - `size` must be a multiple of 8 and at most 16777216 (16 MiB).
///
/// Requires `sm_100`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-bulk>
#[cfg(target_feature = "sm_100")]
#[inline]
pub unsafe fn zero_fill_shared(ptr: *mut u8, size: usize) {
    unsafe { nv::st_bulk::<{ nv::StateSpace::Generic }>(ptr.cast(), size as u64) }
}

/// The kind of global memory targeted by [`store_async_release`] and
/// [`red_async_add_release`](crate::atomic::red_async_add_release).
#[cfg(target_feature = "sm_100")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum AsyncStoreTarget {
    /// A regular global memory address.
    Global,
    /// A multimem address: the operation is performed on the memory location on every GPU that
    /// the address refers to (`multimem.st.async`, `multimem.red.async`).
    Multimem,
    /// Memory-mapped I/O (`.mmio`). Requires [`Scope::System`].
    Mmio,
}

/// Types that can be stored with [`store_async_release`].
///
/// This trait is sealed: it is implemented for the 16-, 32- and 64-bit integers and floats.
#[cfg(target_feature = "sm_100")]
pub trait AsyncStoreElement: Copy + sealed::AsyncStore {
    #[doc(hidden)]
    unsafe fn store_async_release_impl<const SCOPE: Scope, const TARGET: AsyncStoreTarget>(
        dst: *mut Self,
        value: Self,
    );
}

#[cfg(target_feature = "sm_100")]
macro_rules! impl_async_store {
    ($($ty:ty => $bits:ty, |$v:ident| $to_bits:expr, $gpu:ident, $sys:ident, $mmio:ident);+ $(;)?) => {$(
        impl sealed::AsyncStore for $ty {}
        impl AsyncStoreElement for $ty {
            #[inline(always)]
            unsafe fn store_async_release_impl<const SCOPE: Scope, const TARGET: AsyncStoreTarget>(
                dst: *mut Self,
                $v: Self,
            ) {
                let dst = dst.cast::<$bits>();
                let value: $bits = $to_bits;
                unsafe {
                    match (SCOPE, TARGET) {
                        (Scope::Gpu, AsyncStoreTarget::Global) => nv::$gpu::<false>(dst, value),
                        (Scope::Gpu, AsyncStoreTarget::Multimem) => nv::$gpu::<true>(dst, value),
                        (Scope::System, AsyncStoreTarget::Global) => nv::$sys::<false>(dst, value),
                        (Scope::System, AsyncStoreTarget::Multimem) => nv::$sys::<true>(dst, value),
                        (Scope::System, AsyncStoreTarget::Mmio) => nv::$mmio(dst, value),
                        _ => unreachable!(),
                    }
                }
            }
        }
    )+};
}

#[cfg(target_feature = "sm_100")]
impl_async_store! {
    u16 => u16, |v| v, st_async_gpu_u16, st_async_sys_u16, st_async_mmio_sys_u16;
    i16 => u16, |v| v as u16, st_async_gpu_u16, st_async_sys_u16, st_async_mmio_sys_u16;
    f16 => u16, |v| v.to_bits(), st_async_gpu_u16, st_async_sys_u16, st_async_mmio_sys_u16;
    bf16 => u16, |v| v.to_bits(), st_async_gpu_u16, st_async_sys_u16, st_async_mmio_sys_u16;
    u32 => u32, |v| v, st_async_gpu_u32, st_async_sys_u32, st_async_mmio_sys_u32;
    i32 => u32, |v| v as u32, st_async_gpu_u32, st_async_sys_u32, st_async_mmio_sys_u32;
    f32 => u32, |v| v.to_bits(), st_async_gpu_u32, st_async_sys_u32, st_async_mmio_sys_u32;
    u64 => u64, |v| v, st_async_gpu_u64, st_async_sys_u64, st_async_mmio_sys_u64;
    i64 => u64, |v| v as u64, st_async_gpu_u64, st_async_sys_u64, st_async_mmio_sys_u64;
    f64 => u64, |v| v.to_bits(), st_async_gpu_u64, st_async_sys_u64, st_async_mmio_sys_u64;
}

#[cfg(target_feature = "sm_100")]
pub(crate) const fn assert_async_store_supported(scope: Scope, target: AsyncStoreTarget) {
    assert!(
        matches!(scope, Scope::Gpu | Scope::System),
        "only the GPU and system scopes are supported"
    );
    assert!(
        !matches!(target, AsyncStoreTarget::Mmio) || matches!(scope, Scope::System),
        "MMIO stores require the system scope"
    );
}

/// Asynchronously stores `value` to the global memory address `dst` with `.release` semantics at
/// scope `SCOPE`.
///
/// `st.async` is non-blocking: the store is initiated and may complete after this function
/// returns. It is performed in the generic proxy and is a strong memory operation with release
/// semantics at `SCOPE`. `SCOPE` must be [`Scope::Gpu`] or [`Scope::System`]. `TARGET` selects
/// between a regular global address, a multimem address (`multimem.st.async`) and an MMIO store;
/// the latter requires [`Scope::System`].
///
/// # Safety
///
/// - `dst` must be valid for writes and aligned, and point into global memory. With
///   [`AsyncStoreTarget::Multimem`] it must be a multimem address (which must not be accessed
///   by anything other than `multimem.*` operations); with
///   [`AsyncStoreTarget::Mmio`] it must refer to memory-mapped I/O.
/// - The store must not race with other non-atomic accesses to `dst`.
///
/// Requires `sm_100`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st-async>
#[cfg(target_feature = "sm_100")]
#[inline]
pub unsafe fn store_async_release<
    T: AsyncStoreElement,
    const SCOPE: Scope,
    const TARGET: AsyncStoreTarget,
>(
    dst: *mut T,
    value: T,
) {
    const { assert_async_store_supported(SCOPE, TARGET) };
    unsafe { T::store_async_release_impl::<SCOPE, TARGET>(dst, value) }
}
