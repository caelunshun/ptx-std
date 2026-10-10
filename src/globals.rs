// --- LLM-generated --- //
//! Thread, block, cluster and grid geometry, and other special registers.
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers>

use bytemuck::{Pod, Zeroable};
use core::arch::{asm, global_asm, nvptx as nv};

/// Number of threads in a warp.
pub const WARP_SIZE: u32 = 32;

/// A three-dimensional extent or index, used for thread, block and cluster geometry.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    /// Creates a `Dim3` from its components.
    #[inline(always)]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Returns the product `x * y * z`.
    #[inline(always)]
    pub const fn volume(self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    #[inline(always)]
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self { x, y, z }
    }
}

impl From<[u32; 3]> for Dim3 {
    #[inline(always)]
    fn from([x, y, z]: [u32; 3]) -> Self {
        Self { x, y, z }
    }
}

impl From<Dim3> for (u32, u32, u32) {
    #[inline(always)]
    fn from(d: Dim3) -> Self {
        (d.x, d.y, d.z)
    }
}

impl From<Dim3> for [u32; 3] {
    #[inline(always)]
    fn from(d: Dim3) -> Self {
        [d.x, d.y, d.z]
    }
}

/// Returns the index of the executing thread within its block (`%tid`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-tid>
#[inline(always)]
pub fn thread_id() -> Dim3 {
    unsafe { Dim3::new(nv::tid_x(), nv::tid_y(), nv::tid_z()) }
}

/// Returns the dimensions of the block, in threads (`%ntid`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-ntid>
#[inline(always)]
pub fn block_dim() -> Dim3 {
    unsafe { Dim3::new(nv::ntid_x(), nv::ntid_y(), nv::ntid_z()) }
}

/// Returns the index of the executing block within the grid (`%ctaid`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-ctaid>
#[inline(always)]
pub fn block_id() -> Dim3 {
    unsafe { Dim3::new(nv::ctaid_x(), nv::ctaid_y(), nv::ctaid_z()) }
}

/// Returns the dimensions of the grid, in blocks (`%nctaid`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nctaid>
#[inline(always)]
pub fn grid_dim() -> Dim3 {
    unsafe { Dim3::new(nv::nctaid_x(), nv::nctaid_y(), nv::nctaid_z()) }
}

/// Returns the linear index of the executing thread within its block, in row-major order
/// (`x` varies fastest): `tid.x + ntid.x * (tid.y + ntid.y * tid.z)`.
#[inline(always)]
pub fn linear_thread_id() -> u32 {
    let tid = thread_id();
    let dim = block_dim();
    tid.x + dim.x * (tid.y + dim.y * tid.z)
}

/// Returns the total number of threads in the block.
#[inline(always)]
pub fn block_size() -> u32 {
    let dim = block_dim();
    dim.x * dim.y * dim.z
}

/// Returns the index of the executing warp within its block.
///
/// This is computed from [`linear_thread_id`], so unlike [`hardware_warp_id`] it is stable
/// for the lifetime of the thread, and consecutive for threads with consecutive linear indices.
#[inline(always)]
pub fn warp_id() -> u32 {
    linear_thread_id() / WARP_SIZE
}

/// Returns the number of warps in the block, rounding up if the block size is not a multiple of
/// [`WARP_SIZE`].
#[inline(always)]
pub fn num_warps() -> u32 {
    block_size().div_ceil(WARP_SIZE)
}

/// Returns the index of the executing thread within its warp, in `0..32` (`%laneid`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-laneid>
#[inline(always)]
pub fn lane_id() -> u32 {
    unsafe { nv::laneid() }
}

/// Returns the number of threads in a warp (`WARP_SZ`). This is always [`WARP_SIZE`].
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-wsize>
#[inline(always)]
pub fn warp_size() -> u32 {
    unsafe { nv::warpsize() }
}

/// Returns the identifier of the hardware warp slot the executing thread occupies in its block
/// (`%warpid`).
///
/// The value may change during execution, e.g. due to preemption, and is not necessarily
/// contiguous; it is intended for profiling and diagnostics only. Use [`warp_id`] for
/// the logical warp index.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-warpid>
#[inline(always)]
pub fn hardware_warp_id() -> u32 {
    unsafe { nv::warpid() }
}

/// Returns the maximum number of hardware warp identifiers (`%nwarpid`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nwarpid>
#[inline(always)]
pub fn max_hardware_warp_ids() -> u32 {
    unsafe { nv::nwarpid() }
}

/// Returns the identifier of the SM the thread is executing on (`%smid`).
///
/// The value may change during execution, e.g. due to preemption; it is intended for profiling
/// and diagnostics only.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-smid>
#[inline(always)]
pub fn sm_id() -> u32 {
    unsafe { nv::smid() }
}

/// Returns the maximum number of SM identifiers (`%nsmid`).
///
/// This may be larger than the number of SMs on the device, as identifiers are not necessarily
/// contiguous.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nsmid>
#[inline(always)]
pub fn max_sm_ids() -> u32 {
    unsafe { nv::nsmid() }
}

/// Returns the temporal grid launch number of the executing grid (`%gridid`), truncated to
/// 32 bits.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-gridid>
#[inline(always)]
pub fn grid_id() -> u32 {
    unsafe { nv::gridid() }
}

/// Returns the index of the executing thread block cluster within the grid (`%clusterid`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clusterid>
#[cfg(target_feature = "sm_90")]
#[inline(always)]
pub fn cluster_id() -> Dim3 {
    unsafe { Dim3::new(nv::clusterid_x(), nv::clusterid_y(), nv::clusterid_z()) }
}

/// Returns the dimensions of the grid, in clusters (`%nclusterid`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nclusterid>
#[cfg(target_feature = "sm_90")]
#[inline(always)]
pub fn grid_dim_in_clusters() -> Dim3 {
    unsafe { Dim3::new(nv::nclusterid_x(), nv::nclusterid_y(), nv::nclusterid_z()) }
}

/// Returns the index of the executing block within its cluster (`%cluster_ctaid`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-ctaid>
#[cfg(target_feature = "sm_90")]
#[inline(always)]
pub fn block_id_in_cluster() -> Dim3 {
    unsafe {
        Dim3::new(
            nv::cluster_ctaid_x(),
            nv::cluster_ctaid_y(),
            nv::cluster_ctaid_z(),
        )
    }
}

/// Returns the dimensions of the cluster, in blocks (`%cluster_nctaid`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-nctaid>
#[cfg(target_feature = "sm_90")]
#[inline(always)]
pub fn cluster_dim() -> Dim3 {
    unsafe {
        Dim3::new(
            nv::cluster_nctaid_x(),
            nv::cluster_nctaid_y(),
            nv::cluster_nctaid_z(),
        )
    }
}

/// Returns the linear rank of the executing block within its cluster (`%cluster_ctarank`).
///
/// This is the rank used to address a block's shared memory with `mapa` and to select
/// multicast destinations.
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-ctarank>
#[cfg(target_feature = "sm_90")]
#[inline(always)]
pub fn block_rank_in_cluster() -> u32 {
    unsafe { nv::cluster_ctarank() }
}

/// Returns the number of blocks in the cluster (`%cluster_nctarank`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-cluster-nctarank>
#[cfg(target_feature = "sm_90")]
#[inline(always)]
pub fn cluster_size() -> u32 {
    unsafe { nv::cluster_nctarank() }
}

/// Returns whether the grid was launched with an explicit cluster size (`%is_explicit_cluster`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-is-explicit-cluster>
#[cfg(target_feature = "sm_90")]
#[inline(always)]
pub fn is_explicit_cluster() -> bool {
    unsafe { nv::is_explicit_cluster() }
}

/// Returns the total shared memory allocated for the block at launch, in bytes, excluding memory
/// reserved for system software (`%total_smem_size`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-total-smem-size>
#[inline(always)]
pub fn total_smem_size() -> u32 {
    unsafe { nv::total_smem_size() }
}

/// Returns the total shared memory allocated for the block at launch, in bytes, including memory
/// reserved for system software (`%aggr_smem_size`).
///
/// Requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-aggr-smem-size>
#[cfg(target_feature = "sm_90")]
#[inline(always)]
pub fn aggregate_smem_size() -> u32 {
    unsafe { nv::aggr_smem_size() }
}

/// Returns the size in bytes of the dynamically allocated shared memory that was requested at
/// launch (`%dynamic_smem_size`), i.e. the size of the region starting at
/// [`dynamic_smem_start`].
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-dynamic-smem-size>
#[inline(always)]
pub fn dynamic_smem_size() -> u32 {
    unsafe { nv::dynamic_smem_size() }
}

// Declares the (zero-sized) extern shared-memory symbol that the driver places the dynamic shared
// memory allocation at, equivalent to CUDA's `extern __shared__ char smem[]`. The symbol lives in
// the `.shared` state space, which Rust cannot express for statics, so it is declared in module
// level assembly and only ever referenced by `dynamic_smem_start`.
global_asm!(".extern .shared .align 1024 .b8 __ptx_std_dynamic_smem[];");

/// Returns a generic pointer to the start of the block's dynamically allocated shared memory, the
/// region whose size was given at launch (see [`dynamic_smem_size`]).
///
/// The pointer is aligned to 1024 bytes, which satisfies the alignment requirements of TMA and
/// of 128-byte swizzled tiles. Any padding needed to reach this alignment after the kernel's
/// static shared memory counts toward its shared memory usage.
///
/// Every call returns the same pointer, so all users of dynamic shared memory in a kernel must
/// partition the region among themselves. The memory begins uninitialized.
#[inline(always)]
pub fn dynamic_smem_start() -> *mut u8 {
    let offset: u32;
    unsafe {
        asm!(
            "mov.u32 {}, __ptx_std_dynamic_smem;",
            out(reg32) offset,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    shared_offset_to_generic(offset)
}

/// Converts an offset within the shared memory window to a generic pointer.
#[inline(always)]
fn shared_offset_to_generic(offset: u32) -> *mut u8 {
    nv::shared_offset_to_ptr(offset)
}

/// Returns a generic pointer to the start of the shared memory region reserved for system
/// software (`%reserved_smem_offset_begin`).
///
/// This memory is not available to user code; the pointer is only useful for comparisons.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-reserved-smem>
#[inline(always)]
pub fn reserved_smem_start() -> *mut u8 {
    shared_offset_to_generic(unsafe { nv::reserved_smem_offset_begin() })
}

/// Returns a generic pointer to the end (one past the last byte) of the shared memory region
/// reserved for system software (`%reserved_smem_offset_end`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-reserved-smem>
#[inline(always)]
pub fn reserved_smem_end() -> *mut u8 {
    shared_offset_to_generic(unsafe { nv::reserved_smem_offset_end() })
}

/// Returns the total size in bytes of the shared memory region reserved for system software
/// (`%reserved_smem_offset_cap`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-reserved-smem>
#[inline(always)]
pub fn reserved_smem_capacity() -> u32 {
    unsafe { nv::reserved_smem_offset_cap() }
}

/// Returns a generic pointer for the `N`th (`0` or `1`) offset in the shared memory region
/// reserved for system software (`%reserved_smem_offset_<N>`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-reserved-smem>
#[inline(always)]
pub fn reserved_smem_offset<const N: u32>() -> *mut u8 {
    const {
        assert!(
            N < 2,
            "there are only two reserved shared memory offset registers"
        )
    };
    shared_offset_to_generic(unsafe { nv::reserved_smem_offset::<N>() })
}

/// Reads the driver-defined read-only register `%envreg<N>` (`N < 32`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-envreg-32>
#[inline(always)]
pub fn envreg<const N: u32>() -> u32 {
    const { assert!(N < 32, "there are only 32 environment registers") };
    unsafe { nv::envreg::<N>() }
}
