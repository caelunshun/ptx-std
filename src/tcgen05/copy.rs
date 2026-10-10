// --- LLM-generated --- //
//! Copies from shared memory to Tensor Memory (`tcgen05.cp`) and row shifts within Tensor Memory
//! (`tcgen05.shift`).

use core::{arch::asm, marker::ConstParamTy};

use super::{SmemDescriptor, TmemAddr, assert_cta_group, cta_group_count};
use crate::modifiers::CtaGroup;

/// The shape of the data copied by [`copy`] (lanes × bits per lane), and how it is distributed
/// over the four lane quarters of Tensor Memory ("warps": quarter `w` holds lanes
/// `32 * w..32 * w + 32`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-movement-shape>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum CopyShape {
    /// 128 lanes × 256 bits (`.128x256b`).
    S128x256b,
    /// 4 lanes × 256 bits (`.4x256b`).
    S4x256b,
    /// 128 lanes × 128 bits (`.128x128b`).
    S128x128b,
    /// 64 lanes × 128 bits, multicast to the warp pairs (0, 2) and (1, 3), where each warp of a
    /// pair receives half of the data (`.64x128b.warpx2::02_13`).
    S64x128bWarpPairs02_13,
    /// 64 lanes × 128 bits, multicast to the warp pairs (0, 1) and (2, 3), where each warp of a
    /// pair receives half of the data (`.64x128b.warpx2::01_23`).
    S64x128bWarpPairs01_23,
    /// 32 lanes × 128 bits, multicast to all four warps (`.32x128b.warpx4`).
    S32x128bWarpx4,
}

/// Optional decompression performed by [`copy`] from a packed 4-bit or 6-bit floating-point
/// format in shared memory to an 8-bit container format in Tensor Memory.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-optional-decompression>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum CopyDecompression {
    /// The data is copied as is.
    None,
    /// Each group of 16 6-bit elements followed by 4 bytes of padding is decompressed to 16
    /// 8-bit elements (`.b8x16.b6x16_p32`). The elements are `e3m2` or `e2m3` values.
    B6x16P32,
    /// Each group of 16 4-bit elements followed by 8 bytes of padding is decompressed to 16
    /// 8-bit elements (`.b8x16.b4x16_p64`). The elements are `e2m1` values.
    B4x16P64,
}

/// Initiates an asynchronous copy of the matrix described by `src` from shared memory to Tensor
/// Memory at `dst` (`tcgen05.cp`).
///
/// `SHAPE` gives the dimensions of the copied data and how it is distributed over the lane
/// quarters, and `DECOMPRESS` an optional decompression of 4-bit or 6-bit elements to 8 bits.
/// With [`CtaGroup::Two`], both the executing CTA and its peer CTA copy from their own shared
/// memory to their own Tensor Memory.
///
/// The copy is issued by a single thread. Its completion is tracked with [`commit`](super::commit),
/// and a following [`mma`](super::mma) of the same thread that reads the copied data as matrix A
/// (or sparsity metadata or scale factors) at the same address is ordered after the copy without
/// further synchronization. The shared memory is read through the async proxy.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-cp>
///
/// # Safety
///
/// - `src` must describe a valid matrix in the shared memory of the executing CTA (and of the peer
///   CTA with [`CtaGroup::Two`]), which must not be modified until the copy has completed.
/// - The destination must lie within allocated Tensor Memory and must not be accessed by other
///   operations until the copy has completed, except as allowed by pipelining.
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn copy<
    const CTA_GROUP: CtaGroup,
    const SHAPE: CopyShape,
    const DECOMPRESS: CopyDecompression,
>(
    dst: TmemAddr,
    src: SmemDescriptor,
) {
    const { assert_cta_group(CTA_GROUP) };
    macro_rules! emit {
        ($shape:literal, $decompress:literal) => {
            unsafe {
                asm!(
                    concat!("tcgen05.cp.cta_group::{cg}", $shape, $decompress, " [{dst}], {src};"),
                    cg = const cta_group_count(CTA_GROUP),
                    dst = in(reg32) dst.to_bits(),
                    src = in(reg64) src.to_bits(),
                    options(nostack),
                )
            }
        };
    }
    asm_variants!(emit;
        SHAPE => {
            CopyShape::S128x256b => ".128x256b",
            CopyShape::S4x256b => ".4x256b",
            CopyShape::S128x128b => ".128x128b",
            CopyShape::S64x128bWarpPairs02_13 => ".64x128b.warpx2::02_13",
            CopyShape::S64x128bWarpPairs01_23 => ".64x128b.warpx2::01_23",
            CopyShape::S32x128bWarpx4 => ".32x128b.warpx4",
        },
        DECOMPRESS => {
            CopyDecompression::None => "",
            CopyDecompression::B6x16P32 => ".b8x16.b6x16_p32",
            CopyDecompression::B4x16P64 => ".b8x16.b4x16_p64",
        },
    )
}

/// Initiates an asynchronous shift of the 32-byte elements of the matrix at `addr` down by one
/// lane, across all lanes except the last (`tcgen05.shift.down`).
///
/// The shift reads lanes `0..31` and writes lanes `1..32` of the 32 lanes × 256 bits starting at
/// `addr`. With [`CtaGroup::Two`], the shift is performed in the Tensor Memory of both the
/// executing CTA and its peer CTA.
///
/// The shift is issued by a single thread. Its completion is tracked with
/// [`commit`](super::commit), and a following [`mma`](super::mma) of the same thread that reads
/// the shifted data as matrix A is ordered after the shift without further synchronization.
///
/// Requires `sm_100a`, `sm_101a`, `sm_103a` or `sm_110a`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-shift>
///
/// # Safety
///
/// - The lane of `addr` must be a multiple of 32.
/// - The 32 lanes × 256 bits starting at `addr` must lie within allocated Tensor Memory and must not be accessed by other
///   operations until the shift has completed, except as allowed by pipelining.
/// - See also the [safety contract](super#safety-contract).
#[cfg(any(
    target_feature = "sm_100a",
    target_feature = "sm_101a",
    target_feature = "sm_103a",
    target_feature = "sm_110a"
))]
#[inline(always)]
pub unsafe fn shift_down<const CTA_GROUP: CtaGroup>(addr: TmemAddr) {
    const { assert_cta_group(CTA_GROUP) };
    unsafe {
        asm!(
            "tcgen05.shift.cta_group::{cg}.down [{addr}];",
            cg = const cta_group_count(CTA_GROUP),
            addr = in(reg32) addr.to_bits(),
            options(nostack),
        )
    }
}
