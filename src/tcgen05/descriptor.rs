// --- LLM-generated --- //
//! Matrix descriptors of the `tcgen05` instructions: shared memory descriptors, instruction
//! descriptors and zero-column mask descriptors.
//!
//! The builders check that their arguments are encodable with `assert!`. All of them except
//! [`SmemDescriptor::new`] are `const fn`s, so misuse becomes a compile-time error when the
//! descriptor is built in a `const` item.
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-descriptors>

use core::marker::ConstParamTy;

use bytemuck::{Pod, Zeroable};

use super::MmaKind;
use crate::memory::to_shared_address;

/// The swizzling mode of a matrix in shared memory, stored in a [`SmemDescriptor`].
///
/// The swizzle pattern repeats every 256 bytes (32-byte swizzling), 512 bytes (64-byte swizzling)
/// or 1024 bytes (128-byte swizzling), and the matrix must start at such an address, or the
/// [base offset](SmemDescriptor::with_base_offset) must be set accordingly. Not all combinations of
/// element size, major-ness and swizzling are valid:
///
/// | Element size | Major-ness of A / B | Supported swizzling |
/// |---|---|---|
/// | 4, 6, 8, 16, 32 bits | K-major | All modes |
/// | 8, 16 bits | M / N-major | All modes except [`Bytes128Atom32B`](Self::Bytes128Atom32B) |
/// | 32 bits (`tf32`) | M / N-major | Only [`Bytes128Atom32B`](Self::Bytes128Atom32B) |
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum Swizzle {
    /// No swizzling (or interleaved layout).
    None = 0,
    /// 128-byte swizzling with 32-byte atomicity. The only mode supported for M / N-major `tf32`
    /// matrices; not supported for M / N-major matrices of any other kind.
    Bytes128Atom32B = 1,
    /// 128-byte swizzling (with 16-byte atomicity).
    Bytes128 = 2,
    /// 64-byte swizzling.
    Bytes64 = 4,
    /// 32-byte swizzling.
    Bytes32 = 6,
}

/// A shared memory descriptor: a 64-bit value describing a matrix in the shared memory of the
/// executing CTA (its start address, strides and swizzling mode), used as an operand of
/// [`mma`](super::mma) and [`copy`](super::copy).
///
/// Matrix elements are normalized to 128-bit units, so all addresses and strides are multiples of
/// 16 bytes, and the descriptor encodes them as `(x & 0x3FFFF) >> 4` (14 bits, covering at most
/// 256 KiB of shared memory).
///
/// The layout of the descriptor is: the start address in bits 0..15, the leading dimension byte
/// offset in bits 16..31, the stride dimension byte offset in bits 32..46, the fixed constant
/// `0b001` in bits 46..49, the base offset in bits 49..52, the leading dimension stride mode in
/// bit 52, and the [`Swizzle`] mode in bits 61..64. All other bits are zero.
///
/// # Strides
///
/// A matrix is composed of *swizzle atoms* of 128-bit units that are swizzled as a whole: 8 rows by
/// (swizzle bytes / 16) columns for K-major matrices, (swizzle bytes / 16) rows by 8 columns for
/// M / N-major matrices, and 8x1 / 1x8 without swizzling. The *leading dimension byte offset* (LBO)
/// and *stride dimension byte offset* (SBO) locate the atoms relative to each other:
///
/// | Major-ness | Swizzling | LBO | SBO |
/// |---|---|---|---|
/// | K-major | none | Stride from the first to the second column of the 8x2 tile | Offset from the first 8 rows to the next 8 rows |
/// | K-major | any | Unused (assumed to be 1) | Offset from the first 8 rows to the next 8 rows |
/// | M / N-major | none | Stride from the first 8 columns to the next 8 columns | Offset from the first row to the next row |
/// | M / N-major | any | Stride from the first (swizzle bytes / 16) rows to the next (swizzle bytes / 16) rows | Offset from the first 8 columns to the next 8 columns |
///
/// Here, "row" and "column" refer to the matrix normalized to 128-bit units along its contiguous
/// (major) dimension: rows run along M / N and columns along K. For example, the canonical layouts
/// in CuTe notation, with `T` the number of elements in 128 bits, `m` the number of repeating
/// patterns across rows and `k` across columns, are:
///
/// | Major-ness | Swizzling | Canonical layout |
/// |---|---|---|
/// | M / N | none | `((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))` |
/// | M / N | 32 / 64 / 128 bytes | `((T,S,m),(8,k)):((1,T,LBO),(S*T,SBO))` with `S` = 2 / 4 / 8 |
/// | K | none | `((8,m),(T,2k)):((1T,SBO),(1,LBO))` |
/// | K | 32 / 64 / 128 bytes | `((8,m),(T,2k)):((S*T,SBO),(1,T))` with `S` = 2 / 4 / 8 |
///
/// The start address, LBO and SBO must be multiples of 16 bytes; this is checked by the builders.
///
/// # Advancing along K
///
/// To issue several MMAs along the K dimension from the same tile, [`add_start_bytes`] advances
/// the start address of an existing descriptor.
///
/// [`add_start_bytes`]: Self::add_start_bytes
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct SmemDescriptor(u64);

/// Encodes a 16-byte-aligned shared memory byte offset (up to 256 KiB) as a descriptor field.
#[inline(always)]
const fn encode_smem_offset(bytes: u32) -> u64 {
    assert!(
        bytes.is_multiple_of(16),
        "offsets must be multiples of 16 bytes"
    );
    assert!(bytes <= 0x3FFFF, "offsets must be smaller than 256 KiB");
    (bytes >> 4) as u64
}

impl SmemDescriptor {
    /// Bit 52: leading dimension stride mode (0: byte offset relative, 1: byte address absolute).
    const ABSOLUTE_LEADING_BIT: u64 = 1 << 52;

    /// Creates a descriptor for the matrix starting at `start`, which must point into the shared
    /// memory of the executing CTA, using relative leading dimension byte offsets and a base offset
    /// of 0.
    ///
    /// `leading_byte_offset` and `stride_byte_offset` are the strides in bytes described in the
    /// [type documentation](SmemDescriptor#strides). `start` must be a multiple of 16 (not
    /// checked, see [`from_shared_address`](Self::from_shared_address)). The offsets must be
    /// multiples of 16 less than 256 KiB, and are checked at runtime, so they should be
    /// compile-time constants after inlining for the checks to fold away. To check them at compile
    /// time, build a descriptor with [`from_shared_address`](Self::from_shared_address) in a
    /// `const` item and set the start address at runtime instead.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor>
    #[inline(always)]
    pub fn new<T>(
        start: *const T,
        leading_byte_offset: u32,
        stride_byte_offset: u32,
        swizzle: Swizzle,
    ) -> Self {
        Self::from_shared_address(
            to_shared_address(start),
            leading_byte_offset,
            stride_byte_offset,
            swizzle,
        )
    }

    /// Like [`new`](Self::new), but takes the start of the matrix as an address in the
    /// `.shared::cta` state space (see [`to_shared_address`]).
    ///
    /// `start` is not checked, as it is typically only known at runtime: it must be a multiple of
    /// 16, and only its bits 4 to 17 are encoded.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor>
    #[inline(always)]
    pub const fn from_shared_address(
        start: u32,
        leading_byte_offset: u32,
        stride_byte_offset: u32,
        swizzle: Swizzle,
    ) -> Self {
        let start = ((start & 0x3FFFF) >> 4) as u64;
        let lbo = encode_smem_offset(leading_byte_offset);
        let sbo = encode_smem_offset(stride_byte_offset);
        Self(start | (lbo << 16) | (sbo << 32) | (0b001 << 46) | ((swizzle as u64) << 61))
    }

    /// Sets the matrix base offset, for matrices whose swizzle pattern does not start at the
    /// boundary given in [`Swizzle`] (256, 512 or 1024 bytes for 32-, 64- or 128-byte swizzling).
    ///
    /// For such matrices, `base_offset` is `(pattern_start_address >> 7) & 0x7`, where
    /// `pattern_start_address` is the address at which the repeating pattern of the swizzle mode
    /// starts. It must be 0 (the default) if the pattern starts at the usual boundary.
    ///
    /// The base offset must be 0 in the [absolute leading address
    /// mode](Self::with_absolute_leading_address).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor>
    #[inline(always)]
    pub const fn with_base_offset(self, base_offset: u32) -> Self {
        assert!(base_offset < 8, "the base offset is a 3-bit value");
        assert!(
            base_offset == 0 || self.0 & Self::ABSOLUTE_LEADING_BIT == 0,
            "the base offset must be 0 in the absolute leading address mode"
        );
        Self((self.0 & !(0b111 << 49)) | ((base_offset as u64) << 49))
    }

    /// Switches the descriptor to the *absolute leading dimension address* mode, in which the
    /// leading dimension field holds the shared memory address `address` (in the `.shared::cta`
    /// state space, a multiple of 16 bytes that is not checked) of the second chunk of the matrix instead of a byte
    /// offset relative to the start.
    ///
    /// This mode exists for `tcgen05.mma` with a K extent of 48 bytes (e.g. `.kind::mxf4` with
    /// K = 96), which would cross a 128-byte boundary if its data were packed contiguously: the data
    /// is split into two chunks that are each contained in an aligned 128-byte region, and the
    /// leading dimension address points to the second one.
    ///
    /// Restrictions, all checked by this function except for the second one:
    /// - Only supported on `sm_103a`.
    /// - Only for K-major matrices, i.e. the transpose bit of the matrix in the
    ///   [`InstrDescriptor`] must be 0.
    /// - Only for [`Swizzle::Bytes128`] descriptors, so call this after the descriptor has been
    ///   created with that swizzle mode.
    /// - The base offset must be 0.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor>
    #[inline(always)]
    pub const fn with_absolute_leading_address(self, address: u32) -> Self {
        if !cfg!(target_feature = "sm_103a") {
            panic!("the absolute leading dimension address mode requires sm_103a");
        }
        assert!(
            (self.0 >> 61) & 0b111 == Swizzle::Bytes128 as u64,
            "the absolute leading dimension address mode requires 128-byte swizzling"
        );
        assert!(
            (self.0 >> 49) & 0b111 == 0,
            "the absolute leading dimension address mode requires a base offset of 0"
        );
        let address = ((address & 0x3FFFF) >> 4) as u64;
        Self((self.0 & !(0x7FFF << 16)) | (address << 16) | Self::ABSOLUTE_LEADING_BIT)
    }

    /// Returns the descriptor with its start address advanced by `bytes`, a multiple of 16.
    ///
    /// This is a single 64-bit addition on the descriptor, which is how the K dimension of a
    /// matrix is traversed with one descriptor: e.g. for K-major `f16` matrices with 128-byte
    /// swizzling, each MMA of K = 16 advances the start address by 32 bytes.
    ///
    /// The start address field is not masked, so the new start address (the sum of the old one
    /// and `bytes`) must stay below 256 KiB, the range covered by the encoding. Any valid address
    /// in the shared memory of the CTA satisfies this.
    #[inline(always)]
    pub const fn add_start_bytes(self, bytes: u32) -> Self {
        assert!(
            bytes.is_multiple_of(16),
            "the offset must be a multiple of 16 bytes"
        );
        Self(self.0.wrapping_add((bytes >> 4) as u64))
    }

    /// Wraps a raw descriptor.
    #[inline(always)]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// Returns the raw descriptor.
    #[inline(always)]
    pub const fn to_bits(self) -> u64 {
        self.0
    }
}

/// The major-ness of a matrix operand in shared memory, stored in the transpose bits of an
/// [`InstrDescriptor`].
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum Major {
    /// K-major (the K dimension is contiguous), the default.
    K = 0,
    /// M-major (for matrix A) or N-major (for matrix B); the matrix is transposed.
    Mn = 1,
}

/// Whether an MMA multiplies by a structured-sparse matrix A (`tcgen05.mma.sp`).
///
/// The `.sp` instruction variants require [`Sparse`](Self::Sparse), and the dense variants
/// [`Dense`](Self::Dense).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum Sparsity {
    /// Dense matrix A.
    Dense = 0,
    /// Structured-sparse matrix A with sparsity metadata.
    Sparse = 1,
}

/// An element type of matrices A and B for [`MmaKind::F16`].
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum F16ElementType {
    /// IEEE half precision (`.f16`).
    F16 = 0,
    /// Brain floating point (`.bf16`).
    Bf16 = 1,
}

/// The element type of the accumulator matrix D for [`MmaKind::F16`] and [`MmaKind::F8f6f4`].
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum FloatAccumulator {
    /// IEEE half precision (`.f16`).
    F16 = 0,
    /// IEEE single precision (`.f32`).
    F32 = 1,
}

/// An element type of matrices A and B for [`MmaKind::F8f6f4`] and [`MmaKind::Mxf8f6f4`].
///
/// In shared memory, the elements of the 4-bit and 6-bit types are packed contiguously in groups
/// of 16 elements, each followed by padding to 16 bytes (8 bytes after 16 `e2m1` elements, 4 bytes
/// after 16 `e2m3` or `e3m2` elements), for the dense K = 32 of these kinds. This is also the
/// source format of the decompressing [`copy`](super::copy) variants
/// ([`CopyDecompression`](super::CopyDecompression)). Matrix A in Tensor Memory instead holds each
/// element in an 8-bit container, as produced by such a copy.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-packing-formats-mxf8f6f4-smem>
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum F8f6f4ElementType {
    /// 8-bit floating point with 4 exponent and 3 mantissa bits (`.e4m3`).
    E4m3 = 0,
    /// 8-bit floating point with 5 exponent and 2 mantissa bits (`.e5m2`).
    E5m2 = 1,
    /// 6-bit floating point with 2 exponent and 3 mantissa bits (`.e2m3`).
    E2m3 = 3,
    /// 6-bit floating point with 3 exponent and 2 mantissa bits (`.e3m2`).
    E3m2 = 4,
    /// 4-bit floating point with 2 exponent and 1 mantissa bits (`.e2m1`).
    E2m1 = 5,
}

/// An element type of matrices A and B for [`MmaKind::I8`].
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum I8ElementType {
    /// Unsigned 8-bit integer (`.u8`).
    U8 = 0,
    /// Signed 8-bit integer (`.s8`).
    S8 = 1,
}

/// The maximum number of columns by which a weight-stationary MMA ([`mma_ws`](super::mma_ws)) may
/// shift while attempting to reuse the B matrix in the collector buffer.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum WsMaxShift {
    /// No shift.
    None = 0,
    /// A maximum shift of 8.
    Shift8 = 1,
    /// A maximum shift of 16.
    Shift16 = 2,
    /// A maximum shift of 32.
    Shift32 = 3,
}

/// The K dimension of a [`MmaKind::Mxf4`] or [`MmaKind::Mxf4nvf4`] MMA.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum Mxf4K {
    /// K = 64 for dense matrices, K = 128 for sparse matrices.
    K64 = 0,
    /// K = 96 for dense matrices, K = 192 for sparse matrices. Only supported on `sm_103a`. The
    /// scale factors of the K = 96 case have 3 (`.block32`) or 6 (`.block16`) columns, see the
    /// scale factor layouts in the PTX ISA.
    K96 = 1,
}

/// The data type of the scale factors of a [`MmaKind::Mxf4nvf4`] MMA, for both scale_A and scale_B.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum Nvf4ScaleType {
    /// Unsigned 8-bit floating point with 4 exponent and 3 mantissa bits (`.ue4m3`). With
    /// `.block32` this requires `sm_107f` or higher.
    Ue4m3 = 0,
    /// Unsigned 8-bit floating point with 8 exponent bits (`.ue8m0`).
    Ue8m0 = 1,
    /// Unsigned 8-bit floating point with 5 exponent and 3 mantissa bits (`.ue5m3`). Requires
    /// `sm_107f` or higher.
    Ue5m3 = 2,
}

/// An instruction descriptor: a 32-bit value describing the shape, element types and other
/// details of a matrix multiply-accumulate operation of kind `KIND`.
///
/// The layout of the descriptor depends on the kind, which is why it is part of the type. The MMA
/// functions infer their `.kind` qualifier from the descriptor. Descriptors are created with the
/// `new` function of the concrete kind, e.g. `InstrDescriptor::<{ MmaKind::F16 }>::new(..)`, and
/// then refined with the `with_*` functions, which exist only for the kinds where the
/// corresponding field is defined:
///
/// | Setter | Kinds |
/// |---|---|
/// | [`with_sparsity`](Self::with_sparsity) | all |
/// | `with_sparsity_selector` | `F16`, `Tf32` (the selector must be 0 for the other kinds) |
/// | `with_saturate` | `I8` |
/// | `with_negate_a`, `with_negate_b` | all but `I8` |
/// | `with_major_a`, `with_major_b` | `F16`, `Tf32`, `F8f6f4`, `I8`, `Mxf8f6f4` |
/// | `with_max_b_shift` | `F16`, `Tf32`, `F8f6f4`, `I8` |
/// | `with_scale_a_id`, `with_scale_b_id` | `Mxf8f6f4`, `Mxf4`, `Mxf4nvf4` |
/// | `with_k` | `Mxf4`, `Mxf4nvf4` |
///
/// The fields that are only defined on `sm_107` (the K dimension bits of the non-block-scaled
/// kinds and `Mxf8f6f4`, the scale factor layout of A, and the sparsity version of `Mxf4` and
/// `Mxf4nvf4`) are always 0.
///
/// # Shapes
///
/// The M x N x K shape of the MMA is given by `m` and `n`, which must be encodable, and by the
/// kind, sparsity and `.cta_group` of the instruction. The valid combinations for the
/// architectures supported by this crate are (`.ws` is [`mma_ws`](super::mma_ws), which uses
/// one CTA):
///
/// | Kind | CTAs | M | N | K (dense / sparse) |
/// |---|---|---|---|---|
/// | `F16`, `Tf32`, `F8f6f4` | 1 | 64, 128 | multiple of 8, up to 256 | 16 / 32, 8 / 16, 32 / 64 |
/// | `F16`, `Tf32`, `F8f6f4` | 2 | 128, 256 | multiple of 16, up to 256 | |
/// | `F16`, `Tf32`, `F8f6f4`, `I8` `.ws` | 1 | 32, 64, 128 | 64, 128, 256 (sparse: 64, 128) | |
/// | `I8` | 1 | 64, 128 | 8, 16, 24, 32, then multiple of 16 up to 256 | 32 / 64 |
/// | `I8` | 2 | 128, 256 | multiple of 32, up to 256 | 32 / 64 |
/// | `Mxf8f6f4` | 1 | 128 | multiple of 8, up to 256 | 32 / 64 |
/// | `Mxf8f6f4` | 2 | 128, 256 | multiple of 16, up to 256 | 32 / 64 |
/// | `Mxf4`, `Mxf4nvf4` | 1 | 128 | multiple of 8, up to 256 | 64 / 128 (96 / 192) |
/// | `Mxf4`, `Mxf4nvf4` | 2 | 128, 256 | multiple of 16, up to 256 | 64 / 128 (96 / 192) |
///
/// An 8-bit M / N-major matrix B (`F8f6f4`, `Mxf8f6f4`, `I8`) additionally requires N to be a
/// multiple of 16 (1 CTA) or 32 (2 CTAs). The sparse 2-CTA block-scaled MMAs require M = 256.
/// Only the block-scaled kinds have no `.ws` variant.
/// The values for M and N in the descriptor are the per-instruction ones: for two CTAs, M is the
/// total number of rows of both CTAs.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct InstrDescriptor<const KIND: MmaKind>(u32);

// SAFETY: `InstrDescriptor` is a transparent wrapper of a `u32`.
unsafe impl<const KIND: MmaKind> Zeroable for InstrDescriptor<KIND> {}
// SAFETY: `InstrDescriptor` is a transparent wrapper of a `u32`.
unsafe impl<const KIND: MmaKind> Pod for InstrDescriptor<KIND> {}

/// Encodes the `n` dimension of an instruction descriptor (bits 17..23).
#[inline(always)]
const fn encode_n(n: u32) -> u32 {
    assert!(
        n.is_multiple_of(8) && n >= 8 && n <= 256,
        "N must be a multiple of 8 between 8 and 256"
    );
    (n >> 3) << 17
}

/// Encodes the `m` dimension of an instruction descriptor of a non-block-scaled kind (bits 24..29).
#[inline(always)]
const fn encode_m(m: u32) -> u32 {
    assert!(
        m.is_multiple_of(16) && m >= 16 && m <= 256,
        "M must be a multiple of 16 between 16 and 256"
    );
    (m >> 4) << 24
}

/// Encodes the `m` dimension of an instruction descriptor of a block-scaled kind (bits 27..29).
#[inline(always)]
const fn encode_m_block_scaled(m: u32) -> u32 {
    assert!(
        m == 128 || m == 256,
        "M must be 128 or 256 for block-scaled MMAs"
    );
    (m >> 7) << 27
}

impl<const KIND: MmaKind> InstrDescriptor<KIND> {
    /// Wraps a raw descriptor, which must have the layout for `KIND`.
    #[inline(always)]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the raw descriptor.
    #[inline(always)]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    /// Replaces the `width`-bit field at bit `shift` by `value`.
    #[inline(always)]
    const fn with_field(self, shift: u32, width: u32, value: u32) -> Self {
        assert!(value < (1 << width), "the value does not fit in the field");
        let mask = ((1 << width) - 1) << shift;
        Self((self.0 & !mask) | (value << shift))
    }

    /// Sets whether matrix A is structured-sparse, which must match the instruction variant
    /// (`tcgen05.mma.sp` or dense).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
    #[inline(always)]
    pub const fn with_sparsity(self, sparsity: Sparsity) -> Self {
        self.with_field(2, 1, sparsity as u32)
    }
}

impl InstrDescriptor<{ MmaKind::F16 }> {
    /// Creates a descriptor for `.kind::f16` with an M x N x K shape, matrices A and B of the
    /// types `a` and `b`, and an accumulator D of type `d`. Matrices A and B are K-major, are not
    /// negated, and the MMA is dense.
    ///
    /// An `f16` accumulator is only supported if both `a` and `b` are [`F16ElementType::F16`].
    ///
    /// The shape rules are given in the [type documentation](InstrDescriptor#shapes).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
    #[inline(always)]
    pub const fn new(
        a: F16ElementType,
        b: F16ElementType,
        d: FloatAccumulator,
        m: u32,
        n: u32,
    ) -> Self {
        Self(((d as u32) << 4) | ((a as u32) << 7) | ((b as u32) << 10) | encode_n(n) | encode_m(m))
    }
}

impl InstrDescriptor<{ MmaKind::Tf32 }> {
    /// Creates a descriptor for `.kind::tf32` with an M x N x K shape. Matrices A and B are `tf32`
    /// and K-major and are not negated, the accumulator D is `f32`, and the MMA is dense.
    ///
    /// The shape rules are given in the [type documentation](InstrDescriptor#shapes).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
    #[inline(always)]
    pub const fn new(m: u32, n: u32) -> Self {
        // D is f32 (1), A and B are tf32 (2).
        Self((1 << 4) | (2 << 7) | (2 << 10) | encode_n(n) | encode_m(m))
    }
}

impl InstrDescriptor<{ MmaKind::F8f6f4 }> {
    /// Creates a descriptor for `.kind::f8f6f4` with an M x N x K shape, matrices A and B of the
    /// types `a` and `b`, and an accumulator D of type `d`. Matrices A and B are K-major, are not
    /// negated, and the MMA is dense.
    ///
    /// The shape rules are given in the [type documentation](InstrDescriptor#shapes).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
    #[inline(always)]
    pub const fn new(
        a: F8f6f4ElementType,
        b: F8f6f4ElementType,
        d: FloatAccumulator,
        m: u32,
        n: u32,
    ) -> Self {
        Self(((d as u32) << 4) | ((a as u32) << 7) | ((b as u32) << 10) | encode_n(n) | encode_m(m))
    }
}

impl InstrDescriptor<{ MmaKind::I8 }> {
    /// Creates a descriptor for `.kind::i8` with an M x N x K shape and matrices A and B of the
    /// types `a` and `b`. The accumulator D is `s32`, matrices A and B are K-major, and the MMA is
    /// dense and does not saturate.
    ///
    /// The shape rules are given in the [type documentation](InstrDescriptor#shapes).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
    #[inline(always)]
    pub const fn new(a: I8ElementType, b: I8ElementType, m: u32, n: u32) -> Self {
        // D is s32 (2).
        Self((2 << 4) | ((a as u32) << 7) | ((b as u32) << 10) | encode_n(n) | encode_m(m))
    }

    /// Sets whether the accumulation saturates (clamps) on overflow instead of wrapping.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
    #[inline(always)]
    pub const fn with_saturate(self, saturate: bool) -> Self {
        self.with_field(3, 1, saturate as u32)
    }
}

impl InstrDescriptor<{ MmaKind::Mxf8f6f4 }> {
    /// Creates a descriptor for `.kind::mxf8f6f4` with an M x N x K shape (K = 32 for dense
    /// matrices and 64 for sparse matrices) and matrices A and B of the types `a` and `b`. The scale
    /// factors are `ue8m0` and read at byte 0 of the Tensor Memory words (see
    /// [`with_scale_a_id`](Self::with_scale_a_id)), matrices A and B are K-major and not negated,
    /// and the MMA is dense. The accumulator D is `f32`.
    ///
    /// M must be 128 or 256. The other shape rules are given in the [type
    /// documentation](InstrDescriptor#shapes).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
    #[inline(always)]
    pub const fn new(a: F8f6f4ElementType, b: F8f6f4ElementType, m: u32, n: u32) -> Self {
        // The scale factors are ue8m0 (1).
        Self(
            ((a as u32) << 7)
                | ((b as u32) << 10)
                | (1 << 23)
                | encode_n(n)
                | encode_m_block_scaled(m),
        )
    }
}

impl InstrDescriptor<{ MmaKind::Mxf4 }> {
    /// Creates a descriptor for `.kind::mxf4` with an M x N shape and K = 64 for dense matrices
    /// (see [`with_k`](Self::with_k)). Matrices A and B are `e2m1`, K-major (the only supported
    /// major-ness) and not negated, the scale factors are `ue8m0` and read at byte 0 of the Tensor
    /// Memory words (see [`with_scale_a_id`](Self::with_scale_a_id)), and the MMA is dense. The
    /// accumulator D is `f32`.
    ///
    /// M must be 128 or 256. The other shape rules are given in the [type
    /// documentation](InstrDescriptor#shapes).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
    #[inline(always)]
    pub const fn new(m: u32, n: u32) -> Self {
        // A and B are e2m1 (1), the scale factors are ue8m0 (1).
        Self((1 << 7) | (1 << 10) | (1 << 23) | encode_n(n) | encode_m_block_scaled(m))
    }
}

impl InstrDescriptor<{ MmaKind::Mxf4nvf4 }> {
    /// Creates a descriptor for `.kind::mxf4nvf4` with an M x N shape and K = 64 for dense
    /// matrices (see [`with_k`](Self::with_k)), and scale factors of the type `scale_type`.
    /// Matrices A and B are `e2m1`, K-major (the only supported major-ness) and not negated, the
    /// scale factors are read at byte 0 of the Tensor Memory words (see
    /// [`with_scale_a_id`](Self::with_scale_a_id)), and the MMA is dense. The accumulator D is
    /// `f32`.
    ///
    /// M must be 128 or 256. The other shape rules are given in the [type
    /// documentation](InstrDescriptor#shapes).
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
    #[inline(always)]
    pub const fn new(scale_type: Nvf4ScaleType, m: u32, n: u32) -> Self {
        // A and B are e2m1 (1).
        Self(
            (1 << 7)
                | (1 << 10)
                | ((scale_type as u32) << 23)
                | encode_n(n)
                | encode_m_block_scaled(m),
        )
    }
}

impl InstrDescriptor<{ MmaKind::F16 }> {
    /// Sets the sparsity metadata selector (0 to 3), which selects the sub-columns of Tensor Memory
    /// that form the sparsity metadata matrix of a sparse MMA, see [`with_sparsity`].
    ///
    /// The selector is ignored for dense MMAs.
    ///
    /// [`with_sparsity`]: Self::with_sparsity
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-sparse-matrices-sparsity-selector>
    #[inline(always)]
    pub const fn with_sparsity_selector(self, selector: u32) -> Self {
        self.with_field(0, 2, selector)
    }
}

impl InstrDescriptor<{ MmaKind::Tf32 }> {
    /// Sets the sparsity metadata selector (0 to 3), which selects the sub-columns of Tensor Memory
    /// that form the sparsity metadata matrix of a sparse MMA, see [`with_sparsity`].
    ///
    /// The selector is ignored for dense MMAs.
    ///
    /// [`with_sparsity`]: Self::with_sparsity
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-sparse-matrices-sparsity-selector>
    #[inline(always)]
    pub const fn with_sparsity_selector(self, selector: u32) -> Self {
        self.with_field(0, 2, selector)
    }
}

macro_rules! impl_negate {
    ($($kind:ident),*) => {$(
        impl InstrDescriptor<{ MmaKind::$kind }> {
            /// Sets whether the elements of matrix A are negated.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
            #[inline(always)]
            pub const fn with_negate_a(self, negate: bool) -> Self {
                self.with_field(13, 1, negate as u32)
            }

            /// Sets whether the elements of matrix B are negated.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
            #[inline(always)]
            pub const fn with_negate_b(self, negate: bool) -> Self {
                self.with_field(14, 1, negate as u32)
            }
        }
    )*};
}

// Negation is not supported by `.kind::i8`.
impl_negate!(F16, Tf32, F8f6f4, Mxf8f6f4, Mxf4, Mxf4nvf4);

macro_rules! impl_major {
    ($($kind:ident),*) => {$(
        impl InstrDescriptor<{ MmaKind::$kind }> {
            /// Sets the major-ness of matrix A in shared memory ([`Major::Mn`] sets the transpose
            /// bit).
            ///
            /// The PTX ISA lists M-major matrices only for 8-, 16- and 32-bit element types, and
            /// not for the 6-bit and 4-bit types of `.kind::f8f6f4` and `.kind::mxf8f6f4` with
            /// dense K = 64. They are not supported with the [`Swizzle::Bytes128Atom32B`] swizzle
            /// mode, except for `.kind::tf32`, which supports them only with that mode.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
            #[inline(always)]
            pub const fn with_major_a(self, major: Major) -> Self {
                self.with_field(15, 1, major as u32)
            }

            /// Sets the major-ness of matrix B in shared memory ([`Major::Mn`] sets the transpose
            /// bit).
            ///
            /// The restrictions of [`with_major_a`](Self::with_major_a) apply. Additionally,
            /// N-major matrices B of 8-bit elements require N to be a multiple of 16 for one CTA
            /// and of 32 for two CTAs.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
            #[inline(always)]
            pub const fn with_major_b(self, major: Major) -> Self {
                self.with_field(16, 1, major as u32)
            }
        }
    )*};
}

// Transposition is not supported by `.kind::mxf4` and `.kind::mxf4nvf4`.
impl_major!(F16, Tf32, F8f6f4, I8, Mxf8f6f4);

macro_rules! impl_max_b_shift {
    ($($kind:ident),*) => {$(
        impl InstrDescriptor<{ MmaKind::$kind }> {
            /// Sets the maximum shift while attempting B matrix reuse in a weight-stationary MMA
            /// ([`mma_ws`](super::mma_ws)).
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
            #[inline(always)]
            pub const fn with_max_b_shift(self, shift: WsMaxShift) -> Self {
                self.with_field(30, 2, shift as u32)
            }
        }
    )*};
}

// The block-scaled kinds have no `.ws` variant, and use these bits for the A scale factor ID.
impl_max_b_shift!(F16, Tf32, F8f6f4, I8);

macro_rules! impl_scale_ids {
    ($($kind:ident),*) => {$(
        impl InstrDescriptor<{ MmaKind::$kind }> {
            /// Sets the ID (0 to 3) of the scale factors of matrix A, which selects the sub-columns
            /// of the Tensor Memory words of the scale_A matrix that hold the scale factors:
            /// the byte offset for one scale factor per row (`.scale_vec::1X`, `.block32` with
            /// `.kind::mxf8f6f4`) or for three (`.block32` with K = 96), and the half-word offset
            /// (0 or 2) for two (`.scale_vec::2X`) or six (`.block16` with K = 96). It must be 0
            /// for four or eight scale factors (`.scale_vec::4X`, `.block16`). For
            /// `.kind::mxf4` and `.kind::mxf4nvf4`, the PTX ISA only lists 0 and 2 as values of
            /// this field.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-block-scaling>
            #[inline(always)]
            pub const fn with_scale_a_id(self, id: u32) -> Self {
                self.with_field(29, 2, id)
            }

            /// Sets the ID (0 to 3) of the scale factors of matrix B, which selects the
            /// sub-columns of the Tensor Memory words of the scale_B matrix that hold the scale
            /// factors, like [`with_scale_a_id`](Self::with_scale_a_id).
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-block-scaling>
            #[inline(always)]
            pub const fn with_scale_b_id(self, id: u32) -> Self {
                self.with_field(4, 2, id)
            }
        }
    )*};
}

impl_scale_ids!(Mxf8f6f4, Mxf4, Mxf4nvf4);

macro_rules! impl_k {
    ($($kind:ident),*) => {$(
        impl InstrDescriptor<{ MmaKind::$kind }> {
            /// Selects the K dimension of the MMA. Only [`Mxf4K::K64`] (the default) is supported
            /// outside of `sm_103a`.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor>
            #[inline(always)]
            pub const fn with_k(self, k: Mxf4K) -> Self {
                self.with_field(31, 1, k as u32)
            }
        }
    )*};
}

impl_k!(Mxf4, Mxf4nvf4);

/// Which kind of columns the first span of a sub-mask of a [`ZeroColumnMask`] consists of.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-zero-column-mask-descriptor>
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, ConstParamTy)]
#[repr(u8)]
pub enum ColumnSpan {
    /// The first span consists of columns of matrix B that are used.
    Used = 0,
    /// The first span consists of columns that are zero.
    Zeroed = 1,
}

/// A zero-column mask descriptor: a 64-bit value from which the hardware generates a mask of the
/// columns of matrix B to be treated as zero by a weight-stationary MMA
/// ([`mma_ws`](super::mma_ws)), regardless of the values in shared memory.
///
/// The generated mask has one bit per column of B: a 1 bit means that the column is zero, a 0 bit
/// that its values are used. It is periodic, alternating between *runs* of columns that are
/// zeroed and runs of columns that are used, whose lengths are the same for the whole mask
/// ([`new`](Self::new)). Where the pattern starts is set separately for each *sub-mask*
/// ([`with_sub_mask_start`](Self::with_sub_mask_start)).
///
/// The mask is made up of one or more sub-masks, depending on the M of the MMA, each covering a
/// part of the N columns:
///
/// | M | Sub-masks | Columns of B covered by sub-mask `i` |
/// |---|---|---|
/// | 128 | `mask0` | all `N` |
/// | 64 | `mask0`, `mask1` | `i * N/2 .. (i + 1) * N/2` |
/// | 32 | `mask0` .. `mask3` | `i * N/4 .. (i + 1) * N/4` |
///
/// For sub-mask `i`, the mask is generated starting at its first column: the first run has the
/// type given by the first span (`fs_i`) with its first `sc_i` (the start count) columns removed,
/// and the runs then alternate. For example, with 3 zeroed columns, 4 used columns, a start
/// count of 0 and [`ColumnSpan::Zeroed`], the sub-mask is `1110000 1110000 ...` (the first column
/// is the least significant bit, here on the left). The start counts and first spans of
/// sub-masks that do not exist for the M of the MMA are ignored.
///
/// Independently of the mask, the whole B matrix can be shifted to the right by a number of columns
/// ([`with_column_shift`](Self::with_column_shift)).
///
/// The zero value is [`NONE`](Self::NONE), which uses all columns without shifting.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-zero-column-mask-descriptor>
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct ZeroColumnMask(u64);

impl ZeroColumnMask {
    /// The descriptor that zeroes no columns and does not shift: the generated mask is all 0s, so
    /// all columns of B are used.
    pub const NONE: Self = Self(0);

    /// Creates a descriptor whose sub-masks alternate between runs of `used_columns` used columns
    /// (1 to 256) and runs of `zeroed_columns` zeroed columns (1 to 256), starting with a run of
    /// used columns.
    ///
    /// All start counts are 0, the first spans are [`ColumnSpan::Used`], and the column shift is
    /// 0. Use [`with_sub_mask_start`](Self::with_sub_mask_start) to change where each sub-mask
    /// starts.
    ///
    /// The runs are stored as their lengths minus one, in the "skip span" (zeroed columns) and
    /// "use span" (used columns) fields, and the non-zero mask bit that enables the generation of
    /// the mask is set. Note that the PTX ISA documents the two span fields the other way around,
    /// which contradicts its examples.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-zero-column-mask-descriptor>
    #[inline(always)]
    pub const fn new(zeroed_columns: u32, used_columns: u32) -> Self {
        assert!(
            zeroed_columns >= 1 && zeroed_columns <= 256,
            "the zeroed run must be 1 to 256 columns long"
        );
        assert!(
            used_columns >= 1 && used_columns <= 256,
            "the used run must be 1 to 256 columns long"
        );
        Self(
            (1 << 39) | (((zeroed_columns - 1) as u64) << 40) | (((used_columns - 1) as u64) << 48),
        )
    }

    /// Sets the start of sub-mask `sub_mask` (0 to 3): the type of its first run, `first_span`,
    /// and the number of columns `start_count` (0 to 255) that are removed from the beginning of
    /// that run.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-zero-column-mask-descriptor>
    #[inline(always)]
    pub const fn with_sub_mask_start(
        self,
        sub_mask: u32,
        first_span: ColumnSpan,
        start_count: u32,
    ) -> Self {
        assert!(sub_mask < 4, "there are at most 4 sub-masks");
        assert!(start_count < 256, "the start count is an 8-bit value");
        let count_shift = 8 * sub_mask;
        let span_shift = 32 + sub_mask;
        Self(
            (self.0 & !((0xFF << count_shift) | (1 << span_shift)))
                | ((start_count as u64) << count_shift)
                | ((first_span as u64) << span_shift),
        )
    }

    /// Shifts the B matrix to the right by `shift` columns (at most 32, or 16 for M = 32), so
    /// that column `i` of the MMA reads column `i + shift` of B in shared memory. This allows the
    /// MMA to start at a column other than 0.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-zero-column-mask-descriptor>
    #[inline(always)]
    pub const fn with_column_shift(self, shift: u32) -> Self {
        assert!(shift <= 32, "the column shift is at most 32");
        Self((self.0 & !(0x3F << 56)) | ((shift as u64) << 56))
    }

    /// Wraps a raw descriptor.
    #[inline(always)]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// Returns the raw descriptor.
    #[inline(always)]
    pub const fn to_bits(self) -> u64 {
        self.0
    }
}
