// --- LLM-generated --- //
//! Matrix multiply-accumulate operations (`tcgen05.mma`, `tcgen05.mma.sp`, `tcgen05.mma.ws`,
//! `tcgen05.mma.ws.sp`).
//!
//! All of these operations are asynchronous and are issued by a single thread. The `.kind` of the
//! operation is inferred from the [`InstrDescriptor`] passed to them. The optional modifiers of
//! the instructions are selected with the const generic structs [`MmaModifiers`] and
//! [`MmaWsModifiers`].
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions>

use core::{arch::asm, marker::ConstParamTy};

use bytemuck::{Pod, Zeroable};

use super::{
    InstrDescriptor, MmaKind, SmemDescriptor, TmemAddr, ZeroColumnMask, assert_cta_group,
    cta_group_count,
};
use crate::modifiers::CtaGroup;

/// Usage of a collector buffer: a small cache inside the TensorCore that lets a sequence of MMAs
/// reuse the same matrix without reloading it from memory.
///
/// The collector qualifier only grants the hardware *permission* to reuse a matrix; it may reload
/// the matrix anyway. The source memory of a matrix must therefore not be modified while an MMA
/// that uses it has not completed, regardless of the collector usage.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum CollectorUsage {
    /// The matrix is read from memory and filled into the collector buffer (`::fill`).
    Fill,
    /// The matrix is read from the collector buffer (`::use`). Requires a previous fill that is
    /// still valid.
    Use,
    /// The matrix is read from the collector buffer, after which the buffer contents can be
    /// discarded (`::lastuse`). Requires a previous fill that is valid until the buffer is read.
    LastUse,
    /// The contents of the collector buffer can be discarded (`::discard`). This is the default
    /// when no collector usage is specified.
    Discard,
}

/// Whether the rows of matrix A are shifted down by one row (`.ashift`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum AShift {
    /// Matrix A is used as is.
    Disabled,
    /// The rows of matrix A are shifted down by one row, except for the last row. Only allowed
    /// with matrix A in Tensor Memory, `M = 128` or `M = 256`, and not together with
    /// [`CollectorUsage::Fill`] or [`CollectorUsage::Use`] for matrix A.
    Enabled,
}

/// The size of the block of matrix elements that share a scale factor in a block-scaled MMA
/// ([`mma_block_scaled`]), i.e. the shape of the scale factor matrices (`.scale_vectorsize`).
///
/// `.scale_vec::NX` specifies that the scale factor matrix of A has `N` columns and that of B
/// has `N` rows. `.blockN` specifies that one scale factor is applied to each block of `N`
/// elements along K. `.block32` is an alias for `.scale_vec::1X` with
/// [`MmaKind::Mxf8f6f4`] and for `.scale_vec::2X` with [`MmaKind::Mxf4`] and
/// [`MmaKind::Mxf4nvf4`], and `.block16` is an alias for `.scale_vec::4X` with
/// [`MmaKind::Mxf4nvf4`]; for the 4-bit kinds, the aliases only hold for `K = 64` or `K = 128`
/// (with `K = 96`, the `.blockN` forms imply 3 or 6 scale factors per row).
///
/// | Kind | Valid values |
/// |---|---|
/// | [`MmaKind::Mxf8f6f4`] | `Default`, `Vec1X`\*, `Block32` |
/// | [`MmaKind::Mxf4`] | `Default`, `Vec2X`\*, `Block32` |
/// | [`MmaKind::Mxf4nvf4`] | `Vec2X`\*, `Vec4X`\*, `Block16`, `Block32` |
///
/// \* Requires an architecture-specific (`a`) target; the `.blockN` forms are also available on
/// the family-specific (`f`) targets.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum ScaleVectorSize {
    /// No `.scale_vectorsize` qualifier. For [`MmaKind::Mxf4`], this defaults to `Block32`, and
    /// for [`MmaKind::Mxf8f6f4`] it is equivalent to `Block32`. Not allowed with
    /// [`MmaKind::Mxf4nvf4`].
    Default,
    /// `.scale_vec::1X`.
    Vec1X,
    /// `.scale_vec::2X`.
    Vec2X,
    /// `.scale_vec::4X`.
    Vec4X,
    /// `.block16`.
    Block16,
    /// `.block32`.
    Block32,
}

/// The optional modifiers of [`mma`], [`mma_sparse`], [`mma_block_scaled`] and
/// [`mma_block_scaled_sparse`], passed as a const generic argument.
///
/// ```ignore
/// const MODS: MmaModifiers = MmaModifiers::DEFAULT
///     .with_collector_a(CollectorUsage::Fill)
///     .with_scale_input_d(2);
/// tcgen05::mma::<{ CtaGroup::One }, MODS, _>(d, a, b, idesc, None, true);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub struct MmaModifiers {
    /// The usage of the collector buffer for matrix A.
    pub collector_a: CollectorUsage,
    /// Whether the rows of matrix A are shifted by one.
    pub a_shift: AShift,
    /// The immediate `scale-input-d` in `0..=15`: if non-zero, the input matrix D is scaled by
    /// `2^-scale_input_d` before the accumulation, i.e. `D = A*B + D * 2^-scale_input_d`. Only
    /// valid for [`MmaKind::F16`] and [`MmaKind::Tf32`], and only on `sm_100`-family targets.
    pub scale_input_d: u8,
}

impl MmaModifiers {
    /// No modifiers: [`CollectorUsage::Discard`], [`AShift::Disabled`] and no scaling of D.
    pub const DEFAULT: Self = Self {
        collector_a: CollectorUsage::Discard,
        a_shift: AShift::Disabled,
        scale_input_d: 0,
    };

    /// Sets [`collector_a`](Self::collector_a).
    #[inline(always)]
    pub const fn with_collector_a(mut self, collector_a: CollectorUsage) -> Self {
        self.collector_a = collector_a;
        self
    }

    /// Sets [`a_shift`](Self::a_shift).
    #[inline(always)]
    pub const fn with_a_shift(mut self, a_shift: AShift) -> Self {
        self.a_shift = a_shift;
        self
    }

    /// Sets [`scale_input_d`](Self::scale_input_d).
    #[inline(always)]
    pub const fn with_scale_input_d(mut self, scale_input_d: u8) -> Self {
        self.scale_input_d = scale_input_d;
        self
    }
}

/// The optional modifiers of [`mma_ws`] and [`mma_ws_sparse`], passed as a const generic argument.
///
/// A weight-stationary MMA has four collector buffers for matrix B (which holds the activations
/// instead of the weights), selected by [`collector_b_buffer`](Self::collector_b_buffer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub struct MmaWsModifiers {
    /// The index of the collector buffer for matrix B, in `0..4` (`.collector::b0` to
    /// `.collector::b3`).
    pub collector_b_buffer: u8,
    /// The usage of that collector buffer.
    pub collector_b: CollectorUsage,
}

impl MmaWsModifiers {
    /// Collector buffer 0 with [`CollectorUsage::Discard`], which is the default of the
    /// instruction.
    pub const DEFAULT: Self = Self {
        collector_b_buffer: 0,
        collector_b: CollectorUsage::Discard,
    };

    /// Sets [`collector_b_buffer`](Self::collector_b_buffer).
    #[inline(always)]
    pub const fn with_collector_b_buffer(mut self, collector_b_buffer: u8) -> Self {
        self.collector_b_buffer = collector_b_buffer;
        self
    }

    /// Sets [`collector_b`](Self::collector_b).
    #[inline(always)]
    pub const fn with_collector_b(mut self, collector_b: CollectorUsage) -> Self {
        self.collector_b = collector_b;
        self
    }
}

/// A mask of the lanes of Tensor Memory that [`mma`] and [`mma_sparse`] must not update with the
/// result matrix D (the `disable-output-lane` operand).
///
/// Bit `j` of word `i` corresponds to lane `32 * i + j`; if the bit is set, that lane of D is not
/// updated. With [`CtaGroup::One`] only the first four words are used (the instruction takes a
/// vector of 4 registers) and the last four are ignored, with [`CtaGroup::Two`] all eight words are
/// used.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct OutputLaneMask(pub [u32; 8]);

impl OutputLaneMask {
    /// No lane is disabled.
    pub const NONE: Self = Self([0; 8]);

    /// Creates a mask from its words.
    #[inline(always)]
    pub const fn new(words: [u32; 8]) -> Self {
        Self(words)
    }

    /// Returns the words of the mask.
    #[inline(always)]
    pub const fn to_words(self) -> [u32; 8] {
        self.0
    }

    /// Returns the mask with `lane` (less than 256) additionally disabled.
    #[inline(always)]
    pub const fn with_lane_disabled(mut self, lane: u32) -> Self {
        assert!(lane < 256, "lane out of range");
        self.0[(lane / 32) as usize] |= 1 << (lane % 32);
        self
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A location of matrix A of an MMA: a shared memory matrix described by a [`SmemDescriptor`], or
/// a matrix in Tensor Memory at a [`TmemAddr`].
///
/// This trait is sealed.
pub trait MmaOperandA: sealed::Sealed + Copy {
    #[doc(hidden)]
    const IS_TMEM: bool;
    #[doc(hidden)]
    fn operand_bits(self) -> u64;
}

impl sealed::Sealed for SmemDescriptor {}
impl sealed::Sealed for TmemAddr {}

impl MmaOperandA for SmemDescriptor {
    const IS_TMEM: bool = false;

    #[inline(always)]
    fn operand_bits(self) -> u64 {
        self.to_bits()
    }
}

impl MmaOperandA for TmemAddr {
    const IS_TMEM: bool = true;

    #[inline(always)]
    fn operand_bits(self) -> u64 {
        self.to_bits() as u64
    }
}

/// `.kind::i8` requires an architecture-specific target (ptxas rejects it on `sm_103a` and the
/// family-specific targets).
const I8_SUPPORTED: bool = cfg!(any(
    target_feature = "sm_100a",
    target_feature = "sm_101a",
    target_feature = "sm_110a"
));

/// Non-zero `scale-input-d` is only supported by the `sm_100` family (ptxas rejects it on
/// `sm_110`).
const SCALE_INPUT_D_SUPPORTED: bool = cfg!(target_feature = "sm_100f");

const fn check_mma(cta_group: CtaGroup, mods: MmaModifiers, kind: MmaKind, a_is_tmem: bool) {
    assert_cta_group(cta_group);
    assert!(
        !kind.is_block_scaled(),
        "the block-scaled MMA kinds require mma_block_scaled or mma_block_scaled_sparse"
    );
    assert!(
        !matches!(kind, MmaKind::I8) || I8_SUPPORTED,
        "MmaKind::I8 requires sm_100a, sm_101a or sm_110a"
    );
    assert!(mods.scale_input_d <= 15, "scale_input_d must be in 0..=15");
    if mods.scale_input_d != 0 {
        assert!(
            matches!(kind, MmaKind::F16 | MmaKind::Tf32),
            "scale_input_d is only valid for MmaKind::F16 and MmaKind::Tf32"
        );
        if !SCALE_INPUT_D_SUPPORTED {
            panic!(
                "scale_input_d requires an sm_100 family target (sm_100a, sm_100f, sm_103a, sm_103f)"
            );
        }
    }
    check_a_shift(mods, a_is_tmem);
}

const fn check_a_shift(mods: MmaModifiers, a_is_tmem: bool) {
    if matches!(mods.a_shift, AShift::Enabled) {
        assert!(
            a_is_tmem,
            "AShift::Enabled requires matrix A in Tensor Memory (a TmemAddr operand)"
        );
        assert!(
            !matches!(mods.collector_a, CollectorUsage::Fill | CollectorUsage::Use),
            "AShift::Enabled cannot be combined with CollectorUsage::Fill or CollectorUsage::Use"
        );
    }
}

const fn check_block_scaled(
    cta_group: CtaGroup,
    mods: MmaModifiers,
    scale_vec: ScaleVectorSize,
    kind: MmaKind,
    sparse: bool,
) {
    assert_cta_group(cta_group);
    assert!(
        kind.is_block_scaled(),
        "block-scaled MMAs require MmaKind::Mxf8f6f4, MmaKind::Mxf4 or MmaKind::Mxf4nvf4"
    );
    assert!(
        mods.scale_input_d == 0,
        "scale_input_d is not supported by block-scaled MMAs"
    );
    assert!(
        matches!(mods.a_shift, AShift::Disabled),
        "AShift::Enabled is not supported by block-scaled MMAs"
    );
    let valid = match kind {
        MmaKind::Mxf8f6f4 => matches!(
            scale_vec,
            ScaleVectorSize::Default | ScaleVectorSize::Vec1X | ScaleVectorSize::Block32
        ),
        MmaKind::Mxf4 => matches!(
            scale_vec,
            ScaleVectorSize::Default | ScaleVectorSize::Vec2X | ScaleVectorSize::Block32
        ),
        _ => matches!(
            scale_vec,
            ScaleVectorSize::Vec2X
                | ScaleVectorSize::Vec4X
                | ScaleVectorSize::Block16
                | ScaleVectorSize::Block32
        ),
    };
    assert!(
        valid,
        "invalid ScaleVectorSize for this MmaKind (MmaKind::Mxf4nvf4 requires an explicit size)"
    );
    let needs_arch_specific = matches!(
        scale_vec,
        ScaleVectorSize::Vec1X | ScaleVectorSize::Vec2X | ScaleVectorSize::Vec4X
    );
    if needs_arch_specific && !super::ARCH_SPECIFIC {
        panic!(
            "ScaleVectorSize::Vec1X, Vec2X and Vec4X require an architecture-specific target \
             (sm_100a, sm_101a, sm_103a or sm_110a); use Block16 or Block32 otherwise"
        );
    }
    if sparse && matches!(kind, MmaKind::Mxf4 | MmaKind::Mxf4nvf4) && !super::ARCH_SPECIFIC {
        panic!(
            "mma_block_scaled_sparse with MmaKind::Mxf4 or MmaKind::Mxf4nvf4 requires an \
             architecture-specific target (sm_100a, sm_101a, sm_103a or sm_110a)"
        );
    }
}

const fn check_ws(mods: MmaWsModifiers, kind: MmaKind) {
    assert!(
        !kind.is_block_scaled(),
        "weight-stationary MMAs do not support the block-scaled kinds"
    );
    assert!(
        !matches!(kind, MmaKind::I8) || I8_SUPPORTED,
        "MmaKind::I8 requires sm_100a, sm_101a or sm_110a"
    );
    assert!(
        mods.collector_b_buffer < 4,
        "collector_b_buffer must be in 0..4"
    );
}

/// Matrix multiply-accumulate `D = A*B + D` (`tcgen05.mma`).
///
/// Initiates an asynchronous `M×N×K` multiply-accumulate of the `M×K` matrix A, the `K×N` matrix B
/// and the `M×N` matrix D. The shape, element types, transposition and negation of the matrices
/// are given by the instruction descriptor `idesc` (see [`InstrDescriptor`]), whose `KIND` selects
/// the `.kind` qualifier and must be [`MmaKind::F16`], [`MmaKind::Tf32`], [`MmaKind::F8f6f4`] or
/// [`MmaKind::I8`] (use [`mma_block_scaled`] for the block-scaled kinds).
///
/// - `d`: the address of the destination and accumulator matrix D in Tensor Memory.
/// - `a`: matrix A, either in shared memory ([`SmemDescriptor`]) or in Tensor Memory
///   ([`TmemAddr`]).
/// - `b`: matrix B in shared memory (of the executing CTA and, with [`CtaGroup::Two`], also in the
///   peer CTA).
/// - `disable_output_lanes`: if `Some`, the lanes of D marked in the mask are not updated (see
///   [`OutputLaneMask`]). If it is a constant `None` or `Some`, the matching form of the
///   instruction is selected at compile time.
/// - `enable_input_d`: if `false`, the accumulator is not read and `D = A*B` is computed instead.
///   Use this for the first MMA of an accumulation.
///
/// [`MmaModifiers`] selects the collector buffer usage of matrix A, an optional shift of the rows
/// of A (`.ashift`) and an optional scaling of the input accumulator (`scale-input-d`). With a
/// non-zero `scale_input_d`, `D = A*B + D * 2^-scale_input_d`.
///
/// The MMA is issued by a single thread, with the [`CtaGroup`] determining whether it operates on
/// the Tensor Memory of the executing CTA only, or of the CTA pair. Its completion is tracked with
/// [`commit`](super::commit). Consecutive MMAs of the same thread with the same accumulator
/// address and shape are pipelined: they execute in issue order without further synchronization.
/// The shared memory matrices are read through the async proxy and must not be modified until the
/// MMA has completed, regardless of the collector buffer usage.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`, with these exceptions, checked at compile time:
/// - [`MmaKind::I8`] requires `sm_100a`, `sm_101a` or `sm_110a`.
/// - A non-zero `scale_input_d` requires an `sm_100` family target (`sm_100a`, `sm_100f`,
///   `sm_103a` or `sm_103f`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma>
///
/// # Safety
///
/// - `a` and `b` must describe valid matrices matching `idesc` in shared memory (or Tensor
///   Memory for `a`), and shared memory must not be modified until the MMA has completed.
/// - `d` and the Tensor Memory matrix A must lie within allocated Tensor Memory and must not be
///   accessed by other operations until the MMA has completed, except as allowed by pipelining.
/// - The lane of `d` (and of a Tensor Memory matrix A and the sparsity metadata) must be 0,
///   except for the layouts that use half of the datapath lanes (M = 64 with [`CtaGroup::One`]
///   and not weight-stationary, and sparse M = 128 with [`CtaGroup::Two`]), which allow 0 or 16,
///   the same for A, D and the sparsity metadata.
/// - `idesc` must be a valid descriptor.
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn mma<const CTA_GROUP: CtaGroup, const MODS: MmaModifiers, const KIND: MmaKind>(
    d: TmemAddr,
    a: impl MmaOperandA,
    b: SmemDescriptor,
    idesc: InstrDescriptor<KIND>,
    disable_output_lanes: Option<OutputLaneMask>,
    enable_input_d: bool,
) {
    unsafe {
        mma_impl::<false, CTA_GROUP, MODS, KIND, _>(
            d,
            a,
            b,
            TmemAddr::from_bits(0),
            idesc,
            disable_output_lanes,
            enable_input_d,
        )
    }
}

/// Sparse matrix multiply-accumulate `D = A*B + D` (`tcgen05.mma.sp`).
///
/// Like [`mma`], but matrix A is a sparse `M×(K/2)` matrix whose non-zero elements are selected by
/// the metadata at `sparse_metadata` in Tensor Memory. The sparsity selector in the instruction
/// descriptor determines which part of the metadata is used.
///
/// See [`mma`] for the other operands, modifiers and semantics.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`, with these exceptions, checked at compile time:
/// - [`MmaKind::I8`] requires `sm_100a`, `sm_101a` or `sm_110a`.
/// - A non-zero `scale_input_d` requires an `sm_100` family target (`sm_100a`, `sm_100f`,
///   `sm_103a` or `sm_103f`).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma-sp>
///
/// # Safety
///
/// - `sparse_metadata` must point to valid sparsity metadata for matrix A in allocated Tensor
///   Memory, which must not be modified until the MMA has completed, except as allowed by
///   pipelining.
/// - See also the safety requirements of [`mma`] and the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn mma_sparse<
    const CTA_GROUP: CtaGroup,
    const MODS: MmaModifiers,
    const KIND: MmaKind,
>(
    d: TmemAddr,
    a: impl MmaOperandA,
    b: SmemDescriptor,
    sparse_metadata: TmemAddr,
    idesc: InstrDescriptor<KIND>,
    disable_output_lanes: Option<OutputLaneMask>,
    enable_input_d: bool,
) {
    unsafe {
        mma_impl::<true, CTA_GROUP, MODS, KIND, _>(
            d,
            a,
            b,
            sparse_metadata,
            idesc,
            disable_output_lanes,
            enable_input_d,
        )
    }
}

#[inline(always)]
unsafe fn mma_impl<
    const SPARSE: bool,
    const CTA_GROUP: CtaGroup,
    const MODS: MmaModifiers,
    const KIND: MmaKind,
    A: MmaOperandA,
>(
    d: TmemAddr,
    a: A,
    b: SmemDescriptor,
    sparse_metadata: TmemAddr,
    idesc: InstrDescriptor<KIND>,
    disable_output_lanes: Option<OutputLaneMask>,
    enable_input_d: bool,
) {
    const { check_mma(CTA_GROUP, MODS, KIND, A::IS_TMEM) };
    let d = d.to_bits();
    let b = b.to_bits();
    let meta = sparse_metadata.to_bits();
    let idesc = idesc.to_bits();
    let e = enable_input_d as u32;

    macro_rules! leaf {
        ($kind:literal, $ashift:literal, $collector:literal, $a:literal, [$($a_op:tt)*],
         $lanes:literal, [$($lane_op:tt)*], $scale:literal, [$($scale_op:tt)*]) => {
            if const { SPARSE } {
                unsafe {
                    asm!(
                        concat!(
                            "{{ .reg .pred p; setp.ne.b32 p, {e}, 0; ",
                            "tcgen05.mma.sp.cta_group::{cg}", $kind, $ashift, $collector,
                            " [{d}], ", $a, ", {b}, [{meta}], {idesc}", $lanes, ", p", $scale,
                            "; }}"
                        ),
                        cg = const cta_group_count(CTA_GROUP),
                        d = in(reg32) d,
                        $($a_op)*
                        b = in(reg64) b,
                        meta = in(reg32) meta,
                        idesc = in(reg32) idesc,
                        e = in(reg32) e,
                        $($lane_op)*
                        $($scale_op)*
                        options(nostack),
                    )
                }
            } else {
                unsafe {
                    asm!(
                        concat!(
                            "{{ .reg .pred p; setp.ne.b32 p, {e}, 0; ",
                            "tcgen05.mma.cta_group::{cg}", $kind, $ashift, $collector,
                            " [{d}], ", $a, ", {b}, {idesc}", $lanes, ", p", $scale,
                            "; }}"
                        ),
                        cg = const cta_group_count(CTA_GROUP),
                        d = in(reg32) d,
                        $($a_op)*
                        b = in(reg64) b,
                        idesc = in(reg32) idesc,
                        e = in(reg32) e,
                        $($lane_op)*
                        $($scale_op)*
                        options(nostack),
                    )
                }
            }
        };
    }

    macro_rules! with_scale {
        ($kind:literal, $ashift:literal, $collector:literal, $a:literal, [$($a_op:tt)*],
         $lanes:literal, [$($lane_op:tt)*]) => {
            if const { MODS.scale_input_d == 0 } {
                leaf!($kind, $ashift, $collector, $a, [$($a_op)*], $lanes, [$($lane_op)*], "", [])
            } else {
                leaf!(
                    $kind, $ashift, $collector, $a, [$($a_op)*], $lanes, [$($lane_op)*],
                    ", {scale}", [scale = const MODS.scale_input_d as u32,]
                )
            }
        };
    }

    macro_rules! with_lanes {
        ($kind:literal, $ashift:literal, $collector:literal, $a:literal, [$($a_op:tt)*]) => {
            match disable_output_lanes {
                None => with_scale!($kind, $ashift, $collector, $a, [$($a_op)*], "", []),
                Some(mask) => {
                    let mask = mask.0;
                    if const { matches!(CTA_GROUP, CtaGroup::One) } {
                        with_scale!(
                            $kind, $ashift, $collector, $a, [$($a_op)*],
                            ", {{{l0}, {l1}, {l2}, {l3}}}",
                            [
                                l0 = in(reg32) mask[0],
                                l1 = in(reg32) mask[1],
                                l2 = in(reg32) mask[2],
                                l3 = in(reg32) mask[3],
                            ]
                        )
                    } else {
                        with_scale!(
                            $kind, $ashift, $collector, $a, [$($a_op)*],
                            ", {{{l0}, {l1}, {l2}, {l3}, {l4}, {l5}, {l6}, {l7}}}",
                            [
                                l0 = in(reg32) mask[0],
                                l1 = in(reg32) mask[1],
                                l2 = in(reg32) mask[2],
                                l3 = in(reg32) mask[3],
                                l4 = in(reg32) mask[4],
                                l5 = in(reg32) mask[5],
                                l6 = in(reg32) mask[6],
                                l7 = in(reg32) mask[7],
                            ]
                        )
                    }
                }
            }
        };
    }

    macro_rules! emit {
        ($kind:literal, $ashift:literal, $collector:literal) => {
            if const { A::IS_TMEM } {
                let a = a.operand_bits() as u32;
                with_lanes!($kind, $ashift, $collector, "[{a}]", [a = in(reg32) a,])
            } else {
                let a = a.operand_bits();
                with_lanes!($kind, $ashift, $collector, "{a}", [a = in(reg64) a,])
            }
        };
    }

    asm_variants!(emit;
        KIND => {
            MmaKind::F16 => ".kind::f16",
            MmaKind::Tf32 => ".kind::tf32",
            MmaKind::F8f6f4 => ".kind::f8f6f4",
            MmaKind::I8 => ".kind::i8",
        },
        MODS.a_shift => {
            AShift::Disabled => "",
            AShift::Enabled => ".ashift",
        },
        MODS.collector_a => {
            CollectorUsage::Fill => ".collector::a::fill",
            CollectorUsage::Use => ".collector::a::use",
            CollectorUsage::LastUse => ".collector::a::lastuse",
            CollectorUsage::Discard => "",
        },
    )
}

/// Block-scaled matrix multiply-accumulate `D = (A * scale_A) * (B * scale_B) + D`
/// (`tcgen05.mma` with `.block_scale`).
///
/// Like [`mma`], for the block-scaled kinds [`MmaKind::Mxf8f6f4`], [`MmaKind::Mxf4`] and
/// [`MmaKind::Mxf4nvf4`], which scale the matrices A and B with the scale factor matrices
/// `scale_A` (`M×SFA_N`) and `scale_B` (`SFB_M×N`) in Tensor Memory before the multiplication.
/// Each row of A is divided into `SFA_N` chunks, each multiplied with the corresponding element
/// of `scale_A` of the same row, and likewise for each column of B. The exact shape of the scale
/// factor matrices follows from `SCALE_VEC` (see [`ScaleVectorSize`]), and their data types and
/// the scale factor IDs selecting them within the Tensor Memory words from the instruction
/// descriptor. The scale factors of B must be replicated to all four 32-lane partitions of
/// Tensor Memory.
///
/// - `scale_a`, `scale_b`: the addresses of the scale factor matrices in Tensor Memory.
///
/// `disable_output_lanes`, the shift of A and `scale_input_d` are not available for block-scaled
/// MMAs, and `MODS` must not enable them. Only [`MmaModifiers::collector_a`] is used.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`. The `.scale_vec::NX` forms
/// ([`ScaleVectorSize::Vec1X`], [`Vec2X`](ScaleVectorSize::Vec2X) and
/// [`Vec4X`](ScaleVectorSize::Vec4X)) require `sm_100a`, `sm_101a`, `sm_103a` or `sm_110a`, which
/// is checked at compile time. The scale type `ue5m3` and `ue4m3` with
/// [`ScaleVectorSize::Block32`] are only supported by `sm_107`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma>
///
/// # Safety
///
/// - `scale_a` and `scale_b` must point to valid scale factor matrices in allocated Tensor
///   Memory, which must not be modified until the MMA has completed, except as allowed by
///   pipelining.
/// - See also the safety requirements of [`mma`] and the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn mma_block_scaled<
    const CTA_GROUP: CtaGroup,
    const MODS: MmaModifiers,
    const SCALE_VEC: ScaleVectorSize,
    const KIND: MmaKind,
>(
    d: TmemAddr,
    a: impl MmaOperandA,
    b: SmemDescriptor,
    idesc: InstrDescriptor<KIND>,
    scale_a: TmemAddr,
    scale_b: TmemAddr,
    enable_input_d: bool,
) {
    unsafe {
        block_scaled_impl::<false, CTA_GROUP, MODS, SCALE_VEC, KIND, _>(
            d,
            a,
            b,
            TmemAddr::from_bits(0),
            idesc,
            scale_a,
            scale_b,
            enable_input_d,
        )
    }
}

/// Sparse block-scaled matrix multiply-accumulate (`tcgen05.mma.sp` with `.block_scale`).
///
/// Like [`mma_block_scaled`], but matrix A is a sparse `M×(K/2)` matrix whose non-zero elements
/// are selected by the metadata at `sparse_metadata` in Tensor Memory.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f` for [`MmaKind::Mxf8f6f4`], and `sm_100a`, `sm_101a`,
/// `sm_103a` or `sm_110a` for [`MmaKind::Mxf4`] and [`MmaKind::Mxf4nvf4`] and for the
/// `.scale_vec::NX` forms ([`ScaleVectorSize::Vec1X`], [`Vec2X`](ScaleVectorSize::Vec2X) and
/// [`Vec4X`](ScaleVectorSize::Vec4X)), which is checked at compile time.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma-sp>
///
/// # Safety
///
/// - `sparse_metadata` must point to valid sparsity metadata for matrix A in allocated Tensor
///   Memory, which must not be modified until the MMA has completed, except as allowed by
///   pipelining.
/// - See also the safety requirements of [`mma_block_scaled`] and the
///   [safety contract](super#safety-contract).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mma_block_scaled_sparse<
    const CTA_GROUP: CtaGroup,
    const MODS: MmaModifiers,
    const SCALE_VEC: ScaleVectorSize,
    const KIND: MmaKind,
>(
    d: TmemAddr,
    a: impl MmaOperandA,
    b: SmemDescriptor,
    sparse_metadata: TmemAddr,
    idesc: InstrDescriptor<KIND>,
    scale_a: TmemAddr,
    scale_b: TmemAddr,
    enable_input_d: bool,
) {
    unsafe {
        block_scaled_impl::<true, CTA_GROUP, MODS, SCALE_VEC, KIND, _>(
            d,
            a,
            b,
            sparse_metadata,
            idesc,
            scale_a,
            scale_b,
            enable_input_d,
        )
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn block_scaled_impl<
    const SPARSE: bool,
    const CTA_GROUP: CtaGroup,
    const MODS: MmaModifiers,
    const SCALE_VEC: ScaleVectorSize,
    const KIND: MmaKind,
    A: MmaOperandA,
>(
    d: TmemAddr,
    a: A,
    b: SmemDescriptor,
    sparse_metadata: TmemAddr,
    idesc: InstrDescriptor<KIND>,
    scale_a: TmemAddr,
    scale_b: TmemAddr,
    enable_input_d: bool,
) {
    const { check_block_scaled(CTA_GROUP, MODS, SCALE_VEC, KIND, SPARSE) };
    let d = d.to_bits();
    let b = b.to_bits();
    let meta = sparse_metadata.to_bits();
    let idesc = idesc.to_bits();
    let sa = scale_a.to_bits();
    let sb = scale_b.to_bits();
    let e = enable_input_d as u32;

    macro_rules! leaf {
        ($kind:literal, $scale_vec:literal, $collector:literal, $a:literal, [$($a_op:tt)*]) => {
            if const { SPARSE } {
                unsafe {
                    asm!(
                        concat!(
                            "{{ .reg .pred p; setp.ne.b32 p, {e}, 0; ",
                            "tcgen05.mma.sp.cta_group::{cg}", $kind, ".block_scale", $scale_vec,
                            $collector, " [{d}], ", $a,
                            ", {b}, [{meta}], {idesc}, [{sa}], [{sb}], p; }}"
                        ),
                        cg = const cta_group_count(CTA_GROUP),
                        d = in(reg32) d,
                        $($a_op)*
                        b = in(reg64) b,
                        meta = in(reg32) meta,
                        idesc = in(reg32) idesc,
                        sa = in(reg32) sa,
                        sb = in(reg32) sb,
                        e = in(reg32) e,
                        options(nostack),
                    )
                }
            } else {
                unsafe {
                    asm!(
                        concat!(
                            "{{ .reg .pred p; setp.ne.b32 p, {e}, 0; ",
                            "tcgen05.mma.cta_group::{cg}", $kind, ".block_scale", $scale_vec,
                            $collector, " [{d}], ", $a,
                            ", {b}, {idesc}, [{sa}], [{sb}], p; }}"
                        ),
                        cg = const cta_group_count(CTA_GROUP),
                        d = in(reg32) d,
                        $($a_op)*
                        b = in(reg64) b,
                        idesc = in(reg32) idesc,
                        sa = in(reg32) sa,
                        sb = in(reg32) sb,
                        e = in(reg32) e,
                        options(nostack),
                    )
                }
            }
        };
    }

    macro_rules! emit {
        ($kind:literal, $scale_vec:literal, $collector:literal) => {
            if const { A::IS_TMEM } {
                let a = a.operand_bits() as u32;
                leaf!($kind, $scale_vec, $collector, "[{a}]", [a = in(reg32) a,])
            } else {
                let a = a.operand_bits();
                leaf!($kind, $scale_vec, $collector, "{a}", [a = in(reg64) a,])
            }
        };
    }

    asm_variants!(emit;
        KIND => {
            MmaKind::Mxf8f6f4 => ".kind::mxf8f6f4",
            MmaKind::Mxf4 => ".kind::mxf4",
            MmaKind::Mxf4nvf4 => ".kind::mxf4nvf4",
        },
        SCALE_VEC => {
            ScaleVectorSize::Default => "",
            ScaleVectorSize::Vec1X => ".scale_vec::1X",
            ScaleVectorSize::Vec2X => ".scale_vec::2X",
            ScaleVectorSize::Vec4X => ".scale_vec::4X",
            ScaleVectorSize::Block16 => ".block16",
            ScaleVectorSize::Block32 => ".block32",
        },
        MODS.collector_a => {
            CollectorUsage::Fill => ".collector::a::fill",
            CollectorUsage::Use => ".collector::a::use",
            CollectorUsage::LastUse => ".collector::a::lastuse",
            CollectorUsage::Discard => "",
        },
    )
}

/// Weight-stationary matrix multiply-accumulate `D = A*B + D` (`tcgen05.mma.ws`).
///
/// A convolution-oriented MMA where matrix B holds the activations and matrix A the weights,
/// which stay in the TensorCore across a sequence of MMAs: the collector buffers apply to matrix
/// B instead of A. There are four collector buffers for B, selected and used according to
/// [`MmaWsModifiers`].
///
/// This is always a `.cta_group::1` operation, so a kernel using it must use
/// [`CtaGroup::One`] for all its `tcgen05` instructions. `KIND` is inferred from `idesc` and
/// must be [`MmaKind::F16`], [`MmaKind::Tf32`], [`MmaKind::F8f6f4`] or [`MmaKind::I8`]. The
/// block-scaled kinds, `disable_output_lanes`, `.ashift` and `scale-input-d` are not supported.
///
/// - `zero_column_mask`: if `Some`, the descriptor from which the hardware generates a mask of
///   the columns of matrix B that are treated as zero, regardless of their values in shared
///   memory (see [`ZeroColumnMask`]). If it is a constant `None` or `Some`, the matching form of
///   the instruction is selected at compile time.
/// - See [`mma`] for the other operands and the general semantics.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`; [`MmaKind::I8`] requires `sm_100a`, `sm_101a` or
/// `sm_110a`. This is checked at compile time.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma-ws>
///
/// # Safety
///
/// - If a collector buffer is used (`Use` or `LastUse`), a previous fill of the same buffer must
///   still be valid.
/// - See also the safety requirements of [`mma`] and the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn mma_ws<const MODS: MmaWsModifiers, const KIND: MmaKind>(
    d: TmemAddr,
    a: impl MmaOperandA,
    b: SmemDescriptor,
    idesc: InstrDescriptor<KIND>,
    zero_column_mask: Option<ZeroColumnMask>,
    enable_input_d: bool,
) {
    unsafe {
        ws_impl::<false, MODS, KIND, _>(
            d,
            a,
            b,
            TmemAddr::from_bits(0),
            idesc,
            zero_column_mask,
            enable_input_d,
        )
    }
}

/// Weight-stationary sparse matrix multiply-accumulate `D = A*B + D` (`tcgen05.mma.ws.sp`).
///
/// Like [`mma_ws`], but matrix A is a sparse `M×(K/2)` matrix whose non-zero elements are
/// selected by the metadata at `sparse_metadata` in Tensor Memory.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`; [`MmaKind::I8`] requires `sm_100a`, `sm_101a` or
/// `sm_110a`. This is checked at compile time.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma-ws-sp>
///
/// # Safety
///
/// - `sparse_metadata` must point to valid sparsity metadata for matrix A in allocated Tensor
///   Memory, which must not be modified until the MMA has completed, except as allowed by
///   pipelining.
/// - See also the safety requirements of [`mma_ws`] and the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn mma_ws_sparse<const MODS: MmaWsModifiers, const KIND: MmaKind>(
    d: TmemAddr,
    a: impl MmaOperandA,
    b: SmemDescriptor,
    sparse_metadata: TmemAddr,
    idesc: InstrDescriptor<KIND>,
    zero_column_mask: Option<ZeroColumnMask>,
    enable_input_d: bool,
) {
    unsafe {
        ws_impl::<true, MODS, KIND, _>(
            d,
            a,
            b,
            sparse_metadata,
            idesc,
            zero_column_mask,
            enable_input_d,
        )
    }
}

#[inline(always)]
unsafe fn ws_impl<
    const SPARSE: bool,
    const MODS: MmaWsModifiers,
    const KIND: MmaKind,
    A: MmaOperandA,
>(
    d: TmemAddr,
    a: A,
    b: SmemDescriptor,
    sparse_metadata: TmemAddr,
    idesc: InstrDescriptor<KIND>,
    zero_column_mask: Option<ZeroColumnMask>,
    enable_input_d: bool,
) {
    const { check_ws(MODS, KIND) };
    let d = d.to_bits();
    let b = b.to_bits();
    let meta = sparse_metadata.to_bits();
    let idesc = idesc.to_bits();
    let e = enable_input_d as u32;

    macro_rules! leaf {
        ($kind:literal, $collector:literal, $a:literal, [$($a_op:tt)*],
         $mask:literal, [$($mask_op:tt)*]) => {
            if const { SPARSE } {
                unsafe {
                    asm!(
                        concat!(
                            "{{ .reg .pred p; setp.ne.b32 p, {e}, 0; ",
                            "tcgen05.mma.ws.sp.cta_group::1", $kind, $collector,
                            " [{d}], ", $a, ", {b}, [{meta}], {idesc}, p", $mask, "; }}"
                        ),
                        buf = const MODS.collector_b_buffer as u32,
                        d = in(reg32) d,
                        $($a_op)*
                        b = in(reg64) b,
                        meta = in(reg32) meta,
                        idesc = in(reg32) idesc,
                        e = in(reg32) e,
                        $($mask_op)*
                        options(nostack),
                    )
                }
            } else {
                unsafe {
                    asm!(
                        concat!(
                            "{{ .reg .pred p; setp.ne.b32 p, {e}, 0; ",
                            "tcgen05.mma.ws.cta_group::1", $kind, $collector,
                            " [{d}], ", $a, ", {b}, {idesc}, p", $mask, "; }}"
                        ),
                        buf = const MODS.collector_b_buffer as u32,
                        d = in(reg32) d,
                        $($a_op)*
                        b = in(reg64) b,
                        idesc = in(reg32) idesc,
                        e = in(reg32) e,
                        $($mask_op)*
                        options(nostack),
                    )
                }
            }
        };
    }

    macro_rules! with_mask {
        ($kind:literal, $collector:literal, $a:literal, [$($a_op:tt)*]) => {
            match zero_column_mask {
                None => leaf!($kind, $collector, $a, [$($a_op)*], "", []),
                Some(mask) => {
                    let mask = mask.to_bits();
                    leaf!(
                        $kind, $collector, $a, [$($a_op)*],
                        ", {mask}", [mask = in(reg64) mask,]
                    )
                }
            }
        };
    }

    macro_rules! emit {
        ($kind:literal, $collector:literal) => {
            if const { A::IS_TMEM } {
                let a = a.operand_bits() as u32;
                with_mask!($kind, $collector, "[{a}]", [a = in(reg32) a,])
            } else {
                let a = a.operand_bits();
                with_mask!($kind, $collector, "{a}", [a = in(reg64) a,])
            }
        };
    }

    asm_variants!(emit;
        KIND => {
            MmaKind::F16 => ".kind::f16",
            MmaKind::Tf32 => ".kind::tf32",
            MmaKind::F8f6f4 => ".kind::f8f6f4",
            MmaKind::I8 => ".kind::i8",
        },
        MODS.collector_b => {
            CollectorUsage::Fill => ".collector::b{buf}::fill",
            CollectorUsage::Use => ".collector::b{buf}::use",
            CollectorUsage::LastUse => ".collector::b{buf}::lastuse",
            CollectorUsage::Discard => ".collector::b{buf}::discard",
        },
    )
}
