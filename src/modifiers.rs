// --- LLM-generated --- //
//! Instruction modifiers shared by several instruction families, used as const-generic
//! parameters to select between instruction variants.

use core::marker::ConstParamTy;

/// Floating-point rounding mode.
///
/// Not every instruction supports every mode; unsupported modes are rejected at compile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum Rounding {
    /// Round to nearest, ties to even (`.rn`, or `.rni` for integer rounding).
    NearestEven,
    /// Round to nearest, ties away from zero (`.rna`).
    NearestAway,
    /// Round towards zero (`.rz`, or `.rzi` for integer rounding).
    TowardZero,
    /// Round towards negative infinity (`.rm`, or `.rmi` for integer rounding).
    Down,
    /// Round towards positive infinity (`.rp`, or `.rpi` for integer rounding).
    Up,
    /// Stochastic rounding using caller-provided random bits (`.rs`).
    Stochastic,
}

/// The set of threads that a memory operation or fence is performed with respect to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum Scope {
    /// The threads of the executing CTA (`.cta`).
    Cta,
    /// The threads of the executing cluster (`.cluster`).
    Cluster,
    /// The threads of the current program on the executing device (`.gpu`).
    Gpu,
    /// The threads of the current program on all devices and the host (`.sys`).
    System,
}

/// Memory ordering semantics of a memory operation or fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum MemoryOrdering {
    /// `.relaxed`
    Relaxed,
    /// `.acquire`
    Acquire,
    /// `.release`
    Release,
    /// `.acq_rel`
    AcqRel,
    /// Sequentially consistent (`.sc`).
    SeqCst,
}

/// Handling of subnormal floating-point inputs and results (`.ftz`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum Denormals {
    /// Subnormal values are supported.
    Preserve,
    /// Subnormal inputs and results are flushed to sign-preserving zero (`.ftz`).
    FlushToZero,
}

/// Saturation of floating-point results to `[0.0, 1.0]` (`.sat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum Saturation {
    /// The result is not clamped.
    None,
    /// The result is clamped to `[0.0, 1.0]`; NaN results become `+0.0` (`.sat`).
    ClampToUnit,
}

/// Handling of results that fall outside the finite range of a narrow destination format
/// (`.satfinite`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum Overflow {
    /// Out-of-range results become infinity (or NaN for formats without infinity).
    Infinity,
    /// Out-of-range results are clamped to the largest finite value of matching sign, and NaN
    /// inputs produce NaN (`.satfinite`).
    SaturateFinite,
}

/// Clamping of negative results to zero (`.relu`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum Relu {
    /// Negative results are kept.
    Disabled,
    /// Negative results are clamped to zero (`.relu`).
    Enabled,
}

/// NaN handling of floating-point min/max operations (`.NaN`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum NanHandling {
    /// If exactly one input is NaN, the other input is returned (IEEE 754 `minNum`/`maxNum`).
    Ignore,
    /// If any input is NaN, the result is canonical NaN (`.NaN`).
    Propagate,
}

/// Sign and magnitude handling of floating-point min/max operations (`.xorsign.abs`).
///
/// Requires `sm_86`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum XorSignAbs {
    /// The usual min/max semantics.
    Disabled,
    /// The magnitude of the result is the min/max of the absolute values of the inputs, and its
    /// sign is the XOR of the input signs. The sign is ignored if the result is NaN
    /// (`.xorsign.abs`).
    Enabled,
}

/// The `.cta_group` qualifier, selecting whether an operation is performed on behalf of a single CTA
/// or of a CTA pair (the 2-CTA execution mode of Blackwell).
///
/// Two CTAs of a cluster whose `%cluster_ctarank` differs only in the lowest bit form a *CTA pair*;
/// each is the *peer CTA* of the other.
///
/// Used by TMA loads into `.shared::cluster` memory and by the `tcgen05` instructions. The
/// `tcgen05` instructions require [`CtaGroup::One`] or [`CtaGroup::Two`], and all `tcgen05`
/// instructions of a kernel must use the same value. Anything other than [`CtaGroup::Unspecified`]
/// requires the Blackwell family of targets (`sm_100f`, `sm_101f` or `sm_110f`, or the
/// corresponding `a` targets).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-cta-pair>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum CtaGroup {
    /// No `.cta_group` qualifier. Only valid for TMA operations.
    Unspecified,
    /// `.cta_group::1`: the operation is issued on behalf of the executing CTA only. TMA loads
    /// signal completion on the mbarrier of the executing CTA.
    One,
    /// `.cta_group::2`: the operation is issued on behalf of the CTA pair of the executing CTA.
    /// TMA loads signal completion on the mbarrier of either CTA of the pair, as defined by the
    /// PTX ISA, and `tcgen05` operations access the Tensor Memory of both CTAs of the pair.
    Two,
}
