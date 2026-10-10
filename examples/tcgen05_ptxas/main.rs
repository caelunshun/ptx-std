// --- LLM-generated --- //
//! Instantiates the `tcgen05` wrappers of `ptx_std` in kernels, so that the inline PTX they
//! generate can be validated by `ptxas` for every supported target. No GPU is needed.
//!
//! Run `examples/tcgen05_ptxas/check.sh [arch...]` from anywhere; it emits the PTX of this crate
//! for each target (default: `sm_100a sm_100f sm_103a sm_110a`) and assembles it with `ptxas`.
//!
//! Every kernel takes the operands as parameters so that nothing is constant-folded, and uses a
//! single `CtaGroup`, as all `tcgen05` instructions of a kernel must use the same one. Variants
//! that are only available on some targets are instantiated under the same `cfg`s as in the
//! library.

#![no_std]
#![feature(abi_ptx)]
#![allow(clippy::missing_safety_doc)]

mod alloc;
mod copy;
mod descriptor;
mod ld_st;
mod mma;
mod sync;
