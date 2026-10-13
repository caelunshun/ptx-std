//! A `no_std` standard library for writing GPU kernels in Rust, targeting
//! `nvptx64-nvidia-cuda` and exposing PTX instructions and CUDA runtime facilities (special
//! registers, warp and cluster collectives, barriers, memory operations, atomics, TMA and
//! tensor cores, ...) as Rust functions.
//!
//! The crate requires PTX ISA 9.3 and an `sm_80` or later target. Some items are only available
//! on newer architectures and are gated on the corresponding `target_feature`s; their
//! documentation lists the requirement.
//!
//! The PTX ISA reference is at <https://docs.nvidia.com/cuda/parallel-thread-execution/>.

#![feature(
    stdarch_nvptx,
    asm_experimental_arch,
    adt_const_params,
    f16,
    loop_hints,
    portable_simd,
    gca_min_const_items,
    gca_macroless_args
)]
#![allow(incomplete_features)]
#![no_std]

#[cfg(not(target_arch = "nvptx64"))]
compile_error!("this crate only supports the nvptx64-nvidia-cuda target");

#[cfg(not(target_feature = "ptx93"))]
compile_error!("this crate requires a minimum of PTX 9.3 enabled via target_feature");

#[cfg(not(target_feature = "sm_80"))]
compile_error!("this crate requires an Ampere (sm_80) or later GPU target");

#[macro_use]
mod macros;

pub mod alloc;
pub mod atomic;
pub mod barrier;
pub mod cluster;
pub mod cp_async;
mod data_types;
pub mod debug;
pub mod fence;
mod globals;
pub mod launch;
mod mbarrier;
pub mod memory;
mod modifiers;
#[cfg(any(
    target_feature = "sm_100f",
    target_feature = "sm_101f",
    target_feature = "sm_110f"
))]
pub mod tcgen05;
pub mod time;
pub mod tma;
pub mod warp_collectives;

pub use data_types::*;
pub use globals::*;
pub use mbarrier::*;
pub use modifiers::*;
