#![feature(stdarch_nvptx, const_param_ty_trait, )]
#![no_std]

#[cfg(not(target_arch = "nvptx64"))]
compile_error!("this crate only supports the nvptx64-nvidia-cuda target");

#[cfg(not(target_feature = "ptx93"))]
compile_error!("this crate requires a minimum of PTX 9.3 enabled via target_feature");

#[cfg(not(target_feature = "sm_80"))]
compile_error!("this crate requires an Ampere (sm_80) or later GPU target");

mod barrier;
mod data_types;
mod globals;
mod mbarrier;
mod memory;
mod tma;
mod warp_collectives;

pub use data_types::*;
pub use globals::*;
pub use mbarrier::*;
pub use tma::*;
