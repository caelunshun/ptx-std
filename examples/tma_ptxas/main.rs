// --- LLM-generated --- //
//! Instantiates the `tma` bulk reduction wrappers of `ptx_std` in kernels, so that the inline PTX
//! they generate can be validated by `ptxas` for several targets. No GPU is needed.
//!
//! Run `examples/tma_ptxas/check.sh [arch...]` from anywhere; it emits the PTX of this crate
//! for each target (default: `sm_90 sm_100a sm_103a`) and assembles it with `ptxas`.

#![no_std]
#![feature(abi_ptx)]
#![allow(clippy::missing_safety_doc)]

use ptx_std::{MBarrier, Scope as S, bf16, f16, memory::CachePolicy, tma::*};

macro_rules! global {
    ($ty:ty, $p:ident, $policy:ident; $($op:ident)+) => {$(
        unsafe {
            bulk_reduce::<$ty, { TmaReduceOp::$op }, { S::Cta }>($p.cast(), $p.cast(), 16, None);
            bulk_reduce::<$ty, { TmaReduceOp::$op }, { S::Cluster }>($p.cast(), $p.cast(), 16, None);
            bulk_reduce::<$ty, { TmaReduceOp::$op }, { S::Gpu }>($p.cast(), $p.cast(), 16, None);
            bulk_reduce::<$ty, { TmaReduceOp::$op }, { S::System }>(
                $p.cast(),
                $p.cast(),
                16,
                Some($policy),
            );
        }
    )+};
}

macro_rules! cluster {
    ($ty:ty, $p:ident, $mbar:ident; $($op:ident)+) => {$(
        unsafe {
            bulk_reduce_to_cluster::<$ty, { TmaReduceOp::$op }, { S::Cta }>(
                $p.cast(),
                $p.cast(),
                16,
                $mbar,
            );
            bulk_reduce_to_cluster::<$ty, { TmaReduceOp::$op }, { S::Cluster }>(
                $p.cast(),
                $p.cast(),
                16,
                $mbar,
            );
        }
    )+};
}

/// Every supported combination of `bulk_reduce`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn bulk_reduce_global(p: *mut u8, policy: u64) {
    let policy = CachePolicy::from_bits(policy);
    global!(u32, p, policy; Add Min Max WrapIncrement WrapDecrement And Or Xor);
    global!(i32, p, policy; Add Min Max And Or Xor);
    global!(u64, p, policy; Add Min Max And Or Xor);
    global!(i64, p, policy; Add Min Max And Or Xor);
    global!(f32, p, policy; Add);
    global!(f64, p, policy; Add);
    global!(f16, p, policy; Add Min Max);
    global!(bf16, p, policy; Add Min Max);
}

/// Every supported combination of `bulk_reduce_to_cluster`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn bulk_reduce_cluster(p: *mut u8, mbar: *const MBarrier) {
    let mbar = unsafe { &*mbar };
    cluster!(u32, p, mbar; Add Min Max WrapIncrement WrapDecrement And Or Xor);
    cluster!(i32, p, mbar; Add Min Max And Or Xor);
    cluster!(u64, p, mbar; Add);
    cluster!(i64, p, mbar; Add);
}
