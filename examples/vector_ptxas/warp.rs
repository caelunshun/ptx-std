// --- LLM-generated --- //
//! Warp collectives on the vector types: shuffles and matches of values of any size (including
//! types with a partial last word), and element-wise `redux.sync` on integer `Simd` vectors and
//! `f32` vectors. Each lane reads its own operand, since `ptxas` folds collectives on values it
//! can prove to be warp-uniform.

use core::simd::Simd;
use ptx_std::{
    bf16x8, f4e2m1, f4e2m1x2, f4e2m1x32, f8e4m3x2, f16, f16x2, f16x16, f16x32, f32x8, lane_id,
    warp_collectives::{
        LaneMask, ReduxOp, ShuffleMode, match_all, match_any, reduce, shuffle, shuffle_down,
        shuffle_segmented, shuffle_up, shuffle_xor, try_shuffle_down,
    },
};

/// Instantiates the shuffles and matches of `$ty`.
macro_rules! exchange_kernel {
    ($name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            v: *const $ty,
            operand: u32,
            out: *mut [$ty; 6],
            lanes: *mut u32,
            equal: *mut bool,
        ) {
            unsafe {
                let (v, mask) = (*v.add(lane_id() as usize), LaneMask::FULL);
                *out = [
                    shuffle(mask, v, operand),
                    shuffle_up(mask, v, operand),
                    shuffle_down(mask, v, operand),
                    shuffle_xor(mask, v, operand),
                    shuffle_segmented::<{ ShuffleMode::Xor }, 8, $ty>(mask, v, operand),
                    try_shuffle_down(mask, v, operand).unwrap_or(v),
                ];
                *lanes = match_any(mask, v).bits();
                *equal = match_all(mask, v);
            }
        }
    };
}

exchange_kernel!(warp_f4e2m1, f4e2m1);
exchange_kernel!(warp_f4e2m1x2, f4e2m1x2);
exchange_kernel!(warp_f16, f16);
exchange_kernel!(warp_f8e4m3x2, f8e4m3x2);
exchange_kernel!(warp_bytes3, [u8; 3]);
exchange_kernel!(warp_f16x2, f16x2);
exchange_kernel!(warp_u16x3, [u16; 3]);
exchange_kernel!(warp_u64, u64);
exchange_kernel!(warp_bytes13, [u8; 13]);
exchange_kernel!(warp_bf16x8, bf16x8);
exchange_kernel!(warp_f16x16, f16x16);
exchange_kernel!(warp_f4e2m1x32, f4e2m1x32);
exchange_kernel!(warp_f16x32, f16x32);
exchange_kernel!(warp_f32_array3, [f32; 3]);
exchange_kernel!(warp_f32x8, f32x8);

/// Instantiates every integer `redux.sync` operation on `$ty`.
macro_rules! redux_kernel {
    ($name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(v: *const $ty, out: *mut [$ty; 6]) {
            unsafe {
                let (v, mask) = (*v.add(lane_id() as usize), LaneMask::FULL);
                *out = [
                    reduce::<{ ReduxOp::Add }, $ty>(mask, v),
                    reduce::<{ ReduxOp::Min }, $ty>(mask, v),
                    reduce::<{ ReduxOp::Max }, $ty>(mask, v),
                    reduce::<{ ReduxOp::And }, $ty>(mask, v),
                    reduce::<{ ReduxOp::Or }, $ty>(mask, v),
                    reduce::<{ ReduxOp::Xor }, $ty>(mask, v),
                ];
            }
        }
    };
}

redux_kernel!(redux_u32, u32);
redux_kernel!(redux_i32, i32);
redux_kernel!(redux_u32x4, Simd<u32, 4>);
redux_kernel!(redux_i32x8, Simd<i32, 8>);
redux_kernel!(redux_u32x32, Simd<u32, 32>);

#[cfg(target_feature = "sm_100f")]
mod float {
    use ptx_std::{
        NanHandling, f32x4, lane_id,
        warp_collectives::{Absolute, LaneMask, ReduxOp, reduce_f32},
    };

    /// Instantiates `reduce_f32` on `$ty` with several modifier combinations.
    macro_rules! redux_f32_kernel {
        ($name:ident, $ty:ty) => {
            #[unsafe(no_mangle)]
            pub unsafe extern "ptx-kernel" fn $name(v: *const $ty, out: *mut [$ty; 4]) {
                unsafe {
                    let (v, mask) = (*v.add(lane_id() as usize), LaneMask::FULL);
                    *out = [
                        reduce_f32::<
                            { ReduxOp::Min },
                            { Absolute::Disabled },
                            { NanHandling::Ignore },
                            $ty,
                        >(mask, v),
                        reduce_f32::<
                            { ReduxOp::Max },
                            { Absolute::Enabled },
                            { NanHandling::Ignore },
                            $ty,
                        >(mask, v),
                        reduce_f32::<
                            { ReduxOp::Min },
                            { Absolute::Enabled },
                            { NanHandling::Propagate },
                            $ty,
                        >(mask, v),
                        reduce_f32::<
                            { ReduxOp::Max },
                            { Absolute::Disabled },
                            { NanHandling::Propagate },
                            $ty,
                        >(mask, v),
                    ];
                }
            }
        };
    }

    redux_f32_kernel!(redux_f32, f32);
    redux_f32_kernel!(redux_f32x4, f32x4);
}
