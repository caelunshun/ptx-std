// --- LLM-generated --- //

use ptx_std::{
    CtaGroup,
    tcgen05::{self, CopyDecompression, CopyShape, SmemDescriptor, TmemAddr},
};

macro_rules! copy_kernel {
    ($name:ident, $cg:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(dst: u32, src: u64) {
            let (dst, src) = (TmemAddr::from_bits(dst), SmemDescriptor::from_bits(src));
            macro_rules! all_decompressions {
                ($shape:expr) => {
                    unsafe {
                        tcgen05::copy::<{ $cg }, { $shape }, { CopyDecompression::None }>(dst, src);
                        tcgen05::copy::<{ $cg }, { $shape }, { CopyDecompression::B6x16P32 }>(
                            dst, src,
                        );
                        tcgen05::copy::<{ $cg }, { $shape }, { CopyDecompression::B4x16P64 }>(
                            dst, src,
                        );
                    }
                };
            }
            all_decompressions!(CopyShape::S128x256b);
            all_decompressions!(CopyShape::S4x256b);
            all_decompressions!(CopyShape::S128x128b);
            all_decompressions!(CopyShape::S64x128bWarpPairs02_13);
            all_decompressions!(CopyShape::S64x128bWarpPairs01_23);
            all_decompressions!(CopyShape::S32x128bWarpx4);
            #[cfg(any(
                target_feature = "sm_100a",
                target_feature = "sm_101a",
                target_feature = "sm_103a",
                target_feature = "sm_110a"
            ))]
            unsafe {
                tcgen05::shift_down::<{ $cg }>(dst)
            };
        }
    };
}

copy_kernel!(copy_cta_group_1, CtaGroup::One);
copy_kernel!(copy_cta_group_2, CtaGroup::Two);
