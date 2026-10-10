// --- LLM-generated --- //

use ptx_std::{
    CtaGroup,
    tcgen05::{self, TmemAddr},
};

macro_rules! alloc_kernel {
    ($name:ident, $cg:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(dst: *mut TmemAddr, addr: u32, columns: u32) {
            unsafe {
                tcgen05::alloc::<{ $cg }>(dst, columns);
                tcgen05::dealloc::<{ $cg }>(TmemAddr::from_bits(addr), columns);
                tcgen05::relinquish_alloc_permit::<{ $cg }>();
            }
        }
    };
}

alloc_kernel!(alloc_cta_group_1, CtaGroup::One);
alloc_kernel!(alloc_cta_group_2, CtaGroup::Two);
