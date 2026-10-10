// --- LLM-generated --- //

use ptx_std::{CtaGroup, MBarrier, tcgen05, tma::CtaMask};

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn sync_fences() {
    tcgen05::fence_before_thread_sync();
    tcgen05::fence_after_thread_sync();
}

macro_rules! commit_kernel {
    ($name:ident, $cg:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(mbar: *const MBarrier, ctas: u16) {
            unsafe {
                tcgen05::commit::<{ $cg }>(&*mbar);
                tcgen05::commit_multicast::<{ $cg }>(&*mbar, CtaMask::from_bits(ctas));
            }
        }
    };
}

commit_kernel!(sync_commit_cta_group_1, CtaGroup::One);
commit_kernel!(sync_commit_cta_group_2, CtaGroup::Two);
