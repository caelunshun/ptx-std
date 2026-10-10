use core::{cell::UnsafeCell, mem::MaybeUninit};

#[derive(Debug)]
#[repr(transparent)]
pub struct MBarrier(UnsafeCell<MaybeUninit<u64>>);

impl MBarrier {
    pub unsafe fn init(&self, num_threads: u32 /* might be wrong type */) {
        todo!()
    }

    // TODO expose remaining intrinsics: arrive, arrive_expect_tx, try_wait, test_wait, etc. 
}
