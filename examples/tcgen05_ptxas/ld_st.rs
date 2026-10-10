// --- LLM-generated --- //

use core::hint::black_box;

use ptx_std::{
    bf16x2, f16x2,
    tcgen05::{self, LdStShape, Packing, TmemAddr},
};

macro_rules! load_store {
    ($addr:ident, $shape:expr, [$($n:literal)*]) => {
        $(
            let v = unsafe {
                tcgen05::load::<{ $shape }, { Packing::None }, u32, $n>($addr)
            };
            unsafe { tcgen05::store::<{ $shape }, { Packing::None }, u32, $n>($addr, black_box(v)) };
            let v = unsafe {
                tcgen05::load::<{ $shape }, { Packing::Pack16b }, f16x2, $n>($addr)
            };
            unsafe {
                tcgen05::store::<{ $shape }, { Packing::Pack16b }, f16x2, $n>($addr, black_box(v))
            };
        )*
    };
}

macro_rules! load_store_16x32bx2 {
    ($addr:ident, [$($n:literal)*]) => {
        $(
            let v = unsafe {
                tcgen05::load_16x32bx2::<{ $n }, { Packing::None }, f32, $n>($addr)
            };
            unsafe {
                tcgen05::store_16x32bx2::<{ $n }, { Packing::None }, f32, $n>($addr, black_box(v))
            };
            let v = unsafe {
                tcgen05::load_16x32bx2::<16, { Packing::Pack16b }, bf16x2, $n>($addr)
            };
            unsafe {
                tcgen05::store_16x32bx2::<16, { Packing::Pack16b }, bf16x2, $n>($addr, black_box(v))
            };
        )*
    };
}

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn ld_st_32x32b(addr: u32) {
    let addr = TmemAddr::from_bits(addr);
    load_store!(addr, LdStShape::S32x32b, [1 2 4 8 16 32 64 128]);
}

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn ld_st_16x64b(addr: u32) {
    let addr = TmemAddr::from_bits(addr);
    load_store!(addr, LdStShape::S16x64b, [1 2 4 8 16 32 64 128]);
}

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn ld_st_16x128b(addr: u32) {
    let addr = TmemAddr::from_bits(addr);
    load_store!(addr, LdStShape::S16x128b, [2 4 8 16 32 64 128]);
}

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn ld_st_16x256b(addr: u32) {
    let addr = TmemAddr::from_bits(addr);
    load_store!(addr, LdStShape::S16x256b, [4 8 16 32 64 128]);
}

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn ld_st_16x32bx2(addr: u32) {
    let addr = TmemAddr::from_bits(addr);
    load_store_16x32bx2!(addr, [1 2 4 8 16 32 64 128]);
}

#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn ld_st_waits() {
    unsafe {
        tcgen05::wait_load();
        tcgen05::wait_store();
    }
}

#[cfg(any(
    target_feature = "sm_101f",
    target_feature = "sm_103f",
    target_feature = "sm_110f"
))]
mod load_reduce {
    use core::hint::black_box;

    use ptx_std::{
        NanHandling,
        tcgen05::{self, TmemAddr},
        warp_collectives::{Absolute, ReduxOp},
    };

    macro_rules! reduce_ops {
        ($addr:ident, $op:expr, $n:literal) => {
            black_box(unsafe {
                tcgen05::load_reduce::<
                    { $op },
                    { Absolute::Disabled },
                    { NanHandling::Ignore },
                    f32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce::<
                    { $op },
                    { Absolute::Enabled },
                    { NanHandling::Ignore },
                    f32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce::<
                    { $op },
                    { Absolute::Disabled },
                    { NanHandling::Propagate },
                    f32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce::<
                    { $op },
                    { Absolute::Enabled },
                    { NanHandling::Propagate },
                    f32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce::<
                    { $op },
                    { Absolute::Disabled },
                    { NanHandling::Ignore },
                    u32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce::<
                    { $op },
                    { Absolute::Disabled },
                    { NanHandling::Ignore },
                    i32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce_16x32bx2::<
                    { $n },
                    { $op },
                    { Absolute::Disabled },
                    { NanHandling::Ignore },
                    f32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce_16x32bx2::<
                    { $n },
                    { $op },
                    { Absolute::Enabled },
                    { NanHandling::Ignore },
                    f32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce_16x32bx2::<
                    { $n },
                    { $op },
                    { Absolute::Disabled },
                    { NanHandling::Propagate },
                    f32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce_16x32bx2::<
                    { $n },
                    { $op },
                    { Absolute::Enabled },
                    { NanHandling::Propagate },
                    f32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce_16x32bx2::<
                    { $n },
                    { $op },
                    { Absolute::Disabled },
                    { NanHandling::Ignore },
                    u32,
                    $n,
                >($addr)
            });
            black_box(unsafe {
                tcgen05::load_reduce_16x32bx2::<
                    { $n },
                    { $op },
                    { Absolute::Disabled },
                    { NanHandling::Ignore },
                    i32,
                    $n,
                >($addr)
            });
        };
    }

    macro_rules! reduce_all {
        ($addr:ident, [$($n:literal)*]) => {
            $(
                reduce_ops!($addr, ReduxOp::Min, $n);
                reduce_ops!($addr, ReduxOp::Max, $n);
            )*
        };
    }

    #[unsafe(no_mangle)]
    pub extern "ptx-kernel" fn ld_red(addr: u32) {
        let addr = TmemAddr::from_bits(addr);
        reduce_all!(addr, [2 4 8 16 32 64 128]);
    }
}
