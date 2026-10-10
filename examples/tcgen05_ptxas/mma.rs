// --- LLM-generated --- //

use ptx_std::{
    CtaGroup,
    tcgen05::{
        self, AShift, CollectorUsage, InstrDescriptor, MmaKind, MmaModifiers, MmaWsModifiers,
        OutputLaneMask, ScaleVectorSize, SmemDescriptor, TmemAddr, ZeroColumnMask,
    },
};

struct Ops {
    d: TmemAddr,
    adesc: SmemDescriptor,
    atmem: TmemAddr,
    b: SmemDescriptor,
    meta: TmemAddr,
    idesc: u32,
    scale_a: TmemAddr,
    scale_b: TmemAddr,
    lanes: OutputLaneMask,
    zcm: ZeroColumnMask,
    enable: bool,
}

macro_rules! kernel {
    ($name:ident, $o:ident => $body:block) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(
            d: u32,
            adesc: u64,
            atmem: u32,
            b: u64,
            meta: u32,
            idesc: u32,
            scale_a: u32,
            scale_b: u32,
            lanes: *const OutputLaneMask,
            zcm: u64,
            enable: u32,
        ) {
            let $o = Ops {
                d: TmemAddr::from_bits(d),
                adesc: SmemDescriptor::from_bits(adesc),
                atmem: TmemAddr::from_bits(atmem),
                b: SmemDescriptor::from_bits(b),
                meta: TmemAddr::from_bits(meta),
                idesc,
                scale_a: TmemAddr::from_bits(scale_a),
                scale_b: TmemAddr::from_bits(scale_b),
                lanes: unsafe { *lanes },
                zcm: ZeroColumnMask::from_bits(zcm),
                enable: enable != 0,
            };
            $body
        }
    };
}

macro_rules! each_collector {
    ($mac:ident, $($args:tt)*) => {
        $mac!($($args)*, MmaModifiers::DEFAULT.with_collector_a(CollectorUsage::Fill));
        $mac!($($args)*, MmaModifiers::DEFAULT.with_collector_a(CollectorUsage::Use));
        $mac!($($args)*, MmaModifiers::DEFAULT.with_collector_a(CollectorUsage::LastUse));
        $mac!($($args)*, MmaModifiers::DEFAULT.with_collector_a(CollectorUsage::Discard));
    };
}

// A in shared memory and Tensor Memory, with and without disable-output-lane, dense and sparse.
macro_rules! dense {
    ($cg:expr, $kind:expr, $o:ident, $mods:expr) => {{
        let idesc = InstrDescriptor::<{ $kind }>::from_bits($o.idesc);
        unsafe {
            tcgen05::mma::<{ $cg }, { $mods }, _>($o.d, $o.adesc, $o.b, idesc, None, $o.enable);
            tcgen05::mma::<{ $cg }, { $mods }, _>(
                $o.d,
                $o.atmem,
                $o.b,
                idesc,
                Some($o.lanes),
                $o.enable,
            );
            tcgen05::mma_sparse::<{ $cg }, { $mods }, _>(
                $o.d,
                $o.adesc,
                $o.b,
                $o.meta,
                idesc,
                Some($o.lanes),
                $o.enable,
            );
            tcgen05::mma_sparse::<{ $cg }, { $mods }, _>(
                $o.d, $o.atmem, $o.b, $o.meta, idesc, None, $o.enable,
            );
        }
    }};
}

// Only A in Tensor Memory is valid with `.ashift`.
macro_rules! dense_tmem {
    ($cg:expr, $kind:expr, $o:ident, $mods:expr) => {{
        let idesc = InstrDescriptor::<{ $kind }>::from_bits($o.idesc);
        unsafe {
            tcgen05::mma::<{ $cg }, { $mods }, _>($o.d, $o.atmem, $o.b, idesc, None, $o.enable);
            tcgen05::mma::<{ $cg }, { $mods }, _>(
                $o.d,
                $o.atmem,
                $o.b,
                idesc,
                Some($o.lanes),
                $o.enable,
            );
            tcgen05::mma_sparse::<{ $cg }, { $mods }, _>(
                $o.d, $o.atmem, $o.b, $o.meta, idesc, None, $o.enable,
            );
        }
    }};
}

macro_rules! ashift_variants {
    ($cg:expr, $kind:expr, $o:ident) => {
        dense_tmem!(
            $cg,
            $kind,
            $o,
            MmaModifiers::DEFAULT.with_a_shift(AShift::Enabled)
        );
        dense_tmem!(
            $cg,
            $kind,
            $o,
            MmaModifiers::DEFAULT
                .with_a_shift(AShift::Enabled)
                .with_collector_a(CollectorUsage::LastUse)
        );
        dense_tmem!(
            $cg,
            $kind,
            $o,
            MmaModifiers::DEFAULT
                .with_a_shift(AShift::Enabled)
                .with_collector_a(CollectorUsage::Discard)
        );
    };
}

macro_rules! dense_kernels {
    ($cg:expr, $f16:ident, $tf32:ident, $f8f6f4:ident, $i8:ident) => {
        kernel!($f16, o => {
            each_collector!(dense, $cg, MmaKind::F16, o);
            ashift_variants!($cg, MmaKind::F16, o);
            #[cfg(target_feature = "sm_100f")]
            {
                dense!($cg, MmaKind::F16, o, MmaModifiers::DEFAULT.with_scale_input_d(1));
                dense!($cg, MmaKind::F16, o, MmaModifiers::DEFAULT.with_scale_input_d(15));
                dense_tmem!(
                    $cg,
                    MmaKind::F16,
                    o,
                    MmaModifiers::DEFAULT
                        .with_scale_input_d(3)
                        .with_a_shift(AShift::Enabled)
                        .with_collector_a(CollectorUsage::LastUse)
                );
            }
        });
        kernel!($tf32, o => {
            each_collector!(dense, $cg, MmaKind::Tf32, o);
            ashift_variants!($cg, MmaKind::Tf32, o);
            #[cfg(target_feature = "sm_100f")]
            dense!($cg, MmaKind::Tf32, o, MmaModifiers::DEFAULT.with_scale_input_d(7));
        });
        kernel!($f8f6f4, o => {
            each_collector!(dense, $cg, MmaKind::F8f6f4, o);
            ashift_variants!($cg, MmaKind::F8f6f4, o);
        });
        #[cfg(any(
            target_feature = "sm_100a",
            target_feature = "sm_101a",
            target_feature = "sm_110a"
        ))]
        kernel!($i8, o => {
            each_collector!(dense, $cg, MmaKind::I8, o);
            ashift_variants!($cg, MmaKind::I8, o);
        });
    };
}

dense_kernels!(
    CtaGroup::One,
    mma_cg1_f16,
    mma_cg1_tf32,
    mma_cg1_f8f6f4,
    mma_cg1_i8
);
dense_kernels!(
    CtaGroup::Two,
    mma_cg2_f16,
    mma_cg2_tf32,
    mma_cg2_f8f6f4,
    mma_cg2_i8
);

macro_rules! block_scaled {
    ($cg:expr, $kind:expr, $sv:expr, $o:ident, $mods:expr) => {{
        let idesc = InstrDescriptor::<{ $kind }>::from_bits($o.idesc);
        unsafe {
            tcgen05::mma_block_scaled::<{ $cg }, { $mods }, { $sv }, _>(
                $o.d, $o.adesc, $o.b, idesc, $o.scale_a, $o.scale_b, $o.enable,
            );
            tcgen05::mma_block_scaled::<{ $cg }, { $mods }, { $sv }, _>(
                $o.d, $o.atmem, $o.b, idesc, $o.scale_a, $o.scale_b, $o.enable,
            );
        }
    }};
}

macro_rules! block_scaled_sparse {
    ($cg:expr, $kind:expr, $sv:expr, $o:ident, $mods:expr) => {{
        let idesc = InstrDescriptor::<{ $kind }>::from_bits($o.idesc);
        unsafe {
            tcgen05::mma_block_scaled_sparse::<{ $cg }, { $mods }, { $sv }, _>(
                $o.d, $o.adesc, $o.b, $o.meta, idesc, $o.scale_a, $o.scale_b, $o.enable,
            );
            tcgen05::mma_block_scaled_sparse::<{ $cg }, { $mods }, { $sv }, _>(
                $o.d, $o.atmem, $o.b, $o.meta, idesc, $o.scale_a, $o.scale_b, $o.enable,
            );
        }
    }};
}

// Dense and sparse, all collector usages.
macro_rules! block_scaled_all {
    ($cg:expr, $kind:expr, $sv:expr, $o:ident) => {
        each_collector!(block_scaled, $cg, $kind, $sv, $o);
        each_collector!(block_scaled_sparse, $cg, $kind, $sv, $o);
    };
}

// Dense only (sparse needs an architecture-specific target for `mxf4` and `mxf4nvf4`).
macro_rules! block_scaled_dense {
    ($cg:expr, $kind:expr, $sv:expr, $o:ident) => {
        each_collector!(block_scaled, $cg, $kind, $sv, $o);
    };
}

macro_rules! block_scaled_kernels {
    ($cg:expr, $mxf8f6f4:ident, $mxf4:ident, $mxf4nvf4:ident) => {
        kernel!($mxf8f6f4, o => {
            block_scaled_all!($cg, MmaKind::Mxf8f6f4, ScaleVectorSize::Default, o);
            block_scaled_all!($cg, MmaKind::Mxf8f6f4, ScaleVectorSize::Block32, o);
            #[cfg(any(
                target_feature = "sm_100a",
                target_feature = "sm_101a",
                target_feature = "sm_103a",
                target_feature = "sm_110a"
            ))]
            block_scaled_all!($cg, MmaKind::Mxf8f6f4, ScaleVectorSize::Vec1X, o);
        });
        kernel!($mxf4, o => {
            block_scaled_dense!($cg, MmaKind::Mxf4, ScaleVectorSize::Default, o);
            block_scaled_dense!($cg, MmaKind::Mxf4, ScaleVectorSize::Block32, o);
            #[cfg(any(
                target_feature = "sm_100a",
                target_feature = "sm_101a",
                target_feature = "sm_103a",
                target_feature = "sm_110a"
            ))]
            {
                block_scaled_all!($cg, MmaKind::Mxf4, ScaleVectorSize::Default, o);
                block_scaled_all!($cg, MmaKind::Mxf4, ScaleVectorSize::Block32, o);
                block_scaled_all!($cg, MmaKind::Mxf4, ScaleVectorSize::Vec2X, o);
            }
        });
        kernel!($mxf4nvf4, o => {
            block_scaled_dense!($cg, MmaKind::Mxf4nvf4, ScaleVectorSize::Block16, o);
            block_scaled_dense!($cg, MmaKind::Mxf4nvf4, ScaleVectorSize::Block32, o);
            #[cfg(any(
                target_feature = "sm_100a",
                target_feature = "sm_101a",
                target_feature = "sm_103a",
                target_feature = "sm_110a"
            ))]
            {
                block_scaled_all!($cg, MmaKind::Mxf4nvf4, ScaleVectorSize::Block16, o);
                block_scaled_all!($cg, MmaKind::Mxf4nvf4, ScaleVectorSize::Block32, o);
                block_scaled_all!($cg, MmaKind::Mxf4nvf4, ScaleVectorSize::Vec2X, o);
                block_scaled_all!($cg, MmaKind::Mxf4nvf4, ScaleVectorSize::Vec4X, o);
            }
        });
    };
}

block_scaled_kernels!(
    CtaGroup::One,
    mma_block_scaled_cg1_mxf8f6f4,
    mma_block_scaled_cg1_mxf4,
    mma_block_scaled_cg1_mxf4nvf4
);
block_scaled_kernels!(
    CtaGroup::Two,
    mma_block_scaled_cg2_mxf8f6f4,
    mma_block_scaled_cg2_mxf4,
    mma_block_scaled_cg2_mxf4nvf4
);

macro_rules! ws {
    ($kind:expr, $o:ident, $mods:expr) => {{
        let idesc = InstrDescriptor::<{ $kind }>::from_bits($o.idesc);
        unsafe {
            tcgen05::mma_ws::<{ $mods }, _>($o.d, $o.adesc, $o.b, idesc, None, $o.enable);
            tcgen05::mma_ws::<{ $mods }, _>($o.d, $o.atmem, $o.b, idesc, Some($o.zcm), $o.enable);
            tcgen05::mma_ws_sparse::<{ $mods }, _>(
                $o.d,
                $o.adesc,
                $o.b,
                $o.meta,
                idesc,
                Some($o.zcm),
                $o.enable,
            );
            tcgen05::mma_ws_sparse::<{ $mods }, _>(
                $o.d, $o.atmem, $o.b, $o.meta, idesc, None, $o.enable,
            );
        }
    }};
}

macro_rules! ws_buffer {
    ($kind:expr, $o:ident, $buffer:expr) => {
        ws!(
            $kind,
            $o,
            MmaWsModifiers::DEFAULT
                .with_collector_b_buffer($buffer)
                .with_collector_b(CollectorUsage::Fill)
        );
        ws!(
            $kind,
            $o,
            MmaWsModifiers::DEFAULT
                .with_collector_b_buffer($buffer)
                .with_collector_b(CollectorUsage::Use)
        );
        ws!(
            $kind,
            $o,
            MmaWsModifiers::DEFAULT
                .with_collector_b_buffer($buffer)
                .with_collector_b(CollectorUsage::LastUse)
        );
        ws!(
            $kind,
            $o,
            MmaWsModifiers::DEFAULT
                .with_collector_b_buffer($buffer)
                .with_collector_b(CollectorUsage::Discard)
        );
    };
}

macro_rules! ws_all {
    ($kind:expr, $o:ident) => {
        ws!($kind, $o, MmaWsModifiers::DEFAULT);
        ws_buffer!($kind, $o, 0);
        ws_buffer!($kind, $o, 1);
        ws_buffer!($kind, $o, 2);
        ws_buffer!($kind, $o, 3);
    };
}

// Weight-stationary MMAs are always `.cta_group::1`.
kernel!(mma_ws_f16, o => { ws_all!(MmaKind::F16, o); });
kernel!(mma_ws_tf32, o => { ws_all!(MmaKind::Tf32, o); });
kernel!(mma_ws_f8f6f4, o => { ws_all!(MmaKind::F8f6f4, o); });
#[cfg(any(
    target_feature = "sm_100a",
    target_feature = "sm_101a",
    target_feature = "sm_110a"
))]
kernel!(mma_ws_i8, o => { ws_all!(MmaKind::I8, o); });
