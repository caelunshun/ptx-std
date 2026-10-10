// --- LLM-generated --- //

//! Compile-time tests of the descriptor encodings. The expected values were computed with the
//! bitfield unions of CUTLASS (`cute/arch/mma_sm100_desc.hpp`: `InstrDescriptor`,
//! `InstrDescriptorBlockScaled`, `SmemDescriptor` and `MaskAndShiftB`) using its
//! `make_instr_desc*` helpers, except where noted (the PTX ISA is authoritative there).

use ptx_std::tcgen05::{
    ColumnSpan, F8f6f4ElementType as E, F16ElementType as H, FloatAccumulator as Acc,
    I8ElementType as I, InstrDescriptor, Major, MmaKind, Mxf4K, Nvf4ScaleType, SmemDescriptor,
    Sparsity, Swizzle, WsMaxShift, ZeroColumnMask,
};

type F16 = InstrDescriptor<{ MmaKind::F16 }>;
type Tf32 = InstrDescriptor<{ MmaKind::Tf32 }>;
type F8f6f4 = InstrDescriptor<{ MmaKind::F8f6f4 }>;
type I8 = InstrDescriptor<{ MmaKind::I8 }>;
type Mxf8f6f4 = InstrDescriptor<{ MmaKind::Mxf8f6f4 }>;
type Mxf4 = InstrDescriptor<{ MmaKind::Mxf4 }>;
type Mxf4nvf4 = InstrDescriptor<{ MmaKind::Mxf4nvf4 }>;

// `.kind::f16`
const _: () = assert!(F16::new(H::F16, H::Bf16, Acc::F32, 128, 256).to_bits() == 0x0840_0410);
const _: () = assert!(F16::new(H::F16, H::F16, Acc::F16, 64, 8).to_bits() == 0x0402_0000);
const F16_ALL: F16 = F16::new(H::Bf16, H::Bf16, Acc::F32, 256, 128)
    .with_major_a(Major::Mn)
    .with_major_b(Major::Mn)
    .with_negate_a(true)
    .with_negate_b(true)
    .with_sparsity(Sparsity::Sparse)
    .with_max_b_shift(WsMaxShift::Shift16);
const _: () = assert!(F16_ALL.to_bits() == 0x9021_e494);
const _: () = assert!(F16_ALL.with_sparsity_selector(3).to_bits() == 0x9021_e497);
// Setters replace the previous value.
const _: () = assert!(
    F16_ALL
        .with_sparsity_selector(3)
        .with_sparsity_selector(0)
        .with_major_a(Major::K)
        .with_major_b(Major::K)
        .with_negate_a(false)
        .with_negate_b(false)
        .with_sparsity(Sparsity::Dense)
        .with_max_b_shift(WsMaxShift::None)
        .to_bits()
        == F16::new(H::Bf16, H::Bf16, Acc::F32, 256, 128).to_bits()
);

// `.kind::tf32`
const _: () = assert!(Tf32::new(128, 64).to_bits() == 0x0810_0910);
const _: () = assert!(
    Tf32::new(64, 24)
        .with_major_a(Major::Mn)
        .with_negate_a(true)
        .with_sparsity(Sparsity::Sparse)
        .with_max_b_shift(WsMaxShift::Shift32)
        .to_bits()
        == 0xc406_a914
);
const _: () = assert!(
    Tf32::new(64, 24)
        .with_sparsity_selector(2)
        .with_major_b(Major::Mn)
        .with_negate_b(true)
        .with_max_b_shift(WsMaxShift::Shift8)
        .to_bits()
        == Tf32::new(64, 24).to_bits() | 2 | (1 << 16) | (1 << 14) | (1 << 30)
);

// `.kind::f8f6f4`
const _: () = assert!(F8f6f4::new(E::E4m3, E::E5m2, Acc::F32, 128, 128).to_bits() == 0x0820_0410);
const _: () = assert!(F8f6f4::new(E::E2m3, E::E3m2, Acc::F16, 64, 256).to_bits() == 0x0440_1180);
const _: () = assert!(
    F8f6f4::new(E::E2m1, E::E4m3, Acc::F32, 128, 32)
        .with_major_a(Major::Mn)
        .with_major_b(Major::Mn)
        .with_negate_a(true)
        .with_negate_b(true)
        .with_sparsity(Sparsity::Sparse)
        .with_max_b_shift(WsMaxShift::Shift8)
        .to_bits()
        == 0x4809_e294
);

// `.kind::i8`
const _: () = assert!(I8::new(I::S8, I::U8, 128, 256).to_bits() == 0x0840_00a0);
const _: () = assert!(
    I8::new(I::U8, I::S8, 64, 16)
        .with_major_a(Major::Mn)
        .with_major_b(Major::Mn)
        .with_saturate(true)
        .with_sparsity(Sparsity::Sparse)
        .with_max_b_shift(WsMaxShift::Shift16)
        .to_bits()
        == 0x8405_842c
);

// `.kind::mxf8f6f4`
const _: () = assert!(Mxf8f6f4::new(E::E4m3, E::E5m2, 128, 256).to_bits() == 0x08c0_0400);
const MXF8_ALL: Mxf8f6f4 = Mxf8f6f4::new(E::E2m3, E::E2m1, 256, 16)
    .with_major_a(Major::Mn)
    .with_negate_a(true)
    .with_sparsity(Sparsity::Sparse);
const _: () = assert!(MXF8_ALL.to_bits() == 0x1084_b584);
const _: () = assert!(MXF8_ALL.with_scale_a_id(3).with_scale_b_id(2).to_bits() == 0x7084_b5a4);
const _: () = assert!(
    Mxf8f6f4::new(E::E5m2, E::E3m2, 128, 8)
        .with_major_b(Major::Mn)
        .with_negate_b(true)
        .to_bits()
        == Mxf8f6f4::new(E::E5m2, E::E3m2, 128, 8).to_bits() | (1 << 16) | (1 << 14)
);

// `.kind::mxf4`
const _: () = assert!(Mxf4::new(128, 8).to_bits() == 0x0882_0480);
const MXF4_ALL: Mxf4 = Mxf4::new(256, 256)
    .with_negate_a(true)
    .with_negate_b(true)
    .with_sparsity(Sparsity::Sparse);
const _: () = assert!(MXF4_ALL.to_bits() == 0x10c0_6484);
const _: () = assert!(
    MXF4_ALL
        .with_scale_a_id(2)
        .with_scale_b_id(2)
        .with_k(Mxf4K::K96)
        .to_bits()
        == 0xd0c0_64a4
);
const _: () = assert!(MXF4_ALL.with_k(Mxf4K::K96).with_k(Mxf4K::K64).to_bits() == 0x10c0_6484);

// `.kind::mxf4nvf4`
const _: () = assert!(Mxf4nvf4::new(Nvf4ScaleType::Ue4m3, 128, 64).to_bits() == 0x0810_0480);
const _: () = assert!(Mxf4nvf4::new(Nvf4ScaleType::Ue8m0, 128, 64).to_bits() == 0x0890_0480);
// CUTLASS has no UE5M3; the ISA encodes it as 2 in the 2-bit scale type field (bits 23-24).
const _: () = assert!(Mxf4nvf4::new(Nvf4ScaleType::Ue5m3, 128, 64).to_bits() == 0x0910_0480);
const _: () = assert!(
    Mxf4nvf4::new(Nvf4ScaleType::Ue8m0, 256, 192)
        .with_scale_a_id(2)
        .with_scale_b_id(2)
        .with_k(Mxf4K::K96)
        .with_negate_a(true)
        .to_bits()
        == Mxf4nvf4::new(Nvf4ScaleType::Ue8m0, 256, 192).to_bits()
            | (2 << 29)
            | (2 << 4)
            | (1 << 31)
            | (1 << 13)
);

// Raw conversions round-trip.
const _: () = assert!(F16::from_bits(0x1234_5678).to_bits() == 0x1234_5678);

// Shared memory descriptors
const _: () = assert!(
    SmemDescriptor::from_shared_address(0, 0, 0, Swizzle::None).to_bits() == 0x0000_4000_0000_0000
);
const _: () = assert!(
    SmemDescriptor::from_shared_address(0x1230, 16, 1024, Swizzle::Bytes128).to_bits()
        == 0x4000_4040_0001_0123
);
const _: () = assert!(
    SmemDescriptor::from_shared_address(0x3FFF0, 0x3FFF0, 0x3FFF0, Swizzle::Bytes32).to_bits()
        == 0xc000_7fff_3fff_3fff
);
const _: () = assert!(
    SmemDescriptor::from_shared_address(0x400, 256, 512, Swizzle::Bytes64)
        .with_base_offset(5)
        .to_bits()
        == 0x800a_4020_0010_0040
);
const _: () = assert!(
    SmemDescriptor::from_shared_address(0x20, 0, 64, Swizzle::Bytes128Atom32B).to_bits()
        == 0x2000_4004_0000_0002
);
const _: () = assert!(
    SmemDescriptor::from_shared_address(0x1000, 0, 128, Swizzle::Bytes128)
        .add_start_bytes(0x20)
        .to_bits()
        == 0x4000_4008_0000_0102
);
// Only the low 18 bits of the address are encoded.
const _: () = assert!(
    SmemDescriptor::from_shared_address(0xFFFC_1230, 16, 1024, Swizzle::Bytes128).to_bits()
        == 0x4000_4040_0001_0123
);
#[cfg(target_feature = "sm_103a")]
const _: () = assert!(
    SmemDescriptor::from_shared_address(0x1000, 0, 128, Swizzle::Bytes128)
        .with_absolute_leading_address(0x2040)
        .to_bits()
        == 0x4010_4008_0204_0100
);

// Zero-column masks, with the examples of the PTX ISA
const _: () = assert!(ZeroColumnMask::NONE.to_bits() == 0);
const _: () = assert!(ZeroColumnMask::from_bits(0x1234).to_bits() == 0x1234);
// Example 2 (M = 128): mask0 = ...111 0000 111 0000
const _: () = assert!(ZeroColumnMask::new(3, 4).to_bits() == 0x0003_0280_0000_0000);
// Example 3 (M = 64): start counts {-, -, 0, 0}, first spans {-, -, 0, 1}
const _: () = assert!(
    ZeroColumnMask::new(3, 4)
        .with_sub_mask_start(0, ColumnSpan::Zeroed, 0)
        .with_sub_mask_start(1, ColumnSpan::Used, 0)
        .to_bits()
        == 0x0003_0281_0000_0000
);
// Example 4 (M = 32): start counts {1, 2, 1, 0}, first spans {0, 0, 1, 1}, shift 2
const _: () = assert!(
    ZeroColumnMask::new(3, 4)
        .with_sub_mask_start(0, ColumnSpan::Zeroed, 0)
        .with_sub_mask_start(1, ColumnSpan::Zeroed, 1)
        .with_sub_mask_start(2, ColumnSpan::Used, 2)
        .with_sub_mask_start(3, ColumnSpan::Used, 1)
        .with_column_shift(2)
        .to_bits()
        == 0x0203_0283_0102_0100
);
const _: () =
    assert!(ZeroColumnMask::NONE.with_column_shift(32).to_bits() == 0x2000_0000_0000_0000);
const _: () = assert!(
    ZeroColumnMask::new(256, 256)
        .with_sub_mask_start(3, ColumnSpan::Zeroed, 255)
        .with_column_shift(32)
        .to_bits()
        == 0x20ff_ff88_ff00_0000
);
// Setting a sub-mask start again replaces it.
const _: () = assert!(
    ZeroColumnMask::new(1, 1)
        .with_sub_mask_start(2, ColumnSpan::Zeroed, 7)
        .with_sub_mask_start(2, ColumnSpan::Used, 0)
        .to_bits()
        == ZeroColumnMask::new(1, 1).to_bits()
);

/// Builds descriptors from runtime pointers and K-loop advances, to assemble their code.
#[unsafe(no_mangle)]
pub extern "ptx-kernel" fn descriptor_smem_runtime(matrix: *const u8, out: *mut u64, k_steps: u32) {
    let desc = SmemDescriptor::new(matrix, 16, 1024, Swizzle::Bytes128);
    let mut desc_k = desc;
    for i in 0..k_steps {
        unsafe { out.add(i as usize).write(desc_k.to_bits()) };
        desc_k = desc_k.add_start_bytes(32);
    }
    let other =
        SmemDescriptor::new(matrix, 16, 1024, Swizzle::Bytes64).with_base_offset(k_steps & 7);
    unsafe { out.add(32).write(other.to_bits()) };
    #[cfg(target_feature = "sm_103a")]
    {
        let abs = SmemDescriptor::new(matrix, 0, 128, Swizzle::Bytes128)
            .with_absolute_leading_address(to_shared(matrix.wrapping_add(64)));
        unsafe { out.add(33).write(abs.to_bits()) };
    }
}

#[cfg(target_feature = "sm_103a")]
fn to_shared(ptr: *const u8) -> u32 {
    ptx_std::memory::to_shared_address(ptr)
}
