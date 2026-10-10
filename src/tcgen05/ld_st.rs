// --- LLM-generated --- //
//! Loads from and stores to Tensor Memory (`tcgen05.ld`, `tcgen05.st`, `tcgen05.wait`).

use core::{arch::asm, marker::ConstParamTy};

use bytemuck::Pod;

use super::TmemAddr;
#[cfg(any(
    target_feature = "sm_101f",
    target_feature = "sm_103f",
    target_feature = "sm_110f"
))]
use crate::{NanHandling, warp_collectives::Absolute, warp_collectives::ReduxOp};

/// The base shape of the data moved by [`load`] and [`store`] (lanes × bits per lane): the
/// dimensions of the data accessed by a warp with a repeat factor of 1 (`.x1`). The data accessed
/// by a call is this shape repeated along the columns `N / registers per .x1` times (`.xNUM`).
///
/// The 32-lane shape accesses the 32 lanes of the warp's lane quarter. The 16-lane shapes access
/// the 16 lanes starting at the lane of the address, which must lie in the lane quarter of the
/// warp (for the MMA layouts that use half of the datapath lanes, 0 or 16 lanes into the quarter).
///
/// With a repeat factor of 1 (`.x1`), thread `t` of the warp holds the following, where `r_k` is
/// the `k`-th register of the thread and lanes are counted from the first lane accessed (see the
/// figures of the matrix fragments of each shape in the PTX ISA):
///
/// | Shape | Elements of thread `t` |
/// |---|---|
/// | `S32x32b` | `r0`: the 32-bit cell of lane `t`. |
/// | `S16x64b` | `r0`: the low (`t % 4 < 2`) or high (`t % 4 >= 2`) 32 bits of the 64-bit row of lane `t / 4 + 8 * (t % 2)`. |
/// | `S16x128b` | `r0`: word `t % 4` of the 128-bit row of lane `t / 4`. `r1`: the same word of lane `t / 4 + 8`. |
/// | `S16x256b` | `r0`, `r1`: words `2 * (t % 4)` and `2 * (t % 4) + 1` of the 256-bit row of lane `t / 4`. `r2`, `r3`: the same words of lane `t / 4 + 8`. |
/// | `load_16x32bx2` / `store_16x32bx2` | `r0`: the 32-bit cell of lane `t % 16` of the first region (`t < 16`) or of the second region (`t >= 16`). |
///
/// Larger repeat factors repeat this pattern along the columns, with the registers continuing in
/// order (for the 16x64b shape with `.x2`, `r1` is the next 64-bit column).
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-movement-shape>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum LdStShape {
    /// 32 lanes × 32 bits (`.32x32b`). 1 register per thread per repeat; `N` up to 128.
    S32x32b,
    /// 16 lanes × 64 bits (`.16x64b`). 1 register per thread per repeat; `N` up to 128.
    S16x64b,
    /// 16 lanes × 128 bits (`.16x128b`). 2 registers per thread per repeat; `N` from 2 up to 128.
    S16x128b,
    /// 16 lanes × 256 bits (`.16x256b`). 4 registers per thread per repeat; `N` from 4 up to 128.
    S16x256b,
}

/// Optional packing of 16-bit elements, performed by [`load`] (`.pack::16b`) or by [`store`]
/// (`.unpack::16b`).
///
/// This does not change the number of registers moved per repeat, but each register then
/// corresponds to two adjacent 32-bit columns of Tensor Memory (holding a 16-bit element each), so
/// the accessed region spans twice as many columns.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-tensor-memory-ld-st-packing-unpacking>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum Packing {
    /// The 32-bit elements are moved as they are.
    None,
    /// [`load`] packs two 16-bit elements from adjacent columns into a single 32-bit register
    /// element (`.pack::16b`). [`store`] unpacks a single 32-bit register element into two 16-bit
    /// elements stored in adjacent columns (`.unpack::16b`).
    Pack16b,
}

mod sealed {
    pub enum ReduceType {
        F32,
        U32,
        S32,
    }

    pub trait Sealed {
        const TYPE: ReduceType;
    }
}

/// Element types that can be reduced by [`load_reduce`]: `f32`, `u32` and `i32`.
///
/// `Min` and `Max` compare according to the type; [`Absolute`] and [`NanHandling`] only apply to
/// `f32`.
pub trait TmemReduceElement: Pod + sealed::Sealed {}

impl sealed::Sealed for f32 {
    const TYPE: sealed::ReduceType = sealed::ReduceType::F32;
}
impl sealed::Sealed for u32 {
    const TYPE: sealed::ReduceType = sealed::ReduceType::U32;
}
impl sealed::Sealed for i32 {
    const TYPE: sealed::ReduceType = sealed::ReduceType::S32;
}
impl TmemReduceElement for f32 {}
impl TmemReduceElement for u32 {}
impl TmemReduceElement for i32 {}

/// The number of 32-bit registers per thread of the base shape (`.x1`), from Table 59.
const fn regs_per_x1(shape: LdStShape) -> usize {
    match shape {
        LdStShape::S32x32b | LdStShape::S16x64b => 1,
        LdStShape::S16x128b => 2,
        LdStShape::S16x256b => 4,
    }
}

/// Returns the repeat factor `.xNUM` for `regs` registers per thread. Fails const evaluation if it
/// is not one of `1, 2, 4, ..., 128`, or if the shape does not support it.
const fn repeat_count(regs: usize, regs_per_x1: usize) -> u32 {
    assert!(
        regs >= regs_per_x1 && regs <= 128 && regs.is_multiple_of(regs_per_x1),
        "N must be a multiple of the registers per repeat of the shape (1, 2 or 4) and at most 128"
    );
    let num = regs / regs_per_x1;
    assert!(
        num.is_power_of_two(),
        "N divided by the registers per repeat of the shape must be a power of two"
    );
    num as u32
}

/// Builds the brace-enclosed register list `{%0, %1, ...}` of an `asm!` template.
macro_rules! asm_reg_list {
    ($first:literal $($rest:literal)*) => {
        concat!("{{{", $first, "}", $(", {", $rest, "}",)* "}}")
    };
}

/// Emits an asm block with `$k` output registers as the first positional operands, and evaluates
/// to them transmuted to `$out`.
macro_rules! asm_load_regs {
    (
        [$($head:literal),+] [$($tail:literal),+] [$($named:tt)*] $out:ty;
        $k:literal; $($i:literal)+
    ) => {{
        let mut regs = [0u32; $k];
        unsafe {
            asm!(
                concat!($($head,)+ asm_reg_list!($($i)+), $($tail),+),
                $(out(reg32) regs[$i],)+
                $($named)*
                options(nostack),
            );
            core::mem::transmute_copy::<[u32; $k], $out>(&regs)
        }
    }};
}

/// Emits an asm block with the `$k` registers of `$values` as the first positional input operands.
macro_rules! asm_store_regs {
    (
        [$($head:literal),+] [$($tail:literal),+] [$($named:tt)*] $values:ident;
        $k:literal; $($i:literal)+
    ) => {{
        unsafe {
            let regs = core::mem::transmute_copy::<_, [u32; $k]>(&$values);
            asm!(
                concat!($($head,)+ asm_reg_list!($($i)+), $($tail),+),
                $(in(reg32) regs[$i],)+
                $($named)*
                options(nostack),
            );
        }
    }};
}

/// Expands `$mac!(args; K; 0 1 .. K-1)` for the register count `$n` (a const expression), which
/// must be one of `1, 2, 4, ..., 128`. Only the branch of `$n` is codegen'd.
macro_rules! by_num_regs {
    ($n:expr, $mac:ident!($($args:tt)*)) => {
        if const { $n == 1 } {
            $mac!($($args)*; 1;
                0
            )
        } else if const { $n == 2 } {
            $mac!($($args)*; 2;
                0 1
            )
        } else if const { $n == 4 } {
            $mac!($($args)*; 4;
                0 1 2 3
            )
        } else if const { $n == 8 } {
            $mac!($($args)*; 8;
                0 1 2 3 4 5 6 7
            )
        } else if const { $n == 16 } {
            $mac!($($args)*; 16;
                0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
            )
        } else if const { $n == 32 } {
            $mac!($($args)*; 32;
                0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
                27 28 29 30 31
            )
        } else if const { $n == 64 } {
            $mac!($($args)*; 64;
                0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
                27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
                51 52 53 54 55 56 57 58 59 60 61 62 63
            )
        } else if const { $n == 128 } {
            $mac!($($args)*; 128;
                0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
                27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
                51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
                75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98
                99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116
                117 118 119 120 121 122 123 124 125 126 127
            )
        } else {
            unreachable!()
        }
    };
}

/// Loads data from Tensor Memory at `addr` into the registers of the threads of the warp
/// (`tcgen05.ld`).
///
/// `N` is the number of 32-bit registers per thread, and `T` the 4-byte type they are
/// reinterpreted as (any [`Pod`] type of size 4, e.g. `u32`, `f32`, or a pair of 16-bit
/// floating-point values). `SHAPE` and `N` together determine the accessed data: the base shape
/// repeated `N / (registers per .x1 of SHAPE)` times along the columns starting at the column of
/// `addr`. `N` must be a power of two between the registers per repeat of the shape (1, 2 or 4,
/// see [`LdStShape`]) and 128, which is checked at compile time. With [`Packing::Pack16b`], pairs
/// of 16-bit elements from adjacent columns are packed into each 32-bit register, so the accessed
/// region spans twice as many columns.
///
/// The load is collective across the warp, and accesses the lanes of the warp's quarter of Tensor
/// Memory (the warp with index `w` in the CTA accesses the lanes `32 * (w % 4)..32 * (w % 4) +
/// 32`). It is asynchronous: later instructions of the thread may use the returned registers
/// without further synchronization, as register dependencies are respected, but [`wait_load`] is
/// needed before another operation overwrites the loaded Tensor Memory, or before the thread
/// signals to other threads that the Tensor Memory may be reused.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld>
///
/// # Safety
///
/// - All threads of the warp must call this function convergently with the same `addr`.
/// - The accessed region must lie within allocated Tensor Memory, and within the lane quarter of
///   the executing warp.
/// - The data must have been written by operations whose completion has been observed (see the
///   [memory consistency](super#memory-consistency) section of the module documentation).
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn load<const SHAPE: LdStShape, const PACK: Packing, T: Pod, const N: usize>(
    addr: TmemAddr,
) -> [T; N] {
    const {
        assert!(size_of::<T>() == 4, "T must be a 4-byte type");
        repeat_count(N, regs_per_x1(SHAPE));
    };
    macro_rules! emit {
        ($shape:literal, $pack:literal) => {
            by_num_regs!(
                N,
                asm_load_regs!(
                    ["tcgen05.ld.sync.aligned", $shape, ".x{num}", $pack, ".b32 "]
                    [", [{addr}];"]
                    [
                        num = const repeat_count(N, regs_per_x1(SHAPE)),
                        addr = in(reg32) addr.to_bits(),
                    ]
                    [T; N]
                )
            )
        };
    }
    asm_variants!(emit;
        SHAPE => {
            LdStShape::S32x32b => ".32x32b",
            LdStShape::S16x64b => ".16x64b",
            LdStShape::S16x128b => ".16x128b",
            LdStShape::S16x256b => ".16x256b",
        },
        PACK => {
            Packing::None => "",
            Packing::Pack16b => ".pack::16b",
        },
    )
}

/// Loads data from two 16-lane by 32-bit regions of Tensor Memory into the registers of the
/// threads of the warp (`tcgen05.ld.16x32bx2`).
///
/// The first region is located at `addr`, and the second at the address `addr +
/// HALF_SPLIT_OFFSET` (an immediate operand of the instruction). Each region is repeated `N` times
/// along the columns (`.xN`), and each thread receives `N` registers: threads 0 to 15 receive the
/// first region and threads 16 to 31 the second (see [`LdStShape`]). Typically, the offset is
/// chosen so that the second region continues the first, i.e. `HALF_SPLIT_OFFSET = N` (`2 * N` with
/// [`Packing::Pack16b`], under which each region spans twice as many columns). `N` must be a power
/// of two between 1 and 128. The other parameters are as for [`load`], including the asynchrony
/// and the lane quarter restriction.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld>
///
/// # Safety
///
/// - All threads of the warp must call this function convergently with the same `addr`.
/// - Both regions must lie within allocated Tensor Memory, and within the lane quarter of the
///   executing warp.
/// - The data must have been written by operations whose completion has been observed.
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn load_16x32bx2<
    const HALF_SPLIT_OFFSET: u32,
    const PACK: Packing,
    T: Pod,
    const N: usize,
>(
    addr: TmemAddr,
) -> [T; N] {
    const {
        assert!(size_of::<T>() == 4, "T must be a 4-byte type");
        repeat_count(N, 1);
    };
    macro_rules! emit {
        ($pack:literal) => {
            by_num_regs!(
                N,
                asm_load_regs!(
                    ["tcgen05.ld.sync.aligned.16x32bx2.x{num}", $pack, ".b32 "]
                    [", [{addr}], {imm};"]
                    [
                        num = const N as u32,
                        addr = in(reg32) addr.to_bits(),
                        imm = const HALF_SPLIT_OFFSET,
                    ]
                    [T; N]
                )
            )
        };
    }
    asm_variants!(emit;
        PACK => {
            Packing::None => "",
            Packing::Pack16b => ".pack::16b",
        },
    )
}

/// Stores the registers of the threads of the warp to Tensor Memory at `addr` (`tcgen05.st`).
///
/// This is the inverse of [`load`], with the same meaning of `SHAPE`, `T` and `N`, the same
/// collective access to the lane quarter of the warp, and the same asynchrony. With
/// [`Packing::Pack16b`], each 32-bit register element is unpacked into two 16-bit elements stored
/// in adjacent columns (`.unpack::16b`), so the accessed region spans twice as many columns.
/// [`wait_store`] blocks until the store has completed, which is required before an operation that
/// is not pipelined with the store (e.g. an MMA, or an access by another warp) reads or overwrites
/// the stored Tensor Memory; later loads and stores of the same warp are ordered after it.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st>
///
/// # Safety
///
/// - All threads of the warp must call this function convergently with the same `addr`.
/// - The accessed region must lie within allocated Tensor Memory, and within the lane quarter of
///   the executing warp.
/// - The region must not be accessed by operations that are not ordered after the store (see the
///   [memory consistency](super#memory-consistency) section of the module documentation) until
///   the store has completed.
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn store<const SHAPE: LdStShape, const PACK: Packing, T: Pod, const N: usize>(
    addr: TmemAddr,
    values: [T; N],
) {
    const {
        assert!(size_of::<T>() == 4, "T must be a 4-byte type");
        repeat_count(N, regs_per_x1(SHAPE));
    };
    macro_rules! emit {
        ($shape:literal, $pack:literal) => {
            by_num_regs!(
                N,
                asm_store_regs!(
                    ["tcgen05.st.sync.aligned", $shape, ".x{num}", $pack, ".b32 [{addr}], "]
                    [";"]
                    [
                        num = const repeat_count(N, regs_per_x1(SHAPE)),
                        addr = in(reg32) addr.to_bits(),
                    ]
                    values
                )
            )
        };
    }
    asm_variants!(emit;
        SHAPE => {
            LdStShape::S32x32b => ".32x32b",
            LdStShape::S16x64b => ".16x64b",
            LdStShape::S16x128b => ".16x128b",
            LdStShape::S16x256b => ".16x256b",
        },
        PACK => {
            Packing::None => "",
            Packing::Pack16b => ".unpack::16b",
        },
    )
}

/// Stores the registers of the threads of the warp to two 16-lane by 32-bit regions of Tensor
/// Memory (`tcgen05.st.16x32bx2`).
///
/// This is the inverse of [`load_16x32bx2`]: the first region is located at `addr`, and the second
/// at the address `addr + HALF_SPLIT_OFFSET`. The other parameters are as for [`store`].
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st>
///
/// # Safety
///
/// - All threads of the warp must call this function convergently with the same `addr`.
/// - Both regions must lie within allocated Tensor Memory, and within the lane quarter of the
///   executing warp.
/// - The regions must not be accessed by operations that are not ordered after the store until the
///   store has completed.
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn store_16x32bx2<
    const HALF_SPLIT_OFFSET: u32,
    const PACK: Packing,
    T: Pod,
    const N: usize,
>(
    addr: TmemAddr,
    values: [T; N],
) {
    const {
        assert!(size_of::<T>() == 4, "T must be a 4-byte type");
        repeat_count(N, 1);
    };
    macro_rules! emit {
        ($pack:literal) => {
            by_num_regs!(
                N,
                asm_store_regs!(
                    ["tcgen05.st.sync.aligned.16x32bx2.x{num}", $pack, ".b32 [{addr}], {imm}, "]
                    [";"]
                    [
                        num = const N as u32,
                        addr = in(reg32) addr.to_bits(),
                        imm = const HALF_SPLIT_OFFSET,
                    ]
                    values
                )
            )
        };
    }
    asm_variants!(emit;
        PACK => {
            Packing::None => "",
            Packing::Pack16b => ".unpack::16b",
        },
    )
}

/// Maps the reduction parameters of [`load_reduce`] to the PTX modifiers `.type.redOp{.abs}{.NaN}`.
/// Rejects the combinations that PTX does not support at compile time.
#[allow(unused_macros)]
macro_rules! asm_reduce_variants {
    ($cb:ident; $op:expr, $t:ty, $abs:expr, $nan:expr) => {{
        const {
            assert!(
                matches!($op, ReduxOp::Min | ReduxOp::Max),
                "tcgen05.ld.red only supports ReduxOp::Min and ReduxOp::Max"
            );
            assert!(
                matches!(<$t as sealed::Sealed>::TYPE, sealed::ReduceType::F32)
                    || (matches!($abs, Absolute::Disabled) && matches!($nan, NanHandling::Ignore)),
                "Absolute::Enabled and NanHandling::Propagate are only supported for f32"
            );
        };
        asm_variants!($cb;
            ($op, <$t as sealed::Sealed>::TYPE, $abs, $nan) => {
                (ReduxOp::Min, sealed::ReduceType::F32, Absolute::Disabled, NanHandling::Ignore) => ".f32.min",
                (ReduxOp::Max, sealed::ReduceType::F32, Absolute::Disabled, NanHandling::Ignore) => ".f32.max",
                (ReduxOp::Min, sealed::ReduceType::F32, Absolute::Enabled, NanHandling::Ignore) => ".f32.min.abs",
                (ReduxOp::Max, sealed::ReduceType::F32, Absolute::Enabled, NanHandling::Ignore) => ".f32.max.abs",
                (ReduxOp::Min, sealed::ReduceType::F32, Absolute::Disabled, NanHandling::Propagate) => ".f32.min.NaN",
                (ReduxOp::Max, sealed::ReduceType::F32, Absolute::Disabled, NanHandling::Propagate) => ".f32.max.NaN",
                (ReduxOp::Min, sealed::ReduceType::F32, Absolute::Enabled, NanHandling::Propagate) => ".f32.min.abs.NaN",
                (ReduxOp::Max, sealed::ReduceType::F32, Absolute::Enabled, NanHandling::Propagate) => ".f32.max.abs.NaN",
                (ReduxOp::Min, sealed::ReduceType::U32, _, _) => ".u32.min",
                (ReduxOp::Max, sealed::ReduceType::U32, _, _) => ".u32.max",
                (ReduxOp::Min, sealed::ReduceType::S32, _, _) => ".s32.min",
                (ReduxOp::Max, sealed::ReduceType::S32, _, _) => ".s32.max",
            },
        )
    }};
}

/// Loads data from Tensor Memory at `addr` like [`load`] with [`LdStShape::S32x32b`] and no
/// packing, and additionally reduces the loaded values of each lane (`tcgen05.ld.red.32x32b`).
///
/// Returns the `N` loaded registers and the result of the reduction of these values across the
/// columns of the thread's lane, performed with `OP`, which must be [`ReduxOp::Min`] or
/// [`ReduxOp::Max`]. `T` is `f32`, `u32` or `i32`. For `f32`, with [`Absolute::Enabled`] the
/// absolute values of the loaded data are reduced (`.abs`), and with [`NanHandling::Propagate`]
/// the result is canonical NaN if any input is NaN (`.NaN`); these are rejected for the integer
/// types. `N` must be a power of two between 2 and 128.
///
/// The load is asynchronous like [`load`]: both results may be used directly by later
/// instructions of the thread, and [`wait_load`] is required before the Tensor Memory is
/// overwritten by other operations.
///
/// Requires `sm_101f`, `sm_103f` or `sm_110f` (or `sm_101a`, `sm_103a`, `sm_110a`). It is not
/// supported on `sm_100a` or `sm_100f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld>
///
/// # Safety
///
/// - All threads of the warp must call this function convergently with the same `addr`.
/// - The accessed region must lie within allocated Tensor Memory, and within the lane quarter of
///   the executing warp.
/// - The data must have been written by operations whose completion has been observed.
/// - See also the [safety contract](super#safety-contract).
#[cfg(any(
    target_feature = "sm_101f",
    target_feature = "sm_103f",
    target_feature = "sm_110f"
))]
#[inline(always)]
pub unsafe fn load_reduce<
    const OP: ReduxOp,
    const ABS: Absolute,
    const NAN: NanHandling,
    T: TmemReduceElement,
    const N: usize,
>(
    addr: TmemAddr,
) -> ([T; N], T) {
    const {
        assert!(N >= 2, "tcgen05.ld.red requires at least 2 registers (.x2)");
        repeat_count(N, 1);
    };
    let mut red = 0u32;
    macro_rules! emit {
        ($mods:literal) => {
            by_num_regs!(
                N,
                asm_load_regs!(
                    ["tcgen05.ld.red.sync.aligned.32x32b.x{num}", $mods, " "]
                    [", {red}, [{addr}];"]
                    [
                        num = const N as u32,
                        addr = in(reg32) addr.to_bits(),
                        red = out(reg32) red,
                    ]
                    [T; N]
                )
            )
        };
    }
    let values = asm_reduce_variants!(emit; OP, T, ABS, NAN);
    (values, unsafe { core::mem::transmute_copy(&red) })
}

/// Loads data from two regions of Tensor Memory like [`load_16x32bx2`] with no packing, and
/// additionally reduces the loaded values of each lane (`tcgen05.ld.red.16x32bx2`).
///
/// The reduction parameters and the results are as for [`load_reduce`].
///
/// Requires `sm_101f`, `sm_103f` or `sm_110f` (or `sm_101a`, `sm_103a`, `sm_110a`). It is not
/// supported on `sm_100a` or `sm_100f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld>
///
/// # Safety
///
/// - All threads of the warp must call this function convergently with the same `addr`.
/// - Both regions must lie within allocated Tensor Memory, and within the lane quarter of the
///   executing warp.
/// - The data must have been written by operations whose completion has been observed.
/// - See also the [safety contract](super#safety-contract).
#[cfg(any(
    target_feature = "sm_101f",
    target_feature = "sm_103f",
    target_feature = "sm_110f"
))]
#[inline(always)]
pub unsafe fn load_reduce_16x32bx2<
    const HALF_SPLIT_OFFSET: u32,
    const OP: ReduxOp,
    const ABS: Absolute,
    const NAN: NanHandling,
    T: TmemReduceElement,
    const N: usize,
>(
    addr: TmemAddr,
) -> ([T; N], T) {
    const {
        assert!(N >= 2, "tcgen05.ld.red requires at least 2 registers (.x2)");
        repeat_count(N, 1);
    };
    let mut red = 0u32;
    macro_rules! emit {
        ($mods:literal) => {
            by_num_regs!(
                N,
                asm_load_regs!(
                    ["tcgen05.ld.red.sync.aligned.16x32bx2.x{num}", $mods, " "]
                    [", {red}, [{addr}], {imm};"]
                    [
                        num = const N as u32,
                        addr = in(reg32) addr.to_bits(),
                        imm = const HALF_SPLIT_OFFSET,
                        red = out(reg32) red,
                    ]
                    [T; N]
                )
            )
        };
    }
    let values = asm_reduce_variants!(emit; OP, T, ABS, NAN);
    (values, unsafe { core::mem::transmute_copy(&red) })
}

/// Blocks the executing thread until all prior [`load`]s (and their variants) of the thread have
/// completed (`tcgen05.wait::ld`).
///
/// This is needed before the loaded Tensor Memory is overwritten by another operation (for
/// example an MMA issued by another thread, after synchronizing with it using
/// [`fence_before_thread_sync`](super::fence_before_thread_sync)), but not to use the loaded
/// registers.
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait>
///
/// # Safety
///
/// - All threads of the warp must call this function convergently.
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn wait_load() {
    unsafe { asm!("tcgen05.wait::ld.sync.aligned;", options(nostack)) }
}

/// Blocks the executing thread until all prior [`store`]s (and their variants) of the thread have
/// completed (`tcgen05.wait::st`).
///
/// This is needed before the stored Tensor Memory is read or written by an operation that is not
/// pipelined with the stores (for example an MMA that reads it; later loads and stores of the same
/// warp are ordered after the stores without it), and before synchronizing with other threads
/// through [`fence_before_thread_sync`](super::fence_before_thread_sync).
///
/// Requires `sm_100f`, `sm_101f` or `sm_110f`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-wait>
///
/// # Safety
///
/// - All threads of the warp must call this function convergently.
/// - See also the [safety contract](super#safety-contract).
#[inline(always)]
pub unsafe fn wait_store() {
    unsafe { asm!("tcgen05.wait::st.sync.aligned;", options(nostack)) }
}
