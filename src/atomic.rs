// --- LLM-generated --- //
//! Atomic read-modify-write operations (`atom`) and reductions (`red`, `red.async`,
//! `multimem.red.async`) on global and shared memory.
//!
//! `atom_*` functions atomically update a memory location and return its previous value. `red_*`
//! functions perform the same update without returning anything, which can be cheaper.
//!
//! # Memory ordering and scope
//!
//! The const-generic parameter `SEM` selects the memory synchronizing effect of the operation, as
//! described by the memory consistency model. `atom` supports [`MemoryOrdering::Relaxed`],
//! [`MemoryOrdering::Acquire`], [`MemoryOrdering::Release`] and [`MemoryOrdering::AcqRel`]; `red`
//! supports [`MemoryOrdering::Relaxed`] and [`MemoryOrdering::Release`].
//! [`MemoryOrdering::SeqCst`] is not supported.
//!
//! `SCOPE` is the set of threads that can directly observe the memory synchronizing effect of the
//! operation. Two atomic operations (`atom` or `red`) are performed atomically with respect to
//! each other only if each operation specifies a scope that includes the other. Otherwise, each
//! operation observes the other as if it were split into a read followed by a dependent write.
//! [`Scope::Cluster`] requires `sm_90`.
//!
//! # State spaces
//!
//! All functions take generic addresses. `SPACE` ([`AtomicSpace`]) declares the state space that
//! the address points into, so that the instruction can use the address of that state space
//! rather than generic addressing. [`AtomicSpace::Generic`] accepts addresses in either global or
//! shared memory.
//!
//! # Packed and vector types
//!
//! Operations on the packed types [`f16x2`] and [`bf16x2`], and on the vector types [`f16x4`],
//! [`f16x8`], [`bf16x4`], [`bf16x8`] and `Simd<f32, 2 | 4>` (the vector forms `.v2` and `.v4`),
//! access adjacent scalar elements in memory. Atomicity is guaranteed separately for each scalar
//! element; the operation as a whole is not guaranteed to be atomic as a single access. The
//! vector forms require `sm_90` and global memory: `SPACE` must be [`AtomicSpace::Generic`] or
//! [`AtomicSpace::Global`].
//!
//! The state space requirement depends on the operation, not just the type: [`f16x2`] and
//! [`bf16x2`] are added with the packed `.f16x2`/`.bf16x2` forms, which support every state space,
//! but their minimum and maximum use the vector forms `.v2.f16`/`.v2.bf16`, which require `sm_90`
//! and global memory.
//!
//! # L2 cache hints
//!
//! The `*_with_cache_hint` variants apply the L2 cache eviction policy `cache_policy` to the
//! memory access (`.L2::cache_hint`). The policy is only a performance hint and does not affect
//! the memory consistency behavior of the program. Cache hints are only supported for global
//! memory: `SPACE` must be [`AtomicSpace::Generic`] or [`AtomicSpace::Global`], and the address
//! must point into global memory. `atom_cas` has no cache-hint variant.
//!
//! # Safety contract
//!
//! The `atom_*` and `red_*` functions require that:
//!
//! - `ptr` is valid for reads and writes of `T` and aligned for `T`. The alignment of every type
//!   supporting atomic operations is equal to its size, as required by the vector forms.
//! - `ptr` points into the state space selected by `SPACE`: global memory for
//!   [`AtomicSpace::Global`], the shared memory of the executing CTA for
//!   [`AtomicSpace::SharedCta`], the shared memory of any CTA of the executing cluster for
//!   [`AtomicSpace::SharedCluster`], and global or shared memory for [`AtomicSpace::Generic`].
//! - The access does not cause a data race, i.e. conflicting concurrent accesses by other threads
//!   are also atomic or otherwise morally strong operations.
//!
//! <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom>

#[cfg(target_feature = "sm_90")]
use core::simd::Simd;
use core::{arch::asm, marker::ConstParamTy};

#[cfg(target_feature = "sm_90")]
use crate::MBarrier;
#[cfg(target_feature = "sm_100")]
use crate::memory::{AsyncStoreTarget, assert_async_store_supported};
use crate::{
    bf16, bf16x2, f16, f16x2,
    memory::CachePolicy,
    modifiers::{MemoryOrdering, Scope},
};
#[cfg(target_feature = "sm_90")]
use crate::{bf16x4, bf16x8, f16x4, f16x8};
use sealed::Op;

/// The state space that the address of an atomic operation points into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, ConstParamTy)]
pub enum AtomicSpace {
    /// Generic addressing: the address may point into global or shared memory.
    Generic,
    /// Global memory (`.global`).
    Global,
    /// Shared memory of the executing CTA (`.shared::cta`).
    SharedCta,
    /// Shared memory of any CTA in the executing cluster (`.shared::cluster`).
    ///
    /// Requires `sm_90`.
    SharedCluster,
}

mod sealed {
    use super::*;

    /// A read-modify-write operation of `atom`, `red` and `red.async`.
    #[derive(Clone, Copy, PartialEq, Eq, ConstParamTy)]
    pub enum Op {
        Add,
        Min,
        Max,
        And,
        Or,
        Xor,
        Inc,
        Dec,
    }

    /// `atom` and `red` with the operations of [`Op`].
    pub trait Rmw: Copy {
        unsafe fn atom<
            const OP: Op,
            const SEM: MemoryOrdering,
            const SCOPE: Scope,
            const SPACE: AtomicSpace,
        >(
            ptr: *mut Self,
            value: Self,
        ) -> Self;

        unsafe fn atom_hint<
            const OP: Op,
            const SEM: MemoryOrdering,
            const SCOPE: Scope,
            const SPACE: AtomicSpace,
        >(
            ptr: *mut Self,
            value: Self,
            policy: u64,
        ) -> Self;

        unsafe fn red<
            const OP: Op,
            const SEM: MemoryOrdering,
            const SCOPE: Scope,
            const SPACE: AtomicSpace,
        >(
            ptr: *mut Self,
            value: Self,
        );

        unsafe fn red_hint<
            const OP: Op,
            const SEM: MemoryOrdering,
            const SCOPE: Scope,
            const SPACE: AtomicSpace,
        >(
            ptr: *mut Self,
            value: Self,
            policy: u64,
        );
    }

    /// `atom.exch`.
    pub trait Exch: Copy {
        unsafe fn exch<const SEM: MemoryOrdering, const SCOPE: Scope, const SPACE: AtomicSpace>(
            ptr: *mut Self,
            value: Self,
        ) -> Self;

        unsafe fn exch_hint<
            const SEM: MemoryOrdering,
            const SCOPE: Scope,
            const SPACE: AtomicSpace,
        >(
            ptr: *mut Self,
            value: Self,
            policy: u64,
        ) -> Self;
    }

    /// `atom.cas`.
    pub trait Cas: Copy {
        unsafe fn cas<const SEM: MemoryOrdering, const SCOPE: Scope, const SPACE: AtomicSpace>(
            ptr: *mut Self,
            compare: Self,
            new: Self,
        ) -> Self;
    }

    /// `red.async` with `.mbarrier::complete_tx::bytes` completion.
    #[cfg(target_feature = "sm_90")]
    pub trait RedAsync: Copy {
        unsafe fn red_async<const OP: Op, const SPACE: AtomicSpace>(
            ptr: *mut Self,
            value: Self,
            mbar: *mut MBarrier,
        );
    }

    /// `red.async.release` and `multimem.red.async.release`.
    #[cfg(target_feature = "sm_100")]
    pub trait RedAsyncRelease: Copy {
        unsafe fn red_async_release<
            const SCOPE: Scope,
            const TARGET: AsyncStoreTarget,
            const SPACE: AtomicSpace,
        >(
            ptr: *mut Self,
            value: Self,
        );
    }
}

/// Types supporting [`atom_add`] and [`red_add`].
///
/// This trait is sealed. It is implemented for `u32`, `i32`, `u64`, `i64`, `f32`, `f64`,
/// [`f16`](struct@f16) and [`f16x2`], and with `sm_90` also for [`bf16`], [`bf16x2`], the vector
/// types [`f16x4`], [`f16x8`], [`bf16x4`] and [`bf16x8`], and `Simd<f32, 2>` and `Simd<f32, 4>`.
///
/// `i64` is added with the `.u64` form, as PTX has no `.add.s64`; the result is identical in two's
/// complement. Floating-point addition on `f64` requires `sm_60`, on `f16x2` `sm_60` and on
/// `f16` `sm_70`. The vector types (all except [`f16x2`] and [`bf16x2`]) only support global
/// memory.
pub trait AtomicAdd: sealed::Rmw {}

/// Types supporting [`atom_min`], [`atom_max`], [`red_min`] and [`red_max`].
///
/// This trait is sealed. It is implemented for `u32`, `i32`, `u64` and `i64`, and with `sm_90`
/// also for [`f16x2`], [`bf16x2`], [`f16x4`], [`f16x8`], [`bf16x4`] and [`bf16x8`]. These
/// floating-point types only support global memory, including [`f16x2`] and [`bf16x2`] (which
/// can be added in any state space).
pub trait AtomicMinMax: sealed::Rmw {}

/// Types supporting the bitwise operations [`atom_and`], [`atom_or`], [`atom_xor`],
/// [`red_and`], [`red_or`] and [`red_xor`].
///
/// This trait is sealed. It is implemented for `u32`, `i32`, `u64` and `i64`.
pub trait AtomicBitwise: sealed::Rmw {}

/// Types supporting [`atom_exch`].
///
/// This trait is sealed. It is implemented for the 32- and 64-bit integers and floats,
/// [`f16x2`], [`bf16x2`] and raw pointers, and with `sm_90` also for `u128` and `i128`.
pub trait AtomicExchange: sealed::Exch {}

/// Types supporting [`atom_cas`].
///
/// This trait is sealed. It is implemented for the 16-, 32- and 64-bit integers and floats,
/// [`f16`](struct@f16), [`bf16`], [`f16x2`], [`bf16x2`] and raw pointers, and with `sm_90` also for `u128` and
/// `i128`. The 16-bit types require `sm_70`.
pub trait AtomicCompareExchange: sealed::Cas {}

/// Types supporting [`red_async_add`]: `u32`, `i32` and `u64`. This trait is sealed.
#[cfg(target_feature = "sm_90")]
pub trait RedAsyncAdd: sealed::RedAsync {}

/// Types supporting [`red_async_min`] and [`red_async_max`]: `u32` and `i32`. This trait is
/// sealed.
#[cfg(target_feature = "sm_90")]
pub trait RedAsyncMinMax: sealed::RedAsync {}

/// Types supporting [`red_async_and`], [`red_async_or`] and [`red_async_xor`]: `u32` and `i32`.
/// This trait is sealed.
#[cfg(target_feature = "sm_90")]
pub trait RedAsyncBitwise: sealed::RedAsync {}

/// Types supporting [`red_async_add_release`]: `u32`, `i32`, `u64` and `i64`. This trait is
/// sealed.
#[cfg(target_feature = "sm_100")]
pub trait RedAsyncReleaseAdd: sealed::RedAsyncRelease {}

/// Converts the generic address `ptr` to an address in the state space `SPACE` (`cvta.to`).
#[inline(always)]
pub(crate) fn space_address<const SPACE: AtomicSpace, T>(ptr: *mut T) -> u64 {
    let ptr = ptr.expose_provenance() as u64;
    macro_rules! cvta {
        ($space:literal) => {{
            let address: u64;
            // SAFETY: `cvta` only reads and writes registers.
            unsafe {
                asm!(
                    concat!("cvta.to", $space, ".u64 {address}, {ptr};"),
                    address = out(reg64) address,
                    ptr = in(reg64) ptr,
                    options(pure, nomem, nostack),
                );
            }
            address
        }};
    }
    if const { matches!(SPACE, AtomicSpace::Generic) } {
        ptr
    } else if const { matches!(SPACE, AtomicSpace::Global) } {
        cvta!(".global")
    } else if const { matches!(SPACE, AtomicSpace::SharedCta) } {
        cvta!(".shared::cta")
    } else {
        cvta!(".shared::cluster")
    }
}

#[derive(PartialEq, Eq)]
enum Kind {
    Atom,
    Red,
}

const fn assert_qualifiers_supported(
    kind: Kind,
    sem: MemoryOrdering,
    scope: Scope,
    space: AtomicSpace,
    cache_hint: bool,
) {
    match kind {
        Kind::Atom => assert!(
            !matches!(sem, MemoryOrdering::SeqCst),
            "atom does not support MemoryOrdering::SeqCst"
        ),
        Kind::Red => assert!(
            matches!(sem, MemoryOrdering::Relaxed | MemoryOrdering::Release),
            "red only supports MemoryOrdering::Relaxed and MemoryOrdering::Release"
        ),
    }
    assert!(
        !matches!(scope, Scope::Cluster) || cfg!(target_feature = "sm_90"),
        "Scope::Cluster requires sm_90"
    );
    assert!(
        !matches!(space, AtomicSpace::SharedCluster) || cfg!(target_feature = "sm_90"),
        "AtomicSpace::SharedCluster requires sm_90"
    );
    assert!(
        !cache_hint || matches!(space, AtomicSpace::Generic | AtomicSpace::Global),
        "cache hints are only supported for AtomicSpace::Generic and AtomicSpace::Global"
    );
}

const fn assert_global_space(space: AtomicSpace) {
    assert!(
        matches!(space, AtomicSpace::Generic | AtomicSpace::Global),
        "only AtomicSpace::Generic and AtomicSpace::Global are supported"
    );
}

/// Builds the register operand of an `asm!` template from positional operand indices: `{0}` for a
/// single register, or a brace-enclosed vector `{{0}, {1}, ...}`.
macro_rules! reg_list {
    ($only:literal) => {
        concat!("{", $only, "}")
    };
    ($first:literal $($rest:literal)+) => {
        concat!("{{{", $first, "}", $(", {", $rest, "}",)+ "}}")
    };
}

/// Expands `$cb!(tokens... sem, scope, space)` with the qualifiers selected by the const
/// parameters `SEM`, `SCOPE` and `SPACE` in scope. The first argument selects the supported
/// memory orderings (`atom` or `red`) and state spaces (`all` or `global`).
macro_rules! with_qualifiers {
    (atom all, $cb:ident!$pre:tt) => {
        asm_variants!($cb!$pre;
            SEM => {
                MemoryOrdering::Relaxed => ".relaxed",
                MemoryOrdering::Acquire => ".acquire",
                MemoryOrdering::Release => ".release",
                MemoryOrdering::AcqRel => ".acq_rel",
            },
            SCOPE => {
                Scope::Cta => ".cta",
                Scope::Cluster => ".cluster",
                Scope::Gpu => ".gpu",
                Scope::System => ".sys",
            },
            SPACE => {
                AtomicSpace::Generic => "",
                AtomicSpace::Global => ".global",
                AtomicSpace::SharedCta => ".shared::cta",
                AtomicSpace::SharedCluster => ".shared::cluster",
            },
        )
    };
    (atom global, $cb:ident!$pre:tt) => {
        asm_variants!($cb!$pre;
            SEM => {
                MemoryOrdering::Relaxed => ".relaxed",
                MemoryOrdering::Acquire => ".acquire",
                MemoryOrdering::Release => ".release",
                MemoryOrdering::AcqRel => ".acq_rel",
            },
            SCOPE => {
                Scope::Cta => ".cta",
                Scope::Cluster => ".cluster",
                Scope::Gpu => ".gpu",
                Scope::System => ".sys",
            },
            SPACE => {
                AtomicSpace::Generic => "",
                AtomicSpace::Global => ".global",
            },
        )
    };
    (red all, $cb:ident!$pre:tt) => {
        asm_variants!($cb!$pre;
            SEM => {
                MemoryOrdering::Relaxed => ".relaxed",
                MemoryOrdering::Release => ".release",
            },
            SCOPE => {
                Scope::Cta => ".cta",
                Scope::Cluster => ".cluster",
                Scope::Gpu => ".gpu",
                Scope::System => ".sys",
            },
            SPACE => {
                AtomicSpace::Generic => "",
                AtomicSpace::Global => ".global",
                AtomicSpace::SharedCta => ".shared::cta",
                AtomicSpace::SharedCluster => ".shared::cluster",
            },
        )
    };
    (red global, $cb:ident!$pre:tt) => {
        asm_variants!($cb!$pre;
            SEM => {
                MemoryOrdering::Relaxed => ".relaxed",
                MemoryOrdering::Release => ".release",
            },
            SCOPE => {
                Scope::Cta => ".cta",
                Scope::Cluster => ".cluster",
                Scope::Gpu => ".gpu",
                Scope::System => ".sys",
            },
            SPACE => {
                AtomicSpace::Generic => "",
                AtomicSpace::Global => ".global",
            },
        )
    };
}

/// Emits an `atom` or `red` instruction on the registers of the arrays `$d` (destination) and `$b`
/// (source), whose positional operand indices are given by `$regs`.
macro_rules! emit_rmw {
    (
        atom $cls:ident ([$($o:literal)+] [$($i:literal)+]) $op:literal $ty:literal
        $a:ident $d:ident $b:ident $sem:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!(
                    "atom", $sem, $scope, $space, $op, $ty, " ",
                    reg_list!($($o)+), ", [{a}], ", reg_list!($($i)+), ";"
                ),
                $(out($cls) $d[$o],)+
                $(in($cls) $b[$o],)+
                a = in(reg64) $a,
                options(nostack),
            )
        }
    };
    (
        atom_hint $cls:ident ([$($o:literal)+] [$($i:literal)+]) $op:literal $ty:literal
        $a:ident $d:ident $b:ident $p:ident $sem:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!(
                    "atom", $sem, $scope, $space, $op, ".L2::cache_hint", $ty, " ",
                    reg_list!($($o)+), ", [{a}], ", reg_list!($($i)+), ", {p};"
                ),
                $(out($cls) $d[$o],)+
                $(in($cls) $b[$o],)+
                a = in(reg64) $a,
                p = in(reg64) $p,
                options(nostack),
            )
        }
    };
    (
        red $cls:ident ([$($o:literal)+] $_i:tt) $op:literal $ty:literal
        $a:ident $b:ident $sem:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!(
                    "red", $sem, $scope, $space, $op, $ty, " [{a}], ", reg_list!($($o)+), ";"
                ),
                $(in($cls) $b[$o],)+
                a = in(reg64) $a,
                options(nostack),
            )
        }
    };
    (
        red_hint $cls:ident ([$($o:literal)+] $_i:tt) $op:literal $ty:literal
        $a:ident $b:ident $p:ident $sem:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!(
                    "red", $sem, $scope, $space, $op, ".L2::cache_hint", $ty, " [{a}], ",
                    reg_list!($($o)+), ", {p};"
                ),
                $(in($cls) $b[$o],)+
                a = in(reg64) $a,
                p = in(reg64) $p,
                options(nostack),
            )
        }
    };
}

/// Implements [`sealed::Rmw`] and the given marker traits.
///
/// Each entry gives the marker traits implemented for the type (each with optional `cfg`
/// attributes), followed by groups of operations sharing a register layout. A group gives the
/// register class, the positional operand indices of the destination and source registers, the
/// state spaces supported by its operations (`all` or `global`), conversions to and from an array
/// of registers, and the qualifiers (operation and type) of each operation of the group. The
/// layout and the supported state spaces can thus differ between the operations of a type.
macro_rules! impl_rmw {
    (@markers $attrs:tt $ty:ty; $($(#[$mattr:meta])* $marker:ident),+) => {
        $(impl_rmw!(@marker $attrs $ty, $(#[$mattr])* $marker);)+
    };
    (@marker [$(#[$attr:meta])*] $ty:ty, $(#[$mattr:meta])* $marker:ident) => {
        $(#[$attr])*
        $(#[$mattr])*
        impl $marker for $ty {}
    };
    (@check all) => {};
    (@check global) => {
        assert_global_space(SPACE);
    };
    ($(
        $(#[$attr:meta])*
        $ty:ty: $($(#[$mattr:meta])* $marker:ident),+ {$(
            $cls:ident $regs:tt, $spaces:ident, |$v:ident| $to:expr, |$r:ident| $from:expr;
                { $($op:ident => $op_lit:literal $ty_lit:literal),+ $(,)? }
        )+}
    )+) => {$(
        $(#[$attr])*
        impl sealed::Rmw for $ty {
            #[inline(always)]
            unsafe fn atom<
                const OP: Op,
                const SEM: MemoryOrdering,
                const SCOPE: Scope,
                const SPACE: AtomicSpace,
            >(ptr: *mut Self, value: Self) -> Self {
                const { $($(if matches!(OP, Op::$op) { impl_rmw!(@check $spaces); })+)+ };
                let a = space_address::<SPACE, _>(ptr);
                $($(if const { matches!(OP, Op::$op) } {
                    let $v = value;
                    let b = $to;
                    let mut d = b;
                    with_qualifiers!(atom $spaces, emit_rmw!(atom $cls $regs $op_lit $ty_lit a d b));
                    let $r = d;
                    $from
                } else)+)+ {
                    unreachable!()
                }
            }

            #[inline(always)]
            unsafe fn atom_hint<
                const OP: Op,
                const SEM: MemoryOrdering,
                const SCOPE: Scope,
                const SPACE: AtomicSpace,
            >(ptr: *mut Self, value: Self, policy: u64) -> Self {
                let a = space_address::<SPACE, _>(ptr);
                $($(if const { matches!(OP, Op::$op) } {
                    let $v = value;
                    let b = $to;
                    let mut d = b;
                    with_qualifiers!(
                        atom global,
                        emit_rmw!(atom_hint $cls $regs $op_lit $ty_lit a d b policy)
                    );
                    let $r = d;
                    $from
                } else)+)+ {
                    unreachable!()
                }
            }

            #[inline(always)]
            unsafe fn red<
                const OP: Op,
                const SEM: MemoryOrdering,
                const SCOPE: Scope,
                const SPACE: AtomicSpace,
            >(ptr: *mut Self, value: Self) {
                const { $($(if matches!(OP, Op::$op) { impl_rmw!(@check $spaces); })+)+ };
                let a = space_address::<SPACE, _>(ptr);
                $($(if const { matches!(OP, Op::$op) } {
                    let $v = value;
                    let b = $to;
                    with_qualifiers!(red $spaces, emit_rmw!(red $cls $regs $op_lit $ty_lit a b))
                } else)+)+ {
                    unreachable!()
                }
            }

            #[inline(always)]
            unsafe fn red_hint<
                const OP: Op,
                const SEM: MemoryOrdering,
                const SCOPE: Scope,
                const SPACE: AtomicSpace,
            >(ptr: *mut Self, value: Self, policy: u64) {
                let a = space_address::<SPACE, _>(ptr);
                $($(if const { matches!(OP, Op::$op) } {
                    let $v = value;
                    let b = $to;
                    with_qualifiers!(
                        red global,
                        emit_rmw!(red_hint $cls $regs $op_lit $ty_lit a b policy)
                    )
                } else)+)+ {
                    unreachable!()
                }
            }
        }

        impl_rmw!(@markers [$(#[$attr])*] $ty; $($(#[$mattr])* $marker),+);
    )+};
}

impl_rmw! {
    u32: AtomicAdd, AtomicMinMax, AtomicBitwise {
        reg32 ([0] [1]), all, |v| [v], |r| r[0];
            {
                Add => ".add" ".u32",
                Min => ".min" ".u32",
                Max => ".max" ".u32",
                And => ".and" ".b32",
                Or => ".or" ".b32",
                Xor => ".xor" ".b32",
                Inc => ".inc" ".u32",
                Dec => ".dec" ".u32",
            }
    }
    i32: AtomicAdd, AtomicMinMax, AtomicBitwise {
        reg32 ([0] [1]), all, |v| [v as u32], |r| r[0] as i32;
            {
                Add => ".add" ".s32",
                Min => ".min" ".s32",
                Max => ".max" ".s32",
                And => ".and" ".b32",
                Or => ".or" ".b32",
                Xor => ".xor" ".b32",
            }
    }
    u64: AtomicAdd, AtomicMinMax, AtomicBitwise {
        reg64 ([0] [1]), all, |v| [v], |r| r[0];
            {
                Add => ".add" ".u64",
                Min => ".min" ".u64",
                Max => ".max" ".u64",
                And => ".and" ".b64",
                Or => ".or" ".b64",
                Xor => ".xor" ".b64",
            }
    }
    // There is no `.add.s64`; two's complement addition is the same operation as `.add.u64`.
    i64: AtomicAdd, AtomicMinMax, AtomicBitwise {
        reg64 ([0] [1]), all, |v| [v as u64], |r| r[0] as i64;
            {
                Add => ".add" ".u64",
                Min => ".min" ".s64",
                Max => ".max" ".s64",
                And => ".and" ".b64",
                Or => ".or" ".b64",
                Xor => ".xor" ".b64",
            }
    }
    f32: AtomicAdd {
        reg32 ([0] [1]), all, |v| [v.to_bits()], |r| f32::from_bits(r[0]);
            { Add => ".add" ".f32" }
    }
    f64: AtomicAdd {
        reg64 ([0] [1]), all, |v| [v.to_bits()], |r| f64::from_bits(r[0]);
            { Add => ".add" ".f64" }
    }
    f16: AtomicAdd {
        reg16 ([0] [1]), all, |v| [v.to_bits()], |r| f16::from_bits(r[0]);
            { Add => ".add.noftz" ".f16" }
    }
    // The packed forms add in all state spaces, but only the `.v2.f16` form has min and max.
    f16x2: AtomicAdd, #[cfg(target_feature = "sm_90")] AtomicMinMax {
        reg32 ([0] [1]), all, |v| [v.to_bits()], |r| f16x2::from_bits(r[0]);
            { Add => ".add.noftz" ".f16x2" }
        reg16 ([0 1] [2 3]), global,
            |v| {
                let bits = v.to_bits();
                [bits as u16, (bits >> 16) as u16]
            },
            |r| f16x2::from_bits(r[0] as u32 | (r[1] as u32) << 16);
            {
                Min => ".min.noftz" ".v2.f16",
                Max => ".max.noftz" ".v2.f16",
            }
    }
    #[cfg(target_feature = "sm_90")]
    bf16: AtomicAdd {
        reg16 ([0] [1]), all, |v| [v.to_bits()], |r| bf16::from_bits(r[0]);
            { Add => ".add.noftz" ".bf16" }
    }
    #[cfg(target_feature = "sm_90")]
    bf16x2: AtomicAdd, AtomicMinMax {
        reg32 ([0] [1]), all, |v| [v.to_bits()], |r| bf16x2::from_bits(r[0]);
            { Add => ".add.noftz" ".bf16x2" }
        reg16 ([0 1] [2 3]), global,
            |v| {
                let bits = v.to_bits();
                [bits as u16, (bits >> 16) as u16]
            },
            |r| bf16x2::from_bits(r[0] as u32 | (r[1] as u32) << 16);
            {
                Min => ".min.noftz" ".v2.bf16",
                Max => ".max.noftz" ".v2.bf16",
            }
    }
    #[cfg(target_feature = "sm_90")]
    f16x4: AtomicAdd, AtomicMinMax {
        reg32 ([0 1] [2 3]), global,
            |v| v.to_x2s().map(f16x2::to_bits), |r| f16x4::from_x2s(r.map(f16x2::from_bits));
            {
                Add => ".add.noftz" ".v2.f16x2",
                Min => ".min.noftz" ".v2.f16x2",
                Max => ".max.noftz" ".v2.f16x2",
            }
    }
    #[cfg(target_feature = "sm_90")]
    f16x8: AtomicAdd, AtomicMinMax {
        reg32 ([0 1 2 3] [4 5 6 7]), global,
            |v| v.to_x2s().map(f16x2::to_bits), |r| f16x8::from_x2s(r.map(f16x2::from_bits));
            {
                Add => ".add.noftz" ".v4.f16x2",
                Min => ".min.noftz" ".v4.f16x2",
                Max => ".max.noftz" ".v4.f16x2",
            }
    }
    #[cfg(target_feature = "sm_90")]
    bf16x4: AtomicAdd, AtomicMinMax {
        reg32 ([0 1] [2 3]), global,
            |v| v.to_x2s().map(bf16x2::to_bits), |r| bf16x4::from_x2s(r.map(bf16x2::from_bits));
            {
                Add => ".add.noftz" ".v2.bf16x2",
                Min => ".min.noftz" ".v2.bf16x2",
                Max => ".max.noftz" ".v2.bf16x2",
            }
    }
    #[cfg(target_feature = "sm_90")]
    bf16x8: AtomicAdd, AtomicMinMax {
        reg32 ([0 1 2 3] [4 5 6 7]), global,
            |v| v.to_x2s().map(bf16x2::to_bits), |r| bf16x8::from_x2s(r.map(bf16x2::from_bits));
            {
                Add => ".add.noftz" ".v4.bf16x2",
                Min => ".min.noftz" ".v4.bf16x2",
                Max => ".max.noftz" ".v4.bf16x2",
            }
    }
    #[cfg(target_feature = "sm_90")]
    Simd<f32, 2>: AtomicAdd {
        reg32 ([0 1] [2 3]), global,
            |v| v.to_array().map(f32::to_bits), |r| Simd::from_array(r.map(f32::from_bits));
            { Add => ".add" ".v2.f32" }
    }
    #[cfg(target_feature = "sm_90")]
    Simd<f32, 4>: AtomicAdd {
        reg32 ([0 1 2 3] [4 5 6 7]), global,
            |v| v.to_array().map(f32::to_bits), |r| Simd::from_array(r.map(f32::from_bits));
            { Add => ".add" ".v4.f32" }
    }
}

/// Emits `atom.exch` (`$kind` = `exch` or `exch_hint`) or `atom.cas` (`$kind` = `cas`), for a
/// single register of class `$cls` or for a `.b128` value given as an array of two 64-bit
/// registers (`(b128)`).
macro_rules! emit_exch_cas {
    (
        exch (b128) $a:ident $d:ident $b:ident
        $sem:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!(
                    "{{\n\t.reg .b128 %atom_d, %atom_b;\n\t",
                    "mov.b128 %atom_b, {{{b0}, {b1}}};\n\t",
                    "atom", $sem, $scope, $space, ".exch.b128 %atom_d, [{a}], %atom_b;\n\t",
                    "mov.b128 {{{d0}, {d1}}}, %atom_d;\n}}"
                ),
                d0 = out(reg64) $d[0],
                d1 = out(reg64) $d[1],
                a = in(reg64) $a,
                b0 = in(reg64) $b[0],
                b1 = in(reg64) $b[1],
                options(nostack),
            )
        }
    };
    (
        exch_hint (b128) $a:ident $d:ident $b:ident $p:ident
        $sem:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!(
                    "{{\n\t.reg .b128 %atom_d, %atom_b;\n\t",
                    "mov.b128 %atom_b, {{{b0}, {b1}}};\n\t",
                    "atom", $sem, $scope, $space,
                    ".exch.L2::cache_hint.b128 %atom_d, [{a}], %atom_b, {p};\n\t",
                    "mov.b128 {{{d0}, {d1}}}, %atom_d;\n}}"
                ),
                d0 = out(reg64) $d[0],
                d1 = out(reg64) $d[1],
                a = in(reg64) $a,
                b0 = in(reg64) $b[0],
                b1 = in(reg64) $b[1],
                p = in(reg64) $p,
                options(nostack),
            )
        }
    };
    (
        cas (b128) $a:ident $d:ident $b:ident $c:ident
        $sem:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!(
                    "{{\n\t.reg .b128 %atom_d, %atom_b, %atom_c;\n\t",
                    "mov.b128 %atom_b, {{{b0}, {b1}}};\n\t",
                    "mov.b128 %atom_c, {{{c0}, {c1}}};\n\t",
                    "atom", $sem, $scope, $space, ".cas.b128 %atom_d, [{a}], %atom_b, %atom_c;\n\t",
                    "mov.b128 {{{d0}, {d1}}}, %atom_d;\n}}"
                ),
                d0 = out(reg64) $d[0],
                d1 = out(reg64) $d[1],
                a = in(reg64) $a,
                b0 = in(reg64) $b[0],
                b1 = in(reg64) $b[1],
                c0 = in(reg64) $c[0],
                c1 = in(reg64) $c[1],
                options(nostack),
            )
        }
    };
    (
        exch ($cls:ident) $ty:literal $a:ident $d:ident $b:ident
        $sem:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!("atom", $sem, $scope, $space, ".exch", $ty, " {d}, [{a}], {b};"),
                d = out($cls) $d,
                a = in(reg64) $a,
                b = in($cls) $b,
                options(nostack),
            )
        }
    };
    (
        exch_hint ($cls:ident) $ty:literal $a:ident $d:ident $b:ident $p:ident
        $sem:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!(
                    "atom", $sem, $scope, $space, ".exch.L2::cache_hint", $ty,
                    " {d}, [{a}], {b}, {p};"
                ),
                d = out($cls) $d,
                a = in(reg64) $a,
                b = in($cls) $b,
                p = in(reg64) $p,
                options(nostack),
            )
        }
    };
    (
        cas ($cls:ident) $ty:literal $a:ident $d:ident $b:ident $c:ident
        $sem:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!("atom", $sem, $scope, $space, ".cas", $ty, " {d}, [{a}], {b}, {c};"),
                d = out($cls) $d,
                a = in(reg64) $a,
                b = in($cls) $b,
                c = in($cls) $c,
                options(nostack),
            )
        }
    };
}

/// Implements [`sealed::Exch`] and [`AtomicExchange`].
///
/// `$shape` is `(reg_class) ".type"` for a single register, or `(b128)` for a 128-bit value
/// converted to and from an array of two 64-bit registers (low half first).
macro_rules! impl_exch {
    ($(
        $(#[$attr:meta])*
        impl[$($g:ident),*] $ty:ty => ($($shape:tt)*), |$v:ident| $to:expr, |$r:ident| $from:expr;
    )+) => {$(
        $(#[$attr])*
        impl<$($g),*> sealed::Exch for $ty {
            #[inline(always)]
            unsafe fn exch<
                const SEM: MemoryOrdering,
                const SCOPE: Scope,
                const SPACE: AtomicSpace,
            >(ptr: *mut Self, $v: Self) -> Self {
                let a = space_address::<SPACE, _>(ptr);
                let b = $to;
                let mut d = b;
                with_qualifiers!(atom all, emit_exch_cas!(exch $($shape)* a d b));
                let $r = d;
                $from
            }

            #[inline(always)]
            unsafe fn exch_hint<
                const SEM: MemoryOrdering,
                const SCOPE: Scope,
                const SPACE: AtomicSpace,
            >(ptr: *mut Self, $v: Self, policy: u64) -> Self {
                let a = space_address::<SPACE, _>(ptr);
                let b = $to;
                let mut d = b;
                with_qualifiers!(atom global, emit_exch_cas!(exch_hint $($shape)* a d b policy));
                let $r = d;
                $from
            }
        }

        $(#[$attr])*
        impl<$($g),*> AtomicExchange for $ty {}
    )+};
}

/// Implements [`sealed::Cas`] and [`AtomicCompareExchange`]. See [`impl_exch`].
macro_rules! impl_cas {
    ($(
        $(#[$attr:meta])*
        impl[$($g:ident),*] $ty:ty => ($($shape:tt)*), |$v:ident| $to:expr, |$r:ident| $from:expr;
    )+) => {$(
        $(#[$attr])*
        impl<$($g),*> sealed::Cas for $ty {
            #[inline(always)]
            unsafe fn cas<
                const SEM: MemoryOrdering,
                const SCOPE: Scope,
                const SPACE: AtomicSpace,
            >(ptr: *mut Self, compare: Self, new: Self) -> Self {
                let a = space_address::<SPACE, _>(ptr);
                let b = {
                    let $v = compare;
                    $to
                };
                let c = {
                    let $v = new;
                    $to
                };
                let mut d = b;
                with_qualifiers!(atom all, emit_exch_cas!(cas $($shape)* a d b c));
                let $r = d;
                $from
            }
        }

        $(#[$attr])*
        impl<$($g),*> AtomicCompareExchange for $ty {}
    )+};
}

/// Invokes `$mac!` with the entries for the types supporting both `exch` and `cas`.
macro_rules! exch_cas_types {
    ($mac:ident) => {
        $mac! {
            impl[] u32 => ((reg32) ".b32"), |v| v, |r| r;
            impl[] i32 => ((reg32) ".b32"), |v| v as u32, |r| r as i32;
            impl[] f32 => ((reg32) ".b32"), |v| v.to_bits(), |r| f32::from_bits(r);
            impl[] f16x2 => ((reg32) ".b32"), |v| v.to_bits(), |r| f16x2::from_bits(r);
            impl[] bf16x2 => ((reg32) ".b32"), |v| v.to_bits(), |r| bf16x2::from_bits(r);
            impl[] u64 => ((reg64) ".b64"), |v| v, |r| r;
            impl[] i64 => ((reg64) ".b64"), |v| v as u64, |r| r as i64;
            impl[] f64 => ((reg64) ".b64"), |v| v.to_bits(), |r| f64::from_bits(r);
            impl[T] *mut T => ((reg64) ".b64"),
                |v| v.expose_provenance() as u64,
                |r| core::ptr::with_exposed_provenance_mut(r as usize);
            impl[T] *const T => ((reg64) ".b64"),
                |v| v.expose_provenance() as u64,
                |r| core::ptr::with_exposed_provenance(r as usize);
            #[cfg(target_feature = "sm_90")]
            impl[] u128 => ((b128)), |v| [v as u64, (v >> 64) as u64],
                |r| r[0] as u128 | (r[1] as u128) << 64;
            #[cfg(target_feature = "sm_90")]
            impl[] i128 => ((b128)), |v| [v as u64, (v >> 64) as u64],
                |r| (r[0] as u128 | (r[1] as u128) << 64) as i128;
        }
    };
}

exch_cas_types!(impl_exch);
exch_cas_types!(impl_cas);

impl_cas! {
    impl[] u16 => ((reg16) ".b16"), |v| v, |r| r;
    impl[] i16 => ((reg16) ".b16"), |v| v as u16, |r| r as i16;
    impl[] f16 => ((reg16) ".b16"), |v| v.to_bits(), |r| f16::from_bits(r);
    impl[] bf16 => ((reg16) ".b16"), |v| v.to_bits(), |r| bf16::from_bits(r);
}

/// Defines the `atom_*` and `red_*` functions (with and without cache hint) of an operation.
macro_rules! rmw_fns {
    (
        $(#[doc = $doc:literal])*
        [$($generics:tt)*] $ty:ty, $op:ident,
        $atom:ident, $atom_hint:ident, $red:ident, $red_hint:ident
    ) => {
        $(#[doc = $doc])*
        ///
        /// Returns the value of `*ptr` before the operation.
        ///
        /// See the [module documentation](crate::atomic) for the meaning of `SEM`, `SCOPE` and
        /// `SPACE`.
        ///
        /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom>
        ///
        /// # Safety
        ///
        /// See the [safety contract](crate::atomic#safety-contract).
        #[inline]
        pub unsafe fn $atom<
            $($generics)*
            const SEM: MemoryOrdering,
            const SCOPE: Scope,
            const SPACE: AtomicSpace,
        >(
            ptr: *mut $ty,
            value: $ty,
        ) -> $ty {
            const { assert_qualifiers_supported(Kind::Atom, SEM, SCOPE, SPACE, false) };
            unsafe { <$ty as sealed::Rmw>::atom::<{ Op::$op }, SEM, SCOPE, SPACE>(ptr, value) }
        }

        #[doc = concat!(
            "Like [`", stringify!($atom), "`], with the L2 cache eviction policy `cache_policy` ",
            "applied to the memory access (`.L2::cache_hint`)."
        )]
        ///
        /// `SPACE` must be [`AtomicSpace::Generic`] or [`AtomicSpace::Global`].
        ///
        /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom>
        ///
        /// # Safety
        ///
        /// See the [safety contract](crate::atomic#safety-contract). In addition, `ptr` must
        /// point into global memory.
        #[inline]
        pub unsafe fn $atom_hint<
            $($generics)*
            const SEM: MemoryOrdering,
            const SCOPE: Scope,
            const SPACE: AtomicSpace,
        >(
            ptr: *mut $ty,
            value: $ty,
            cache_policy: CachePolicy,
        ) -> $ty {
            const { assert_qualifiers_supported(Kind::Atom, SEM, SCOPE, SPACE, true) };
            unsafe {
                <$ty as sealed::Rmw>::atom_hint::<{ Op::$op }, SEM, SCOPE, SPACE>(
                    ptr,
                    value,
                    cache_policy.to_bits(),
                )
            }
        }

        $(#[doc = $doc])*
        ///
        /// Unlike
        #[doc = concat!("[`", stringify!($atom), "`],")]
        /// this does not return the previous value. `SEM` must be [`MemoryOrdering::Relaxed`] or
        /// [`MemoryOrdering::Release`].
        ///
        /// See the [module documentation](crate::atomic) for the meaning of `SEM`, `SCOPE` and
        /// `SPACE`.
        ///
        /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red>
        ///
        /// # Safety
        ///
        /// See the [safety contract](crate::atomic#safety-contract).
        #[inline]
        pub unsafe fn $red<
            $($generics)*
            const SEM: MemoryOrdering,
            const SCOPE: Scope,
            const SPACE: AtomicSpace,
        >(
            ptr: *mut $ty,
            value: $ty,
        ) {
            const { assert_qualifiers_supported(Kind::Red, SEM, SCOPE, SPACE, false) };
            unsafe { <$ty as sealed::Rmw>::red::<{ Op::$op }, SEM, SCOPE, SPACE>(ptr, value) }
        }

        #[doc = concat!(
            "Like [`", stringify!($red), "`], with the L2 cache eviction policy `cache_policy` ",
            "applied to the memory access (`.L2::cache_hint`)."
        )]
        ///
        /// `SPACE` must be [`AtomicSpace::Generic`] or [`AtomicSpace::Global`].
        ///
        /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red>
        ///
        /// # Safety
        ///
        /// See the [safety contract](crate::atomic#safety-contract). In addition, `ptr` must
        /// point into global memory.
        #[inline]
        pub unsafe fn $red_hint<
            $($generics)*
            const SEM: MemoryOrdering,
            const SCOPE: Scope,
            const SPACE: AtomicSpace,
        >(
            ptr: *mut $ty,
            value: $ty,
            cache_policy: CachePolicy,
        ) {
            const { assert_qualifiers_supported(Kind::Red, SEM, SCOPE, SPACE, true) };
            unsafe {
                <$ty as sealed::Rmw>::red_hint::<{ Op::$op }, SEM, SCOPE, SPACE>(
                    ptr,
                    value,
                    cache_policy.to_bits(),
                )
            }
        }
    };
}

rmw_fns! {
    /// Atomically adds `value` to `*ptr` (`.add`).
    ///
    /// Integer addition wraps around on overflow. Floating-point addition rounds to nearest
    /// even. `f32` addition (including `Simd<f32, 2 | 4>`) flushes subnormal inputs and results to
    /// sign-preserving zero in global memory, but preserves them in shared memory.
    /// Addition of all other floating-point types preserves subnormals (`.noftz`).
    [T: AtomicAdd,] T, Add,
    atom_add, atom_add_with_cache_hint, red_add, red_add_with_cache_hint
}

rmw_fns! {
    /// Atomically replaces `*ptr` with the minimum of `*ptr` and `value` (`.min`).
    ///
    /// Integers are compared as signed or unsigned depending on `T`. The floating-point types
    /// have the same semantics as the non-atomic `min` instruction and preserve subnormals
    /// (`.noftz`). They require `sm_90` and global memory.
    [T: AtomicMinMax,] T, Min,
    atom_min, atom_min_with_cache_hint, red_min, red_min_with_cache_hint
}

rmw_fns! {
    /// Atomically replaces `*ptr` with the maximum of `*ptr` and `value` (`.max`).
    ///
    /// Integers are compared as signed or unsigned depending on `T`. The floating-point types
    /// have the same semantics as the non-atomic `max` instruction and preserve subnormals
    /// (`.noftz`). They require `sm_90` and global memory.
    [T: AtomicMinMax,] T, Max,
    atom_max, atom_max_with_cache_hint, red_max, red_max_with_cache_hint
}

rmw_fns! {
    /// Atomically replaces `*ptr` with the bitwise AND of `*ptr` and `value` (`.and`).
    [T: AtomicBitwise,] T, And,
    atom_and, atom_and_with_cache_hint, red_and, red_and_with_cache_hint
}

rmw_fns! {
    /// Atomically replaces `*ptr` with the bitwise OR of `*ptr` and `value` (`.or`).
    [T: AtomicBitwise,] T, Or,
    atom_or, atom_or_with_cache_hint, red_or, red_or_with_cache_hint
}

rmw_fns! {
    /// Atomically replaces `*ptr` with the bitwise XOR of `*ptr` and `value` (`.xor`).
    [T: AtomicBitwise,] T, Xor,
    atom_xor, atom_xor_with_cache_hint, red_xor, red_xor_with_cache_hint
}

rmw_fns! {
    /// Atomically increments `*ptr`, wrapping around to zero once it reaches `value`
    /// (`.inc.u32`): `*ptr = if *ptr >= value { 0 } else { *ptr + 1 }`.
    [] u32, Inc,
    atom_inc, atom_inc_with_cache_hint, red_inc, red_inc_with_cache_hint
}

rmw_fns! {
    /// Atomically decrements `*ptr`, wrapping around to `value` once it reaches zero or if it
    /// exceeds `value` (`.dec.u32`): `*ptr = if *ptr == 0 || *ptr > value { value } else { *ptr -
    /// 1 }`.
    [] u32, Dec,
    atom_dec, atom_dec_with_cache_hint, red_dec, red_dec_with_cache_hint
}

/// Atomically replaces `*ptr` with `value`, returning the previous value of `*ptr` (`atom.exch`).
///
/// See the [module documentation](crate::atomic) for the meaning of `SEM`, `SCOPE` and `SPACE`.
/// The 128-bit exchange of `u128` and `i128` requires `sm_90`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom>
///
/// # Safety
///
/// See the [safety contract](crate::atomic#safety-contract).
#[inline]
pub unsafe fn atom_exch<
    T: AtomicExchange,
    const SEM: MemoryOrdering,
    const SCOPE: Scope,
    const SPACE: AtomicSpace,
>(
    ptr: *mut T,
    value: T,
) -> T {
    const { assert_qualifiers_supported(Kind::Atom, SEM, SCOPE, SPACE, false) };
    unsafe { T::exch::<SEM, SCOPE, SPACE>(ptr, value) }
}

/// Like [`atom_exch`], with the L2 cache eviction policy `cache_policy` applied to the memory
/// access (`.L2::cache_hint`).
///
/// `SPACE` must be [`AtomicSpace::Generic`] or [`AtomicSpace::Global`].
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom>
///
/// # Safety
///
/// See the [safety contract](crate::atomic#safety-contract). In addition, `ptr` must point into
/// global memory.
#[inline]
pub unsafe fn atom_exch_with_cache_hint<
    T: AtomicExchange,
    const SEM: MemoryOrdering,
    const SCOPE: Scope,
    const SPACE: AtomicSpace,
>(
    ptr: *mut T,
    value: T,
    cache_policy: CachePolicy,
) -> T {
    const { assert_qualifiers_supported(Kind::Atom, SEM, SCOPE, SPACE, true) };
    unsafe { T::exch_hint::<SEM, SCOPE, SPACE>(ptr, value, cache_policy.to_bits()) }
}

/// Atomically compares `*ptr` with `compare` and, if they are bitwise equal, replaces `*ptr` with
/// `new` (`atom.cas`). Returns the previous value of `*ptr`; the exchange succeeded if and only if
/// it is bitwise equal to `compare`.
///
/// See the [module documentation](crate::atomic) for the meaning of `SEM`, `SCOPE` and `SPACE`.
/// The 128-bit compare-and-swap of `u128` and `i128` requires `sm_90`, and the 16-bit types
/// require `sm_70`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-atom>
///
/// # Safety
///
/// See the [safety contract](crate::atomic#safety-contract).
#[inline]
pub unsafe fn atom_cas<
    T: AtomicCompareExchange,
    const SEM: MemoryOrdering,
    const SCOPE: Scope,
    const SPACE: AtomicSpace,
>(
    ptr: *mut T,
    compare: T,
    new: T,
) -> T {
    const { assert_qualifiers_supported(Kind::Atom, SEM, SCOPE, SPACE, false) };
    unsafe { T::cas::<SEM, SCOPE, SPACE>(ptr, compare, new) }
}

#[cfg(target_feature = "sm_90")]
const fn assert_red_async_space_supported(space: AtomicSpace) {
    assert!(
        matches!(space, AtomicSpace::Generic | AtomicSpace::SharedCluster),
        "red.async only supports AtomicSpace::Generic and AtomicSpace::SharedCluster"
    );
}

/// Emits `red.async` with `.mbarrier::complete_tx::bytes` completion.
#[cfg(target_feature = "sm_90")]
macro_rules! emit_red_async {
    (
        $cls:ident $op:literal $ty:literal $a:ident $b:ident $mbar:ident
        $space:literal
    ) => {
        unsafe {
            asm!(
                concat!(
                    "red.async.relaxed.cluster", $space, ".mbarrier::complete_tx::bytes", $op, $ty,
                    " [{a}], {b}, [{mbar}];"
                ),
                a = in(reg64) $a,
                b = in($cls) $b,
                mbar = in(reg64) $mbar,
                options(nostack),
            )
        }
    };
}

/// Implements [`sealed::RedAsync`] and the given marker traits.
#[cfg(target_feature = "sm_90")]
macro_rules! impl_red_async {
    ($(
        $ty:ty => $cls:ident, |$v:ident| $to:expr;
            { $($op:ident => $op_lit:literal $ty_lit:literal),+ $(,)? }
            impl $($marker:ident),+;
    )+) => {$(
        impl sealed::RedAsync for $ty {
            #[inline(always)]
            unsafe fn red_async<const OP: Op, const SPACE: AtomicSpace>(
                ptr: *mut Self,
                $v: Self,
                mbar: *mut MBarrier,
            ) {
                let a = space_address::<SPACE, _>(ptr);
                let mbar = space_address::<SPACE, _>(mbar);
                let b = $to;
                $(if const { matches!(OP, Op::$op) } {
                    asm_variants!(emit_red_async!($cls $op_lit $ty_lit a b mbar);
                        SPACE => {
                            AtomicSpace::Generic => "",
                            AtomicSpace::SharedCluster => ".shared::cluster",
                        },
                    )
                } else)+ {
                    unreachable!()
                }
            }
        }

        $(impl $marker for $ty {})+
    )+};
}

#[cfg(target_feature = "sm_90")]
impl_red_async! {
    u32 => reg32, |v| v;
        {
            Add => ".add" ".u32",
            Min => ".min" ".u32",
            Max => ".max" ".u32",
            And => ".and" ".b32",
            Or => ".or" ".b32",
            Xor => ".xor" ".b32",
            Inc => ".inc" ".u32",
            Dec => ".dec" ".u32",
        }
        impl RedAsyncAdd, RedAsyncMinMax, RedAsyncBitwise;
    i32 => reg32, |v| v as u32;
        {
            Add => ".add" ".s32",
            Min => ".min" ".s32",
            Max => ".max" ".s32",
            And => ".and" ".b32",
            Or => ".or" ".b32",
            Xor => ".xor" ".b32",
        }
        impl RedAsyncAdd, RedAsyncMinMax, RedAsyncBitwise;
    u64 => reg64, |v| v;
        { Add => ".add" ".u64" }
        impl RedAsyncAdd;
}

/// Defines a `red_async_*` function.
#[cfg(target_feature = "sm_90")]
macro_rules! red_async_fn {
    ($(#[doc = $doc:literal])* [$($generics:tt)*] $ty:ty, $op:ident, $name:ident) => {
        $(#[doc = $doc])*
        ///
        /// The reduction is performed asynchronously on `*ptr` in the shared memory of another CTA
        /// of the cluster. Upon its completion, a complete-tx operation is performed on the
        /// mbarrier `mbar` in the same CTA as `ptr`, with a transaction count of
        /// `size_of::<T>()` bytes (`.mbarrier::complete_tx::bytes`). The receiving CTA typically
        /// registers the number of bytes it expects with [`MBarrier::expect_tx`], then waits on
        /// the barrier to observe the reductions.
        ///
        /// The reduction is a relaxed memory operation at cluster scope, performed in the generic
        /// proxy. The complete-tx operation on the mbarrier has release semantics at cluster
        /// scope.
        ///
        /// `SPACE` must be [`AtomicSpace::Generic`] or [`AtomicSpace::SharedCluster`]; it applies
        /// to both `ptr` and `mbar`.
        ///
        /// Requires `sm_90`.
        ///
        /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red-async>
        ///
        /// # Safety
        ///
        /// - `ptr` and `mbar` must be generic addresses in the shared memory of the same CTA of
        ///   the executing cluster (e.g. obtained from
        ///   [`map_to_cta`](crate::cluster::map_to_cta)), which must not be the executing CTA.
        /// - `ptr` must be valid for reads and writes of `T` and aligned to `size_of::<T>()`, and
        ///   the reduction must not cause a data race.
        /// - `mbar` must point to an initialized mbarrier (see the
        ///   [safety contract](MBarrier#safety-contract-of-all-operations)).
        #[inline]
        pub unsafe fn $name<$($generics)* const SPACE: AtomicSpace>(
            ptr: *mut $ty,
            value: $ty,
            mbar: *mut MBarrier,
        ) {
            const { assert_red_async_space_supported(SPACE) };
            unsafe {
                <$ty as sealed::RedAsync>::red_async::<{ Op::$op }, SPACE>(ptr, value, mbar)
            }
        }
    };
}

#[cfg(target_feature = "sm_90")]
red_async_fn! {
    /// Asynchronously adds `value` to `*ptr` (`red.async.add`).
    [T: RedAsyncAdd,] T, Add, red_async_add
}

#[cfg(target_feature = "sm_90")]
red_async_fn! {
    /// Asynchronously replaces `*ptr` with the minimum of `*ptr` and `value`
    /// (`red.async.min`). Integers are compared as signed or unsigned depending on `T`.
    [T: RedAsyncMinMax,] T, Min, red_async_min
}

#[cfg(target_feature = "sm_90")]
red_async_fn! {
    /// Asynchronously replaces `*ptr` with the maximum of `*ptr` and `value`
    /// (`red.async.max`). Integers are compared as signed or unsigned depending on `T`.
    [T: RedAsyncMinMax,] T, Max, red_async_max
}

#[cfg(target_feature = "sm_90")]
red_async_fn! {
    /// Asynchronously replaces `*ptr` with the bitwise AND of `*ptr` and `value`
    /// (`red.async.and`).
    [T: RedAsyncBitwise,] T, And, red_async_and
}

#[cfg(target_feature = "sm_90")]
red_async_fn! {
    /// Asynchronously replaces `*ptr` with the bitwise OR of `*ptr` and `value`
    /// (`red.async.or`).
    [T: RedAsyncBitwise,] T, Or, red_async_or
}

#[cfg(target_feature = "sm_90")]
red_async_fn! {
    /// Asynchronously replaces `*ptr` with the bitwise XOR of `*ptr` and `value`
    /// (`red.async.xor`).
    [T: RedAsyncBitwise,] T, Xor, red_async_xor
}

#[cfg(target_feature = "sm_90")]
red_async_fn! {
    /// Asynchronously increments `*ptr`, wrapping around to zero once it reaches `value`
    /// (`red.async.inc.u32`): `*ptr = if *ptr >= value { 0 } else { *ptr + 1 }`.
    [] u32, Inc, red_async_inc
}

#[cfg(target_feature = "sm_90")]
red_async_fn! {
    /// Asynchronously decrements `*ptr`, wrapping around to `value` once it reaches zero or if
    /// it exceeds `value` (`red.async.dec.u32`):
    /// `*ptr = if *ptr == 0 || *ptr > value { value } else { *ptr - 1 }`.
    [] u32, Dec, red_async_dec
}

/// Emits `red.async.release` or `multimem.red.async.release`.
#[cfg(target_feature = "sm_100")]
macro_rules! emit_red_async_release {
    (
        $cls:ident $ty:literal $a:ident $b:ident
        $prefix:literal, $scope:literal, $space:literal
    ) => {
        unsafe {
            asm!(
                concat!($prefix, ".release", $scope, $space, ".add", $ty, " [{a}], {b};"),
                a = in(reg64) $a,
                b = in($cls) $b,
                options(nostack),
            )
        }
    };
}

#[cfg(target_feature = "sm_100")]
macro_rules! impl_red_async_release {
    ($($ty:ty => $cls:ident $ty_lit:literal, |$v:ident| $to:expr;)+) => {$(
        impl sealed::RedAsyncRelease for $ty {
            #[inline(always)]
            unsafe fn red_async_release<
                const SCOPE: Scope,
                const TARGET: AsyncStoreTarget,
                const SPACE: AtomicSpace,
            >(
                ptr: *mut Self,
                $v: Self,
            ) {
                let a = space_address::<SPACE, _>(ptr);
                let b = $to;
                asm_variants!(emit_red_async_release!($cls $ty_lit a b);
                    TARGET => {
                        AsyncStoreTarget::Global => "red.async",
                        AsyncStoreTarget::Multimem => "multimem.red.async",
                        AsyncStoreTarget::Mmio => "red.async.mmio",
                    },
                    SCOPE => {
                        Scope::Gpu => ".gpu",
                        Scope::System => ".sys",
                    },
                    SPACE => {
                        AtomicSpace::Generic => "",
                        AtomicSpace::Global => ".global",
                    },
                )
            }
        }

        impl RedAsyncReleaseAdd for $ty {}
    )+};
}

// `multimem.red.async` has no `.s64` type; two's complement addition is the same operation as
// `.add.u64`.
#[cfg(target_feature = "sm_100")]
impl_red_async_release! {
    u32 => reg32 ".u32", |v| v;
    i32 => reg32 ".s32", |v| v as u32;
    u64 => reg64 ".u64", |v| v;
    i64 => reg64 ".u64", |v| v as u64;
}

/// Asynchronously adds `value` to `*ptr` in global memory, with `.release` semantics at scope
/// `SCOPE` (`red.async.release.add`).
///
/// `red.async` is non-blocking: the reduction is initiated and may complete after this function
/// returns. It is a strong memory operation with release semantics at `SCOPE`, which must be
/// [`Scope::Gpu`] or [`Scope::System`]. `TARGET` selects between a regular global address, a
/// multimem address (`multimem.red.async`, which performs the reduction on the memory location on
/// every GPU that the address refers to) and an MMIO operation (`.mmio`); the latter requires
/// [`Scope::System`]. `red.async` is performed in the generic proxy, and its completion
/// synchronizes with the end of the CTA.
///
/// `SPACE` must be [`AtomicSpace::Generic`] or [`AtomicSpace::Global`].
///
/// Requires `sm_100`.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red-async>
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-multimem-red-async>
///
/// # Safety
///
/// - `ptr` must be valid for reads and writes of `T`, aligned, and point into global memory.
///   With [`AsyncStoreTarget::Multimem`] it must be a multimem address (which must not be
///   accessed by anything other than `multimem.*` operations); with
///   [`AsyncStoreTarget::Mmio`] it must refer to memory-mapped I/O.
/// - The reduction must not cause a data race.
#[cfg(target_feature = "sm_100")]
#[inline]
pub unsafe fn red_async_add_release<
    T: RedAsyncReleaseAdd,
    const SCOPE: Scope,
    const TARGET: AsyncStoreTarget,
    const SPACE: AtomicSpace,
>(
    ptr: *mut T,
    value: T,
) {
    const {
        assert_async_store_supported(SCOPE, TARGET);
        assert_global_space(SPACE);
    };
    unsafe { T::red_async_release::<SCOPE, TARGET, SPACE>(ptr, value) }
}
