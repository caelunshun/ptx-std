// --- LLM-generated --- //
//! Instantiates the `atomic` wrappers of `ptx_std` in kernels, so that the inline PTX they
//! generate can be validated by `ptxas` for several targets. No GPU is needed.
//!
//! Run `examples/atomic_ptxas/check.sh [arch...]` from anywhere; it emits the PTX of this crate
//! for each target (default: `sm_80 sm_90 sm_100a sm_103a`) and assembles it with `ptxas`.

#![no_std]
#![feature(abi_ptx)]
#![allow(clippy::missing_safety_doc)]

use ptx_std::{
    MemoryOrdering as M, Scope as S, atomic::*, bf16, bf16x2, f16, f16x2, memory::CachePolicy,
};

/// Expands `$cb!(x y ...)` for every element of the cartesian product of the given lists.
macro_rules! product {
    (@call $cb:ident [$($acc:tt)*]) => {
        $cb!($($acc)*);
    };
    (@each $cb:ident $acc:tt $rest:tt $($x:tt)*) => {
        $(product!(@push $cb $acc $x $rest);)*
    };
    (@push $cb:ident [$($acc:tt)*] $x:tt $rest:tt) => {
        product!($cb [$($acc)* $x] $rest);
    };
    ($cb:ident $acc:tt []) => {
        product!(@call $cb $acc);
    };
    ($cb:ident $acc:tt [[$($x:tt)*] $($rest:tt)*]) => {
        product!(@each $cb $acc [$($rest)*] $($x)*);
    };
}

/// All qualifier combinations of `atom.add.u32`/`red.add.u32` and `atom.exch`/`atom.cas`.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn all_qualifiers(p: *mut u32, v: u32, policy: u64) {
    let policy = CachePolicy::from_bits(policy);
    macro_rules! atom {
        ($sem:ident $scope:ident $space:ident) => {
            unsafe {
                atom_add::<u32, { M::$sem }, { S::$scope }, { AtomicSpace::$space }>(p, v);
                atom_exch::<u32, { M::$sem }, { S::$scope }, { AtomicSpace::$space }>(p, v);
                atom_cas::<u32, { M::$sem }, { S::$scope }, { AtomicSpace::$space }>(p, v, v);
            }
        };
    }
    macro_rules! atom_hint {
        ($sem:ident $scope:ident $space:ident) => {
            unsafe {
                atom_add_with_cache_hint::<u32, { M::$sem }, { S::$scope }, { AtomicSpace::$space }>(
                    p, v, policy,
                );
                atom_exch_with_cache_hint::<u32, { M::$sem }, { S::$scope }, { AtomicSpace::$space }>(
                    p, v, policy,
                );
            }
        };
    }
    macro_rules! red {
        ($sem:ident $scope:ident $space:ident) => {
            unsafe { red_add::<u32, { M::$sem }, { S::$scope }, { AtomicSpace::$space }>(p, v) }
        };
    }
    macro_rules! red_hint {
        ($sem:ident $scope:ident $space:ident) => {
            unsafe {
                red_add_with_cache_hint::<u32, { M::$sem }, { S::$scope }, { AtomicSpace::$space }>(
                    p, v, policy,
                )
            }
        };
    }
    #[cfg(target_feature = "sm_90")]
    {
        product!(atom [] [[Relaxed Acquire Release AcqRel] [Cta Cluster Gpu System] [Generic Global SharedCta SharedCluster]]);
        product!(atom_hint [] [[Relaxed Acquire Release AcqRel] [Cta Cluster Gpu System] [Generic Global]]);
        product!(red [] [[Relaxed Release] [Cta Cluster Gpu System] [Generic Global SharedCta SharedCluster]]);
        product!(red_hint [] [[Relaxed Release] [Cta Cluster Gpu System] [Generic Global]]);
    }
    #[cfg(not(target_feature = "sm_90"))]
    {
        product!(atom [] [[Relaxed Acquire Release AcqRel] [Cta Gpu System] [Generic Global SharedCta]]);
        product!(atom_hint [] [[Relaxed Acquire Release AcqRel] [Cta Gpu System] [Generic Global]]);
        product!(red [] [[Relaxed Release] [Cta Gpu System] [Generic Global SharedCta]]);
        product!(red_hint [] [[Relaxed Release] [Cta Gpu System] [Generic Global]]);
    }
}

/// Instantiates `atom_$op`, `red_$op` and their cache hint variants for `$ty`, in the given spaces.
macro_rules! rmw_kernel {
    ($name:ident, $ty:ty, [$($space:ident)*], $($op:ident)*) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(p: *mut $ty, out: *mut $ty, policy: u64) {
            let policy = CachePolicy::from_bits(policy);
            macro_rules! op {
                (add $space2:ident) => {
                    op!(@ atom_add atom_add_with_cache_hint red_add red_add_with_cache_hint $space2)
                };
                (min $space2:ident) => {
                    op!(@ atom_min atom_min_with_cache_hint red_min red_min_with_cache_hint $space2)
                };
                (max $space2:ident) => {
                    op!(@ atom_max atom_max_with_cache_hint red_max red_max_with_cache_hint $space2)
                };
                (and $space2:ident) => {
                    op!(@ atom_and atom_and_with_cache_hint red_and red_and_with_cache_hint $space2)
                };
                (or $space2:ident) => {
                    op!(@ atom_or atom_or_with_cache_hint red_or red_or_with_cache_hint $space2)
                };
                (xor $space2:ident) => {
                    op!(@ atom_xor atom_xor_with_cache_hint red_xor red_xor_with_cache_hint $space2)
                };
                (inc $space2:ident) => {
                    op!(@u32 atom_inc atom_inc_with_cache_hint red_inc red_inc_with_cache_hint $space2)
                };
                (dec $space2:ident) => {
                    op!(@u32 atom_dec atom_dec_with_cache_hint red_dec red_dec_with_cache_hint $space2)
                };
                (@u32 $atom:ident $atom_hint:ident $red:ident $red_hint:ident $space2:ident) => {
                    unsafe {
                        let v = *out;
                        *out = $atom::<{ M::AcqRel }, { S::Gpu }, { AtomicSpace::$space2 }>(p, v);
                        *out = $atom_hint::<{ M::Relaxed }, { S::Gpu }, { AtomicSpace::Global }>(
                            p, v, policy,
                        );
                        $red::<{ M::Release }, { S::Gpu }, { AtomicSpace::$space2 }>(p, v);
                        $red_hint::<{ M::Relaxed }, { S::Gpu }, { AtomicSpace::Generic }>(
                            p, v, policy,
                        );
                    }
                };
                (@ $atom:ident $atom_hint:ident $red:ident $red_hint:ident $space2:ident) => {
                    unsafe {
                        let v = *out;
                        *out = $atom::<$ty, { M::AcqRel }, { S::Gpu }, { AtomicSpace::$space2 }>(p, v);
                        *out = $atom_hint::<$ty, { M::Relaxed }, { S::Gpu }, { AtomicSpace::Global }>(
                            p, v, policy,
                        );
                        $red::<$ty, { M::Release }, { S::Gpu }, { AtomicSpace::$space2 }>(p, v);
                        $red_hint::<$ty, { M::Relaxed }, { S::Gpu }, { AtomicSpace::Generic }>(
                            p, v, policy,
                        );
                    }
                };
            }
            product!(op [] [[$($op)*] [$($space)*]]);
        }
    };
}

rmw_kernel!(rmw_u32, u32, [Generic Global SharedCta], add min max and or xor inc dec);
rmw_kernel!(rmw_i32, i32, [Generic Global SharedCta], add min max and or xor);
rmw_kernel!(rmw_u64, u64, [Generic Global SharedCta], add min max and or xor);
rmw_kernel!(rmw_i64, i64, [Generic Global SharedCta], add min max and or xor);
rmw_kernel!(rmw_f32, f32, [Generic Global SharedCta], add);
rmw_kernel!(rmw_f64, f64, [Generic Global SharedCta], add);
rmw_kernel!(rmw_f16, f16, [Generic Global SharedCta], add);
rmw_kernel!(rmw_f16x2, f16x2, [Generic Global SharedCta], add);

#[cfg(target_feature = "sm_90")]
mod sm90 {
    use super::*;

    rmw_kernel!(rmw_bf16, bf16, [Generic Global SharedCta SharedCluster], add);
    rmw_kernel!(rmw_bf16x2, bf16x2, [Generic Global SharedCta SharedCluster], add);
    rmw_kernel!(rmw_f32_cluster, f32, [SharedCluster], add);
    rmw_kernel!(rmw_u32_cluster, u32, [SharedCluster], add min max and or xor inc dec);
    rmw_kernel!(rmw_v2_f16, [f16; 2], [Generic Global], add min max);
    rmw_kernel!(rmw_v4_f16, [f16; 4], [Generic Global], add min max);
    rmw_kernel!(rmw_v8_f16, [f16; 8], [Generic Global], add min max);
    rmw_kernel!(rmw_v2_bf16, [bf16; 2], [Generic Global], add min max);
    rmw_kernel!(rmw_v4_bf16, [bf16; 4], [Generic Global], add min max);
    rmw_kernel!(rmw_v8_bf16, [bf16; 8], [Generic Global], add min max);
    rmw_kernel!(rmw_v2_f16x2, [f16x2; 2], [Generic Global], add min max);
    rmw_kernel!(rmw_v4_f16x2, [f16x2; 4], [Generic Global], add min max);
    rmw_kernel!(rmw_v2_bf16x2, [bf16x2; 2], [Generic Global], add min max);
    rmw_kernel!(rmw_v4_bf16x2, [bf16x2; 4], [Generic Global], add min max);
    rmw_kernel!(rmw_v2_f32, [f32; 2], [Generic Global], add);
    rmw_kernel!(rmw_v4_f32, [f32; 4], [Generic Global], add);
}

/// Instantiates `atom_exch` (if `exch` is given) and `atom_cas` for `$ty`.
macro_rules! exch_cas_kernel {
    ($name:ident, $ty:ty, $spaces:tt $(, $exch:ident)?) => {
        exch_cas_kernel!(@kernel $name, $ty, $spaces, $spaces $(, $exch)?);
    };
    (@kernel $name:ident, $ty:ty, [$($space:ident)*], $spaces:tt $(, $exch:ident)?) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "ptx-kernel" fn $name(p: *mut $ty, out: *mut $ty, policy: u64) {
            let policy = CachePolicy::from_bits(policy);
            let _ = policy;
            unsafe {
                let v = *out;
                $(
                    *out = atom_cas::<$ty, { M::Acquire }, { S::System }, { AtomicSpace::$space }>(
                        p, v, *out,
                    );
                )*
                $(
                    exch_cas_kernel!(@$exch $ty, p, v, out, policy, $spaces);
                )?
            }
        }
    };
    (@exch $ty:ty, $p:ident, $v:ident, $out:ident, $policy:ident, [$($space:ident)*]) => {
        $(
            *$out = atom_exch::<$ty, { M::Release }, { S::Cta }, { AtomicSpace::$space }>($p, $v);
        )*
        *$out = atom_exch_with_cache_hint::<$ty, { M::Relaxed }, { S::Gpu }, { AtomicSpace::Global }>(
            $p, $v, $policy,
        );
    };
}

exch_cas_kernel!(exch_cas_u16, u16, [Generic Global SharedCta]);
exch_cas_kernel!(exch_cas_i16, i16, [Generic Global SharedCta]);
exch_cas_kernel!(exch_cas_f16, f16, [Generic Global SharedCta]);
exch_cas_kernel!(exch_cas_bf16, bf16, [Generic Global SharedCta]);
exch_cas_kernel!(exch_cas_u32, u32, [Generic Global SharedCta], exch);
exch_cas_kernel!(exch_cas_i32, i32, [Generic Global SharedCta], exch);
exch_cas_kernel!(exch_cas_f32, f32, [Generic Global SharedCta], exch);
exch_cas_kernel!(exch_cas_f16x2, f16x2, [Generic Global SharedCta], exch);
exch_cas_kernel!(exch_cas_bf16x2, bf16x2, [Generic Global SharedCta], exch);
exch_cas_kernel!(exch_cas_u64, u64, [Generic Global SharedCta], exch);
exch_cas_kernel!(exch_cas_i64, i64, [Generic Global SharedCta], exch);
exch_cas_kernel!(exch_cas_f64, f64, [Generic Global SharedCta], exch);
exch_cas_kernel!(exch_cas_ptr_mut, *mut u8, [Generic Global SharedCta], exch);
exch_cas_kernel!(exch_cas_ptr_const, *const u8, [Generic Global SharedCta], exch);
#[cfg(target_feature = "sm_90")]
exch_cas_kernel!(exch_cas_u128, u128, [Generic Global SharedCta SharedCluster], exch);
#[cfg(target_feature = "sm_90")]
exch_cas_kernel!(exch_cas_i128, i128, [Generic Global SharedCta SharedCluster], exch);

#[cfg(target_feature = "sm_90")]
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn red_async_cluster(
    p32: *mut u32,
    pi32: *mut i32,
    p64: *mut u64,
    v: u32,
    mbar: *mut ptx_std::MBarrier,
) {
    macro_rules! each {
        ($space:ident) => {
            unsafe {
                red_async_add::<u32, { AtomicSpace::$space }>(p32, v, mbar);
                red_async_add::<i32, { AtomicSpace::$space }>(pi32, v as i32, mbar);
                red_async_add::<u64, { AtomicSpace::$space }>(p64, v as u64, mbar);
                red_async_min::<u32, { AtomicSpace::$space }>(p32, v, mbar);
                red_async_min::<i32, { AtomicSpace::$space }>(pi32, v as i32, mbar);
                red_async_max::<u32, { AtomicSpace::$space }>(p32, v, mbar);
                red_async_max::<i32, { AtomicSpace::$space }>(pi32, v as i32, mbar);
                red_async_and::<u32, { AtomicSpace::$space }>(p32, v, mbar);
                red_async_or::<i32, { AtomicSpace::$space }>(pi32, v as i32, mbar);
                red_async_xor::<u32, { AtomicSpace::$space }>(p32, v, mbar);
                red_async_inc::<{ AtomicSpace::$space }>(p32, v, mbar);
                red_async_dec::<{ AtomicSpace::$space }>(p32, v, mbar);
            }
        };
    }
    each!(Generic);
    each!(SharedCluster);
}

#[cfg(target_feature = "sm_100")]
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn red_async_release(
    p32: *mut u32,
    pi32: *mut i32,
    p64: *mut u64,
    pi64: *mut i64,
    v: u32,
) {
    use ptx_std::memory::AsyncStoreTarget as T;
    macro_rules! each {
        ($scope:ident $target:ident $space:ident) => {
            unsafe {
                red_async_add_release::<u32, { S::$scope }, { T::$target }, { AtomicSpace::$space }>(
                    p32, v,
                );
                red_async_add_release::<i32, { S::$scope }, { T::$target }, { AtomicSpace::$space }>(
                    pi32,
                    v as i32,
                );
                red_async_add_release::<u64, { S::$scope }, { T::$target }, { AtomicSpace::$space }>(
                    p64,
                    v as u64,
                );
                red_async_add_release::<i64, { S::$scope }, { T::$target }, { AtomicSpace::$space }>(
                    pi64,
                    v as i64,
                );
            }
        };
    }
    product!(each [] [[Gpu System] [Global Multimem] [Generic Global]]);
    product!(each [] [[System] [Mmio] [Generic Global]]);
}
