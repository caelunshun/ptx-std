// --- LLM-generated --- //
//! Conversions of the narrow formats (FP8, FP6, FP4, `ue8m0` and `s2f6`) for every width, and for
//! the lengths 3, 6 and 12 (of which 3 is not a multiple of two or four, and 6 is not a multiple
//! of four) through local aliases. Each method group is a kernel named `<type>_<group>`,
//! instantiated under the same `cfg` as in the library with every combination of const
//! parameters, so that the `cfg`s must be repeated by the invocations below.

// The types and aliases are only used on the targets that have their conversions.
#![allow(unused_imports, dead_code, non_camel_case_types)]

use ptx_std::{
    Overflow, Quads, Relu, Rounding, bf16, bf16x2, bf16x4, bf16x8, bf16x16, bf16x32, bf16xN,
    f4e2m1, f4e2m1x2, f4e2m1x4, f4e2m1x8, f4e2m1x16, f4e2m1x32, f4e2m1xN, f6e2m3, f6e2m3x2,
    f6e2m3x4, f6e2m3x8, f6e2m3x16, f6e2m3x32, f6e2m3xN, f6e3m2, f6e3m2x2, f6e3m2x4, f6e3m2x8,
    f6e3m2x16, f6e3m2x32, f6e3m2xN, f8e4m3, f8e4m3x2, f8e4m3x4, f8e4m3x8, f8e4m3x16, f8e4m3x32,
    f8e4m3xN, f8e5m2, f8e5m2x2, f8e5m2x4, f8e5m2x8, f8e5m2x16, f8e5m2x32, f8e5m2xN, f8e8m0,
    f8e8m0x2, f8e8m0x4, f8e8m0x8, f8e8m0x16, f8e8m0x32, f8e8m0xN, f16, f16x2, f16x4, f16x8, f16x16,
    f16x32, f16xN, f32x2, f32x4, f32x8, f32x16, f32x32, f32xN, s2f6, s2f6x2, s2f6x4, s2f6x8,
    s2f6x16, s2f6x32, s2f6xN,
};

/// Defines a kernel `<ty>_<suffix>` with the given arguments and body, available when `cfg` holds.
macro_rules! kernel {
    ($ty:ident, $suffix:literal, $cfg:meta, |$($arg:ident: $argty:ty),*| $body:block) => {
        #[cfg($cfg)]
        const _: () = {
            #[unsafe(export_name = concat!(stringify!($ty), "_", $suffix))]
            pub unsafe extern "ptx-kernel" fn kernel($($arg: $argty),*) {
                unsafe { $body }
            }
        };
    };
}

/// Instantiates the conversions of an FP8, FP6 or FP4 type `$ty` of one width, where `$f32`,
/// `$f16`, `$bf16` and `$scales` are the types of the same width.
macro_rules! cvt_kernels {
    (
        $ty:ident, f32: $f32:ty, f16: $f16:ty, bf16: $bf16:ty, scales: $scales:ty,
        basic: $basic:meta, from_f16: $from_f16:meta, newer: $newer:meta
    ) => {
        kernel!(
            $ty,
            "from_f32",
            $basic,
            |x: *const $f32, out: *mut [$ty; 2]| {
                let x = *x;
                *out = [
                    <$ty>::from_f32::<{ Relu::Disabled }>(x),
                    <$ty>::from_f32::<{ Relu::Enabled }>(x),
                ];
            }
        );
        kernel!(
            $ty,
            "from_f16",
            $from_f16,
            |x: *const $f16, out: *mut [$ty; 2]| {
                let x = *x;
                *out = [
                    <$ty>::from_f16::<{ Relu::Disabled }>(x),
                    <$ty>::from_f16::<{ Relu::Enabled }>(x),
                ];
            }
        );
        kernel!(
            $ty,
            "from_bf16",
            $newer,
            |x: *const $bf16, out: *mut [$ty; 2]| {
                let x = *x;
                *out = [
                    <$ty>::from_bf16::<{ Relu::Disabled }>(x),
                    <$ty>::from_bf16::<{ Relu::Enabled }>(x),
                ];
            }
        );
        kernel!(
            $ty,
            "to_f16",
            $basic,
            |v: *const $ty, out: *mut [$f16; 2]| {
                let v = *v;
                *out = [
                    v.to_f16::<{ Relu::Disabled }>(),
                    v.to_f16::<{ Relu::Enabled }>(),
                ];
            }
        );
        kernel!(
            $ty,
            "to_f32",
            $basic,
            |v: *const $ty, out: *mut [$f32; 2]| {
                let v = *v;
                *out = [
                    v.to_f32::<{ Relu::Disabled }>(),
                    v.to_f32::<{ Relu::Enabled }>(),
                ];
            }
        );
        kernel!(
            $ty,
            "to_bf16",
            $newer,
            |v: *const $ty, scales: *const $scales, out: *mut [$bf16; 6]| {
                let v = *v;
                let scales = *scales;
                *out = [
                    v.to_bf16::<{ Relu::Disabled }>(),
                    v.to_bf16::<{ Relu::Enabled }>(),
                    v.to_bf16_scaled::<{ Relu::Disabled }, { Overflow::Infinity }>(scales),
                    v.to_bf16_scaled::<{ Relu::Enabled }, { Overflow::Infinity }>(scales),
                    v.to_bf16_scaled::<{ Relu::Disabled }, { Overflow::SaturateFinite }>(scales),
                    v.to_bf16_scaled::<{ Relu::Enabled }, { Overflow::SaturateFinite }>(scales),
                ];
            }
        );
        kernel!(
            $ty,
            "from",
            $basic,
            |v: *const $ty, out_f16: *mut $f16, out_f32: *mut $f32| {
                let v = *v;
                *out_f16 = <$f16>::from(v);
                *out_f32 = <$f32>::from(v);
            }
        );
        kernel!(
            $ty,
            "from_bf16_widening",
            $newer,
            |v: *const $ty, out: *mut $bf16| {
                *out = <$bf16>::from(*v);
            }
        );
    };
}

/// Instantiates `from_f32_stochastic` of an FP8, FP6 or FP4 type `$ty` of `$len` elements.
macro_rules! stochastic_kernel {
    ($ty:ident, f32: $f32:ty, len: $len:literal, stochastic: $stochastic:meta) => {
        kernel!(
            $ty,
            "from_f32_stochastic",
            $stochastic,
            |x: *const $f32, rbits: *const Quads<u32, $len>, out: *mut [$ty; 2]| {
                let x = *x;
                let rbits = *rbits;
                *out = [
                    <$ty>::from_f32_stochastic::<{ Relu::Disabled }>(x, rbits),
                    <$ty>::from_f32_stochastic::<{ Relu::Enabled }>(x, rbits),
                ];
            }
        );
    };
}

/// Instantiates the conversions of an FP8, FP6 or FP4 format for all widths.
macro_rules! narrow_format {
    (
        scalar: $scalar:ident,
        x2: $x2:ident, x4: $x4:ident, x8: $x8:ident, x16: $x16:ident, x32: $x32:ident,
        xn: $xn:ident, $(x3: $x3:ident,)? x6: $x6:ident, x12: $x12:ident,
        basic: $basic:meta,
        from_f16: $from_f16:meta,
        newer: $newer:meta,
        stochastic: $stochastic:meta $(,)?
    ) => {
        $(type $x3 = $xn<3>;)?
        type $x6 = $xn<6>;
        type $x12 = $xn<12>;
        cvt_kernels!($scalar, f32: f32, f16: f16, bf16: bf16, scales: f8e8m0,
            basic: $basic, from_f16: $from_f16, newer: $newer);
        cvt_kernels!($x2, f32: f32x2, f16: f16x2, bf16: bf16x2, scales: f8e8m0x2,
            basic: $basic, from_f16: $from_f16, newer: $newer);
        cvt_kernels!($x4, f32: f32x4, f16: f16x4, bf16: bf16x4, scales: f8e8m0x4,
            basic: $basic, from_f16: $from_f16, newer: $newer);
        cvt_kernels!($x8, f32: f32x8, f16: f16x8, bf16: bf16x8, scales: f8e8m0x8,
            basic: $basic, from_f16: $from_f16, newer: $newer);
        cvt_kernels!($x16, f32: f32x16, f16: f16x16, bf16: bf16x16, scales: f8e8m0x16,
            basic: $basic, from_f16: $from_f16, newer: $newer);
        cvt_kernels!($x32, f32: f32x32, f16: f16x32, bf16: bf16x32, scales: f8e8m0x32,
            basic: $basic, from_f16: $from_f16, newer: $newer);
        $(
            cvt_kernels!($x3, f32: f32xN<3>, f16: f16xN<3>, bf16: bf16xN<3>, scales: f8e8m0xN<3>,
                basic: $basic, from_f16: $from_f16, newer: $newer);
            stochastic_kernel!($x3, f32: f32xN<3>, len: 3, stochastic: $stochastic);
        )?
        cvt_kernels!($x6, f32: f32xN<6>, f16: f16xN<6>, bf16: bf16xN<6>, scales: f8e8m0xN<6>,
            basic: $basic, from_f16: $from_f16, newer: $newer);
        cvt_kernels!($x12, f32: f32xN<12>, f16: f16xN<12>, bf16: bf16xN<12>, scales: f8e8m0xN<12>,
            basic: $basic, from_f16: $from_f16, newer: $newer);
        stochastic_kernel!($x2, f32: f32x2, len: 2, stochastic: $stochastic);
        stochastic_kernel!($x4, f32: f32x4, len: 4, stochastic: $stochastic);
        stochastic_kernel!($x6, f32: f32xN<6>, len: 6, stochastic: $stochastic);
        stochastic_kernel!($x8, f32: f32x8, len: 8, stochastic: $stochastic);
        stochastic_kernel!($x12, f32: f32xN<12>, len: 12, stochastic: $stochastic);
        stochastic_kernel!($x16, f32: f32x16, len: 16, stochastic: $stochastic);
        stochastic_kernel!($x32, f32: f32x32, len: 32, stochastic: $stochastic);
    };
}

narrow_format! {
    scalar: f8e4m3,
    x2: f8e4m3x2, x4: f8e4m3x4, x8: f8e4m3x8, x16: f8e4m3x16, x32: f8e4m3x32,
    xn: f8e4m3xN, x3: f8e4m3x3, x6: f8e4m3x6, x12: f8e4m3x12,
    basic: target_feature = "sm_89",
    from_f16: target_feature = "sm_89",
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
}
narrow_format! {
    scalar: f8e5m2,
    x2: f8e5m2x2, x4: f8e5m2x4, x8: f8e5m2x8, x16: f8e5m2x16, x32: f8e5m2x32,
    xn: f8e5m2xN, x3: f8e5m2x3, x6: f8e5m2x6, x12: f8e5m2x12,
    basic: target_feature = "sm_89",
    from_f16: target_feature = "sm_89",
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
}
narrow_format! {
    scalar: f6e2m3,
    x2: f6e2m3x2, x4: f6e2m3x4, x8: f6e2m3x8, x16: f6e2m3x16, x32: f6e2m3x32,
    xn: f6e2m3xN, x3: f6e2m3x3, x6: f6e2m3x6, x12: f6e2m3x12,
    basic: any(
        target_feature = "sm_100f",
        target_feature = "sm_101f",
        target_feature = "sm_110f",
        target_feature = "sm_120f"
    ),
    from_f16: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
}
narrow_format! {
    scalar: f6e3m2,
    x2: f6e3m2x2, x4: f6e3m2x4, x8: f6e3m2x8, x16: f6e3m2x16, x32: f6e3m2x32,
    xn: f6e3m2xN, x3: f6e3m2x3, x6: f6e3m2x6, x12: f6e3m2x12,
    basic: any(
        target_feature = "sm_100f",
        target_feature = "sm_101f",
        target_feature = "sm_110f",
        target_feature = "sm_120f"
    ),
    from_f16: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
}
narrow_format! {
    scalar: f4e2m1,
    x2: f4e2m1x2, x4: f4e2m1x4, x8: f4e2m1x8, x16: f4e2m1x16, x32: f4e2m1x32,
    xn: f4e2m1xN, x6: f4e2m1x6, x12: f4e2m1x12,
    basic: any(
        target_feature = "sm_100f",
        target_feature = "sm_101f",
        target_feature = "sm_110f",
        target_feature = "sm_120f"
    ),
    from_f16: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    newer: any(target_feature = "sm_100f", target_feature = "sm_110f", target_feature = "sm_120f"),
    stochastic: any(target_feature = "sm_100a", target_feature = "sm_103a"),
}

/// Instantiates the conversions of a `ue8m0` type `$ty` of one width.
macro_rules! ue8m0_kernels {
    ($ty:ident, f32: $f32:ty, bf16: $bf16:ty, cfg: $cfg:meta) => {
        kernel!(
            $ty,
            "from_f32",
            $cfg,
            |x: *const $f32, out: *mut [$ty; 4]| {
                let x = *x;
                *out = [
                    <$ty>::from_f32::<{ Rounding::TowardZero }, { Overflow::Infinity }>(x),
                    <$ty>::from_f32::<{ Rounding::TowardZero }, { Overflow::SaturateFinite }>(x),
                    <$ty>::from_f32::<{ Rounding::Up }, { Overflow::Infinity }>(x),
                    <$ty>::from_f32::<{ Rounding::Up }, { Overflow::SaturateFinite }>(x),
                ];
            }
        );
        kernel!(
            $ty,
            "from_bf16",
            $cfg,
            |x: *const $bf16, out: *mut [$ty; 4]| {
                let x = *x;
                *out = [
                    <$ty>::from_bf16::<{ Rounding::TowardZero }, { Overflow::Infinity }>(x),
                    <$ty>::from_bf16::<{ Rounding::TowardZero }, { Overflow::SaturateFinite }>(x),
                    <$ty>::from_bf16::<{ Rounding::Up }, { Overflow::Infinity }>(x),
                    <$ty>::from_bf16::<{ Rounding::Up }, { Overflow::SaturateFinite }>(x),
                ];
            }
        );
        kernel!(
            $ty,
            "to",
            $cfg,
            |v: *const $ty,
             out_bf16: *mut $bf16,
             out_f32: *mut $f32,
             out_from_bf16: *mut $bf16,
             out_from_f32: *mut $f32| {
                let v = *v;
                *out_bf16 = v.to_bf16();
                *out_f32 = v.to_f32();
                *out_from_bf16 = <$bf16>::from(v);
                *out_from_f32 = <$f32>::from(v);
            }
        );
    };
}

macro_rules! ue8m0_format {
    (
        scalar: $scalar:ident,
        x2: $x2:ident, x4: $x4:ident, x8: $x8:ident, x16: $x16:ident, x32: $x32:ident,
        xn: $xn:ident, x3: $x3:ident, x6: $x6:ident, x12: $x12:ident,
        cfg: $cfg:meta $(,)?
    ) => {
        type $x3 = $xn<3>;
        type $x6 = $xn<6>;
        type $x12 = $xn<12>;
        ue8m0_kernels!($scalar, f32: f32, bf16: bf16, cfg: $cfg);
        ue8m0_kernels!($x2, f32: f32x2, bf16: bf16x2, cfg: $cfg);
        ue8m0_kernels!($x4, f32: f32x4, bf16: bf16x4, cfg: $cfg);
        ue8m0_kernels!($x8, f32: f32x8, bf16: bf16x8, cfg: $cfg);
        ue8m0_kernels!($x16, f32: f32x16, bf16: bf16x16, cfg: $cfg);
        ue8m0_kernels!($x32, f32: f32x32, bf16: bf16x32, cfg: $cfg);
        ue8m0_kernels!($x3, f32: f32xN<3>, bf16: bf16xN<3>, cfg: $cfg);
        ue8m0_kernels!($x6, f32: f32xN<6>, bf16: bf16xN<6>, cfg: $cfg);
        ue8m0_kernels!($x12, f32: f32xN<12>, bf16: bf16xN<12>, cfg: $cfg);
    };
}

ue8m0_format! {
    scalar: f8e8m0,
    x2: f8e8m0x2, x4: f8e8m0x4, x8: f8e8m0x8, x16: f8e8m0x16, x32: f8e8m0x32,
    xn: f8e8m0xN, x3: f8e8m0x3, x6: f8e8m0x6, x12: f8e8m0x12,
    cfg: any(
        target_feature = "sm_100f",
        target_feature = "sm_101f",
        target_feature = "sm_110f",
        target_feature = "sm_120f"
    ),
}

/// Instantiates the conversions of an `s2f6` type `$ty` of one width.
macro_rules! s2f6_kernels {
    (
        $ty:ident, f32: $f32:ty, bf16: $bf16:ty, scales: $scales:ty, cfg: $cfg:meta
    ) => {
        kernel!(
            $ty,
            "from_f32",
            $cfg,
            |x: *const $f32, scales: *const $scales, out: *mut [$ty; 4]| {
                let x = *x;
                let scales = *scales;
                *out = [
                    <$ty>::from_f32::<{ Relu::Disabled }>(x),
                    <$ty>::from_f32::<{ Relu::Enabled }>(x),
                    <$ty>::from_f32_scaled::<{ Relu::Disabled }>(x, scales),
                    <$ty>::from_f32_scaled::<{ Relu::Enabled }>(x, scales),
                ];
            }
        );
        kernel!(
            $ty,
            "from_bf16",
            $cfg,
            |x: *const $bf16, scales: *const $scales, out: *mut [$ty; 4]| {
                let x = *x;
                let scales = *scales;
                *out = [
                    <$ty>::from_bf16::<{ Relu::Disabled }>(x),
                    <$ty>::from_bf16::<{ Relu::Enabled }>(x),
                    <$ty>::from_bf16_scaled::<{ Relu::Disabled }>(x, scales),
                    <$ty>::from_bf16_scaled::<{ Relu::Enabled }>(x, scales),
                ];
            }
        );
        kernel!(
            $ty,
            "to_bf16",
            $cfg,
            |v: *const $ty, scales: *const $scales, out: *mut [$bf16; 6]| {
                let v = *v;
                let scales = *scales;
                *out = [
                    v.to_bf16::<{ Relu::Disabled }>(),
                    v.to_bf16::<{ Relu::Enabled }>(),
                    v.to_bf16_scaled::<{ Relu::Disabled }, { Overflow::Infinity }>(scales),
                    v.to_bf16_scaled::<{ Relu::Enabled }, { Overflow::Infinity }>(scales),
                    v.to_bf16_scaled::<{ Relu::Disabled }, { Overflow::SaturateFinite }>(scales),
                    v.to_bf16_scaled::<{ Relu::Enabled }, { Overflow::SaturateFinite }>(scales),
                ];
            }
        );
        kernel!($ty, "to_f32", $cfg, |v: *const $ty, out: *mut [$f32; 3]| {
            let v = *v;
            *out = [
                v.to_f32::<{ Relu::Disabled }>(),
                v.to_f32::<{ Relu::Enabled }>(),
                <$f32>::from(v),
            ];
        });
        kernel!($ty, "from", $cfg, |v: *const $ty, out: *mut $bf16| {
            *out = <$bf16>::from(*v);
        });
    };
}

macro_rules! s2f6_format {
    (
        scalar: $scalar:ident,
        x2: $x2:ident, x4: $x4:ident, x8: $x8:ident, x16: $x16:ident, x32: $x32:ident,
        xn: $xn:ident, x3: $x3:ident, x6: $x6:ident, x12: $x12:ident,
        cfg: $cfg:meta $(,)?
    ) => {
        type $x3 = $xn<3>;
        type $x6 = $xn<6>;
        type $x12 = $xn<12>;
        s2f6_kernels!($scalar, f32: f32, bf16: bf16, scales: f8e8m0, cfg: $cfg);
        s2f6_kernels!($x2, f32: f32x2, bf16: bf16x2, scales: f8e8m0x2, cfg: $cfg);
        s2f6_kernels!($x4, f32: f32x4, bf16: bf16x4, scales: f8e8m0x4, cfg: $cfg);
        s2f6_kernels!($x8, f32: f32x8, bf16: bf16x8, scales: f8e8m0x8, cfg: $cfg);
        s2f6_kernels!($x16, f32: f32x16, bf16: bf16x16, scales: f8e8m0x16, cfg: $cfg);
        s2f6_kernels!($x32, f32: f32x32, bf16: bf16x32, scales: f8e8m0x32, cfg: $cfg);
        s2f6_kernels!($x3, f32: f32xN<3>, bf16: bf16xN<3>, scales: f8e8m0xN<3>, cfg: $cfg);
        s2f6_kernels!($x6, f32: f32xN<6>, bf16: bf16xN<6>, scales: f8e8m0xN<6>, cfg: $cfg);
        s2f6_kernels!($x12, f32: f32xN<12>, bf16: bf16xN<12>, scales: f8e8m0xN<12>, cfg: $cfg);
    };
}

s2f6_format! {
    scalar: s2f6,
    x2: s2f6x2, x4: s2f6x4, x8: s2f6x8, x16: s2f6x16, x32: s2f6x32,
    xn: s2f6xN, x3: s2f6x3, x6: s2f6x6, x12: s2f6x12,
    cfg: any(
        target_feature = "sm_100a",
        target_feature = "sm_103a",
        target_feature = "sm_110a",
        target_feature = "sm_120a",
        target_feature = "sm_121a"
    ),
}
