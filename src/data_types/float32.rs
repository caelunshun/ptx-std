// --- LLM-generated --- //
use crate::{Denormals, F32Ext, NanHandling, Rounding, Saturation, XorSignAbs};
use core::{
    ops::{Add, Div, Mul, Neg, Sub},
    simd::Simd,
};

use super::vector::{broadcast_ops, vector_element};

vector_element! {
    elem: f32, name: "`f32`", unit: u32, packing: Bytes,
    consts: [],
    note: "Arithmetic operators act element-wise like those of `f32`: they round to nearest, ties to even, and preserve subnormal values. [`F32Ext`](crate::F32Ext) provides the other operations and the instruction modifiers. `From` converts to and from `core::simd::Simd<f32, N>`.",
    eq: elementwise,
    aliases: f32xN, f32x2, f32x4, f32x8, f32x16, f32x32, f32x64,
}

impl<const N: usize> f32xN<N> {
    /// All elements positive zero.
    pub const ZERO: Self = Self::splat(0.0);
    /// All elements negative zero.
    pub const NEG_ZERO: Self = Self::splat(-0.0);
    /// All elements one.
    pub const ONE: Self = Self::splat(1.0);
    /// All elements negative one.
    pub const NEG_ONE: Self = Self::splat(-1.0);
    /// All elements equal to `f32::INFINITY`.
    pub const INFINITY: Self = Self::splat(f32::INFINITY);
    /// All elements equal to `f32::NEG_INFINITY`.
    pub const NEG_INFINITY: Self = Self::splat(f32::NEG_INFINITY);
    /// All elements equal to `f32::NAN`.
    pub const NAN: Self = Self::splat(f32::NAN);
    /// All elements equal to `f32::MAX`.
    pub const MAX: Self = Self::splat(f32::MAX);
    /// All elements equal to `f32::MIN`.
    pub const MIN: Self = Self::splat(f32::MIN);
    /// All elements equal to `f32::MIN_POSITIVE`.
    pub const MIN_POSITIVE: Self = Self::splat(f32::MIN_POSITIVE);
    /// All elements equal to the smallest positive subnormal value (2^-149).
    pub const MIN_POSITIVE_SUBNORMAL: Self = Self::splat(f32::from_bits(1));
    /// All elements equal to `f32::EPSILON`.
    pub const EPSILON: Self = Self::splat(f32::EPSILON);

    /// Computes the absolute value of each element. A NaN input yields an unspecified NaN.
    ///
    /// See [`F32Ext::abs_with`] for flushing subnormal values.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-abs>
    #[inline(always)]
    pub fn abs(self) -> Self {
        self.abs_with::<{ Denormals::Preserve }>()
    }

    /// Element-wise fused multiply-add: computes `self * b + c` with a single rounding (to
    /// nearest, ties to even).
    ///
    /// See [`F32Ext::fma`] for the other rounding modes and modifiers.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-fma>
    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        self.fma::<{ Rounding::NearestEven }, { Denormals::Preserve }, { Saturation::None }>(b, c)
    }

    /// Returns the element-wise minimum. If exactly one input element is NaN, the other is
    /// returned. `-0.0` is treated as less than `+0.0`.
    ///
    /// See [`F32Ext::min_with`] for the modifiers.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-min>
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        self.min_with::<{ Denormals::Preserve }, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(
            other,
        )
    }

    /// Returns the element-wise maximum. If exactly one input element is NaN, the other is
    /// returned. `+0.0` is treated as greater than `-0.0`.
    ///
    /// See [`F32Ext::max_with`] for the modifiers.
    ///
    /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-max>
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        self.max_with::<{ Denormals::Preserve }, { NanHandling::Ignore }, { XorSignAbs::Disabled }>(
            other,
        )
    }
}

impl<const N: usize> From<Simd<f32, N>> for f32xN<N> {
    #[inline(always)]
    fn from(x: Simd<f32, N>) -> Self {
        Self::from_array(x.to_array())
    }
}

impl<const N: usize> From<f32xN<N>> for Simd<f32, N> {
    #[inline(always)]
    fn from(x: f32xN<N>) -> Self {
        Simd::from_array(x.to_array())
    }
}

/// Implements the element-wise arithmetic operators with those of `Simd<f32, N>`, which LLVM
/// lowers like the operators of `f32` (`+`, `-` and `*` to the packed `f32x2` instructions on
/// `sm_100` and later).
macro_rules! simd_ops {
    ($($op:ident $method:ident),*) => {$(
        impl<const N: usize> $op for f32xN<N> {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Self::from($op::$method(Simd::from(self), Simd::from(rhs)))
            }
        }
    )*};
}

simd_ops!(Add add, Sub sub, Mul mul, Div div);

impl<const N: usize> Neg for f32xN<N> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self::from(-Simd::from(self))
    }
}

broadcast_ops!(
    f32: Add add AddAssign add_assign, Sub sub SubAssign sub_assign, Mul mul MulAssign mul_assign,
    Div div DivAssign div_assign,
);
