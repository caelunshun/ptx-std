// --- LLM-generated --- //
//! Shared implementation of the container API of the floating-point scalar and vector types.
//!
//! Every format has a scalar and vectors of any length from 1 to 64 elements (for the 4-bit
//! format, any even length from 2 to 64). A vector occupies exactly `N * size_of::<E>()` bytes
//! (`N / 2` bytes for the 4-bit format), with no padding, and is aligned to the largest power of
//! two dividing its size: its own size for the lengths that are powers of two, so that `f16x8`
//! (16 bytes) is aligned like the 128-bit loads and stores that move it, and the size of one
//! element for odd lengths. This is the layout of a packed `core::simd::Simd`. Element 0 always
//! occupies the least significant bits (the machine is little-endian, so it is also the first
//! element in memory).
//!
//! All vectors of all formats are one type, [`Vector<E, N>`], where `E` implements
//! [`VectorElement<N>`] (every format does for every `N`, except the 4-bit format, which has
//! only even lengths). The familiar names (`f32x8`, `f16x4`, `f8e4m3x32`, ...) are aliases. The
//! storage is [`VectorStorage`], an array of the elements' integer unit with the alignment of
//! the corresponding `Simd`, which also makes lengths of 0 and above 64 errors at compile time
//! (once the vector type is instantiated).
//!
//! The elements are processed in pairs, the unit of the packed PTX instructions (`.f32x2`,
//! `.f16x2`, `.e4m3x2`, ...). For an odd length, the last pair holds one element and a padding
//! lane: it is zero when the pair is read and ignored when it is written.
//!
//! - [`narrow_scalar!`] defines the one-byte scalars of the narrow formats (FP8, FP6, FP4, s2f6).
//!   `f16` and `bf16` are defined by hand, and `f32` is the primitive type.
//! - [`vector_element!`] makes a scalar a [`VectorElement`] and defines the aliases, the splatted
//!   constants and the `From` conversion to arrays of its vectors.
//! - [`broadcast_ops!`] adds the operators with a scalar operand and the compound assignment
//!   operators of the vectors of a format.
//! - [`half_vector!`] adds what the `f16` and `bf16` vectors share: the conversions from and to
//!   the `f32` vectors, `reduce_sum` and the [`broadcast_ops!`] of `+`, `-` and `*`.
//!
//! Format specific operations are written once per format as private methods on the pair type
//! (`Vector<E, 2>`), and once, generically over the length, as public methods that map over the
//! pairs with the crate-private primitives of [`Vector`] (`pair`, `set_pair`, `map_pairs`,
//! `zip_pairs`, `reduce_pairs`, ...). These need no bound on the length, unlike the types that
//! involve `N / 2` (arrays of pairs) and the raw bits, which are described by the table of
//! [`SupportedVectorLen`]. A pair type and a generic impl may not define the same name, so the
//! private methods carry a `pair_` prefix. The macros expand to code that accesses private items,
//! so they must be invoked in the module that defines the scalar type.

use bytemuck::{Pod, Zeroable};
use core::fmt;
use core::simd::{Simd, SimdElement};

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Special value encodings of the small IEEE-like float formats, for [`minifloat_to_f32`].
#[derive(Clone, Copy)]
pub(crate) enum Specials {
    /// No infinities or NaNs (FP6, FP4).
    None,
    /// IEEE conventions: the largest exponent encodes infinity and NaN (FP8 E5M2).
    Ieee,
    /// A single NaN encoding, all exponent and mantissa bits set (FP8 E4M3).
    NanOnly,
}

/// Decodes a sign/exponent/mantissa minifloat (sign bit at bit `exp_bits + man_bits`) to `f32`.
/// Used by the `Debug` impls of the narrow scalars so that they work on any architecture.
pub(crate) const fn minifloat_to_f32(
    bits: u8,
    exp_bits: u32,
    man_bits: u32,
    bias: i32,
    specials: Specials,
) -> f32 {
    let bits = bits as u32;
    let negative = (bits >> (exp_bits + man_bits)) & 1 == 1;
    let exp = (bits >> man_bits) & ((1 << exp_bits) - 1);
    let man = bits & ((1 << man_bits) - 1);
    let max_exp = (1 << exp_bits) - 1;
    let max_man = (1 << man_bits) - 1;

    let magnitude = if exp == max_exp
        && match specials {
            Specials::None => false,
            Specials::Ieee => true,
            Specials::NanOnly => man == max_man,
        } {
        if man == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        // Both scale exponents are in the normal range of `f32` for all supported formats.
        let (fraction, scale_exp) = if exp == 0 {
            (0.0, 1 - bias)
        } else {
            (1.0, exp as i32 - bias)
        };
        let scale = f32::from_bits(((scale_exp + 127) as u32) << 23);
        (fraction + man as f32 / (1u32 << man_bits) as f32) * scale
    };
    if negative { -magnitude } else { magnitude }
}

/// Defines a one-byte scalar of a narrow format.
///
/// - `width`: number of bits used by the format (the low `width` bits of the byte). The unused
///   high bits are always zero, which `from_bits` guarantees.
/// - `decode`: expression computing the `f32` value from the raw bits (a `u8` bound to the given
///   identifier), used by `Debug`.
/// - `consts`: the associated constants, as raw bits.
///
/// ```ignore
/// narrow_scalar! {
///     /// Docs.
///     f8e4m3, width: 8,
///     decode: |bits| minifloat_to_f32(bits, 4, 3, 7, Specials::NanOnly),
///     consts: {
///         /// Positive zero.
///         ZERO = 0x00;
///     }
/// }
/// ```
macro_rules! narrow_scalar {
    (
        $(#[$attr:meta])*
        $name:ident, width: $width:literal,
        decode: |$bits:ident| $decode:expr,
        consts: { $($(#[$cattr:meta])* $cname:ident = $cval:expr;)* } $(,)?
    ) => {
        $(#[$attr])*
        #[derive(Clone, Copy, ::bytemuck::Pod, ::bytemuck::Zeroable)]
        #[repr(transparent)]
        pub struct $name(u8);

        impl $name {
            $($(#[$cattr])* pub const $cname: Self = Self($cval);)*

            #[doc = concat!("Creates a value from its raw bit representation. Only the low ", stringify!($width), " bits are used; higher bits are cleared.")]
            #[inline]
            pub const fn from_bits(bits: u8) -> Self {
                Self(bits & (((1u16 << $width) - 1) as u8))
            }

            #[doc = concat!("Returns the raw bit representation, stored in the low ", stringify!($width), " bits.")]
            #[inline]
            pub const fn to_bits(self) -> u8 {
                self.0
            }

            /// Decodes the value in software, so that `Debug` works on every architecture.
            #[inline]
            const fn soft_f32(self) -> f32 {
                let $bits = self.0;
                $decode
            }
        }

        impl ::core::default::Default for $name {
            /// The all-zero bit pattern.
            #[inline]
            fn default() -> Self {
                Self(0)
            }
        }

        impl ::core::fmt::Debug for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                ::core::fmt::Debug::fmt(&self.soft_f32(), f)
            }
        }
    };
}

/// Reinterprets the bits of `a` as a `B`, for sizes that depend on generic array lengths. The
/// size check runs after monomorphization.
///
/// # Safety
///
/// Every bit pattern of `A` that can occur must be a valid `B` (the types are plain data).
#[inline(always)]
pub(crate) const unsafe fn cast<A: Copy, B: Copy>(a: A) -> B {
    const {
        assert!(size_of::<A>() == size_of::<B>());
    }
    // SAFETY: the sizes are equal and the caller guarantees validity.
    unsafe { cast_unchecked(a) }
}

/// [`cast`] without the size check, for branches that are dead for some instantiations (the check
/// of `cast` is evaluated even in untaken `if const` branches).
///
/// A union read is used instead of `transmute_copy`, which reads unaligned if `B` is more aligned
/// than `A`.
///
/// # Safety
///
/// As for [`cast`], and `A` and `B` must have the same size whenever the branch is taken.
#[inline(always)]
pub(crate) const unsafe fn cast_unchecked<A: Copy, B: Copy>(a: A) -> B {
    union Transmute<A: Copy, B: Copy> {
        a: A,
        b: B,
    }
    // SAFETY: guaranteed by the caller.
    unsafe { Transmute { a }.b }
}

/// The number of elements of a [`Vector`], as a type, to describe the lengths that have the
/// types of [`SupportedVectorLen`]: `VectorLen<N>: SupportedVectorLen` is the bound that code
/// needs when it names [`Pairs`], [`Quads`], [`Bits`] or the raw-bits constructors of a vector of
/// generic length.
pub struct VectorLen<const N: usize>;

/// Implemented for [`VectorLen`] of every length from `1` to `64`. Sealed.
///
/// It provides the types of the vector API that depend on the length through `N / 2`, `N / 4` or
/// the size of the vector: the arrays [`Pairs`] and [`Quads`], and the raw bits [`Bits`].
/// Everything else (arithmetic, conversions, reductions, ...) is available for every length
/// without a bound.
pub trait SupportedVectorLen: sealed::Sealed {
    /// The number of pairs of elements, `N.div_ceil(2)`, usable in the types of arrays. The last
    /// pair of a vector of odd length holds a single element.
    #[rustc_always_gca]
    const PAIRS: usize;

    /// The number of groups of four elements, `N.div_ceil(4)`, usable in the types of arrays.
    /// The last group may hold fewer than four elements.
    #[rustc_always_gca]
    const QUADS: usize;

    /// The raw bits of a vector of `N` elements whose integer unit is `U` (`u32` for `f32`,
    /// `u16` for the 16-bit formats, `u8` for the 8-bit formats): the unsigned integer with the
    /// size of the vector if `N` is a power of two (`u8` ... `u128`, or an array of `u128`), and
    /// an array of such integers if it is not.
    type Bits<U: BitsUnit>: Copy + Pod;
}

/// Implemented for [`VectorLen`] of the even lengths from `4` to `64`, the lengths whose halves
/// are vectors (the halves of a pair are single elements, so `2` is not included). Sealed.
pub trait EvenVectorLen: SupportedVectorLen {
    /// Half of the number of elements, usable in the types of vectors: the length of the halves
    /// returned by [`Vector::lo`] and [`Vector::hi`].
    #[rustc_always_gca]
    const HALF: usize;
}

/// An array of `T` with one element per pair of elements of a vector of `N` elements, `N / 2`
/// rounded up.
pub type Pairs<T, const N: usize> = [T; <VectorLen<N> as SupportedVectorLen>::PAIRS];

/// An array of `T` with one element per group of four elements of a vector of `N` elements, `N / 4`
/// rounded up.
pub type Quads<T, const N: usize> = [T; <VectorLen<N> as SupportedVectorLen>::QUADS];

/// The vector of half the length of an `N`-element vector of `E`, for even `N` of at least `4`.
///
/// Using it requires `E: VectorElement<{ <VectorLen<N> as EvenVectorLen>::HALF }>` (the bound
/// [`EvenVectorElement<N>`] spells), which holds for every format except the 4-bit format, whose
/// halves must have an even length too.
pub type Half<E, const N: usize> = Vector<E, { <VectorLen<N> as EvenVectorLen>::HALF }>;

/// An element type whose vectors of `N` elements have halves: `E: VectorElement<N>` and
/// `E: VectorElement<N / 2>` (as `VectorElement<{ <VectorLen<N> as EvenVectorLen>::HALF }>`) in
/// one trait, so that code generic over the element type that uses [`Vector::new`],
/// [`Vector::lo`] or [`Vector::hi`] can write `E: EvenVectorElement<N>` without spelling the
/// projection, which needs the `gca_min_const_items` and `gca_macroless_args` features in the
/// crate that writes it. Implemented for every `E` and even `N` of at least `4` for which the
/// two bounds hold, so for every format except the 4-bit format, for which `N` must also be a
/// multiple of four.
pub trait EvenVectorElement<const N: usize>:
    VectorElement<N> + VectorElement<{ <VectorLen<N> as EvenVectorLen>::HALF }>
where
    VectorLen<N>: EvenVectorLen,
{
}

impl<E, const N: usize> EvenVectorElement<N> for E
where
    VectorLen<N>: EvenVectorLen,
    E: VectorElement<N> + VectorElement<{ <VectorLen<N> as EvenVectorLen>::HALF }>,
{
}

/// The unsigned integer (or array of integers) with the size of an `N`-element vector of `E`, the
/// type of its raw bits. For example `u32` for `f16x2`, `u128` for `f16x8`, `[u128; 16]` for
/// `f32x64` and `[u32; 3]` for `f16xN<6>`.
pub type Bits<E, const N: usize> =
    <VectorLen<N> as SupportedVectorLen>::Bits<<E as VectorScalar>::BitsUnit>;

macro_rules! vector_lens {
    ($(
        $n:literal => $pairs:literal, $quads:literal, $x:ident $(* $odd:literal)?
            $(, half $half:literal)?;
    )*) => {$(
        impl sealed::Sealed for VectorLen<$n> {}

        impl SupportedVectorLen for VectorLen<$n> {
            const PAIRS: usize = core::gca!($pairs);
            const QUADS: usize = core::gca!($quads);
            type Bits<U: BitsUnit> = vector_lens!(@bits U, $x $(* $odd)?);
        }

        $(impl EvenVectorLen for VectorLen<$n> {
            const HALF: usize = core::gca!($half);
        })?
    )*};
    (@bits $u:ident, $x:ident) => { <$u as BitsUnit>::$x };
    (@bits $u:ident, $x:ident * $odd:literal) => { [<$u as BitsUnit>::$x; $odd] };
}

// Each row gives the number of pairs and quads, and the raw bits as `X{p}` for a power of two
// `p`, or `X{p} * odd` for `N == p * odd`, the largest power of two `p` dividing `N`.
vector_lens! {
    1 => 1, 1, X1;
    2 => 1, 1, X2;
    3 => 2, 1, X1 * 3;
    4 => 2, 1, X4, half 2;
    5 => 3, 2, X1 * 5;
    6 => 3, 2, X2 * 3, half 3;
    7 => 4, 2, X1 * 7;
    8 => 4, 2, X8, half 4;
    9 => 5, 3, X1 * 9;
    10 => 5, 3, X2 * 5, half 5;
    11 => 6, 3, X1 * 11;
    12 => 6, 3, X4 * 3, half 6;
    13 => 7, 4, X1 * 13;
    14 => 7, 4, X2 * 7, half 7;
    15 => 8, 4, X1 * 15;
    16 => 8, 4, X16, half 8;
    17 => 9, 5, X1 * 17;
    18 => 9, 5, X2 * 9, half 9;
    19 => 10, 5, X1 * 19;
    20 => 10, 5, X4 * 5, half 10;
    21 => 11, 6, X1 * 21;
    22 => 11, 6, X2 * 11, half 11;
    23 => 12, 6, X1 * 23;
    24 => 12, 6, X8 * 3, half 12;
    25 => 13, 7, X1 * 25;
    26 => 13, 7, X2 * 13, half 13;
    27 => 14, 7, X1 * 27;
    28 => 14, 7, X4 * 7, half 14;
    29 => 15, 8, X1 * 29;
    30 => 15, 8, X2 * 15, half 15;
    31 => 16, 8, X1 * 31;
    32 => 16, 8, X32, half 16;
    33 => 17, 9, X1 * 33;
    34 => 17, 9, X2 * 17, half 17;
    35 => 18, 9, X1 * 35;
    36 => 18, 9, X4 * 9, half 18;
    37 => 19, 10, X1 * 37;
    38 => 19, 10, X2 * 19, half 19;
    39 => 20, 10, X1 * 39;
    40 => 20, 10, X8 * 5, half 20;
    41 => 21, 11, X1 * 41;
    42 => 21, 11, X2 * 21, half 21;
    43 => 22, 11, X1 * 43;
    44 => 22, 11, X4 * 11, half 22;
    45 => 23, 12, X1 * 45;
    46 => 23, 12, X2 * 23, half 23;
    47 => 24, 12, X1 * 47;
    48 => 24, 12, X16 * 3, half 24;
    49 => 25, 13, X1 * 49;
    50 => 25, 13, X2 * 25, half 25;
    51 => 26, 13, X1 * 51;
    52 => 26, 13, X4 * 13, half 26;
    53 => 27, 14, X1 * 53;
    54 => 27, 14, X2 * 27, half 27;
    55 => 28, 14, X1 * 55;
    56 => 28, 14, X8 * 7, half 28;
    57 => 29, 15, X1 * 57;
    58 => 29, 15, X2 * 29, half 29;
    59 => 30, 15, X1 * 59;
    60 => 30, 15, X4 * 15, half 30;
    61 => 31, 16, X1 * 61;
    62 => 31, 16, X2 * 31, half 31;
    63 => 32, 16, X1 * 63;
    64 => 32, 16, X64, half 32;
}

/// The unit of the packed PTX instructions of an element type: the unsigned integer holding a
/// pair of elements, `u64` for `f32`, `u32` for the 16-bit formats, `u16` for the 8-bit formats
/// and `u8` for the 4-bit formats. Sealed.
pub trait PairBits: Copy + Pod + sealed::Sealed {
    /// All bits zero.
    const ZEROED: Self;
}

macro_rules! pair_bits {
    ($($bits:ident)*) => {$(
        impl sealed::Sealed for $bits {}

        impl PairBits for $bits {
            const ZEROED: Self = 0;
        }
    )*};
}

pair_bits!(u8 u16 u32 u64);

/// The integer unit of the raw bits of the vectors of an element type, which selects the
/// [`Bits`] types: `u8`, `u16` or `u32` for the formats with elements of that size, and
/// [`NibbleBits`] for the 4-bit format. `X{n}` is the type of the raw bits of `n` units (for
/// the 4-bit format, `n` elements). Sealed.
#[doc(hidden)]
pub trait BitsUnit: sealed::Sealed + 'static {
    type X1: Copy + Pod;
    type X2: Copy + Pod;
    type X4: Copy + Pod;
    type X8: Copy + Pod;
    type X16: Copy + Pod;
    type X32: Copy + Pod;
    type X64: Copy + Pod;
}

macro_rules! bits_unit {
    ($($unit:ty => $x1:ty, $x2:ty, $x4:ty, $x8:ty, $x16:ty, $x32:ty, $x64:ty;)*) => {$(
        impl BitsUnit for $unit {
            type X1 = $x1;
            type X2 = $x2;
            type X4 = $x4;
            type X8 = $x8;
            type X16 = $x16;
            type X32 = $x32;
            type X64 = $x64;
        }
    )*};
}

/// The [`BitsUnit`] of the 4-bit format, whose elements are half a byte: `X1` is never used, as
/// its vectors have an even length.
#[doc(hidden)]
pub struct NibbleBits;

impl sealed::Sealed for NibbleBits {}

bits_unit! {
    u8 => u8, u16, u32, u64, u128, [u128; 2], [u128; 4];
    u16 => u16, u32, u64, u128, [u128; 2], [u128; 4], [u128; 8];
    u32 => u32, u64, u128, [u128; 2], [u128; 4], [u128; 8], [u128; 16];
    NibbleBits => (), u8, u16, u32, u64, u128, [u128; 2];
}

/// How the elements of an element type are packed into the bytes of a [`Vector`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Packing {
    /// Every element occupies one or more whole bytes: an array of the elements has the layout of
    /// the vector.
    Bytes,
    /// Two elements share each byte, the first in the low nibble (the 4-bit formats).
    Nibbles,
}

/// Implemented for the scalar types of the floating-point formats: everything about an element
/// type of [`Vector`] that does not depend on the length. Sealed.
///
/// The types are the element types of the vectors if they also implement [`VectorElement<N>`] for
/// the length `N`.
pub trait VectorScalar: Copy + Pod + Default + sealed::Sealed {
    /// The unsigned integer holding a pair of elements.
    type PairBits: PairBits;
    /// The integer unit that selects the type of the raw bits of the vectors, see [`Bits`].
    #[doc(hidden)]
    type BitsUnit: BitsUnit;
    /// The all-zero bit pattern, which is also what `Default` returns.
    const ZEROED: Self;
    /// How the elements are packed into a vector.
    const PACKING: Packing;
    /// The name of the format, as printed by the `Debug` implementation of the vector.
    const NAME: &'static str;
}

/// `E` can be the element type of vectors of `N` elements. Sealed.
///
/// Every format implements it for every `N` (`1` to `64`) except the 4-bit format `f4e2m1`,
/// which implements it for the even `N` from `2` to `64`, so that `Vector<f4e2m1, 3>` is a type
/// error. Code generic over the length of a vector of a given format needs no bound for the
/// formats that implement it for every `N`; code generic over the element type bounds
/// `E: VectorElement<N>`, and `E: VectorElement<2>` as well if it names the pair type
/// `Vector<E, 2>`, as the methods that process the pairs of the elements do.
pub trait VectorElement<const N: usize>: VectorScalar {
    /// The storage of the vector: [`VectorStorage`] of the integer unit of the format, `N`
    /// units long (`N / 2` bytes for the 4-bit format).
    #[doc(hidden)]
    type Storage: Copy + Pod;
}

/// `[U; N]` with the alignment of the `Simd<U, N>` of the same length: the largest power of two
/// dividing its size. The storage of a [`Vector`], `U` being `u32` for `f32`, `u16` for the 16-bit
/// formats and `u8` for the 8-bit formats (`u8` and `N / 2` for the 4-bit format).
///
/// A zero-length array of the `Simd` provides the alignment without taking space, so that the
/// size is exactly that of the array, and the limit of 64 lanes of `Simd` becomes an error for
/// vectors of more than 64 (or of 0) elements.
#[doc(hidden)]
#[repr(C)]
pub struct VectorStorage<U: SimdElement + Pod, const N: usize> {
    #[allow(dead_code)]
    data: [U; N],
    #[allow(dead_code)]
    align: [Simd<U, N>; 0],
}

impl<U: SimdElement + Pod, const N: usize> Clone for VectorStorage<U, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<U: SimdElement + Pod, const N: usize> Copy for VectorStorage<U, N> {}

// SAFETY: the array is plain data, and so is the zero-sized field. There is no padding, as the
// alignment divides the size.
unsafe impl<U: SimdElement + Pod, const N: usize> Zeroable for VectorStorage<U, N> {}

// SAFETY: see `Zeroable`.
unsafe impl<U: SimdElement + Pod, const N: usize> Pod for VectorStorage<U, N> {}

/// `N` packed values of the element type `E`.
///
/// `N` is a length from `1` to `64` (for the 4-bit format `f4e2m1`, an even length from `2` to
/// `64`); other lengths are errors when the type is instantiated. The element types are `f32`
/// and the floating-point formats of this crate, and each has aliases of its vector types:
/// [`f32xN`](crate::f32xN), [`f32x2`](crate::f32x2), ..., and the same for every other format.
///
/// Element 0 occupies the least significant bits. The size is exactly `N` times that of an
/// element (`N / 2` bytes for `f4e2m1`). The alignment is the largest power of two dividing the
/// size (the layout of `core::simd::Simd`): the size itself when it is a power of two, so that
/// vectors of up to 16 bytes are moved with a single memory access, and otherwise the largest
/// power of two that divides it.
///
/// The packed PTX instructions operate on pairs of elements, so the arithmetic of a vector is
/// done pair by pair; for an odd `N`, the last pair is made of the last element and a padding
/// lane, whose result is discarded. Reductions combine the last element separately.
///
/// Code that is generic over the length of a vector of a concrete format (other than `f4e2m1`)
/// needs no bound, for example `fn f<const N: usize>(v: f16xN<N>) -> f32 { v.to_f32().reduce_sum() }`.
/// Code generic over the element type bounds `E: VectorElement<N>`, and
/// `E: VectorElement<2>` to use the pairs ([`to_x2s`](Self::to_x2s)). The types that depend on
/// `N / 2` or on the raw bits additionally need `VectorLen<N>: SupportedVectorLen`, and the
/// halves ([`lo`](Self::lo) and [`hi`](Self::hi)) need `VectorLen<N>: EvenVectorLen` and
/// `E: EvenVectorElement<N>` (for a concrete format, the latter holds without being written).
///
/// # Kernel parameters
///
/// The vector contains a zero-length array of `core::simd::Simd`, which is not FFI-safe, so
/// `extern "ptx-kernel"` functions that take a `Vector` by value trigger the
/// `improper_ctypes_definitions` lint. The lint is spurious here (the type has the layout of
/// an array of integers, and the by-value passing works); allow it on such kernels with
/// `#[allow(improper_ctypes_definitions)]`. Kernels taking pointers to vectors do not trigger
/// it. Note that the compiler cannot pass values aligned to more than 16 bytes by value to
/// kernels, so vectors of 32 or more bytes should be passed by pointer anyway.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct Vector<E, const N: usize>(E::Storage)
where
    E: VectorElement<N>;

/// All-zero value of a plain-data type, in a `const` context (`Zeroable::zeroed` is not `const`).
///
/// # Safety
///
/// The all-zero bit pattern must be a valid `T`.
#[inline(always)]
const unsafe fn zeroed_const<T: Copy>() -> T {
    // SAFETY: guaranteed by the caller.
    unsafe { core::mem::zeroed() }
}

impl<E: VectorElement<N>, const N: usize> Vector<E, N> {
    /// The number of elements.
    pub const LEN: usize = N;

    /// The number of pairs of elements, `N / 2` rounded up. The last pair of a vector of odd
    /// length holds a single element.
    pub const PAIRS: usize = N.div_ceil(2);

    /// The number of groups of four elements, `N / 4` rounded up.
    pub const QUADS: usize = N.div_ceil(4);

    /// Checks the layout assumptions of the vector: the size is that of `N` elements without
    /// padding, the alignment is the largest power of two dividing the size, and the pair of an
    /// element type is as large as two elements (as large as one byte for the 4-bit format).
    const LAYOUT_OK: () = {
        let size = match E::PACKING {
            Packing::Bytes => N * size_of::<E>(),
            Packing::Nibbles => N / 2,
        };
        assert!(size_of::<Self>() == size);
        assert!(align_of::<Self>() == size.isolate_lowest_one());
        assert!(match E::PACKING {
            Packing::Bytes => size_of::<E::PairBits>() == 2 * size_of::<E>(),
            Packing::Nibbles => size_of::<E>() == 1 && size_of::<E::PairBits>() == 1,
        });
    };

    /// The vector with all bits zero.
    pub(crate) const ALL_ZERO: Self = {
        const { Self::LAYOUT_OK };
        // SAFETY: vectors are plain data, valid for every bit pattern.
        unsafe { zeroed_const() }
    };

    /// Pointer to the first byte of the vector.
    #[inline(always)]
    const fn bytes(&self) -> *const u8 {
        (&raw const self.0).cast()
    }

    /// Mutable pointer to the first byte of the vector.
    #[inline(always)]
    const fn bytes_mut(&mut self) -> *mut u8 {
        (&raw mut self.0).cast()
    }

    /// Element `i` of a vector of a format whose elements are whole bytes.
    #[inline(always)]
    pub(crate) const fn elem(&self, i: usize) -> E {
        // SAFETY: the storage is aligned to at least the size of an element, which is a power of
        // two dividing the size of the vector.
        unsafe { self.bytes().cast::<E>().add(i).read() }
    }

    /// Sets element `i` of a vector of a format whose elements are whole bytes.
    #[inline(always)]
    pub(crate) const fn set_elem(&mut self, i: usize, x: E) {
        // SAFETY: see `elem`.
        unsafe { self.bytes_mut().cast::<E>().add(i).write(x) }
    }

    /// The raw bits of pair `i`. The padding lane of the last pair of a vector of odd length is
    /// zero. Not for use on references into global memory: the vector must be a local, so that
    /// the access is optimized to registers.
    #[inline(always)]
    pub(crate) const fn pair_bits(&self, i: usize) -> E::PairBits {
        const { Self::LAYOUT_OK };
        if const { N % 2 == 1 } && 2 * i + 1 == N {
            // SAFETY: odd lengths only exist for formats of whole bytes, whose pair is two
            // elements.
            unsafe { cast_unchecked::<[E; 2], E::PairBits>([self.elem(2 * i), E::ZEROED]) }
        } else {
            // SAFETY: the pair is within the vector, and plain data.
            unsafe { self.bytes().cast::<E::PairBits>().add(i).read_unaligned() }
        }
    }

    /// Sets the raw bits of pair `i`. The padding lane of the last pair of a vector of odd
    /// length is dropped.
    #[inline(always)]
    pub(crate) const fn set_pair_bits(&mut self, i: usize, bits: E::PairBits) {
        const { Self::LAYOUT_OK };
        if const { N % 2 == 1 } && 2 * i + 1 == N {
            // SAFETY: see `pair_bits`.
            let [lo, _]: [E; 2] = unsafe { cast_unchecked(bits) };
            self.set_elem(2 * i, lo);
        } else {
            // SAFETY: see `pair_bits`.
            unsafe {
                self.bytes_mut()
                    .cast::<E::PairBits>()
                    .add(i)
                    .write_unaligned(bits)
            }
        }
    }

    /// Packs an array of elements.
    #[inline(always)]
    pub const fn from_array(x: [E; N]) -> Self {
        const { Self::LAYOUT_OK };
        if const { matches!(E::PACKING, Packing::Bytes) } {
            // SAFETY: the array has the layout of the vector, as checked by `LAYOUT_OK`.
            unsafe { cast_unchecked(x) }
        } else {
            let mut out = Self::ALL_ZERO;
            let mut i = 0;
            #[rustc_unroll(full)]
            while i < N / 2 {
                // SAFETY: nibble formats have one-byte elements and pairs, as checked by
                // `LAYOUT_OK`.
                let (lo, hi): (u8, u8) =
                    unsafe { (cast_unchecked(x[2 * i]), cast_unchecked(x[2 * i + 1])) };
                // The low element is masked in case it was created by a cast that bypassed the
                // masking `from_bits`; the high element loses its high bits in the shift.
                out.set_pair_bits(i, unsafe {
                    cast_unchecked::<u8, E::PairBits>((lo & 0xF) | (hi << 4))
                });
                i += 1;
            }
            out
        }
    }

    /// Unpacks into an array of elements.
    #[inline(always)]
    pub const fn to_array(self) -> [E; N] {
        const { Self::LAYOUT_OK };
        if const { matches!(E::PACKING, Packing::Bytes) } {
            // SAFETY: see `from_array`.
            unsafe { cast_unchecked(self.0) }
        } else {
            let mut out = [E::ZEROED; N];
            let mut i = 0;
            #[rustc_unroll(full)]
            while i < N / 2 {
                // SAFETY: see `from_array`.
                let byte: u8 = unsafe { cast_unchecked(self.pair_bits(i)) };
                out[2 * i] = unsafe { cast_unchecked(byte & 0xF) };
                out[2 * i + 1] = unsafe { cast_unchecked(byte >> 4) };
                i += 1;
            }
            out
        }
    }

    /// Creates a value with all elements set to `x`.
    #[inline(always)]
    pub const fn splat(x: E) -> Self {
        const { Self::LAYOUT_OK };
        // SAFETY: see `from_array`.
        let pair = unsafe {
            if const { matches!(E::PACKING, Packing::Bytes) } {
                cast_unchecked::<[E; 2], E::PairBits>([x, x])
            } else {
                let x: u8 = cast_unchecked(x);
                cast_unchecked::<u8, E::PairBits>((x & 0xF) | (x << 4))
            }
        };
        let mut out = Self::ALL_ZERO;
        let mut i = 0;
        #[rustc_unroll(full)]
        while i < Self::PAIRS {
            out.set_pair_bits(i, pair);
            i += 1;
        }
        out
    }
}

impl<E: VectorElement<N>, const N: usize> Vector<E, N>
where
    VectorLen<N>: SupportedVectorLen,
{
    /// Creates a value from its raw bit representation.
    #[inline(always)]
    pub const fn from_bits(bits: Bits<E, N>) -> Self {
        // SAFETY: both types are plain data of the same size, valid for all bit patterns.
        Self(unsafe { cast(bits) })
    }

    /// Returns the raw bit representation.
    #[inline(always)]
    pub const fn to_bits(self) -> Bits<E, N> {
        // SAFETY: see `from_bits`.
        unsafe { cast(self.0) }
    }

    /// Creates a value from the raw bits of its pairs of elements; `pairs[i]` holds elements
    /// `2 * i` and `2 * i + 1`. For an odd length, the upper half of the last pair is ignored.
    #[inline(always)]
    pub const fn from_pair_bits(pairs: Pairs<E::PairBits, N>) -> Self {
        if const { N.is_multiple_of(2) } {
            // SAFETY: the pairs fill the vector exactly.
            return unsafe { cast_unchecked(pairs) };
        }
        let mut out = Self::ALL_ZERO;
        let mut i = 0;
        #[rustc_unroll(full)]
        while i < pairs.len() {
            out.set_pair_bits(i, pairs[i]);
            i += 1;
        }
        out
    }

    /// Returns the raw bits of the pairs of elements; `pairs[i]` holds elements `2 * i` and
    /// `2 * i + 1`. For an odd length, the upper half of the last pair is zero.
    #[inline(always)]
    pub const fn to_pair_bits(self) -> Pairs<E::PairBits, N> {
        if const { N.is_multiple_of(2) } {
            // SAFETY: the pairs fill the vector exactly.
            return unsafe { cast_unchecked(self.0) };
        }
        let mut out = [E::PairBits::ZEROED; <VectorLen<N> as SupportedVectorLen>::PAIRS];
        let mut i = 0;
        #[rustc_unroll(full)]
        while i < out.len() {
            out[i] = self.pair_bits(i);
            i += 1;
        }
        out
    }
}

impl<E: VectorElement<N> + VectorElement<2>, const N: usize> Vector<E, N>
where
    VectorLen<N>: SupportedVectorLen,
{
    /// Packs pairs of elements; `x2s[i]`, a `Vector<E, 2>`, holds elements `2 * i` and
    /// `2 * i + 1`. For an odd length, the upper half of the last pair is ignored.
    ///
    /// Pairs are the unit of the PTX instructions operating on packed values.
    #[inline(always)]
    pub const fn from_x2s(x2s: Pairs<Vector<E, 2>, N>) -> Self {
        if const { N.is_multiple_of(2) } {
            // SAFETY: the pairs fill the vector exactly.
            return unsafe { cast_unchecked(x2s) };
        }
        let mut out = Self::ALL_ZERO;
        let mut i = 0;
        #[rustc_unroll(full)]
        while i < x2s.len() {
            out.set_pair(i, x2s[i]);
            i += 1;
        }
        out
    }

    /// Splits into pairs of elements; `x2s[i]`, a `Vector<E, 2>`, holds elements `2 * i` and
    /// `2 * i + 1`. For an odd length, the upper half of the last pair is zero.
    #[inline(always)]
    pub const fn to_x2s(self) -> Pairs<Vector<E, 2>, N> {
        if const { N.is_multiple_of(2) } {
            // SAFETY: the pairs fill the vector exactly.
            return unsafe { cast_unchecked(self.0) };
        }
        let mut out = [Vector::<E, 2>::ALL_ZERO; <VectorLen<N> as SupportedVectorLen>::PAIRS];
        let mut i = 0;
        #[rustc_unroll(full)]
        while i < out.len() {
            out[i] = self.pair(i);
            i += 1;
        }
        out
    }
}

// The primitives all formats are written with, which do not need the length as a type.
//
// The pair is read with unaligned pointer accesses to the local copy `self` (methods take the
// vector by value), which are optimized to registers; do not call them on references into global
// memory. Elementwise operations are correct for an odd length because the padding lane of the
// last pair is dropped when the pair is written back; reductions must handle the last element
// explicitly, as a padding lane would be an extra operand.
impl<E: VectorElement<N> + VectorElement<2>, const N: usize> Vector<E, N> {
    /// Pair `i`, zero-padded for the last pair of a vector of odd length.
    #[inline(always)]
    pub(crate) const fn pair(&self, i: usize) -> Vector<E, 2> {
        const { Self::LAYOUT_OK };
        if const { N % 2 == 1 } && 2 * i + 1 == N {
            // SAFETY: odd lengths only exist for formats of whole bytes.
            unsafe { cast_unchecked::<[E; 2], Vector<E, 2>>([self.elem(2 * i), E::ZEROED]) }
        } else {
            // SAFETY: the pair is within the vector, and plain data.
            unsafe { self.bytes().cast::<Vector<E, 2>>().add(i).read_unaligned() }
        }
    }

    /// Sets pair `i`, dropping the padding lane for the last pair of a vector of odd length.
    #[inline(always)]
    pub(crate) const fn set_pair(&mut self, i: usize, pair: Vector<E, 2>) {
        const { Self::LAYOUT_OK };
        if const { N % 2 == 1 } && 2 * i + 1 == N {
            // SAFETY: see `pair`.
            let [lo, _]: [E; 2] = unsafe { cast_unchecked(pair) };
            self.set_elem(2 * i, lo);
        } else {
            // SAFETY: see `pair`.
            unsafe {
                self.bytes_mut()
                    .cast::<Vector<E, 2>>()
                    .add(i)
                    .write_unaligned(pair)
            }
        }
    }

    /// Builds a vector from its pairs, `f(i)` being pair `i`.
    #[inline(always)]
    pub(crate) fn from_pairs_with(mut f: impl FnMut(usize) -> Vector<E, 2>) -> Self {
        let mut out = Self::zeroed();
        #[rustc_unroll(full)]
        for i in 0..Self::PAIRS {
            out.set_pair(i, f(i));
        }
        out
    }

    /// Applies `f` to every pair.
    #[inline(always)]
    pub(crate) fn map_pairs(self, mut f: impl FnMut(Vector<E, 2>) -> Vector<E, 2>) -> Self {
        Self::from_pairs_with(|i| f(self.pair(i)))
    }

    /// Applies `f` to every pair of corresponding pairs of `self` and `other`.
    #[inline(always)]
    pub(crate) fn zip_pairs(
        self,
        other: Self,
        mut f: impl FnMut(Vector<E, 2>, Vector<E, 2>) -> Vector<E, 2>,
    ) -> Self {
        Self::from_pairs_with(|i| f(self.pair(i), other.pair(i)))
    }

    /// Applies `f` to every triple of corresponding pairs of `self`, `b` and `c`.
    #[inline(always)]
    pub(crate) fn zip3_pairs(
        self,
        b: Self,
        c: Self,
        mut f: impl FnMut(Vector<E, 2>, Vector<E, 2>, Vector<E, 2>) -> Vector<E, 2>,
    ) -> Self {
        Self::from_pairs_with(|i| f(self.pair(i), b.pair(i), c.pair(i)))
    }

    /// Reduces the elements to one with `pair_op` combining pairs lane by lane and `lane_op`
    /// combining two elements.
    ///
    /// The `N / 2` whole pairs are combined with `pair_op` in a tree, the two lanes of the
    /// remaining pair with `lane_op`, and then the last element of a vector of odd length.
    ///
    /// The pairs are kept in a scratch array of the maximum number of pairs, used as a ring
    /// buffer holding a FIFO queue: step `k` pops the pairs at queue positions `2 * k` and
    /// `2 * k + 1` and pushes their combination at queue position `N / 2 + k`, i.e. into pair
    /// `k`. All indices are constant after unrolling, so the unused entries vanish.
    #[inline(always)]
    pub(crate) fn reduce_pairs(
        self,
        pair_op: impl Fn(Vector<E, 2>, Vector<E, 2>) -> Vector<E, 2>,
        lane_op: impl Fn(E, E) -> E,
    ) -> E {
        if const { N == 1 } {
            return self.elem(0);
        }
        let full = N / 2;
        let mut x = [Vector::<E, 2>::ALL_ZERO; 32];
        #[rustc_unroll(full)]
        #[allow(clippy::needless_range_loop)]
        for i in 0..full {
            x[i] = self.pair(i);
        }
        #[rustc_unroll(full)]
        for k in 0..full - 1 {
            let (a, b) = (2 * k % full, (2 * k + 1) % full);
            x[k] = pair_op(x[a], x[b]);
        }
        let last = x[(2 * full - 2) % full];
        let reduced = lane_op(last.lo(), last.hi());
        if const { N % 2 == 1 } {
            lane_op(reduced, self.elem(N - 1))
        } else {
            reduced
        }
    }
}

// The primitives over groups of four elements, for the conversions that process four elements
// with one instruction: a quad is two pairs.
impl<E: VectorElement<N> + VectorElement<2> + VectorElement<4>, const N: usize> Vector<E, N> {
    /// Quad `i`, the elements `4 * i` to `4 * i + 3`, zero-padded past the end of the vector.
    #[inline(always)]
    pub(crate) fn quad(&self, i: usize) -> Vector<E, 4> {
        if 4 * i + 4 <= N {
            // SAFETY: the quad is within the vector, and plain data.
            return unsafe { self.bytes().cast::<Vector<E, 4>>().add(i).read_unaligned() };
        }
        let lo = self.pair(2 * i);
        let hi = if 2 * i + 1 < Self::PAIRS {
            self.pair(2 * i + 1)
        } else {
            Vector::<E, 2>::ALL_ZERO
        };
        Vector::<E, 4>::from_x2s([lo, hi])
    }

    /// Sets quad `i`, dropping the elements past the end of the vector.
    #[inline(always)]
    pub(crate) fn set_quad(&mut self, i: usize, quad: Vector<E, 4>) {
        if 4 * i + 4 <= N {
            // SAFETY: see `quad`.
            return unsafe {
                self.bytes_mut()
                    .cast::<Vector<E, 4>>()
                    .add(i)
                    .write_unaligned(quad)
            };
        }
        let [lo, hi] = quad.to_x2s();
        self.set_pair(2 * i, lo);
        if 2 * i + 1 < Self::PAIRS {
            self.set_pair(2 * i + 1, hi);
        }
    }

    /// Builds a vector from its quads, `f(i)` being quad `i`.
    #[inline(always)]
    pub(crate) fn from_quads_with(mut f: impl FnMut(usize) -> Vector<E, 4>) -> Self {
        let mut out = Self::zeroed();
        #[rustc_unroll(full)]
        for i in 0..Self::QUADS {
            out.set_quad(i, f(i));
        }
        out
    }

    /// Applies `f` to every quad.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn map_quads(self, mut f: impl FnMut(Vector<E, 4>) -> Vector<E, 4>) -> Self {
        Self::from_quads_with(|i| f(self.quad(i)))
    }
}

impl<E: VectorElement<2>> Vector<E, 2> {
    /// Creates a pair from its elements.
    #[inline(always)]
    pub const fn new(lo: E, hi: E) -> Self {
        Self::from_array([lo, hi])
    }

    /// Returns the first element.
    #[inline(always)]
    pub const fn lo(self) -> E {
        self.to_array()[0]
    }

    /// Returns the second element.
    #[inline(always)]
    pub const fn hi(self) -> E {
        self.to_array()[1]
    }
}

/// Copies `bytes` bytes (a multiple of `size_of::<T>()`) from `src` to `dst` in units of `T`.
///
/// # Safety
///
/// `src` must be valid for reads and `dst` for writes of `bytes` bytes, without overlap.
#[inline(always)]
const unsafe fn copy_units<T: Copy>(src: *const u8, dst: *mut u8, bytes: usize) {
    let (src, dst) = (src.cast::<T>(), dst.cast::<T>());
    let mut i = 0;
    #[rustc_unroll(full)]
    while i < bytes / size_of::<T>() {
        // SAFETY: guaranteed by the caller.
        unsafe { dst.add(i).write_unaligned(src.add(i).read_unaligned()) };
        i += 1;
    }
}

// The halves are copied in unrolled units of up to 8 bytes through pointers to local copies: a
// cast of `[Half; 2]` or a `copy_nonoverlapping` leaves a copy in local memory for vectors of 48
// bytes and more.
impl<E: EvenVectorElement<N>, const N: usize> Vector<E, N>
where
    VectorLen<N>: EvenVectorLen,
{
    /// The size of a half in bytes.
    const HALF_BYTES: usize = size_of::<Self>() / 2;

    /// Copies a half from `src` to `dst`, in the largest units that divide its size.
    ///
    /// # Safety
    ///
    /// `src` must be valid for reads and `dst` for writes of a half, without overlap.
    #[inline(always)]
    const unsafe fn copy_half(src: *const u8, dst: *mut u8) {
        const { assert!(2 * size_of::<Half<E, N>>() == size_of::<Self>()) };
        // SAFETY: guaranteed by the caller.
        unsafe {
            if const {
                Self::HALF_BYTES.is_multiple_of(8)
                    && (size_of::<E::PairBits>() == 8 || Self::HALF_BYTES == 8)
            } {
                copy_units::<u64>(src, dst, Self::HALF_BYTES)
            } else if const { Self::HALF_BYTES.is_multiple_of(4) } {
                copy_units::<u32>(src, dst, Self::HALF_BYTES)
            } else if const { Self::HALF_BYTES.is_multiple_of(2) } {
                copy_units::<u16>(src, dst, Self::HALF_BYTES)
            } else {
                copy_units::<u8>(src, dst, Self::HALF_BYTES)
            }
        }
    }

    /// Creates a value from two halves; `lo` becomes the first `N / 2` elements and `hi` the last
    /// `N / 2`.
    ///
    /// Pairs have a constructor of the same name taking two elements, so when `N` is not known
    /// from the context, write `Vector::<E, N>::new(lo, hi)` or use one of the aliases.
    #[inline(always)]
    pub const fn new(lo: Half<E, N>, hi: Half<E, N>) -> Self {
        let mut out = Self::ALL_ZERO;
        let dst = out.bytes_mut();
        // SAFETY: the halves and the two halves of the vector have the same size, as checked by
        // `copy_half`.
        unsafe {
            Self::copy_half(lo.bytes(), dst);
            Self::copy_half(hi.bytes(), dst.add(Self::HALF_BYTES));
        }
        out
    }

    /// Returns half `which`, the elements from `which * N / 2`.
    #[inline(always)]
    const fn half(self, which: usize) -> Half<E, N> {
        let mut half = Half::<E, N>::ALL_ZERO;
        // SAFETY: see `new`.
        unsafe { Self::copy_half(self.bytes().add(which * Self::HALF_BYTES), half.bytes_mut()) };
        half
    }

    /// Returns the lower half, the first `N / 2` elements.
    #[inline(always)]
    pub const fn lo(self) -> Half<E, N> {
        self.half(0)
    }

    /// Returns the upper half, the last `N / 2` elements.
    #[inline(always)]
    pub const fn hi(self) -> Half<E, N> {
        self.half(1)
    }
}

impl<E: VectorElement<N>, const N: usize> Default for Vector<E, N> {
    /// All elements set to the default of the element type.
    #[inline(always)]
    fn default() -> Self {
        Self::splat(E::default())
    }
}

impl<E: VectorElement<N> + fmt::Debug, const N: usize> fmt::Debug for Vector<E, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Prints like `f16x4(1.0, 2.0, 3.0, 4.0)`: the name is built from the pieces, and the
        // unnamed tuple provides the (alternate-aware) list of elements.
        write!(f, "{}x{}", E::NAME, N)?;
        let mut t = f.debug_tuple("");
        for e in self.to_array() {
            t.field(&e);
        }
        t.finish()
    }
}

impl<E: VectorElement<N>, const N: usize> From<[E; N]> for Vector<E, N> {
    #[inline(always)]
    fn from(x: [E; N]) -> Self {
        Self::from_array(x)
    }
}

/// Makes the scalar `$elem` a [`VectorElement`] and defines the aliases of its vectors, their
/// splatted constants and the conversions to arrays.
///
/// - `name` (optional): the documentation text naming the scalar, a link to `crate::$elem` by
///   default.
/// - `unit`: the integer unit of the elements: `u32`, `u16` or `u8` for the formats whose
///   elements are 4, 2 or 1 bytes, and `nibble` for `f4e2m1`. It selects the storage, the
///   integer holding a pair of elements and the raw bits of the vectors.
/// - `packing`: `Bytes` if each element occupies a whole number of bytes, or `Nibbles` if two
///   elements share a byte (`f4e2m1`, with `unit: nibble`). A `Nibbles` element is a one-byte
///   scalar whose `from_bits` masks the low nibble.
/// - `consts`: the constants of the scalar to re-export splatted on each vector.
/// - `note` (optional): extra documentation paragraph added to every alias.
/// - `eq` (optional): `elementwise` adds element-wise `PartialEq` (only the `f32`, `f16` and
///   `bf16` vectors have it).
/// - `aliases`: the names of the generic alias and of the aliases of 2, 4, 8, 16, 32 and 64
///   elements.
///
/// The formats with `Bytes` packing are elements of the vectors of every length; the vectors of
/// `Nibbles` have the even lengths from 2 to 64, and code generic over the length needs the bound
/// `$elem: VectorElement<N>` to name them.
///
/// ```ignore
/// vector_element! {
///     elem: f8e4m3, unit: u8, packing: Bytes,
///     consts: [ZERO, ONE],
///     note: "Format description.",
///     aliases: f8e4m3xN, f8e4m3x2, f8e4m3x4, f8e4m3x8, f8e4m3x16, f8e4m3x32, f8e4m3x64,
/// }
/// ```
macro_rules! vector_element {
    (
        elem: $elem:ident, $(name: $name:literal,)? unit: $unit:ident, packing: $packing:ident,
        consts: [$($const:ident),* $(,)?],
        $(note: $note:literal,)?
        $(eq: $eq:ident,)?
        aliases: $xn:ident, $x2:ident, $x4:ident, $x8:ident, $x16:ident, $x32:ident, $x64:ident $(,)?
    ) => {
        impl $crate::data_types::vector::sealed::Sealed for $elem {}

        impl $crate::VectorScalar for $elem {
            type PairBits = vector_element!(@pair $unit);
            type BitsUnit = vector_element!(@bits_unit $unit);
            const ZEROED: Self = <$elem>::from_bits(0);
            const PACKING: $crate::Packing = $crate::Packing::$packing;
            const NAME: &'static str = stringify!($elem);
        }

        vector_element!(@storage $packing, $elem, $unit);

        #[doc = concat!("`N` packed ", vector_element!(@name $elem $($name)?), " values.")]
        ///
        #[doc = concat!("An alias of [`Vector<", stringify!($elem), ", N>`](crate::Vector).")]
        ///
        #[doc = vector_element!(@pack_doc $packing)]
        $(
            ///
            #[doc = $note]
        )?
        pub type $xn<const N: usize> = $crate::Vector<$elem, N>;

        vector_element!(@alias $xn, $x2, 2, "Two", $elem, [$($name)?], $packing, [$($note)?]);
        vector_element!(@alias $xn, $x4, 4, "Four", $elem, [$($name)?], $packing, [$($note)?]);
        vector_element!(@alias $xn, $x8, 8, "Eight", $elem, [$($name)?], $packing, [$($note)?]);
        vector_element!(@alias $xn, $x16, 16, "Sixteen", $elem, [$($name)?], $packing, [$($note)?]);
        vector_element!(@alias $xn, $x32, 32, "Thirty-two", $elem, [$($name)?], $packing, [$($note)?]);
        vector_element!(@alias $xn, $x64, 64, "Sixty-four", $elem, [$($name)?], $packing, [$($note)?]);

        const _: () = {
            assert!(size_of::<$x2>() == align_of::<$x2>());
            assert!(size_of::<$x4>() == align_of::<$x4>());
            assert!(size_of::<$x8>() == align_of::<$x8>());
            assert!(size_of::<$x16>() == align_of::<$x16>());
            assert!(size_of::<$x32>() == align_of::<$x32>());
            assert!(size_of::<$x64>() == align_of::<$x64>());
        };

        impl<const N: usize> $crate::Vector<$elem, N>
        where
            $elem: $crate::VectorElement<N>,
        {
            $(
                #[doc = concat!("All elements equal to [`", stringify!($elem), "::", stringify!($const), "`](crate::", stringify!($elem), "::", stringify!($const), ").")]
                pub const $const: Self = Self::splat($elem::$const);
            )*
        }

        impl<const N: usize> ::core::convert::From<$crate::Vector<$elem, N>> for [$elem; N]
        where
            $elem: $crate::VectorElement<N>,
        {
            #[inline(always)]
            fn from(x: $crate::Vector<$elem, N>) -> Self {
                x.to_array()
            }
        }

        $(vector_element!(@eq $eq $elem);)?
    };

    (
        @alias $xn:ident, $ty:ident, $n:literal, $count:literal, $elem:ident, [$($name:literal)?],
        $packing:ident, [$($note:literal)?]
    ) => {
        #[doc = concat!($count, " packed ", vector_element!(@name $elem $($name)?), " values, aligned to their size.")]
        ///
        #[doc = concat!("An alias of [`", stringify!($xn), "<", stringify!($n), ">`](crate::", stringify!($xn), ").")]
        ///
        #[doc = vector_element!(@pack_doc $packing)]
        $(
            ///
            #[doc = $note]
        )?
        pub type $ty = $xn<$n>;
    };

    (@name $elem:ident) => {
        concat!("[`", stringify!($elem), "`](crate::", stringify!($elem), ")")
    };
    (@name $elem:ident $name:literal) => { $name };

    (@pack_doc Bytes) => { "Element 0 occupies the least significant bits." };
    (@pack_doc Nibbles) => {
        "Two elements share each byte; element 0 occupies the low nibble of the first byte."
    };

    (@pair u8) => { u16 };
    (@pair u16) => { u32 };
    (@pair u32) => { u64 };
    (@pair nibble) => { u8 };

    (@bits_unit u8) => { u8 };
    (@bits_unit u16) => { u16 };
    (@bits_unit u32) => { u32 };
    (@bits_unit nibble) => { $crate::NibbleBits };

    // The elements are whole bytes, so there is a vector of every length.
    (@storage Bytes, $elem:ident, $unit:ident) => {
        impl<const N: usize> $crate::VectorElement<N> for $elem {
            type Storage = $crate::VectorStorage<$unit, N>;
        }
    };
    // Two elements per byte: only the even lengths exist, as one impl each.
    (@storage Nibbles, $elem:ident, $unit:ident) => {
        vector_element!(@nibbles $elem;
            2 => 1; 4 => 2; 6 => 3; 8 => 4; 10 => 5; 12 => 6; 14 => 7; 16 => 8; 18 => 9; 20 => 10;
            22 => 11; 24 => 12; 26 => 13; 28 => 14; 30 => 15; 32 => 16; 34 => 17; 36 => 18;
            38 => 19; 40 => 20; 42 => 21; 44 => 22; 46 => 23; 48 => 24; 50 => 25; 52 => 26;
            54 => 27; 56 => 28; 58 => 29; 60 => 30; 62 => 31; 64 => 32;
        );
    };
    (@nibbles $elem:ident; $($n:literal => $bytes:literal;)*) => {$(
        impl $crate::VectorElement<$n> for $elem {
            type Storage = $crate::VectorStorage<u8, $bytes>;
        }
    )*};

    (@eq elementwise $elem:ident) => {
        impl<const N: usize> ::core::cmp::PartialEq for $crate::Vector<$elem, N>
        where
            $elem: $crate::VectorElement<N>,
        {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                self.to_array() == other.to_array()
            }
        }
    };
}

/// Implements, for the vectors of the scalar `$elem` and each given operator, the operator with a
/// scalar operand on either side, which is broadcast with [`Vector::splat`], and the compound
/// assignment operators with a vector and with a scalar operand. The operator between two vectors
/// must be implemented separately.
///
/// ```ignore
/// broadcast_ops!(f16: Add add AddAssign add_assign, Mul mul MulAssign mul_assign);
/// ```
macro_rules! broadcast_ops {
    ($elem:ident: $($op:ident $method:ident $assign:ident $assign_method:ident),* $(,)?) => {$(
        impl<const N: usize> ::core::ops::$op<$elem> for $crate::Vector<$elem, N>
        where
            $elem: $crate::VectorElement<N>,
        {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: $elem) -> Self {
                ::core::ops::$op::$method(self, Self::splat(rhs))
            }
        }

        impl<const N: usize> ::core::ops::$op<$crate::Vector<$elem, N>> for $elem
        where
            $elem: $crate::VectorElement<N>,
        {
            type Output = $crate::Vector<$elem, N>;

            #[inline(always)]
            fn $method(self, rhs: $crate::Vector<$elem, N>) -> Self::Output {
                ::core::ops::$op::$method($crate::Vector::splat(self), rhs)
            }
        }

        impl<const N: usize> ::core::ops::$assign for $crate::Vector<$elem, N>
        where
            $elem: $crate::VectorElement<N>,
        {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: Self) {
                *self = ::core::ops::$op::$method(*self, rhs);
            }
        }

        impl<const N: usize> ::core::ops::$assign<$elem> for $crate::Vector<$elem, N>
        where
            $elem: $crate::VectorElement<N>,
        {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: $elem) {
                *self = ::core::ops::$op::$method(*self, rhs);
            }
        }
    )*};
}

/// Implements what the `f16` and `bf16` vectors share, for the scalar `$elem` and the name
/// `$ptx` of its PTX type: `from_f32`, `from_f32_with`, `reduce_sum`, the [`broadcast_ops!`] of
/// `+`, `-` and `*`, and the conversion to the `f32` vectors, all generic over the length `N`.
///
/// The scalar must provide `fn add_to_f32(self, acc: f32) -> f32` and the pair type
/// `fn pair_from_f32_with`, with the signature of that of `f16`, and
/// `fn add_to_f32x2(self, acc: [f32; 2]) -> [f32; 2]`, for `sm_100`, which add with the
/// mixed-precision `add.rn.f32.$ptx`.
macro_rules! half_vector {
    ($elem:ident, $ptx:literal) => {
        impl<const N: usize> $crate::Vector<$elem, N> {
            #[doc = concat!("Converts `f32` values to [`", stringify!($elem), "`](crate::", stringify!($elem), "), rounding to nearest, ties to even.")]
            ///
            /// Element `i` of the result is converted from element `i` of `x`.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn from_f32(x: $crate::f32xN<N>) -> Self {
                Self::from_f32_with::<
                    { $crate::Rounding::NearestEven },
                    { $crate::Relu::Disabled },
                    { $crate::Overflow::Infinity },
                >(x)
            }

            #[doc = concat!("Converts `f32` values to [`", stringify!($elem), "`](crate::", stringify!($elem), "), with explicit rounding and modifiers.")]
            ///
            /// Element `i` of the result is converted from element `i` of `x`.
            ///
            /// - `RND` selects the rounding mode and must be [`Rounding::NearestEven`] or
            ///   [`Rounding::TowardZero`].
            /// - `R` optionally clamps negative results to zero (NaN results become canonical NaN).
            /// - `O` selects whether out-of-range results become infinity or are clamped to the
            #[doc = concat!("  finite range of `", stringify!($elem), "` (`.satfinite`; NaN inputs still produce NaN).")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn from_f32_with<
                const RND: $crate::Rounding,
                const R: $crate::Relu,
                const O: $crate::Overflow,
            >(
                x: $crate::f32xN<N>,
            ) -> Self {
                const {
                    assert!(
                        matches!(
                            RND,
                            $crate::Rounding::NearestEven | $crate::Rounding::TowardZero
                        ),
                        "vector conversion only supports NearestEven and TowardZero rounding"
                    );
                }
                Self::from_pairs_with(|i| {
                    $crate::Vector::<$elem, 2>::pair_from_f32_with::<RND, R, O>(
                        x.pair(i).to_array(),
                    )
                })
            }

            /// Sums all elements in `f32`.
            ///
            /// The elements are converted to `f32` (exactly) and added with `add.rn.f32` (round to
            /// nearest, ties to even; subnormal values are preserved) in an unspecified order,
            /// which may differ between architectures. On `sm_100` and later, each element is
            #[doc = concat!("added with the mixed-precision `add.rn.f32.", $ptx, "`, which fuses the conversion into the addition.")]
            ///
            /// On `sm_100`, the elements are accumulated directly with the mixed-precision
            /// instruction, in two independent chains for the even and the odd elements (a
            /// single chain for up to three elements); the last element of a vector of odd length
            /// is added to the chain of the even elements. The chains start at `-0.0`, which
            /// every element (including `-0.0`) is an identity of.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add>
            #[inline(always)]
            pub fn reduce_sum(self) -> f32 {
                #[cfg(target_feature = "sm_100")]
                {
                    if const { N <= 3 } {
                        let mut acc = -0.0;
                        #[rustc_unroll(full)]
                        for x in self.to_array() {
                            acc = x.add_to_f32(acc);
                        }
                        acc
                    } else {
                        let mut acc = [-0.0; 2];
                        #[rustc_unroll(full)]
                        for i in 0..N / 2 {
                            acc = self.pair(i).add_to_f32x2(acc);
                        }
                        if const { N % 2 == 1 } {
                            acc[0] = self.elem(N - 1).add_to_f32(acc[0]);
                        }
                        $crate::F32Ext::add_with::<
                            { $crate::Rounding::NearestEven },
                            { $crate::Denormals::Preserve },
                            { $crate::Saturation::None },
                        >(acc[0], acc[1])
                    }
                }
                #[cfg(not(target_feature = "sm_100"))]
                $crate::F32Ext::reduce_sum(self.to_f32())
            }
        }

        $crate::data_types::vector::broadcast_ops!(
            $elem: Add add AddAssign add_assign, Sub sub SubAssign sub_assign,
            Mul mul MulAssign mul_assign,
        );

        impl<const N: usize> ::core::convert::From<$crate::Vector<$elem, N>> for $crate::f32xN<N> {
            #[inline(always)]
            fn from(x: $crate::Vector<$elem, N>) -> Self {
                x.to_f32()
            }
        }
    };
}

pub(crate) use {broadcast_ops, half_vector, narrow_scalar, vector_element};
