// --- LLM-generated --- //
//! Shared implementation of the container API of the floating-point scalar and vector types.
//!
//! Every format has a scalar and vectors of 2, 4, 8, 16 and 32 elements, all aligned to their own
//! size. Element 0 always occupies the least significant bits (the machine is little-endian, so
//! it is also the first element in memory).
//!
//! All vectors of all formats are one type, [`Vector<E, N>`], where `E` implements
//! [`VectorElement`] and `N` is a length supported by [`SupportedVectorLen`] (`VectorLen<N>`). The
//! familiar names (`f16x4`, `f8e4m3x32`, ...) are aliases. The vector is stored as an array of
//! pairs (`N / 2` words of the unit of the packed PTX instructions, such as `.f16x2` or `.e4m3x2`),
//! inside a storage struct aligned to the size of the whole vector.
//!
//! - [`narrow_scalar!`] defines the one-byte scalars of the narrow formats (FP8, FP6, FP4, s2f6).
//!   `f16` and `bf16` are defined by hand.
//! - [`vector_element!`] makes a scalar a [`VectorElement`] and defines the aliases, the splatted
//!   constants and the `From` conversion to arrays of its vectors.
//! - [`half_vector!`] adds what the `f16` and `bf16` vectors share: the `f32` conversions,
//!   `reduce_sum`, the scalar broadcast operators and the conversion to `Simd<f32, N>`.
//!
//! Format specific operations are written once per format as private methods on the pair type
//! (`Vector<E, 2>`), and once, generically over the length, as public methods that map over the
//! pairs with [`Vector::map_pairs`] and [`Vector::zip_pairs`]. A pair type and a generic impl may
//! not define the same name, so the private methods carry a `pair_` prefix. The macros expand to
//! code that accesses private items, so they must be invoked in the module that defines the scalar
//! type.

use bytemuck::{Pod, Zeroable};
use core::fmt;

use super::array_from_fn;

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

/// The number of elements of a [`Vector`]. `VectorLen<N>: SupportedVectorLen` is the bound that
/// code generic over the length of a vector needs.
pub struct VectorLen<const N: usize>;

/// Implemented for [`VectorLen`] of `2`, `4`, `8`, `16` and `32`, the lengths of the vectors.
/// Sealed.
///
/// Code that is generic over the length writes the bound `VectorLen<N>: SupportedVectorLen`, and
/// names the types that depend on it through the aliases [`Pairs`], [`Quads`] and [`Half`].
pub trait SupportedVectorLen: sealed::Sealed {
    /// The number of pairs of elements, `N / 2`, usable in the types of arrays.
    #[rustc_always_gca]
    const PAIRS: usize;

    /// An array of `N / 2` pair words of `B`, aligned to its size.
    type Storage<B: PairBits>: AlignedStorage;

    /// The vector of `N / 2` elements of `E`, or the scalar `E` itself if `N` is `2`.
    type Half<E: VectorElement>: Copy + Pod;
}

/// Implemented for [`VectorLen`] of `4`, `8`, `16` and `32`, the lengths that are multiples of
/// four, for the conversions that process four elements at a time. Sealed.
pub trait QuadVectorLen: SupportedVectorLen {
    /// The number of groups of four elements, `N / 4`, usable in the types of arrays.
    #[rustc_always_gca]
    const QUADS: usize;
}

/// An array of `T` with one element per pair of elements of a vector of `N` elements.
pub type Pairs<T, const N: usize> = [T; <VectorLen<N> as SupportedVectorLen>::PAIRS];

/// An array of `T` with one element per group of four elements of a vector of `N` elements.
pub type Quads<T, const N: usize> = [T; <VectorLen<N> as QuadVectorLen>::QUADS];

/// The vector of half the length of an `N`-element vector of `E` (the scalar `E` for `N == 2`).
pub type Half<E, const N: usize> = <VectorLen<N> as SupportedVectorLen>::Half<E>;

/// The unsigned integer (or array of `u128`) with the size of an `N`-element vector of `E`, the
/// type of its raw bits: `u8` ... `u128`, `[u128; 2]` or `[u128; 4]`.
pub type Bits<E, const N: usize> = <Storage<E, N> as AlignedStorage>::Bits;

type Storage<E, const N: usize> =
    <VectorLen<N> as SupportedVectorLen>::Storage<<E as VectorElement>::PairBits>;

macro_rules! vector_lens {
    ($($n:literal => pairs: $pairs:literal, storage: $storage:ident, half: $half:tt),* $(,)?) => {$(
        impl sealed::Sealed for VectorLen<$n> {}

        impl SupportedVectorLen for VectorLen<$n> {
            const PAIRS: usize = core::gca!($pairs);
            type Storage<B: PairBits> = B::$storage;
            type Half<E: VectorElement> = vector_lens!(@half E, $half);
        }
    )*};
    (@half $e:ident, scalar) => { $e };
    (@half $e:ident, $m:literal) => { Vector<$e, $m> };
}

vector_lens! {
    2 => pairs: 1, storage: StorageX1, half: scalar,
    4 => pairs: 2, storage: StorageX2, half: 2,
    8 => pairs: 4, storage: StorageX4, half: 4,
    16 => pairs: 8, storage: StorageX8, half: 8,
    32 => pairs: 16, storage: StorageX16, half: 16,
}

macro_rules! quad_vector_lens {
    ($($n:literal => $quads:literal),* $(,)?) => {$(
        impl QuadVectorLen for VectorLen<$n> {
            const QUADS: usize = core::gca!($quads);
        }
    )*};
}

quad_vector_lens!(4 => 1, 8 => 2, 16 => 4, 32 => 8);

/// An array of pair words aligned to its size, the storage of a [`Vector`]. Implemented by the
/// storage structs, one per pair width and number of pairs. Sealed.
pub trait AlignedStorage: Copy + Pod + sealed::Sealed {
    /// The unsigned integer (or array of `u128`) with the size of the storage.
    type Bits: Copy + Pod;
}

/// The unit of the packed PTX instructions of an element type: the unsigned integer holding a pair
/// of elements, `u32` for the 16-bit formats, `u16` for the 8-bit formats and `u8` for the 4-bit
/// formats. Sealed.
pub trait PairBits: Copy + Pod + sealed::Sealed {
    /// All bits zero.
    const ZEROED: Self;
    /// Storage of one pair.
    type StorageX1: AlignedStorage;
    /// Storage of two pairs.
    type StorageX2: AlignedStorage;
    /// Storage of four pairs.
    type StorageX4: AlignedStorage;
    /// Storage of eight pairs.
    type StorageX8: AlignedStorage;
    /// Storage of sixteen pairs.
    type StorageX16: AlignedStorage;
}

macro_rules! pair_storage {
    ($bits:ident: $($name:ident($pairs:literal, $align:literal, $raw:ty)),* $(,)?) => {
        $(
            #[doc = concat!("Storage of ", stringify!($pairs), " pairs of `", stringify!($bits), "`, aligned to its size.")]
            #[derive(Clone, Copy, Pod, Zeroable)]
            #[repr(C, align($align))]
            #[doc(hidden)]
            pub struct $name([$bits; $pairs]);

            impl sealed::Sealed for $name {}

            impl AlignedStorage for $name {
                type Bits = $raw;
            }
        )*

        impl sealed::Sealed for $bits {}

        impl PairBits for $bits {
            const ZEROED: Self = 0;
            pair_storage!(@assoc $($name),*);
        }
    };
    (@assoc $x1:ident, $x2:ident, $x4:ident, $x8:ident, $x16:ident) => {
        type StorageX1 = $x1;
        type StorageX2 = $x2;
        type StorageX4 = $x4;
        type StorageX8 = $x8;
        type StorageX16 = $x16;
    };
}

pair_storage!(u8: U8x1(1, 1, u8), U8x2(2, 2, u16), U8x4(4, 4, u32), U8x8(8, 8, u64), U8x16(16, 16, u128));
pair_storage!(
    u16: U16x1(1, 2, u16), U16x2(2, 4, u32), U16x4(4, 8, u64), U16x8(8, 16, u128),
    U16x16(16, 32, [u128; 2])
);
pair_storage!(
    u32: U32x1(1, 4, u32), U32x2(2, 8, u64), U32x4(4, 16, u128), U32x8(8, 32, [u128; 2]),
    U32x16(16, 64, [u128; 4])
);

/// How the elements of an element type are packed into the bytes of a [`Vector`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Packing {
    /// Every element occupies one or more whole bytes: an array of the elements has the layout of
    /// the vector.
    Bytes,
    /// Two elements share each byte, the first in the low nibble (the 4-bit formats).
    Nibbles,
}

/// Implemented for the scalar types of the floating-point formats, the element types of
/// [`Vector`]. Sealed.
pub trait VectorElement: Copy + Pod + Default + sealed::Sealed {
    /// The unsigned integer holding a pair of elements.
    type PairBits: PairBits;
    /// The all-zero bit pattern, which is also what `Default` returns.
    const ZEROED: Self;
    /// How the elements are packed into a vector.
    const PACKING: Packing;
    /// The name of the format, as printed by the `Debug` implementation of the vector.
    const NAME: &'static str;
}

/// `N` packed values of the element type `E`, aligned to their size.
///
/// `N` is `2`, `4`, `8`, `16` or `32`. The element types are the floating-point formats of this
/// crate, and each has aliases of its vector types: [`f16xN`](crate::f16xN),
/// [`f16x2`](crate::f16x2), ..., and the same for every other format. Code that is generic over
/// the length bounds `VectorLen<N>: SupportedVectorLen`, and may be generic over the element type
/// with `E: VectorElement`.
///
/// Element 0 occupies the least significant bits. Vectors are stored as arrays of pairs of
/// elements, the unit of the PTX instructions operating on packed values, see
/// [`from_x2s`](Self::from_x2s).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(transparent)]
pub struct Vector<E, const N: usize>(Storage<E, N>)
where
    E: VectorElement,
    VectorLen<N>: SupportedVectorLen;

impl<E: VectorElement, const N: usize> Vector<E, N>
where
    VectorLen<N>: SupportedVectorLen,
{
    /// The number of elements.
    pub const LEN: usize = N;

    /// The number of pairs of elements, `N / 2`.
    pub const PAIRS: usize = <VectorLen<N> as SupportedVectorLen>::PAIRS;

    /// Checks the layout assumptions of [`from_array`](Self::from_array) and
    /// [`to_array`](Self::to_array) for the packing of the element type.
    const PACKING_OK: () = assert!(match E::PACKING {
        Packing::Bytes => size_of::<[E; N]>() == size_of::<Self>(),
        Packing::Nibbles => size_of::<E>() == 1 && size_of::<E::PairBits>() == 1,
    });

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
    /// `2 * i` and `2 * i + 1`.
    #[inline(always)]
    pub const fn from_pair_bits(pairs: Pairs<E::PairBits, N>) -> Self {
        // SAFETY: see `from_bits`.
        Self(unsafe { cast(pairs) })
    }

    /// Returns the raw bits of the pairs of elements; `pairs[i]` holds elements `2 * i` and
    /// `2 * i + 1`.
    #[inline(always)]
    pub const fn to_pair_bits(self) -> Pairs<E::PairBits, N> {
        // SAFETY: see `from_bits`.
        unsafe { cast(self.0) }
    }

    /// Packs pairs of elements; `x2s[i]`, a `Vector<E, 2>`, holds elements `2 * i` and `2 * i + 1`.
    ///
    /// Pairs are the unit of the PTX instructions operating on packed values.
    #[inline(always)]
    pub const fn from_x2s(x2s: Pairs<Vector<E, 2>, N>) -> Self {
        // SAFETY: see `from_bits`.
        Self(unsafe { cast(x2s) })
    }

    /// Splits into pairs of elements; `x2s[i]`, a `Vector<E, 2>`, holds elements `2 * i` and
    /// `2 * i + 1`.
    #[inline(always)]
    pub const fn to_x2s(self) -> Pairs<Vector<E, 2>, N> {
        // SAFETY: see `from_bits`.
        unsafe { cast(self.0) }
    }

    /// Packs an array of elements.
    #[inline(always)]
    pub const fn from_array(x: [E; N]) -> Self {
        const { Self::PACKING_OK };
        if const { matches!(E::PACKING, Packing::Bytes) } {
            // SAFETY: the array has the layout of the vector, as checked by `PACKING_OK`.
            unsafe { cast_unchecked(x) }
        } else {
            let mut pairs = [E::PairBits::ZEROED; <VectorLen<N> as SupportedVectorLen>::PAIRS];
            let mut i = 0;
            #[rustc_unroll(full)]
            while i < pairs.len() {
                // SAFETY: nibble formats have one-byte elements and pairs, as checked by
                // `PACKING_OK`.
                let (lo, hi): (u8, u8) =
                    unsafe { (cast_unchecked(x[2 * i]), cast_unchecked(x[2 * i + 1])) };
                // The low element is masked in case it was created by a cast that bypassed the
                // masking `from_bits`; the high element loses its high bits in the shift.
                pairs[i] = unsafe { cast_unchecked::<u8, E::PairBits>((lo & 0xF) | (hi << 4)) };
                i += 1;
            }
            Self::from_pair_bits(pairs)
        }
    }

    /// Unpacks into an array of elements.
    #[inline(always)]
    pub const fn to_array(self) -> [E; N] {
        const { Self::PACKING_OK };
        if const { matches!(E::PACKING, Packing::Bytes) } {
            // SAFETY: see `from_array`.
            unsafe { cast_unchecked(self.0) }
        } else {
            let pairs = self.to_pair_bits();
            let mut out = [E::ZEROED; N];
            let mut i = 0;
            #[rustc_unroll(full)]
            while i < pairs.len() {
                // SAFETY: see `from_array`.
                let byte: u8 = unsafe { cast_unchecked(pairs[i]) };
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
        const { Self::PACKING_OK };
        // SAFETY: see `from_array`; a pair of byte elements is exactly one pair word.
        let pair: E::PairBits = unsafe {
            if const { matches!(E::PACKING, Packing::Bytes) } {
                cast_unchecked([x, x])
            } else {
                let x: u8 = cast_unchecked(x);
                cast_unchecked((x & 0xF) | (x << 4))
            }
        };
        Self::from_pair_bits([pair; <VectorLen<N> as SupportedVectorLen>::PAIRS])
    }

    /// Creates a value from two halves; `lo` becomes the first `N / 2` elements and `hi` the last
    /// `N / 2`. For `N == 2` the halves are single elements.
    #[inline(always)]
    pub const fn new(lo: Half<E, N>, hi: Half<E, N>) -> Self {
        const {
            assert!(N == 2 || 2 * size_of::<Half<E, N>>() == size_of::<Self>());
        }
        if const { N == 2 } {
            // SAFETY: for `N == 2` the halves are elements and the array is `[E; 2]`.
            let (lo, hi): (E, E) = unsafe { (cast_unchecked(lo), cast_unchecked(hi)) };
            Self::from_array(unsafe { cast_unchecked([lo, hi]) })
        } else {
            // SAFETY: the halves are plain data and together exactly fill `Self`, as checked above.
            unsafe { cast_unchecked([lo, hi]) }
        }
    }

    /// Returns the lower half, the first `N / 2` elements (element 0 for `N == 2`).
    #[inline(always)]
    pub const fn lo(self) -> Half<E, N> {
        const {
            assert!(N == 2 || 2 * size_of::<Half<E, N>>() == size_of::<Self>());
        }
        if const { N == 2 } {
            // SAFETY: for `N == 2` the half is an element and the array is `[E; 2]`.
            let elements: [E; 2] = unsafe { cast_unchecked(self.to_array()) };
            unsafe { cast_unchecked(elements[0]) }
        } else {
            // SAFETY: see `new`.
            let halves: [Half<E, N>; 2] = unsafe { cast_unchecked(self) };
            halves[0]
        }
    }

    /// Returns the upper half, the last `N / 2` elements (element 1 for `N == 2`).
    #[inline(always)]
    pub const fn hi(self) -> Half<E, N> {
        const {
            assert!(N == 2 || 2 * size_of::<Half<E, N>>() == size_of::<Self>());
        }
        if const { N == 2 } {
            // SAFETY: see `lo`.
            let elements: [E; 2] = unsafe { cast_unchecked(self.to_array()) };
            unsafe { cast_unchecked(elements[1]) }
        } else {
            // SAFETY: see `new`.
            let halves: [Half<E, N>; 2] = unsafe { cast_unchecked(self) };
            halves[1]
        }
    }

    /// Applies `f` to every pair of elements.
    #[inline(always)]
    pub(crate) fn map_pairs(self, mut f: impl FnMut(Vector<E, 2>) -> Vector<E, 2>) -> Self {
        let x = self.to_x2s();
        Self::from_x2s(array_from_fn(|i| f(x[i])))
    }

    /// Applies `f` to every pair of corresponding pairs of elements of `self` and `other`.
    #[inline(always)]
    pub(crate) fn zip_pairs(
        self,
        other: Self,
        mut f: impl FnMut(Vector<E, 2>, Vector<E, 2>) -> Vector<E, 2>,
    ) -> Self {
        let (x, y) = (self.to_x2s(), other.to_x2s());
        Self::from_x2s(array_from_fn(|i| f(x[i], y[i])))
    }
}

impl<E: VectorElement, const N: usize> Default for Vector<E, N>
where
    VectorLen<N>: SupportedVectorLen,
{
    /// All elements set to the default of the element type.
    #[inline(always)]
    fn default() -> Self {
        Self::splat(E::default())
    }
}

impl<E: VectorElement + fmt::Debug, const N: usize> fmt::Debug for Vector<E, N>
where
    VectorLen<N>: SupportedVectorLen,
{
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

impl<E: VectorElement, const N: usize> From<[E; N]> for Vector<E, N>
where
    VectorLen<N>: SupportedVectorLen,
{
    #[inline(always)]
    fn from(x: [E; N]) -> Self {
        Self::from_array(x)
    }
}

/// Makes the scalar `$elem` a [`VectorElement`] and defines the aliases of its vectors, their
/// splatted constants and the conversions to arrays.
///
/// - `pair`: the unsigned integer holding a pair of elements (`u32`, `u16` or `u8`).
/// - `packing`: `Bytes` if each element occupies a whole number of bytes, or `Nibbles` if two
///   elements share a byte (`f4e2m1`). A `Nibbles` element is a one-byte scalar whose `from_bits`
///   masks the low nibble.
/// - `consts`: the constants of the scalar to re-export splatted on each vector.
/// - `note` (optional): extra documentation paragraph added to every alias.
/// - `eq` (optional): `elementwise` adds element-wise `PartialEq` (only the `f16` and `bf16`
///   vectors have it).
/// - `aliases`: the names of the generic alias and of the aliases of 2, 4, 8, 16 and 32 elements.
///
/// ```ignore
/// vector_element! {
///     elem: f8e4m3, pair: u16, packing: Bytes,
///     consts: [ZERO, ONE],
///     note: "Format description.",
///     aliases: f8e4m3xN, f8e4m3x2, f8e4m3x4, f8e4m3x8, f8e4m3x16, f8e4m3x32,
/// }
/// ```
macro_rules! vector_element {
    (
        elem: $elem:ident, pair: $pair:ident, packing: $packing:ident,
        consts: [$($const:ident),* $(,)?],
        $(note: $note:literal,)?
        $(eq: $eq:ident,)?
        aliases: $xn:ident, $x2:ident, $x4:ident, $x8:ident, $x16:ident, $x32:ident $(,)?
    ) => {
        impl $crate::data_types::vector::sealed::Sealed for $elem {}

        impl $crate::VectorElement for $elem {
            type PairBits = $pair;
            const ZEROED: Self = <$elem>::from_bits(0);
            const PACKING: $crate::Packing = $crate::Packing::$packing;
            const NAME: &'static str = stringify!($elem);
        }

        #[doc = concat!("`N` packed [`", stringify!($elem), "`](crate::", stringify!($elem), ") values, aligned to their size.")]
        ///
        #[doc = concat!("An alias of [`Vector<", stringify!($elem), ", N>`](crate::Vector).")]
        ///
        #[doc = vector_element!(@pack_doc $packing)]
        $(
            ///
            #[doc = $note]
        )?
        pub type $xn<const N: usize> = $crate::Vector<$elem, N>;

        vector_element!(@alias $xn, $x2, 2, "Two", $elem, $packing, [$($note)?]);
        vector_element!(@alias $xn, $x4, 4, "Four", $elem, $packing, [$($note)?]);
        vector_element!(@alias $xn, $x8, 8, "Eight", $elem, $packing, [$($note)?]);
        vector_element!(@alias $xn, $x16, 16, "Sixteen", $elem, $packing, [$($note)?]);
        vector_element!(@alias $xn, $x32, 32, "Thirty-two", $elem, $packing, [$($note)?]);

        const _: () = {
            assert!(size_of::<$x2>() == align_of::<$x2>());
            assert!(size_of::<$x4>() == align_of::<$x4>());
            assert!(size_of::<$x8>() == align_of::<$x8>());
            assert!(size_of::<$x16>() == align_of::<$x16>());
            assert!(size_of::<$x32>() == align_of::<$x32>());
        };

        impl<const N: usize> $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            $(
                #[doc = concat!("All elements equal to [`", stringify!($elem), "::", stringify!($const), "`](crate::", stringify!($elem), "::", stringify!($const), ").")]
                pub const $const: Self = Self::splat($elem::$const);
            )*
        }

        impl<const N: usize> ::core::convert::From<$crate::Vector<$elem, N>> for [$elem; N]
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn from(x: $crate::Vector<$elem, N>) -> Self {
                x.to_array()
            }
        }

        $(vector_element!(@eq $eq $elem);)?
    };

    (@alias $xn:ident, $ty:ident, $n:literal, $count:literal, $elem:ident, $packing:ident, [$($note:literal)?]) => {
        #[doc = concat!($count, " packed [`", stringify!($elem), "`](crate::", stringify!($elem), ") values, aligned to their size.")]
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

    (@pack_doc Bytes) => { "Element 0 occupies the least significant bits." };
    (@pack_doc Nibbles) => {
        "Two elements share each byte; element 0 occupies the low nibble of the first byte."
    };

    (@eq elementwise $elem:ident) => {
        impl<const N: usize> ::core::cmp::PartialEq for $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                self.to_array() == other.to_array()
            }
        }
    };
}

/// Implements what the `f16` and `bf16` vectors share, for the scalar `$elem` and the name
/// `$ptx` of its PTX type: `from_f32`, `from_f32_with`, `reduce_sum`, the scalar broadcast
/// operators and the conversion to `Simd<f32, N>`, all generic over the length `N`.
///
/// The scalar must provide `fn add_to_f32(self, acc: f32) -> f32` and the pair type
/// `fn pair_from_f32_with`, with the signature of that of `f16`, and
/// `fn add_to_f32x2(self, acc: [f32; 2]) -> [f32; 2]`, for `sm_100`, which add with the
/// mixed-precision `add.rn.f32.$ptx`.
macro_rules! half_vector {
    ($elem:ident, $ptx:literal) => {
        impl<const N: usize> $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[doc = concat!("Converts `f32` values to [`", stringify!($elem), "`](crate::", stringify!($elem), "), rounding to nearest, ties to even.")]
            ///
            /// Element `i` of the result is converted from element `i` of `x`.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline(always)]
            pub fn from_f32(x: ::core::simd::Simd<f32, N>) -> Self {
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
                x: ::core::simd::Simd<f32, N>,
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
                let x = x.to_array();
                Self::from_x2s($crate::data_types::array_from_fn(|i| {
                    $crate::Vector::<$elem, 2>::pair_from_f32_with::<RND, R, O>([
                        x[2 * i],
                        x[2 * i + 1],
                    ])
                }))
            }

            /// Sums all elements in `f32`.
            ///
            /// The elements are converted to `f32` (exactly) and added with `add.rn.f32` (round to
            /// nearest, ties to even; subnormal values are preserved) in an unspecified order,
            /// which may differ between architectures. On `sm_100` and later, each element is
            #[doc = concat!("added with the mixed-precision `add.rn.f32.", $ptx, "`, which fuses the conversion into the addition.")]
            ///
            /// On `sm_100`, the elements are accumulated directly with the mixed-precision
            /// instruction, in two independent chains for the even and the odd elements. The
            /// chains start at `-0.0`, which every element (including `-0.0`) is an identity of.
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add>
            #[inline(always)]
            pub fn reduce_sum(self) -> f32 {
                #[cfg(target_feature = "sm_100")]
                {
                    let pairs = self.to_x2s();
                    if const { N == 2 } {
                        pairs[0].hi().add_to_f32(pairs[0].lo().add_to_f32(-0.0))
                    } else {
                        let mut acc = pairs[0].add_to_f32x2([-0.0; 2]);
                        #[rustc_unroll(full)]
                        for p in &pairs[1..] {
                            acc = p.add_to_f32x2(acc);
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

        impl<const N: usize> ::core::ops::Add<$elem> for $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            type Output = Self;

            #[inline(always)]
            fn add(self, rhs: $elem) -> Self {
                self + Self::splat(rhs)
            }
        }

        impl<const N: usize> ::core::ops::Sub<$elem> for $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            type Output = Self;

            #[inline(always)]
            fn sub(self, rhs: $elem) -> Self {
                self - Self::splat(rhs)
            }
        }

        impl<const N: usize> ::core::ops::Mul<$elem> for $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            type Output = Self;

            #[inline(always)]
            fn mul(self, rhs: $elem) -> Self {
                self * Self::splat(rhs)
            }
        }

        impl<const N: usize> ::core::ops::Add<$crate::Vector<$elem, N>> for $elem
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            type Output = $crate::Vector<$elem, N>;

            #[inline(always)]
            fn add(self, rhs: $crate::Vector<$elem, N>) -> Self::Output {
                $crate::Vector::splat(self) + rhs
            }
        }

        impl<const N: usize> ::core::ops::Sub<$crate::Vector<$elem, N>> for $elem
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            type Output = $crate::Vector<$elem, N>;

            #[inline(always)]
            fn sub(self, rhs: $crate::Vector<$elem, N>) -> Self::Output {
                $crate::Vector::splat(self) - rhs
            }
        }

        impl<const N: usize> ::core::ops::Mul<$crate::Vector<$elem, N>> for $elem
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            type Output = $crate::Vector<$elem, N>;

            #[inline(always)]
            fn mul(self, rhs: $crate::Vector<$elem, N>) -> Self::Output {
                $crate::Vector::splat(self) * rhs
            }
        }

        impl<const N: usize> ::core::ops::AddAssign for $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl<const N: usize> ::core::ops::SubAssign for $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl<const N: usize> ::core::ops::MulAssign for $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl<const N: usize> ::core::ops::AddAssign<$elem> for $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn add_assign(&mut self, rhs: $elem) {
                *self = *self + rhs;
            }
        }

        impl<const N: usize> ::core::ops::SubAssign<$elem> for $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: $elem) {
                *self = *self - rhs;
            }
        }

        impl<const N: usize> ::core::ops::MulAssign<$elem> for $crate::Vector<$elem, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: $elem) {
                *self = *self * rhs;
            }
        }

        impl<const N: usize> ::core::convert::From<$crate::Vector<$elem, N>>
            for ::core::simd::Simd<f32, N>
        where
            $crate::VectorLen<N>: $crate::SupportedVectorLen,
        {
            #[inline(always)]
            fn from(x: $crate::Vector<$elem, N>) -> Self {
                x.to_f32()
            }
        }
    };
}

pub(crate) use {half_vector, narrow_scalar, vector_element};
