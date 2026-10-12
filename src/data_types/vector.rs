// --- LLM-generated --- //
//! Shared implementation of the container API of the floating-point scalar and vector types.
//!
//! Every format has a scalar and vectors of 2, 4, 8, 16 and 32 elements, all aligned to their own
//! size. Element 0 always occupies the least significant bits (the machine is little-endian, so
//! it is also the first element in memory).
//!
//! - [`narrow_scalar!`] defines the one-byte scalars of the narrow formats (FP8, FP6, FP4, s2f6).
//!   `f16` and `bf16` are defined by hand.
//! - [`vector_types!`] defines `x2`, `x4`, `x8`, `x16` and `x32` of a format, with the API that
//!   does not depend on the format: raw bits, elements, halves, pairs, constants, `Default`,
//!   `Debug` and `From` with arrays.
//! - [`vector_partial_eq!`] adds element-wise `PartialEq` (only the `f16` and `bf16` vectors have
//!   it).
//! - [`half_f32_wide!`] adds the `f32` conversions of the wide `f16` and `bf16` vectors.
//! - [`half_reduce_sum!`] adds `reduce_sum` to the `f16` and `bf16` vectors.
//!
//! The `x2` types keep their primitive field (`u32`, `u16` or `u8`); wider types are
//! `#[repr(C, align(SIZE))]` arrays of `x2`s. The macros expand to code that accesses private
//! fields, so they must be invoked in the module that defines the scalar type.

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

/// Defines the `x2`, `x4`, `x8`, `x16` and `x32` vectors of a format and their container API.
///
/// - `elem`: the scalar type, with `to_bits`/`from_bits` and (for the `pack: nibbles` format)
///   masked `from_bits`.
/// - `pack`: `bytes` if each element occupies a whole number of bytes (the array of elements has
///   the layout of the vector), or `nibbles` if two elements share a byte (`f4e2m1`).
/// - `consts`: the constants of the scalar to re-export splatted on each vector.
/// - `note` (optional): extra documentation paragraph added to every vector type.
/// - `x2`, `x4`, `x8`, `x16`, `x32`: the types, with the unsigned integer type of the same size
///   used as their raw bits (`[u128; 2]` for 32 bytes and `[u128; 4]` for 64 bytes).
///
/// ```ignore
/// vector_types! {
///     elem: f8e4m3, pack: bytes,
///     consts: [ZERO, ONE],
///     note: "Format description.",
///     x2: f8e4m3x2(u16), x4: f8e4m3x4(u32), x8: f8e4m3x8(u64), x16: f8e4m3x16(u128),
///     x32: f8e4m3x32([u128; 2]),
/// }
/// ```
///
/// Every type `TxN` gets: `from_bits`/`to_bits`, `from_array`/`to_array`, `splat`,
/// `new(lo, hi)`/`lo()`/`hi()`, `from_x2s`/`to_x2s` (for `N >= 4`), the constants, `Default`,
/// `Debug`, and `From` conversions with `[T; N]`. The types are `Pod` and `Zeroable`.
macro_rules! vector_types {
    (
        elem: $elem:ident, pack: $pack:ident,
        consts: [$($const:ident),* $(,)?],
        $(note: $note:literal,)?
        x2: $x2:ident($b2:tt),
        x4: $x4:ident($b4:tt),
        x8: $x8:ident($b8:tt),
        x16: $x16:ident($b16:tt),
        x32: $x32:ident($b32:tt) $(,)?
    ) => {
        vector_types!(@x2 $x2($b2), $elem, $pack, [$($const),*], [$($note)?]);
        vector_types!(@sized $b4 [@wide $x4($b4), $elem, $pack, [$($const),*], [$($note)?],
            n: 4, "Four", half: $x2, half_n: 2, pairs: 2, x2: $x2,]);
        vector_types!(@sized $b8 [@wide $x8($b8), $elem, $pack, [$($const),*], [$($note)?],
            n: 8, "Eight", half: $x4, half_n: 4, pairs: 4, x2: $x2,]);
        vector_types!(@sized $b16 [@wide $x16($b16), $elem, $pack, [$($const),*], [$($note)?],
            n: 16, "Sixteen", half: $x8, half_n: 8, pairs: 8, x2: $x2,]);
        vector_types!(@sized $b32 [@wide $x32($b32), $elem, $pack, [$($const),*], [$($note)?],
            n: 32, "Thirty-two", half: $x16, half_n: 16, pairs: 16, x2: $x2,]);
    };

    // Appends the size in bytes of the given raw bits type to the arguments of `@wide`.
    (@sized u8 [$($args:tt)*]) => { vector_types! { $($args)* size: 1 } };
    (@sized u16 [$($args:tt)*]) => { vector_types! { $($args)* size: 2 } };
    (@sized u32 [$($args:tt)*]) => { vector_types! { $($args)* size: 4 } };
    (@sized u64 [$($args:tt)*]) => { vector_types! { $($args)* size: 8 } };
    (@sized u128 [$($args:tt)*]) => { vector_types! { $($args)* size: 16 } };
    (@sized [u128; 2] [$($args:tt)*]) => { vector_types! { $($args)* size: 32 } };
    (@sized [u128; 4] [$($args:tt)*]) => { vector_types! { $($args)* size: 64 } };

    // Number of bits between the two elements of an x2.
    (@shift bytes, $elem:ident) => { (::core::mem::size_of::<$elem>() as u32 * 8) };
    (@shift nibbles, $elem:ident) => { 4u32 };

    // The bits of the low element of an x2, masked to the element width for nibbles (in case the
    // scalar was created by a cast that bypassed the masking `from_bits`).
    (@lo_bits bytes, $x:expr) => { $x };
    (@lo_bits nibbles, $x:expr) => { ($x & 0xF) };

    (@pack_doc bytes) => { "Element 0 occupies the least significant bits." };
    (@pack_doc nibbles) => {
        "Two elements share each byte; element 0 occupies the low nibble of the first byte."
    };

    (@x2 $ty:ident($bits:tt), $elem:ident, $pack:ident, $consts:tt, [$($note:literal)?]) => {
        #[doc = concat!("Two packed [`", stringify!($elem), "`](crate::", stringify!($elem), ") values, aligned to their size.")]
        ///
        #[doc = vector_types!(@pack_doc $pack)]
        $(
            ///
            #[doc = $note]
        )?
        #[derive(Clone, Copy, ::bytemuck::Pod, ::bytemuck::Zeroable)]
        #[repr(transparent)]
        pub struct $ty($bits);

        vector_types!(@common $ty, $elem, 2, $consts);

        #[allow(clippy::unnecessary_cast)]
        impl $ty {
            /// Creates a value from its raw bit representation.
            #[inline]
            pub const fn from_bits(bits: $bits) -> Self {
                Self(bits)
            }

            /// Returns the raw bit representation.
            #[inline]
            pub const fn to_bits(self) -> $bits {
                self.0
            }

            /// Packs two elements; `lo` becomes element 0 and `hi` element 1.
            #[inline]
            pub const fn new(lo: $elem, hi: $elem) -> Self {
                Self(
                    (vector_types!(@lo_bits $pack, lo.to_bits()) as $bits)
                        | ((hi.to_bits() as $bits) << vector_types!(@shift $pack, $elem)),
                )
            }

            /// Creates a value with both elements set to `x`.
            #[inline]
            pub const fn splat(x: $elem) -> Self {
                Self::new(x, x)
            }

            /// Returns element 0 (the low half of the bits).
            #[inline]
            pub const fn lo(self) -> $elem {
                $elem::from_bits(self.0 as _)
            }

            /// Returns element 1 (the high half of the bits).
            #[inline]
            pub const fn hi(self) -> $elem {
                $elem::from_bits((self.0 >> vector_types!(@shift $pack, $elem)) as _)
            }

            /// Unpacks into an array `[lo, hi]`.
            #[inline]
            pub const fn to_array(self) -> [$elem; 2] {
                [self.lo(), self.hi()]
            }

            /// Packs an array `[lo, hi]`.
            #[inline]
            pub const fn from_array(x: [$elem; 2]) -> Self {
                Self::new(x[0], x[1])
            }
        }
    };

    (
        @wide $ty:ident($bits:tt), $elem:ident, $pack:ident, $consts:tt, [$($note:literal)?],
        n: $n:literal, $count:literal, half: $half:ident, half_n: $half_n:literal, pairs: $pairs:literal,
        x2: $x2:ident,
        size: $size:literal
    ) => {
        #[doc = concat!($count, " packed [`", stringify!($elem), "`](crate::", stringify!($elem), ") values, aligned to their size (", stringify!($size), " bytes).")]
        ///
        #[doc = vector_types!(@pack_doc $pack)]
        ///
        #[doc = concat!("Stored as an array of ", stringify!($pairs), " [`", stringify!($x2), "`] pairs, see [`Self::from_x2s`].")]
        $(
            ///
            #[doc = $note]
        )?
        #[derive(Clone, Copy, ::bytemuck::Pod, ::bytemuck::Zeroable)]
        #[repr(C, align($size))]
        pub struct $ty([$x2; $pairs]);

        vector_types!(@common $ty, $elem, $n, $consts);

        impl $ty {
            /// Creates a value from its raw bit representation.
            #[inline]
            pub const fn from_bits(bits: $bits) -> Self {
                // SAFETY: both types are plain data of the same size, valid for all bit patterns.
                unsafe { ::core::mem::transmute::<$bits, Self>(bits) }
            }

            /// Returns the raw bit representation.
            #[inline]
            pub const fn to_bits(self) -> $bits {
                // SAFETY: see `from_bits`.
                unsafe { ::core::mem::transmute::<Self, $bits>(self) }
            }

            #[doc = concat!("Creates a value from two halves; `lo` becomes the first ", stringify!($half_n), " elements and `hi` the last ", stringify!($half_n), ".")]
            #[inline]
            pub const fn new(lo: $half, hi: $half) -> Self {
                // SAFETY: the halves are plain data and together exactly fill `Self`.
                unsafe { ::core::mem::transmute::<[$half; 2], Self>([lo, hi]) }
            }

            #[doc = concat!("Returns the lower half, the first ", stringify!($half_n), " elements.")]
            #[inline]
            pub const fn lo(self) -> $half {
                // SAFETY: see `new`.
                let halves = unsafe { ::core::mem::transmute::<Self, [$half; 2]>(self) };
                halves[0]
            }

            #[doc = concat!("Returns the upper half, the last ", stringify!($half_n), " elements.")]
            #[inline]
            pub const fn hi(self) -> $half {
                // SAFETY: see `new`.
                let halves = unsafe { ::core::mem::transmute::<Self, [$half; 2]>(self) };
                halves[1]
            }

            /// Packs pairs of elements; `x2s[i]` holds elements `2 * i` and `2 * i + 1`.
            ///
            /// Pairs are the unit of the PTX instructions operating on packed values.
            #[inline]
            pub const fn from_x2s(x2s: [$x2; $pairs]) -> Self {
                Self(x2s)
            }

            /// Splits into pairs of elements; `x2s[i]` holds elements `2 * i` and `2 * i + 1`.
            #[inline]
            pub const fn to_x2s(self) -> [$x2; $pairs] {
                self.0
            }

            /// Creates a value with all elements set to `x`.
            #[inline]
            pub const fn splat(x: $elem) -> Self {
                Self([$x2::splat(x); $pairs])
            }

            vector_types!(@wide_array $pack, $elem, $n, $pairs, $x2);
        }
    };

    (@wide_array bytes, $elem:ident, $n:literal, $pairs:literal, $x2:ident) => {
        /// Packs an array of elements.
        #[inline]
        pub const fn from_array(x: [$elem; $n]) -> Self {
            // SAFETY: the array has the layout of the vector, as the elements are whole bytes.
            unsafe { ::core::mem::transmute::<[$elem; $n], Self>(x) }
        }

        /// Unpacks into an array of elements.
        #[inline]
        pub const fn to_array(self) -> [$elem; $n] {
            // SAFETY: see `from_array`.
            unsafe { ::core::mem::transmute::<Self, [$elem; $n]>(self) }
        }
    };

    (@wide_array nibbles, $elem:ident, $n:literal, $pairs:literal, $x2:ident) => {
        /// Packs an array of elements.
        #[inline]
        pub const fn from_array(x: [$elem; $n]) -> Self {
            let mut pairs = [$x2::from_bits(0); $pairs];
            let mut i = 0;
            #[rustc_unroll(full)]
            while i < $pairs {
                pairs[i] = $x2::new(x[2 * i], x[2 * i + 1]);
                i += 1;
            }
            Self(pairs)
        }

        /// Unpacks into an array of elements.
        #[inline]
        pub const fn to_array(self) -> [$elem; $n] {
            let mut out = [$elem::from_bits(0); $n];
            let mut i = 0;
            #[rustc_unroll(full)]
            while i < $pairs {
                out[2 * i] = self.0[i].lo();
                out[2 * i + 1] = self.0[i].hi();
                i += 1;
            }
            out
        }
    };

    // The format independent parts: constants, `Default`, `Debug` and `From` with arrays.
    (@common $ty:ident, $elem:ident, $n:literal, [$($const:ident),*]) => {
        const _: () = {
            assert!(::core::mem::size_of::<$ty>() == ::core::mem::align_of::<$ty>());
        };

        impl $ty {
            $(
                #[doc = concat!("All elements equal to [`", stringify!($elem), "::", stringify!($const), "`](crate::", stringify!($elem), "::", stringify!($const), ").")]
                pub const $const: Self = Self::splat($elem::$const);
            )*
        }

        impl ::core::default::Default for $ty {
            /// All elements set to the default of the element type.
            #[inline]
            fn default() -> Self {
                Self::splat(<$elem as ::core::default::Default>::default())
            }
        }

        impl ::core::fmt::Debug for $ty {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                let mut t = f.debug_tuple(stringify!($ty));
                for e in self.to_array() {
                    t.field(&e);
                }
                t.finish()
            }
        }

        impl ::core::convert::From<[$elem; $n]> for $ty {
            #[inline]
            fn from(x: [$elem; $n]) -> Self {
                Self::from_array(x)
            }
        }

        impl ::core::convert::From<$ty> for [$elem; $n] {
            #[inline]
            fn from(x: $ty) -> Self {
                x.to_array()
            }
        }
    };
}

/// Implements element-wise `PartialEq` for the given vector types.
macro_rules! vector_partial_eq {
    ($($ty:ident),* $(,)?) => {
        $(
            impl ::core::cmp::PartialEq for $ty {
                #[inline]
                fn eq(&self, other: &Self) -> bool {
                    self.to_array() == other.to_array()
                }
            }
        )*
    };
}

/// Implements `from_f32`, `from_f32_with` and `to_f32` of a wide (`N >= 4`) `f16` or `bf16`
/// vector `$ty` pair-wise in terms of those of its pairs `$x2`.
macro_rules! half_f32_wide {
    ($ty:ident, $x2:ident, $elem:ident, n: $n:literal, pairs: $pairs:literal) => {
        impl $ty {
            #[doc = concat!("Converts ", stringify!($n), " `f32` values to [`", stringify!($elem), "`](crate::", stringify!($elem), "), rounding to nearest, ties to even.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn from_f32(x: ::core::simd::Simd<f32, $n>) -> Self {
                Self::from_f32_with::<
                    { $crate::Rounding::NearestEven },
                    { $crate::Relu::Disabled },
                    { $crate::Overflow::Infinity },
                >(x)
            }

            #[doc = concat!("Converts ", stringify!($n), " `f32` values to [`", stringify!($elem), "`](crate::", stringify!($elem), "), with explicit rounding and modifiers.")]
            ///
            #[doc = concat!("Each pair of elements is converted with [`", stringify!($x2), "::from_f32_with`]; see there for the meaning of the parameters.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt>
            #[inline]
            pub fn from_f32_with<
                const RND: $crate::Rounding,
                const R: $crate::Relu,
                const O: $crate::Overflow,
            >(
                x: ::core::simd::Simd<f32, $n>,
            ) -> Self {
                let x = x.to_array();
                let mut pairs = [$x2::default(); $pairs];
                let mut i = 0;
                #[rustc_unroll(full)]
                while i < $pairs {
                    pairs[i] = $x2::from_f32_with::<RND, R, O>(::core::simd::Simd::from_array([
                        x[2 * i],
                        x[2 * i + 1],
                    ]));
                    i += 1;
                }
                Self::from_x2s(pairs)
            }

            #[doc = concat!("Converts all elements to `f32`. This conversion is exact.")]
            #[inline]
            pub fn to_f32(self) -> ::core::simd::Simd<f32, $n> {
                let pairs = self.to_x2s();
                let mut out = [0.0f32; $n];
                let mut i = 0;
                #[rustc_unroll(full)]
                while i < $pairs {
                    let [lo, hi] = pairs[i].to_f32().to_array();
                    out[2 * i] = lo;
                    out[2 * i + 1] = hi;
                    i += 1;
                }
                ::core::simd::Simd::from_array(out)
            }
        }
    };
}

/// Implements `reduce_sum` of the `f16` or `bf16` vector `$ty`, either an `x2` or a wide vector
/// of `x2` pairs. For `sm_100`, the scalar type must provide `fn add_to_f32(self, acc: f32) -> f32`
/// and the `x2` type `fn add_to_f32x2(self, acc: [f32; 2]) -> [f32; 2]`, which add with the
/// mixed-precision `add.rn.f32.$ptx`.
///
/// On `sm_100`, the elements are accumulated directly with the mixed-precision instruction, in two
/// independent chains for the even and the odd elements of wide vectors. The chains start at
/// `-0.0`, which every element (including `-0.0`) is an identity of.
macro_rules! half_reduce_sum {
    ($ty:ident, $ptx:literal, x2) => {
        half_reduce_sum!(@impl $ty, $ptx, |x: $ty| x.hi().add_to_f32(x.lo().add_to_f32(-0.0)));
    };
    ($ty:ident, $ptx:literal, wide) => {
        half_reduce_sum!(@impl $ty, $ptx, |x: $ty| {
            let pairs = x.to_x2s();
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
        });
    };
    (@impl $ty:ident, $ptx:literal, $sm100:expr) => {
        impl $ty {
            /// Sums all elements in `f32`.
            ///
            /// The elements are converted to `f32` (exactly) and added with `add.rn.f32` (round to
            /// nearest, ties to even; subnormal values are preserved) in an unspecified order,
            /// which may differ between architectures. On `sm_100` and later, each element is
            #[doc = concat!("added with the mixed-precision `add.rn.f32.", $ptx, "`, which fuses the conversion into the addition.")]
            ///
            /// <https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-add>
            #[inline]
            pub fn reduce_sum(self) -> f32 {
                #[cfg(target_feature = "sm_100")]
                {
                    ($sm100)(self)
                }
                #[cfg(not(target_feature = "sm_100"))]
                $crate::F32Ext::reduce_sum(self.to_f32())
            }
        }
    };
}

pub(crate) use {half_f32_wide, half_reduce_sum, narrow_scalar, vector_partial_eq, vector_types};
