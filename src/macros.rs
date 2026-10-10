// --- LLM-generated --- //
//! Internal helpers for forwarding our const-generic parameters to `core::arch::nvptx`.
//!
//! Generic const expressions like `core_fn::<{ R.to_core() }>()` would leak `where` bounds into
//! our public signatures, so instead we expand an `if`/`else if` chain on the (const) parameter
//! with one branch per supported value, each defining a local `const` that can be passed as a
//! const argument.
//! Each branch condition is an inline `const` block, so rustc skips codegen of the untaken
//! branches even in debug builds. This matters because untaken branches may call intrinsics
//! that LLVM cannot select for the current target (a plain `match` on a const generic would
//! still be lowered at `-O0`).
//!
//! Only list branches for values the target intrinsic actually supports: core rejects unsupported
//! values with post-monomorphization asserts, which fire even in dead branches. Pair this with a
//! `const { assert!(...) }` that gives a readable error for unsupported values.

/// Binds a local `const NAME: bool` to the value of a const condition and evaluates `body`.
/// The condition is evaluated in a const context, so use `matches!` rather than `==`.
///
/// ```ignore
/// const_bool!(FTZ = matches!(D, Denormals::FlushToZero) => unsafe { nv::add_f32::<RND, FTZ, false>(a, b) })
/// ```
macro_rules! const_bool {
    ($name:ident = $cond:expr => $body:expr) => {
        if const { $cond } {
            const $name: bool = true;
            $body
        } else {
            const $name: bool = false;
            $body
        }
    };
}

/// Binds a local `const NAME: Ty` by matching on a const value and evaluates `body`.
/// Unlisted values are unreachable; reject them beforehand with a `const { assert!(...) }`.
///
/// ```ignore
/// const_map!(R => RND: nv::Rounding {
///     Rounding::NearestEven => nv::Rounding::Rn,
///     Rounding::TowardZero => nv::Rounding::Rz,
/// } in unsafe { nv::cvt_tf32_f32::<RND, false, false>(a) })
/// ```
macro_rules! const_map {
    ($scrut:expr => $name:ident: $ty:ty { $($pat:pat => $val:expr),+ $(,)? } in $body:expr) => {
        $(if const { matches!($scrut, $pat) } {
            const $name: $ty = $val;
            $body
        } else)+ {
            unreachable!()
        }
    };
}

/// Selects string fragments of an inline assembly template from const values, for instructions
/// whose modifiers are not numeric (numeric modifiers can be substituted with `const` operands of
/// `asm!` instead).
///
/// Each group maps the possible values of a const scrutinee to string literals. The macro expands
/// to nested `if const` chains over all groups (only the selected branch is codegen'd) and invokes
/// the callback macro `$cb` with the selected literal of every group, in order, separated by
/// commas. The callback is typically a local `macro_rules!` that splices the literals into an
/// `asm!` template with `concat!`. Unlisted values are unreachable; reject them beforehand with a
/// `const { assert!(...) }`.
///
/// ```ignore
/// macro_rules! emit {
///     ($shape:literal, $decompress:literal) => {
///         asm!(concat!("tcgen05.cp.cta_group::{cg}", $shape, $decompress, " [{dst}], {src};"), ...)
///     };
/// }
/// asm_variants!(emit;
///     SHAPE => { CopyShape::S128x256b => ".128x256b", CopyShape::S4x256b => ".4x256b" },
///     DECOMPRESS => { CopyDecompression::None => "", CopyDecompression::B4x16P64 => ".b8x16.b4x16_p64" },
/// )
/// ```
#[allow(unused_macros)]
macro_rules! asm_variants {
    ($cb:ident; $($scrut:expr => { $($pat:pat => $lit:literal),+ $(,)? }),* $(,)?) => {
        asm_variants!(@next $cb [] [$([$scrut => { $($pat => $lit),+ }])*])
    };
    (@next $cb:ident [$($acc:literal)*] []) => {
        $cb!($($acc),*)
    };
    (@next $cb:ident $acc:tt [[$($group:tt)*] $($rest:tt)*]) => {
        asm_variants!(@group $cb $acc [$($rest)*] $($group)*)
    };
    // The accumulated literals and remaining groups are passed as single token trees, so that they
    // can be used inside the repetition over the patterns of this group.
    (@group $cb:ident $acc:tt $rest:tt $scrut:expr => { $($pat:pat => $lit:literal),+ }) => {
        $(if const { matches!($scrut, $pat) } {
            asm_variants!(@push $cb $acc $lit $rest)
        } else)+ {
            unreachable!()
        }
    };
    (@push $cb:ident [$($acc:literal)*] $lit:literal $rest:tt) => {
        asm_variants!(@next $cb [$($acc)* $lit] $rest)
    };
}
