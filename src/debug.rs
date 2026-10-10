// --- LLM-generated --- //
//! Debugging and profiling support: trap, printf, assertions, and performance monitor events.

use core::{arch::nvptx as nv, ffi::CStr, mem::MaybeUninit};

/// Terminates the kernel with an abnormal error by executing the `trap` instruction.
///
/// The host observes an "unspecified launch failure" and the CUDA context becomes unusable.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-trap>
#[inline]
pub fn trap() -> ! {
    // SAFETY: `trap` has no preconditions.
    unsafe { nv::trap() }
}

/// Triggers the performance monitor events whose bits are set in `MASK` (`pmevent.mask`).
///
/// Bit `i` of `MASK` triggers event `i`, for events 0 through 15. The events can be counted with
/// profiling tools.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-pmevent>
#[inline]
pub fn trigger_pm_events<const MASK: u16>() {
    // SAFETY: only affects performance counters.
    unsafe { nv::pmevent_mask::<MASK>() }
}

// A `const fn` is needed as `generic_const_exprs` rejects control flow in inline const blocks
// that depend on generic parameters.
const fn assert_pm_counter(n: u32) {
    assert!(
        n < 4,
        "only performance monitoring counters %pm0 to %pm3 are available"
    );
}

/// Reads the performance monitoring counter `%pm<N>`, for `N` in `0..4`.
///
/// The behavior of the counters is currently undefined by the PTX ISA.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-pm0-pm7>
#[inline]
pub fn pm_counter<const N: u32>() -> u32 {
    const { assert_pm_counter(N) };
    // SAFETY: reading a special register has no preconditions.
    unsafe { nv::pm::<N>() }
}

mod sealed {
    pub trait PrintfPromoted {}
    impl PrintfPromoted for i32 {}
    impl PrintfPromoted for u32 {}
    impl PrintfPromoted for i64 {}
    impl PrintfPromoted for u64 {}
    impl PrintfPromoted for f64 {}
}

/// A value that can be passed as an argument to [`printf!`](crate::printf).
///
/// Arguments are promoted like C variadic arguments: integers narrower than 32 bits become
/// 32-bit integers, `f32` becomes `f64`, and pointers become 64-bit integers.
pub trait PrintfArg {
    /// The type the argument is promoted to, which is how it is laid out in the `vprintf`
    /// argument buffer.
    type Promoted: Copy + sealed::PrintfPromoted;

    /// Converts the argument to its promoted type.
    fn promote(self) -> Self::Promoted;
}

macro_rules! impl_printf_arg {
    ($($ty:ty => $promoted:ty),+ $(,)?) => {$(
        impl PrintfArg for $ty {
            type Promoted = $promoted;
            #[inline(always)]
            fn promote(self) -> $promoted {
                self as $promoted
            }
        }
    )+};
}

impl_printf_arg! {
    i8 => i32, i16 => i32, i32 => i32,
    u8 => u32, u16 => u32, u32 => u32,
    i64 => i64, isize => i64,
    u64 => u64, usize => u64,
    f32 => f64, f64 => f64,
    bool => i32,
}

impl<T> PrintfArg for *const T {
    type Promoted = u64;
    #[inline(always)]
    fn promote(self) -> u64 {
        self as usize as u64
    }
}

impl<T> PrintfArg for *mut T {
    type Promoted = u64;
    #[inline(always)]
    fn promote(self) -> u64 {
        self as usize as u64
    }
}

/// Passed as a pointer to the NUL-terminated string, for use with `%s`.
impl PrintfArg for &CStr {
    type Promoted = u64;
    #[inline(always)]
    fn promote(self) -> u64 {
        self.as_ptr() as usize as u64
    }
}

/// Argument buffer for `vprintf`, with `N` arguments of at most 8 bytes each.
#[doc(hidden)]
pub struct PrintfBuffer<const N: usize> {
    data: [MaybeUninit<u64>; N],
    offset: usize,
}

impl<const N: usize> Default for PrintfBuffer<N> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> PrintfBuffer<N> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            data: [MaybeUninit::uninit(); N],
            offset: 0,
        }
    }

    /// Appends an argument, aligned to its own size as in a C struct.
    #[inline(always)]
    pub fn push<A: PrintfArg>(&mut self, arg: A) {
        let promoted = arg.promote();
        let size = size_of::<A::Promoted>();
        let offset = self.offset.next_multiple_of(size);
        assert!(offset + size <= N * 8, "too many printf arguments");
        // SAFETY: the buffer is 8-aligned and `offset` is a multiple of `size` (4 or 8), and the
        // write is in bounds per the assertion above.
        unsafe {
            self.data
                .as_mut_ptr()
                .cast::<u8>()
                .add(offset)
                .cast::<A::Promoted>()
                .write(promoted);
        }
        self.offset = offset + size;
    }

    #[inline(always)]
    pub fn call(&self, format: &CStr) -> i32 {
        // SAFETY: `format` is NUL-terminated and the buffer holds the promoted arguments in the
        // layout expected by `vprintf`.
        unsafe { nv::vprintf(format.as_ptr().cast(), self.data.as_ptr().cast()) }
    }
}

/// Returns the number of arguments consumed by the C `printf` format string `format`.
///
/// Used by [`printf!`](crate::printf) to check the number of arguments at compile time.
#[doc(hidden)]
pub const fn count_format_args(format: &CStr) -> usize {
    let bytes = format.to_bytes();
    let mut count = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            i += 1;
            continue;
        }
        i += 1;
        while i < bytes.len() {
            let c = bytes[i];
            i += 1;
            match c {
                b'%' => break,
                b'*' => count += 1,
                b'd' | b'i' | b'o' | b'u' | b'x' | b'X' | b'f' | b'F' | b'e' | b'E' | b'g'
                | b'G' | b'a' | b'A' | b'c' | b's' | b'p' => {
                    count += 1;
                    break;
                }
                _ => {}
            }
        }
    }
    count
}

/// Prints formatted output from the kernel to the host's standard output, using the CUDA
/// `vprintf` system call.
///
/// The format string must be a C string literal (`c"..."`) using C `printf` syntax. The number
/// of arguments is checked against the number of conversion specifiers at compile time, but
/// their types are not: it is up to the caller to use the right specifier, as in C. Arguments
/// are promoted like C variadic arguments (see [`PrintfArg`](crate::debug::PrintfArg)), so
/// `%d` and `%u` take integers of up to 32 bits, `%ld`/`%lld`/`%lu` take 64-bit integers, `%f`
/// takes `f32` or `f64`, and `%s` takes a `&CStr` or a `*const u8` pointing to a C string.
///
/// Evaluates to the `i32` status returned by `vprintf` (the number of arguments, or a negative
/// value on error).
///
/// ```ignore
/// printf!(c"thread %d of block %d: x = %f\n", tid, bid, x);
/// ```
///
/// <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#formatted-output>
#[macro_export]
macro_rules! printf {
    ($format:expr $(, $arg:expr)* $(,)?) => {{
        const FORMAT: &::core::ffi::CStr = $format;
        const NUM_ARGS: usize = <[()]>::len(&[$($crate::printf!(@unit $arg)),*]);
        const {
            ::core::assert!(
                $crate::debug::count_format_args(FORMAT) == NUM_ARGS,
                "number of printf arguments doesn't match the format string"
            )
        };
        let mut buffer = $crate::debug::PrintfBuffer::<NUM_ARGS>::new();
        $(buffer.push($arg);)*
        buffer.call(FORMAT)
    }};
    (@unit $arg:expr) => {
        ()
    };
}

/// Reports a failed assertion through the CUDA `__assertfail` system call and terminates the
/// kernel.
///
/// The message, file name and function name are printed to the host's standard error stream.
/// See also the [`device_assert!`](crate::device_assert) macro.
///
/// <https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#system-calls>
#[cold]
#[inline(never)]
pub fn assert_failed(message: &CStr, file: &CStr, line: u32, function: &CStr) -> ! {
    // SAFETY: all pointers are valid NUL-terminated strings.
    unsafe {
        nv::__assert_fail(
            message.as_ptr().cast(),
            file.as_ptr().cast(),
            line,
            function.as_ptr().cast(),
        );
    }
    trap()
}

/// Asserts that a condition holds in device code. On failure, reports the assertion with the
/// source location (see [`assert_failed`](crate::debug::assert_failed)) and terminates the
/// kernel.
///
/// An optional string literal can be given as the failure message.
///
/// ```ignore
/// device_assert!(index < len);
/// device_assert!(index < len, "index out of bounds");
/// ```
#[macro_export]
macro_rules! device_assert {
    ($cond:expr $(,)?) => {
        $crate::device_assert!(@impl $cond, "assertion failed: ", ::core::stringify!($cond))
    };
    ($cond:expr, $message:literal $(,)?) => {
        $crate::device_assert!(@impl $cond, $message)
    };
    (@impl $cond:expr, $($message:tt)+) => {
        if !$cond {
            $crate::debug::assert_failed(
                const {
                    // SAFETY: the string ends with a single NUL terminator, assuming that the
                    // message contains no NUL.
                    unsafe {
                        ::core::ffi::CStr::from_bytes_with_nul_unchecked(
                            ::core::concat!($($message)+, "\0").as_bytes(),
                        )
                    }
                },
                const {
                    unsafe {
                        ::core::ffi::CStr::from_bytes_with_nul_unchecked(
                            ::core::concat!(::core::file!(), "\0").as_bytes(),
                        )
                    }
                },
                ::core::line!(),
                const {
                    unsafe {
                        ::core::ffi::CStr::from_bytes_with_nul_unchecked(
                            ::core::concat!(::core::module_path!(), "\0").as_bytes(),
                        )
                    }
                },
            )
        }
    };
}
