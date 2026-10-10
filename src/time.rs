// --- LLM-generated --- //
//! Clocks, timers, and sleeping.

use core::arch::nvptx as nv;

/// Reads the 32-bit per-SM clock cycle counter (`%clock`).
///
/// The counter wraps around silently, so differences should be computed with `wrapping_sub`. Its
/// value is only meaningful relative to other readings taken on the same SM. It is identical to
/// the lower 32 bits of [`clock64`].
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock>
#[inline(always)]
pub fn clock() -> u32 {
    unsafe { nv::clock() }
}

/// Reads the 64-bit per-SM clock cycle counter (`%clock64`).
///
/// The counter wraps around silently. Its value is only meaningful relative to other readings
/// taken on the same SM.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock64>
#[inline(always)]
pub fn clock64() -> u64 {
    unsafe { nv::clock64() }
}

/// Reads the 64-bit global nanosecond timer (`%globaltimer`).
///
/// The timer is shared by all SMs of the device. Its resolution is device dependent and
/// typically much coarser than a nanosecond, often updating only every 32 ns to 1 us.
///
/// This register is intended for use by NVIDIA tools: its behavior is target-specific and may
/// change or be removed in future GPUs, and its value is unspecified when the PTX is JIT-compiled
/// to other targets.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer>
#[inline(always)]
pub fn global_timer() -> u64 {
    unsafe { nv::globaltimer() }
}

/// Reads the lower 32 bits of the global nanosecond timer (`%globaltimer_lo`).
///
/// This wraps around roughly every 4.3 seconds. See [`global_timer`] for caveats.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer>
#[inline(always)]
pub fn global_timer_lo() -> u32 {
    unsafe { nv::globaltimer_lo() }
}

/// Suspends the executing thread for approximately `nanoseconds` nanoseconds (`nanosleep`).
///
/// The actual duration is in the interval `[0, 2 * nanoseconds]`, and is at most one
/// millisecond; larger requests are not honored in full, so callers that need a longer sleep
/// must loop. The implementation may shorten the sleep of individual threads of a warp so that
/// all sleeping threads of the warp wake up together.
///
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-nanosleep>
#[inline(always)]
pub fn nanosleep(nanoseconds: u32) {
    unsafe { nv::nanosleep(nanoseconds) }
}
