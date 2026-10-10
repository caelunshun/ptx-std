Higher-level wrappers around `core::arch::nvptx` intrinsics
in our [rustc fork for NVPTX](https://github.com/caelunshun/rust-nvptx).
Requires the rustc fork to be built and locally added as toolchain name `nvptx-ext`.

Only supports sm_80 (Ampere) and later, and requires building for PTX 9.3 (CUDA 13.3) or later.
Many functions are only supported on later architectures; these are conditionally compiled
to avoid misuse (similar to the `safe_arch` crate).

Most functions use LLVM's native NVPTX intrinsics (exposed in `core::arch::nvptx`).
In theory, this may allow LLVM to perform more optimizations than if we used inline PTX.
Some operations (e.g. `tcgen05`) cannot be called from Rust as LLVM intrinsics  because
they use pointer address spaces that can't be created from generic pointers. For such cases,
we fall back to using inline PTX.
