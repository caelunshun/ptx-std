// --- LLM-generated --- //
//! Device-side heap allocation (`malloc`/`free`), exposed as a `GlobalAlloc`.

use core::{
    alloc::{GlobalAlloc, Layout},
    arch::nvptx as nv,
    ffi::c_void,
    ptr,
};

/// The alignment guaranteed by the device `malloc`.
const MALLOC_ALIGN: usize = 16;

/// An allocator backed by the CUDA in-kernel `malloc` and `free`, which allocate from a
/// fixed-size heap in global memory.
///
/// Register it with `#[global_allocator]`:
///
/// ```ignore
/// #[global_allocator]
/// static ALLOCATOR: ptx_std::alloc::DeviceAllocator = ptx_std::alloc::DeviceAllocator;
/// ```
///
/// Allocations return null if the device heap is exhausted. The heap size can be configured from
/// the host with the `cudaLimitMallocHeapSize` limit (8 MiB by default). Memory allocated by one
/// thread can be used and freed by any other thread, and stays allocated until freed, even
/// across kernel launches.
///
/// `malloc` guarantees an alignment of 16 bytes. Layouts requiring a larger alignment are
/// supported by over-allocating and storing the pointer returned by `malloc` just before the
/// aligned block, at the cost of `align + 8` extra bytes per allocation.
///
/// <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-global-memory-allocation-and-operations>
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceAllocator;

unsafe impl GlobalAlloc for DeviceAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let align = layout.align();
        if align <= MALLOC_ALIGN {
            return unsafe { nv::malloc(layout.size()) }.cast();
        }

        // `malloc` returns a 16-aligned pointer `p`. The aligned pointer is the first address
        // `>= p + 8` that is a multiple of `align`, which leaves room for the original pointer
        // in the 8 bytes before it, and lies less than `align + 8` bytes after `p`.
        let Some(total) = layout
            .size()
            .checked_add(align)
            .and_then(|size| size.checked_add(size_of::<*mut u8>()))
        else {
            return ptr::null_mut();
        };
        let original: *mut u8 = unsafe { nv::malloc(total) }.cast();
        if original.is_null() {
            return ptr::null_mut();
        }
        let address = (original as usize + size_of::<*mut u8>()).next_multiple_of(align);
        let aligned = original.wrapping_add(address - original as usize);
        unsafe { aligned.cast::<*mut u8>().sub(1).write(original) };
        aligned
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let original = if layout.align() <= MALLOC_ALIGN {
            ptr
        } else {
            unsafe { ptr.cast::<*mut u8>().sub(1).read() }
        };
        unsafe { nv::free(original.cast::<c_void>()) }
    }
}
