# CLAUDE.md - CudaBridge Development Guide

## Project Overview

CudaBridge enables CUDA computing on Apple Silicon Macs by bridging external NVIDIA GPUs connected via USB4/Thunderbolt eGPU enclosures. It tunnels PCIe transactions over USB4 protocol, providing a CUDA-compatible API layer for applications.

**Language:** C (C11) with C++17 support
**License:** MIT
**Target Platform:** macOS 13.0+ on Apple Silicon (M1/M2/M3/M4)
**Documentation language:** Korean comments and docs, English code identifiers

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)    # macOS
make -j$(nproc)                 # Linux
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_SHARED_LIBS` | ON | Build dynamic library |
| `BUILD_TESTS` | ON | Build test programs |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `ENABLE_DEBUG` | OFF | Debug logging output |
| `ENABLE_ASAN` | OFF | Address Sanitizer |

### Build Output

- `libcudabridge.dylib` / `libcudabridge.so` (shared)
- `libcudabridge.a` (static)
- Test executables: `test_basic`, `test_memory`, `test_stream`, `test_arm_compat`
- Example executables: `example_vector_add`, `example_device_info`, `example_bandwidth`

## Tests

```bash
cd build && ctest --output-on-failure
```

Four test suites:
- **basic_test** - Device enumeration, initialization, version info
- **memory_test** - Memory allocation/deallocation, memcpy
- **stream_test** - Stream creation, events, synchronization
- **arm_compat_test** - ARM64 architecture compatibility (standalone, no library link)

Tests use a simple macro pattern: `TEST_PASS(name)` / `TEST_FAIL(name, msg)` with `printf`-based output.

## Architecture

Seven-layer stack (top to bottom):

```
User Applications
  └── Public API        (src/userspace/)        cbInit, cbMalloc, cbMemcpy...
      └── CUDA Runtime  (src/cuda/runtime/)     cudaMalloc, cudaMemcpy compatibility
          └── Memory Mgr (src/cuda/memory/)     Pools, page tables, unified memory
              └── GPU Driver (src/driver/nvidia/) Register access, channels, kernel exec
                  └── PCIe Tunnel (src/kernel/pcie/) TLP packets, BAR mapping, DMA
                      └── USB4 Ctrl (src/kernel/usb4/) Router enum, tunnel mgmt
```

## Directory Structure

```
src/
├── kernel/usb4/          # USB4 controller driver
├── kernel/pcie/          # PCIe tunneling layer
├── driver/nvidia/        # NVIDIA GPU hardware abstraction
├── cuda/runtime/         # CUDA Runtime API bridge
├── cuda/memory/          # GPU memory management
└── userspace/
    ├── include/          # Public header (cudabridge.h)
    ├── lib/              # Public API implementation
    └── examples/         # vector_add, device_info, bandwidth_test
tests/                    # Test suite (4 files)
docs/                     # ARCHITECTURE.md
scripts/                  # CMake/pkg-config templates
```

## Code Conventions

### Naming

| Category | Style | Example |
|----------|-------|---------|
| Structs/Typedefs | PascalCase | `USB4Router`, `PCIeDevice`, `NVGpuContext` |
| Functions (internal) | snake_case | `usb4_controller_init`, `pcie_map_bar` |
| Public API | cb prefix + CamelCase | `cbInit`, `cbMalloc`, `cbGetDeviceCount` |
| Constants/Macros | UPPER_SNAKE_CASE | `USB4_MAX_TUNNELS`, `NV_PMC_BOOT_0` |

### Patterns

- **Context pattern**: Global state structs (`USB4ControllerContext`, `NVGpuContext`) passed as first parameter
- **Error handling**: Return int error codes (0 = success, negative = error). Global `g_last_error` for CUDA compat. Per-layer `SET_ERROR()` macros
- **Logging**: Per-layer macros — `USB4_LOG()`, `NV_LOG()`, `PCIE_LOG()`, `CUDA_LOG()` — with tagged prefixes like `[USB4]`, `[NVIDIA]`
- **Header guards**: `#ifndef CUDABRIDGE_<LAYER>_<FILE>_H`
- **Thread safety**: `pthread_mutex_t lock` in context structs
- **Resource management**: Init/shutdown and alloc/free pairs throughout
- **Platform code**: `#ifdef __APPLE__` guards for macOS-specific IOKit usage

### Style

- Compiler flags: `-Wall -Wextra -Wpedantic`
- Comments in Korean (doxygen-style `@param`, `@return`)
- Each `.c` file has a corresponding `.h` with the public interface

## Known Build Warnings

The codebase has several `-Wunused-variable`, `-Wunused-parameter`, and `-Wstringop-truncation` warnings. These are tracked but not yet resolved. When making changes, avoid introducing new warnings.

## Dependencies

- **macOS frameworks**: IOKit, CoreFoundation, Security (linked via CMake on macOS)
- **Standard**: pthread, standard C library
- **Build tools**: CMake 3.20+, C11-capable compiler
- No external third-party library dependencies

## Installation

```bash
cd build && sudo make install
```

Installs to `/usr/local` by default:
- Library to `lib/`
- Header to `include/`
- pkg-config file to `lib/pkgconfig/`
- CMake config to `lib/cmake/CudaBridge/`

Downstream projects can use `find_package(CudaBridge)` or `pkg-config --cflags --libs cudabridge`.
