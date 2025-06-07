# NoirLedgerHash: Optimized CPU/GPU Proof-of-Work Algorithm

## Project Overview

**NoirLedgerHash** is a next-generation, memory- and compute-intensive Proof-of-Work (PoW) algorithm engineered for maximal security and performance on both CPUs and GPUs. Inspired by and surpassing existing solutions like Monero’s RandomX, NoirLedgerHash integrates multiple cryptographic primitives, heavy floating-point operations, and memory-hard techniques to resist ASIC centralization while delivering high throughput on commodity hardware.

## Key Features

* **ASIC Resistance**: Combines memory-hard lookup tables (256 MB) with random ChaCha20 mixing to thwart ASIC and low-memory GPU miners.
* **Parallel Crypto Kernels**: Leverages AES-256 with AES-NI when available for blazing-fast, parallelizable encryption stages.
* **Floating-Point Workload**: Introduces data-dependent floating-point sequences to hinder hardware pipelining and specialized optimizations.
* **Lightweight Finalization**: Uses Blake3 for both seed expansion and final hashing, offering fast, secure output.
* **Cross-Platform Support**: Native C++17 codebase with optional OpenCL and CUDA plugins for GPU acceleration.

## Algorithm Breakdown

1. **Seed Expansion (Blake3)**

   * Expands the input seed into a large pseudorandom buffer using Blake3, ensuring uniform distribution for downstream stages.
2. **Parallel AES-256 Kernels**

   * Executes multiple AES-256 encryption passes on independent state blocks, harnessing AES-NI on modern CPUs.
3. **Floating-Point Dependency Stage**

   * Applies nonlinear floating-point transformations to data, producing unpredictable memory and compute patterns.
4. **Memory-Hard ChaCha20 Mixing**

   * Performs random accesses into a 256 MB lookup table, mixing entries with ChaCha20 to enforce high-bandwidth memory usage.
5. \*\*Final Blake3 Hash

   * Collapses the processed state into a compact 32-byte digest with Blake3, ready for blockchain integration.

## Installation and Build

### Prerequisites

* **C++17-compatible compiler**: GCC (≥9), Clang, or MSVC
* **CMake** ≥3.10
* **Optional**: OpenCL (ICD) / CUDA SDK for GPU acceleration

### Build Steps

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/NoirLedger.git
cd NoirLedger

# 2. Create and enter build directory
mkdir build && cd build

# 3. Configure with CMake
#    Release mode with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..

# 4. Compile
cmake --build .
# or simply: make
```

After compilation, executables are located under `build/`, e.g., `NoirLedger_profiler`.

## Benchmarking and Usage

Use the `NoirLedger_profiler` tool to measure performance and debug:

```bash
# CPU benchmark with 8 threads
./build/NoirLedger_profiler --mode cpu --threads 8

# GPU benchmark with 1_000_000 iterations
./build/NoirLedger_profiler --mode gpu --iterations 1000000

# List available OpenCL platforms/devices
./build/NoirLedger_profiler --diagnose-gpu
```

### Command-Line Options

| Option                   | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `--mode <cpu\|gpu>`      | Selects benchmark mode (default: `cpu`).          |
| `-t`, `--threads <N>`    | Number of CPU threads to use.                     |
| `-i`, `--iterations <N>` | Total number of hash iterations.                  |
| `--gpu-platform <id>`    | OpenCL platform ID (default: `0`).                |
| `--gpu-device <id>`      | OpenCL device ID (default: `0`).                  |
| `--diagnose-gpu`         | Prints OpenCL platform/device info then exits.    |
| `--debug-<stage>`        | Enable detailed debug prints for specific stages. |
| `-h`, `--help`           | Show help message.                                |

## Library Integration

NoirLedgerHash can be used as a standalone library in other C++ projects.

```cpp
#include "noxium_hash/noirledger_hash.h"

int main() {
    NoirLedgerHasher hasher;  // Initializes 256 MB lookup table
    std::vector<uint8_t> data = {...};
    auto digest = hasher(data);
    // digest is std::array<uint8_t, 32>
}
```

In your `CMakeLists.txt`, link against optional OpenCL/CUDA if found:

```cmake
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
  target_link_libraries(my_app PRIVATE OpenCL::OpenCL)
  target_compile_definitions(my_app PRIVATE NOIRLEDGER_ENABLE_OPENCL)
endif()

find_package(CUDA QUIET)
if(CUDA_FOUND)
  enable_language(CUDA)
  target_link_libraries(my_app PRIVATE CUDA::CUDA_CUDA_LIBRARY)
  target_compile_definitions(my_app PRIVATE NOIRLEDGER_ENABLE_CUDA)
endif()
```

## Contributing

We welcome issues, pull requests, and security audits. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

**Experimental Release (Beta).** The NoirLedgerHash algorithm and implementation have not undergone formal cryptographic audits. Use at your own risk and report vulnerabilities via our [SECURITY.md](SECURITY.md).
