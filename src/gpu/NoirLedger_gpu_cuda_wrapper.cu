#include "NoirLedger_gpu_cuda_wrapper.h"
#include <cuda_runtime.h>
#include <cstdint>

// Declaração do kernel que está definido em NoirLedger_hash.cu
// O compilador CUDA (nvcc) garantirá que esta chamada seja resolvida.
__global__ void NoirLedger_hash_main_cu(
    const uint8_t* global_input,
    uint32_t input_len,
    uint8_t* global_output,
    const uint8_t* lookup_table
);

// Definição da função wrapper que será chamada pelo código C++.
void NoirLedger_hash_main_cu_wrapper(
    const unsigned char* input,
    unsigned int input_len,
    unsigned char* output,
    const unsigned char* lookup_table,
    int num_hashes,
    int block_size
) {
    if (num_hashes <= 0 || block_size <= 0) {
        return;
    }

    int grid_size = (num_hashes + block_size - 1) / block_size;
    
    NoirLedger_hash_main_cu<<<grid_size, block_size>>>(
        reinterpret_cast<const uint8_t*>(input),
        input_len,
        reinterpret_cast<uint8_t*>(output),
        reinterpret_cast<const uint8_t*>(lookup_table)
    );
}