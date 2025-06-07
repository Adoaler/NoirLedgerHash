#ifndef NOXIUM_GPU_CUDA_WRAPPER_H
#define NOXIUM_GPU_CUDA_WRAPPER_H

// Este wrapper é a ponte entre o código C++ e o kernel CUDA.
// A declaração extern "C" é crucial para a vinculação correta.
#ifdef __cplusplus
extern "C" {
#endif

void NoirLedger_hash_main_cu_wrapper(
    const unsigned char* input,
    unsigned int input_len,
    unsigned char* output,
    const unsigned char* lookup_table,
    int num_hashes,
    int block_size
);

#ifdef __cplusplus
}
#endif

#endif // NOXIUM_GPU_CUDA_WRAPPER_H