#ifndef NOXIUM_GPU_CUDA_H
#define NOXIUM_GPU_CUDA_H

#ifdef NOXIUM_ENABLE_CUDA

#include <string>
#include <vector>

// Forward declaration para evitar incluir NoirLedger_hash.h aqui
class NoirLedgerHasher;

// A declaração da função wrapper foi movida para NoirLedger_gpu_cuda_wrapper.h
// para garantir que a ligação "C" seja usada consistentemente.

class NoirLedgerGPU_CUDA {
public:
    NoirLedgerGPU_CUDA(NoirLedgerHasher& hasher, int device_id = 0);
    ~NoirLedgerGPU_CUDA();

    bool is_initialized() const { return initialized; }
    std::string get_device_name() const { return device_name; }

    // Função para executar o hash na GPU
    void hash(
        const unsigned char* input,
        size_t input_len,
        unsigned char* output,
        size_t num_hashes
    );

private:
    bool initialized = false;
    NoirLedgerHasher& parent_hasher;

    // Variáveis do dispositivo CUDA
    int cuda_device_id;
    std::string device_name;

    // Buffers de memória da GPU
    unsigned char* d_input_buffer = nullptr;
    unsigned char* d_output_buffer = nullptr;
    unsigned char* d_lookup_table_buffer = nullptr;
    size_t input_buffer_size = 0;
    size_t output_buffer_size = 0;

    void release_resources();
};

#endif // NOXIUM_ENABLE_CUDA
#endif // NOXIUM_GPU_CUDA_H