#ifdef NOXIUM_ENABLE_CUDA

#include "gpu/NoirLedger_gpu_cuda.h"
#include "gpu/NoirLedger_gpu_cuda_wrapper.h" // Inclui a declaração do wrapper
#include "NoirLedger_hash/NoirLedger_hash.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept> // For std::runtime_error

// Custom exception class for CUDA errors
class CudaException : public std::runtime_error {
public:
    CudaException(const char* file, int line, cudaError_t error)
        : std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error) +
                             " at " + file + ":" + std::to_string(line)) {}
};

// Macro para verificação de erros CUDA
#define CUDA_CHECK(err) { \
    cudaError_t error = err; \
    if (error != cudaSuccess) { \
        std::cerr << "Erro CUDA em " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
        throw CudaException(__FILE__, __LINE__, error); \
    } \
}


NoirLedgerGPU_CUDA::NoirLedgerGPU_CUDA(NoirLedgerHasher& hasher, int device_id)
    : parent_hasher(hasher), cuda_device_id(device_id) {
    
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        throw std::runtime_error("Nenhum dispositivo CUDA encontrado.");
    }
    if (device_id >= device_count) {
        throw std::runtime_error("ID de dispositivo CUDA inválido.");
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    device_name = props.name;

    // Aloca a tabela de consulta na VRAM
    const unsigned char* table_ptr = parent_hasher.get_lookup_table();
    size_t table_size = parent_hasher.get_lookup_table_size();
    
    std::cout << "Copiando " << table_size / (1024.0 * 1024.0) << " MB da tabela de consulta para a VRAM da GPU (CUDA)..." << std::endl;
    CUDA_CHECK(cudaMalloc(&d_lookup_table_buffer, table_size));
    CUDA_CHECK(cudaMemcpy(d_lookup_table_buffer, table_ptr, table_size, cudaMemcpyHostToDevice));
    std::cout << "Cópia da tabela de consulta para a VRAM (CUDA) concluída." << std::endl;

    initialized = true;
    std::cout << "NoirLedgerGPU_CUDA inicializado com sucesso no dispositivo: " << device_name << std::endl;
}

NoirLedgerGPU_CUDA::~NoirLedgerGPU_CUDA() {
    release_resources();
}

void NoirLedgerGPU_CUDA::hash(const unsigned char* input, size_t input_len, unsigned char* output, size_t num_hashes) {
    if (!initialized) {
        std::cerr << "Erro: A classe NoirLedgerGPU_CUDA não foi inicializada corretamente." << std::endl;
        return;
    }
    if (num_hashes == 0) return;

    // Realocar buffers de entrada e saída apenas se necessário
    if (d_input_buffer == nullptr || input_buffer_size < num_hashes * input_len) {
        if (d_input_buffer != nullptr) cudaFree(d_input_buffer);
        CUDA_CHECK(cudaMalloc(&d_input_buffer, num_hashes * input_len));
        input_buffer_size = num_hashes * input_len;
    }
    if (d_output_buffer == nullptr || output_buffer_size < num_hashes * NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES) {
        if (d_output_buffer != nullptr) cudaFree(d_output_buffer);
        CUDA_CHECK(cudaMalloc(&d_output_buffer, num_hashes * NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES));
        output_buffer_size = num_hashes * NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES;
    }

    // Copiar dados de entrada para a GPU
    CUDA_CHECK(cudaMemcpy(d_input_buffer, input, num_hashes * input_len, cudaMemcpyHostToDevice));


    // Chamar o wrapper do kernel
    NoirLedger_hash_main_cu_wrapper(
        d_input_buffer,
        input_len,
        d_output_buffer,
        d_lookup_table_buffer,
        num_hashes,
        256 // Tamanho do bloco (pode ser ajustado)
    );

    // Sincronizar para garantir que o kernel terminou
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copiar resultados de volta da GPU
    CUDA_CHECK(cudaMemcpy(output, d_output_buffer, num_hashes * NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES, cudaMemcpyDeviceToHost));

    // Os buffers de entrada/saída agora são reutilizados e liberados apenas no destrutor.
}

void NoirLedgerGPU_CUDA::release_resources() {
    if (d_input_buffer) {
        cudaFree(d_input_buffer);
        d_input_buffer = nullptr;
    }
    if (d_output_buffer) {
        cudaFree(d_output_buffer);
        d_output_buffer = nullptr;
    }
    if (d_lookup_table_buffer) {
        cudaFree(d_lookup_table_buffer); // Não use CUDA_CHECK aqui
        d_lookup_table_buffer = nullptr;
    }
    initialized = false;
}

#endif // NOXIUM_ENABLE_CUDA