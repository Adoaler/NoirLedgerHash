#ifndef NOXIUM_HASH_H
#define NOXIUM_HASH_H

#include <cstddef> // Para size_t
#include <vector>
#include <string>

// Constantes do Algoritmo NoirLedger (movidas para dentro do namespace para melhor escopo)
namespace NoirLedgerConstants {
    // Tamanho da saída do hash em bytes (256 bits = 32 bytes)
    static constexpr size_t HASH_OUTPUT_SIZE_BYTES = 32;

    // Tamanho da tabela de consulta global em MB
    static constexpr size_t LOOKUP_TABLE_SIZE_MB = 256;

    // Memória de trabalho por hash em bytes (ex: 2MB)
    static constexpr size_t WORKING_MEMORY_PER_HASH_BYTES = 2 * 1024 * 1024;

    // Parâmetros do Estágio 2 (AES-256)
    static constexpr int AES_KERNELS = 8; // Número de kernels AES paralelos
    static constexpr int AES_ROUNDS = 10; // Número de rodadas AES (AES-256 padrão é 14, mas podemos ajustar)

    // Parâmetros do Estágio 3 (Ponto Flutuante)
    static constexpr int FP_VALUES = 16; // Número de valores double para operações FP (128 bytes de entrada / 8 bytes por double)
    static constexpr int FP_ROUNDS = 16; // Número de rodadas de operações FP

    // Parâmetros do Estágio 5 (Memory-Hard)
    static constexpr int MEMORY_LOOKUPS = 16; // Número de acessos aleatórios à memória
    static constexpr int CHACHA_ROUNDS = 8; // Número de rodadas duplas para ChaCha20 (ChaCha20 padrão é 10)
}

#ifdef __cplusplus
extern "C" {
#endif

// Interface C-style para compatibilidade.
// A inicialização e liberação de recursos globais agora são gerenciadas
// automaticamente pelo padrão Singleton dentro da implementação.
void NoirLedger_hash(const unsigned char* input, size_t input_len, unsigned char* output);

#ifdef __cplusplus
} // extern "C"

#include <array>
#include <memory> // Para std::unique_ptr

// Forward declarations para as classes da GPU
#ifdef NOXIUM_ENABLE_OPENCL
class NoirLedgerGPU_CL;
#endif
#ifdef NOXIUM_ENABLE_CUDA
class NoirLedgerGPU_CUDA;
#endif

// Classe C++ para encapsular o hasher NoirLedger
class NoirLedgerHasher {
public:
    NoirLedgerHasher();
    ~NoirLedgerHasher();

    // --- Funções de Hashing ---
    void hash_cpu(const unsigned char* input, size_t input_len, unsigned char* output);
    void hash_gpu(const unsigned char* input, size_t input_len, unsigned char* output, size_t num_hashes);

    // --- Funções de Conveniência (Wrappers) ---
    std::array<uint8_t, NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES> operator()(const std::vector<uint8_t>& data);
    std::array<uint8_t, NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES> operator()(const std::string& data_str);

    // --- Gerenciamento de GPU Unificado ---
    bool init_gpu(int device_id = 0, int platform_id = 0);
    bool is_gpu_initialized() const;
    std::string get_gpu_device_name() const;
    std::string get_gpu_backend_name() const;

    // --- Acesso a Dados Internos (para backends de GPU) ---
    const unsigned char* get_lookup_table() const;
    size_t get_lookup_table_size() const;

private:
    // Decompõe a função de hash monolítica em estágios privados
    void run_stage1_seed_expansion(const unsigned char* input, size_t input_len);
    void run_stage2_aes_kernels();
    void run_stage3_floating_point();
    void run_stage4_mixed_logic();
    void run_stage5_memory_hard();
    void run_stage6_final_hash(unsigned char* output);

    enum class GPUBackend { NONE, CUDA, OPENCL };
    GPUBackend m_active_gpu_backend = GPUBackend::NONE;

    // Buffer de trabalho alocado na heap para evitar estouro de pilha
    std::vector<unsigned char> m_working_buffer;
    size_t m_current_data_len = 0; // Rastreia o tamanho dos dados válidos no buffer

    // Ponteiros para as implementações da GPU
#ifdef NOXIUM_ENABLE_OPENCL
    std::unique_ptr<NoirLedgerGPU_CL> m_gpu_cl_hasher;
#endif
#ifdef NOXIUM_ENABLE_CUDA
    std::unique_ptr<NoirLedgerGPU_CUDA> m_gpu_cuda_hasher;
#endif
};

// Funções auxiliares C++ que usam a classe
std::vector<unsigned char> NoirLedger_hash_vector(const std::vector<unsigned char>& input_data);
std::vector<unsigned char> NoirLedger_hash_string(const std::string& input_str);

#endif // __cplusplus

#endif // NOXIUM_HASH_H