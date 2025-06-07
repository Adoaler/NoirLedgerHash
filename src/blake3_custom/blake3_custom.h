#ifndef BLAKE3_CUSTOM_H
#define BLAKE3_CUSTOM_H

#include <cstddef> // Para size_t
#include <cstdint> // Para uint8_t
#include "blake3_internal.h" // Inclui as declarações das funções de compressão

#ifdef __cplusplus
extern "C" {
#endif

// Constantes do BLAKE3 (redefinidas aqui para uso externo, se necessário)
#define BLAKE3_CUSTOM_BLOCK_LEN 64
#define BLAKE3_CUSTOM_OUT_LEN 32 // 256 bits

// Flags de compressão (redefinidas aqui para uso externo, se necessário)
#define CHUNK_CUSTOM_START 1
#define CHUNK_CUSTOM_END 2
#define PARENT_CUSTOM 4
#define ROOT_CUSTOM 8
#define KEYED_HASH 16
#define DERIVE_KEY_CONTEXT 32
#define DERIVE_KEY_MATERIAL 64

// Estrutura do hasher BLAKE3 customizado
typedef struct {
    uint32_t cv[8];
    uint64_t chunk_len;
    uint64_t bytes_hashed;
    uint8_t block[BLAKE3_CUSTOM_BLOCK_LEN];
    uint8_t block_len;
    uint8_t flags;
} blake3_custom_hasher;

// Funções da API BLAKE3 customizada
void blake3_custom_hasher_init(blake3_custom_hasher* hasher);
void blake3_custom_hasher_update(blake3_custom_hasher* hasher, const void* input, size_t input_len);
void blake3_custom_hasher_finalize(blake3_custom_hasher* hasher, void* output, size_t output_len);

// Função de hash direto (para conveniência)
void blake3_custom_hash_direct(const unsigned char* input, size_t input_len, unsigned char* output);

// Função de hash direto com XOF (eXtendable Output Function)
void blake3_custom_hash_direct_xof(const unsigned char* input, size_t input_len, unsigned char* output, size_t output_len);

#ifdef __cplusplus
}
#endif

#endif // BLAKE3_CUSTOM_H