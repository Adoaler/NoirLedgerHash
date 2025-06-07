#ifndef BLAKE3_INTERNAL_H
#define BLAKE3_INTERNAL_H

#include <cstdint> // Para uint32_t, uint64_t
#include <cstring> // Para memcpy, memset

// Constantes do BLAKE3
#define BLAKE3_CUSTOM_IV_LEN 8
#define BLAKE3_CUSTOM_BLOCK_LEN 64
#define BLAKE3_CUSTOM_CHUNK_LEN 1024
#define BLAKE3_CUSTOM_OUT_LEN 32 // 256 bits

// Flags de compressão
#define CHUNK_CUSTOM_START 1
#define CHUNK_CUSTOM_END 2
#define PARENT_CUSTOM 4
#define ROOT_CUSTOM 8
#define KEYED_HASH 16
#define DERIVE_KEY_CONTEXT 32
#define DERIVE_KEY_MATERIAL 64

// Vetor de inicialização (IV) do BLAKE3
static const uint32_t IV[BLAKE3_CUSTOM_IV_LEN] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9EBUL, 0x5BE0CD19UL
};

// Permutação para as rodadas de compressão
static const uint8_t MSG_PERMUTATION[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

// Carrega 4 bytes little-endian como uint32_t
static inline uint32_t load_u32_le(const uint8_t bytes[4]) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

// Armazena um uint32_t como 4 bytes little-endian
static inline void store_u32_le(uint32_t val, uint8_t bytes[4]) {
    bytes[0] = (uint8_t)val;
    bytes[1] = (uint8_t)(val >> 8);
    bytes[2] = (uint8_t)(val >> 16);
    bytes[3] = (uint8_t)(val >> 24);
}


// Funções internas (declaradas aqui, definidas em blake3_compress.cpp)
void blake3_compress_in_place(uint32_t cv[8], const uint8_t block[BLAKE3_CUSTOM_BLOCK_LEN],
                              uint8_t block_len, uint64_t counter, uint8_t flags);

#if defined(__AVX2__)
void blake3_compress_in_place_avx2(uint32_t cv[8], const uint8_t block[BLAKE3_CUSTOM_BLOCK_LEN],
                                   uint8_t block_len, uint64_t counter, uint8_t flags);
#endif

#endif // BLAKE3_INTERNAL_H