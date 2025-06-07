#include "blake3_custom.h"
#include "blake3_internal.h" // Garante que as funções inline como store_u32_le estejam visíveis
#include <cstring> // Para memcpy, memset
#include <algorithm> // Para std::min
#include <stdexcept> // Para std::invalid_argument
#include <cassert>   // Para assert()

// Helper para converter bytes para palavras (uint32_t)
static inline void words_from_bytes(const uint8_t *bytes, uint32_t *words, size_t num_words) {
    for (size_t i = 0; i < num_words; ++i) {
        words[i] = load_u32_le(bytes + i * 4);
    }
}

// Inicializa o hasher BLAKE3
void blake3_custom_hasher_init(blake3_custom_hasher* hasher) {
    memcpy(hasher->cv, IV, sizeof(IV));
    hasher->chunk_len = 0;
    hasher->bytes_hashed = 0;
    hasher->block_len = 0;
    hasher->flags = 0;
}

// Atualiza o hasher com novos dados
void blake3_custom_hasher_update(blake3_custom_hasher* hasher, const void* input, size_t input_len) {
    if (input == nullptr && input_len > 0) {
        throw std::invalid_argument("blake3_custom_hasher_update: input nulo com comprimento maior que zero.");
    }

    const uint8_t* input_bytes = static_cast<const uint8_t*>(input);
    size_t remaining_len = input_len;

    // Se há dados restantes no bloco, preencha-o primeiro
    if (hasher->block_len > 0) {
        size_t take = BLAKE3_CUSTOM_BLOCK_LEN - hasher->block_len;
        if (take > remaining_len) {
            take = remaining_len;
        }
        assert(hasher->block_len + take <= BLAKE3_CUSTOM_BLOCK_LEN);
        memcpy(hasher->block + hasher->block_len, input_bytes, take);
        hasher->block_len += take;
        input_bytes += take;
        remaining_len -= take;

        if (hasher->block_len == BLAKE3_CUSTOM_BLOCK_LEN) {
            blake3_compress_in_place(hasher->cv, hasher->block, BLAKE3_CUSTOM_BLOCK_LEN, hasher->bytes_hashed, hasher->flags);
            hasher->bytes_hashed += BLAKE3_CUSTOM_BLOCK_LEN;
            hasher->block_len = 0;
            hasher->flags &= ~CHUNK_CUSTOM_START;
        }
    }

    // Processa blocos completos
    while (remaining_len >= BLAKE3_CUSTOM_BLOCK_LEN) {
        blake3_compress_in_place(hasher->cv, input_bytes, BLAKE3_CUSTOM_BLOCK_LEN, hasher->bytes_hashed, hasher->flags);
        hasher->bytes_hashed += BLAKE3_CUSTOM_BLOCK_LEN;
        input_bytes += BLAKE3_CUSTOM_BLOCK_LEN;
        remaining_len -= BLAKE3_CUSTOM_BLOCK_LEN;
        hasher->flags &= ~CHUNK_CUSTOM_START;
    }

    // Copia os bytes restantes para o bloco
    if (remaining_len > 0) {
        assert(remaining_len <= BLAKE3_CUSTOM_BLOCK_LEN);
        memcpy(hasher->block, input_bytes, remaining_len);
        hasher->block_len = remaining_len;
    }
}

// Finaliza o hasher e produz a saída
void blake3_custom_hasher_finalize(blake3_custom_hasher* hasher, void* output, size_t output_len) {
    // Adiciona a flag de fim de chunk ao processar o último bloco.
    hasher->flags |= CHUNK_CUSTOM_END;
    if (hasher->bytes_hashed == 0) { // Se for o primeiro e único bloco
        hasher->flags |= ROOT_CUSTOM;
    }
    blake3_compress_in_place(hasher->cv, hasher->block, hasher->block_len, hasher->bytes_hashed, hasher->flags);

    // Converte o CV final para bytes e copia para a saída.
    uint8_t cv_bytes[BLAKE3_CUSTOM_OUT_LEN];
    for (size_t i = 0; i < 8; ++i) {
        store_u32_le(hasher->cv[i], cv_bytes + i * 4);
    }

    size_t bytes_to_copy = std::min(output_len, (size_t)BLAKE3_CUSTOM_OUT_LEN);
    if (output != nullptr && bytes_to_copy > 0) {
        memcpy(output, cv_bytes, bytes_to_copy);
    }
}

// Função de hash direto (para conveniência)
void blake3_custom_hash_direct(const unsigned char* input, size_t input_len, unsigned char* output) {
    blake3_custom_hasher hasher;
    blake3_custom_hasher_init(&hasher);
    blake3_custom_hasher_update(&hasher, input, input_len);
    blake3_custom_hasher_finalize(&hasher, output, BLAKE3_CUSTOM_OUT_LEN);
}

// Função de hash direto com XOF (eXtendable Output Function)
void blake3_custom_hash_direct_xof(const unsigned char* input, size_t input_len, unsigned char* output, size_t output_len) {
    if (output == nullptr && output_len > 0) {
        throw std::invalid_argument("blake3_custom_hash_direct_xof: output nulo com comprimento maior que zero.");
    }
    if (output_len == 0) {
        return; // Nada a fazer
    }

    blake3_custom_hasher hasher;
    blake3_custom_hasher_init(&hasher);
    hasher.flags |= ROOT_CUSTOM;
    blake3_custom_hasher_update(&hasher, input, input_len);

    // Finaliza o hash para obter o CV do nó raiz.
    blake3_custom_hasher_finalize(&hasher, nullptr, 0);

    uint64_t output_block_counter = 0;
    size_t bytes_copied = 0;
    uint8_t out_bytes[BLAKE3_CUSTOM_BLOCK_LEN];
    
    while (bytes_copied < output_len) {
        // Gera o próximo bloco de saída.
        uint32_t temp_cv[8];
        memcpy(temp_cv, hasher.cv, sizeof(hasher.cv));
        blake3_compress_in_place(temp_cv, (const uint8_t*)hasher.cv, 0, output_block_counter, ROOT_CUSTOM);

        for (size_t i = 0; i < 8; ++i) {
            store_u32_le(temp_cv[i], out_bytes + i * 4);
            store_u32_le(temp_cv[i+8], out_bytes + 32 + i * 4);
        }

        size_t bytes_to_copy = std::min((size_t)BLAKE3_CUSTOM_BLOCK_LEN, output_len - bytes_copied);
        memcpy(output + bytes_copied, out_bytes, bytes_to_copy);
        
        bytes_copied += bytes_to_copy;
        output_block_counter++;
    }
}