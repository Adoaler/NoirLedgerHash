#include "NoirLedger_hash.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cstring>
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <mutex>

#include "../blake3_custom/blake3_custom.h"
#include "debug_flags.h"

#ifdef NOXIUM_ENABLE_OPENCL
#include "gpu/NoirLedger_gpu_cl.h"
#endif
#ifdef NOXIUM_ENABLE_CUDA
#include "gpu/NoirLedger_gpu_cuda.h"
#endif

// --- Constantes e Helpers Internos ---

namespace { // Namespace anônimo para helpers internos

// Função auxiliar para rotação de bits
inline uint32_t rotate_right(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// Constantes para o estágio de ponto flutuante
constexpr double K1_s = 3.14159265358979323846;
constexpr double K2_s = 1.61803398874989484820;
constexpr double K_MOD_s = 1000000007.0;

// Constante para o estágio de lógica mista
constexpr uint32_t GOLDEN_RATIO_32 = 0x9E3779B9;

// Constante para o estágio de memória
constexpr uint64_t MIX_CONSTANT_64 = 0x9E3779B97F4A7C15ULL;

// --- Implementações Criptográficas (AES, ChaCha) ---

// Implementação de fallback em C puro para AES-256
namespace AES256_C {
    // ... (Implementação C-pure completa, como estava antes)
    // Tabela S-Box e Rcon para expansão de chave
static const uint8_t sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

static const uint8_t Rcon[11] = { 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

void key_expansion(uint8_t* round_keys, const uint8_t* key) {
    unsigned i, j, k;
    uint8_t tempa[4];
    
    for (i = 0; i < 8; ++i) {
        round_keys[i * 4 + 0] = key[i * 4 + 0];
        round_keys[i * 4 + 1] = key[i * 4 + 1];
        round_keys[i * 4 + 2] = key[i * 4 + 2];
        round_keys[i * 4 + 3] = key[i * 4 + 3];
    }

    for (i = 8; i < (NoirLedgerConstants::AES_ROUNDS + 1) * 4; ++i) {
        for (k = 0; k < 4; ++k) {
            tempa[k] = round_keys[(i - 1) * 4 + k];
        }
        if (i % 8 == 0) {
            k = tempa[0];
            tempa[0] = tempa[1];
            tempa[1] = tempa[2];
            tempa[2] = tempa[3];
            tempa[3] = k;
            
            tempa[0] = sbox[tempa[0]];
            tempa[1] = sbox[tempa[1]];
            tempa[2] = sbox[tempa[2]];
            tempa[3] = sbox[tempa[3]];
            
            tempa[0] = tempa[0] ^ Rcon[i / 8];
        }
        if (i % 8 == 4) {
            tempa[0] = sbox[tempa[0]];
            tempa[1] = sbox[tempa[1]];
            tempa[2] = sbox[tempa[2]];
            tempa[3] = sbox[tempa[3]];
        }
        j = i * 4; k = (i - 8) * 4;
        round_keys[j + 0] = round_keys[k + 0] ^ tempa[0];
        round_keys[j + 1] = round_keys[k + 1] ^ tempa[1];
        round_keys[j + 2] = round_keys[k + 2] ^ tempa[2];
        round_keys[j + 3] = round_keys[k + 3] ^ tempa[3];
    }
}

void add_round_key(uint8_t round, uint8_t state[4][4], const uint8_t* round_keys) {
    for (uint8_t i = 0; i < 4; ++i) {
        for (uint8_t j = 0; j < 4; ++j) {
            state[j][i] ^= round_keys[round * 16 + i * 4 + j];
        }
    }
}

void sub_bytes(uint8_t state[4][4]) {
    for (uint8_t i = 0; i < 4; ++i) {
        for (uint8_t j = 0; j < 4; ++j) {
            state[i][j] = sbox[state[i][j]];
        }
    }
}

void shift_rows(uint8_t state[4][4]) {
    uint8_t temp;
    temp = state[1][0]; state[1][0] = state[1][1]; state[1][1] = state[1][2]; state[1][2] = state[1][3]; state[1][3] = temp;
    temp = state[2][0]; state[2][0] = state[2][2]; state[2][2] = temp;
    temp = state[2][1]; state[2][1] = state[2][3]; state[2][3] = temp;
    temp = state[3][3]; state[3][3] = state[3][2]; state[3][2] = state[3][1]; state[3][1] = state[3][0]; state[3][0] = temp;
}

uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (uint8_t i = 0; i < 8; ++i) {
        if (b & 1) p ^= a;
        uint8_t hi_bit_set = (a & 0x80);
        a <<= 1;
        if (hi_bit_set) a ^= 0x1b;
        b >>= 1;
    }
    return p;
}

void mix_columns(uint8_t state[4][4]) {
    uint8_t t[4];
    for (uint8_t i = 0; i < 4; ++i) {
        t[0] = state[0][i]; t[1] = state[1][i]; t[2] = state[2][i]; t[3] = state[3][i];
        state[0][i] = gmul(t[0], 2) ^ gmul(t[1], 3) ^ gmul(t[2], 1) ^ gmul(t[3], 1);
        state[1][i] = gmul(t[0], 1) ^ gmul(t[1], 2) ^ gmul(t[2], 3) ^ gmul(t[3], 1);
        state[2][i] = gmul(t[0], 1) ^ gmul(t[1], 1) ^ gmul(t[2], 2) ^ gmul(t[3], 3);
        state[3][i] = gmul(t[0], 3) ^ gmul(t[1], 1) ^ gmul(t[2], 1) ^ gmul(t[3], 2);
    }
}

void cipher(uint8_t state[4][4], const uint8_t* round_keys) {
    add_round_key(0, state, round_keys);
    for (uint8_t round = 1; round < NoirLedgerConstants::AES_ROUNDS; ++round) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(round, state, round_keys);
    }
    sub_bytes(state);
    shift_rows(state);
    add_round_key(NoirLedgerConstants::AES_ROUNDS, state, round_keys);
}
} // namespace AES256_C

#if defined(__AES__)
// Função auxiliar para a expansão de chave AES-256
static inline void KEY_256_ASSIST(__m128i* temp1, __m128i * temp2) {
    __m128i temp4;
    *temp2 = _mm_shuffle_epi32(*temp2, 0xff);
    temp4 = _mm_slli_si128(*temp1, 0x4);
    *temp1 = _mm_xor_si128(*temp1, temp4);
    temp4 = _mm_slli_si128(temp4, 0x4);
    *temp1 = _mm_xor_si128(*temp1, temp4);
    temp4 = _mm_slli_si128(temp4, 0x4);
    *temp1 = _mm_xor_si128(*temp1, temp4);
    *temp1 = _mm_xor_si128(*temp1, *temp2);
}

// Expansão de chave AES-256 com AES-NI (versão desenrolada e correta)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("aes,avx")))
#endif
static inline void aes256_ni_key_expansion(const unsigned char *user_key, __m128i round_keys[NoirLedgerConstants::AES_ROUNDS + 1]) {
    __m128i temp1, temp2, temp3;
    temp1 = _mm_loadu_si128((const __m128i*)user_key);
    temp3 = _mm_loadu_si128((const __m128i*)(user_key + 16));
    round_keys[0] = temp1;
    round_keys[1] = temp3;

    // Gera as 10 chaves de rodada restantes (para um total de 11, índices 0-10)
    temp2 = _mm_aeskeygenassist_si128(temp3, 0x01);
    KEY_256_ASSIST(&temp1, &temp2);
    round_keys[2] = temp1;
    temp2 = _mm_aeskeygenassist_si128(temp1, 0x00);
    KEY_256_ASSIST(&temp3, &temp2);
    round_keys[3] = temp3;

    temp2 = _mm_aeskeygenassist_si128(temp3, 0x02);
    KEY_256_ASSIST(&temp1, &temp2);
    round_keys[4] = temp1;
    temp2 = _mm_aeskeygenassist_si128(temp1, 0x00);
    KEY_256_ASSIST(&temp3, &temp2);
    round_keys[5] = temp3;

    temp2 = _mm_aeskeygenassist_si128(temp3, 0x04);
    KEY_256_ASSIST(&temp1, &temp2);
    round_keys[6] = temp1;
    temp2 = _mm_aeskeygenassist_si128(temp1, 0x00);
    KEY_256_ASSIST(&temp3, &temp2);
    round_keys[7] = temp3;

    temp2 = _mm_aeskeygenassist_si128(temp3, 0x08);
    KEY_256_ASSIST(&temp1, &temp2);
    round_keys[8] = temp1;
    temp2 = _mm_aeskeygenassist_si128(temp1, 0x00);
    KEY_256_ASSIST(&temp3, &temp2);
    round_keys[9] = temp3;

    temp2 = _mm_aeskeygenassist_si128(temp3, 0x10);
    KEY_256_ASSIST(&temp1, &temp2);
    round_keys[10] = temp1;
}

// Kernel de criptografia AES-256 com AES-NI
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("aes,avx")))
#endif
void aes256_encrypt_kernel(unsigned char* data_block_16_bytes, const unsigned char* key_32_bytes) {
    __m128i round_keys[NoirLedgerConstants::AES_ROUNDS + 1];
    aes256_ni_key_expansion(key_32_bytes, round_keys);
    __m128i block = _mm_loadu_si128((const __m128i*)data_block_16_bytes);
    block = _mm_xor_si128(block, round_keys[0]);
    for (int r = 1; r < NoirLedgerConstants::AES_ROUNDS; ++r) {
        block = _mm_aesenc_si128(block, round_keys[r]);
    }
    block = _mm_aesenclast_si128(block, round_keys[NoirLedgerConstants::AES_ROUNDS]);
    _mm_storeu_si128((__m128i*)data_block_16_bytes, block);
}
#else
// Fallback C-pure para o kernel de criptografia
void aes256_encrypt_kernel(unsigned char* data_block_16_bytes, const unsigned char* key_32_bytes) {
    uint8_t round_keys[(NoirLedgerConstants::AES_ROUNDS + 1) * 16];
    AES256_C::key_expansion(round_keys, key_32_bytes);
    uint8_t state[4][4];
    for(int i=0; i<16; ++i) {
        state[i % 4][i / 4] = data_block_16_bytes[i];
    }
    AES256_C::cipher(state, round_keys);
    for(int i=0; i<16; ++i) {
        data_block_16_bytes[i] = state[i % 4][i / 4];
    }
}
#endif

// Implementação ChaCha20
namespace ChaCha20Full {
    // ... (Implementação completa do ChaCha20, como estava antes, mas usando constantes do namespace)
    inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

#define CHACHA_QR(a, b, c, d) \
    a += b; d ^= a; d = rotl32(d, 16); \
    c += d; b ^= c; b = rotl32(b, 12); \
    a += b; d ^= a; d = rotl32(d,  8); \
    c += d; b ^= c; b = rotl32(b,  7)

void chacha_block(uint32_t state[16], int num_double_rounds) {
    uint32_t working_state[16];
    memcpy(working_state, state, sizeof(uint32_t) * 16);

    for (int i = 0; i < num_double_rounds; ++i) {
        CHACHA_QR(working_state[0], working_state[4], working_state[ 8], working_state[12]);
        CHACHA_QR(working_state[1], working_state[5], working_state[ 9], working_state[13]);
        CHACHA_QR(working_state[2], working_state[6], working_state[10], working_state[14]);
        CHACHA_QR(working_state[3], working_state[7], working_state[11], working_state[15]);
        CHACHA_QR(working_state[0], working_state[5], working_state[10], working_state[15]);
        CHACHA_QR(working_state[1], working_state[6], working_state[11], working_state[12]);
        CHACHA_QR(working_state[2], working_state[7], working_state[ 8], working_state[13]);
        CHACHA_QR(working_state[3], working_state[4], working_state[ 9], working_state[14]);
    }

    for (int i = 0; i < 16; ++i) {
        state[i] += working_state[i];
    }
}

inline uint32_t u8to32_le(const unsigned char p[4]) {
    return ((uint32_t)(p[0])) | ((uint32_t)(p[1]) << 8) | ((uint32_t)(p[2]) << 16) | ((uint32_t)(p[3]) << 24);
}

inline void u32to8_le(unsigned char p[4], uint32_t v) {
    p[0] = (unsigned char)(v);
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

void chacha_init_state(uint32_t state[16], const unsigned char key[32], const unsigned char nonce[12], uint32_t counter) {
    state[0] = 0x61707865; state[1] = 0x3320646e; state[2] = 0x79622d32; state[3] = 0x6b206574;
    for (int i = 0; i < 8; ++i) state[4 + i] = u8to32_le(key + i * 4);
    state[12] = counter;
    for (int i = 0; i < 3; ++i) state[13 + i] = u8to32_le(nonce + i * 4);
}

void chacha_serialize_state(const uint32_t state[16], unsigned char keystream_block[64]) {
    for (int i = 0; i < 16; ++i) u32to8_le(keystream_block + i * 4, state[i]);
}
} // namespace ChaCha20Full

void chacha20_mix(unsigned char* data, size_t data_len, const unsigned char* key_32_bytes, const unsigned char* nonce_12_bytes) {
    uint32_t initial_state[16];
    unsigned char keystream_block[64];
    uint32_t block_counter = 1;

    size_t offset = 0;
    while (offset < data_len) {
        ChaCha20Full::chacha_init_state(initial_state, key_32_bytes, nonce_12_bytes, block_counter);
        uint32_t working_state[16];
        memcpy(working_state, initial_state, sizeof(initial_state));
        ChaCha20Full::chacha_block(working_state, NoirLedgerConstants::CHACHA_ROUNDS);
        ChaCha20Full::chacha_serialize_state(working_state, keystream_block);
        size_t bytes_to_xor = std::min(data_len - offset, (size_t)64);
        for (size_t i = 0; i < bytes_to_xor; ++i) {
            data[offset + i] ^= keystream_block[i];
        }
        offset += 64;
        block_counter++;
    }
}

// --- Gerenciamento da Tabela de Consulta Global (Singleton) ---
static std::vector<unsigned char>& get_global_lookup_table() {
    static std::vector<unsigned char> lookup_table = []() {
        size_t table_size_bytes = (size_t)NoirLedgerConstants::LOOKUP_TABLE_SIZE_MB * 1024 * 1024;
        std::vector<unsigned char> table;
        table.reserve(table_size_bytes);
        const std::string seed_str = "NoirLedgerLookupTableSeed_v1.0_For_Enhanced_Security_And_Mining_Optimization_And_Decentralization_Goals";
        std::vector<unsigned char> current_hash_input(seed_str.begin(), seed_str.end());
        unsigned char hash_output[NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES];
        while (table.size() < table_size_bytes) {
            blake3_custom_hash_direct(current_hash_input.data(), current_hash_input.size(), hash_output);
            size_t bytes_to_append = std::min((size_t)NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES, table_size_bytes - table.size());
            table.insert(table.end(), hash_output, hash_output + bytes_to_append);
            current_hash_input.assign(hash_output, hash_output + NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES);
        }
        return table;
    }();
    return lookup_table;
}

} // namespace anônimo

// ============================================================================
// Implementação da Classe NoirLedgerHasher
// ============================================================================

NoirLedgerHasher::NoirLedgerHasher() {
    try {
        // Garante que a tabela de consulta seja inicializada
        get_global_lookup_table();
        // Aloca o buffer de trabalho na heap
        m_working_buffer.resize(NoirLedgerConstants::WORKING_MEMORY_PER_HASH_BYTES);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Falha na construção do NoirLedgerHasher: ") + e.what());
    }
}

NoirLedgerHasher::~NoirLedgerHasher() {
    // unique_ptrs cuidam da liberação dos hashers de GPU
}

void NoirLedgerHasher::hash_cpu(const unsigned char* input, size_t input_len, unsigned char* output) {
    if (input == nullptr || output == nullptr) {
        throw std::invalid_argument("Ponteiro nulo recebido em hash_cpu.");
    }
    if (input_len == 0) {
        blake3_custom_hash_direct(nullptr, 0, output);
        return;
    }

    run_stage1_seed_expansion(input, input_len);
    run_stage2_aes_kernels();
    run_stage3_floating_point();
    run_stage4_mixed_logic();
    run_stage5_memory_hard();
    run_stage6_final_hash(output);
}

void NoirLedgerHasher::run_stage1_seed_expansion(const unsigned char* input, size_t input_len) {
    if (NoirLedgerDebug::is_flag_enabled(NoirLedgerDebug::DebugFlags::SEED_EXPANSION)) {
        std::cout << "[STAGE 1] Iniciando Expansão de Semente..." << std::endl;
    }
    blake3_custom_hash_direct_xof(input, input_len, m_working_buffer.data(), NoirLedgerConstants::WORKING_MEMORY_PER_HASH_BYTES);
    m_current_data_len = NoirLedgerConstants::WORKING_MEMORY_PER_HASH_BYTES;
}

void NoirLedgerHasher::run_stage2_aes_kernels() {
    if (NoirLedgerDebug::is_flag_enabled(NoirLedgerDebug::DebugFlags::AES_STAGE)) {
        std::cout << "[STAGE 2] Iniciando Computação Paralela (AES-256)..." << std::endl;
    }
    unsigned char aes_data_blocks[NoirLedgerConstants::AES_KERNELS][16];
    unsigned char aes_keys[NoirLedgerConstants::AES_KERNELS][32];
    for (size_t i = 0; i < NoirLedgerConstants::AES_KERNELS; ++i) {
        memcpy(aes_data_blocks[i], m_working_buffer.data() + (i * 16), 16);
        memcpy(aes_keys[i], m_working_buffer.data() + 128 + (i * 32), 32);
    }
    for (size_t i = 0; i < NoirLedgerConstants::AES_KERNELS; ++i) {
        aes256_encrypt_kernel(aes_data_blocks[i], aes_keys[i]);
    }
    for (size_t i = 0; i < NoirLedgerConstants::AES_KERNELS; ++i) {
        memcpy(m_working_buffer.data() + (i * 16), aes_data_blocks[i], 16);
    }
    m_current_data_len = 128;
}

void NoirLedgerHasher::run_stage3_floating_point() {
    if (NoirLedgerDebug::is_flag_enabled(NoirLedgerDebug::DebugFlags::FP_STAGE)) {
        std::cout << "[STAGE 3] Iniciando Operações de Ponto Flutuante..." << std::endl;
    }
    std::vector<double> fp_data(NoirLedgerConstants::FP_VALUES);
    memcpy(fp_data.data(), m_working_buffer.data(), NoirLedgerConstants::FP_VALUES * sizeof(double));

    for (int r = 0; r < NoirLedgerConstants::FP_ROUNDS; ++r) {
        std::vector<double> fp_data_const_this_round = fp_data;
        for (int i = 0; i < NoirLedgerConstants::FP_VALUES; ++i) {
            int prev_idx = (i + NoirLedgerConstants::FP_VALUES - 1) % NoirLedgerConstants::FP_VALUES;
            int mix_idx1 = (i * 5 + r) % NoirLedgerConstants::FP_VALUES;
            int mix_idx2 = (i * 11 + r * 3) % NoirLedgerConstants::FP_VALUES;
            double term1 = fp_data_const_this_round[i] * K1_s + fp_data_const_this_round[prev_idx];
            double term2 = sqrt(fabs(term1) + 1e-9);
            double term3 = fmod(term2 * K2_s + fp_data_const_this_round[mix_idx1], K_MOD_s);
            double res_before_finite_check = (r % 2 == 0)
                ? fmod(term3 + fp_data_const_this_round[mix_idx2] + (double)r * K1_s, K_MOD_s)
                : fmod(term3 - fp_data_const_this_round[mix_idx2] - (double)i * K2_s, K_MOD_s);
            fp_data[i] = std::isfinite(res_before_finite_check) ? res_before_finite_check : fmod((double)(r*13 + i*7), K_MOD_s);
        }
    }
    unsigned char temp_fp_bytes[NoirLedgerConstants::FP_VALUES * sizeof(double)];
    memcpy(temp_fp_bytes, fp_data.data(), sizeof(temp_fp_bytes));
    for (size_t i = 0; i < 32; ++i) {
        m_working_buffer[i] = temp_fp_bytes[i] ^ temp_fp_bytes[i + 32] ^ temp_fp_bytes[i + 64] ^ temp_fp_bytes[i + 96];
    }
    m_current_data_len = 32;
}

void NoirLedgerHasher::run_stage4_mixed_logic() {
    if (NoirLedgerDebug::is_flag_enabled(NoirLedgerDebug::DebugFlags::LOGIC_STAGE)) {
        std::cout << "[STAGE 4] Iniciando Lógica Mista..." << std::endl;
    }
    uint32_t logic_state[8];
    memcpy(logic_state, m_working_buffer.data(), sizeof(logic_state));

    for (int i = 0; i < 64; ++i) {
        uint32_t idx1 = logic_state[0] % 8;
        uint32_t idx2 = logic_state[1] % 8;
        uint32_t r_val = logic_state[2];
        uint32_t temp = logic_state[idx1];
        logic_state[idx1] = (logic_state[idx2] * GOLDEN_RATIO_32) + r_val;
        if (temp > logic_state[idx1]) {
            logic_state[idx2] = rotate_right(temp, (r_val % 32));
        } else {
            logic_state[idx2] = temp ^ r_val;
        }
        logic_state[2] = logic_state[idx1] + logic_state[idx2];
    }
    memcpy(m_working_buffer.data(), logic_state, sizeof(logic_state));
    m_current_data_len = sizeof(logic_state);
}

void NoirLedgerHasher::run_stage5_memory_hard() {
    if (NoirLedgerDebug::is_flag_enabled(NoirLedgerDebug::DebugFlags::MEMORY_STAGE)) {
        std::cout << "[STAGE 5] Iniciando Operações de Memória Difícil..." << std::endl;
    }
    unsigned char current_state[32];
    memcpy(current_state, m_working_buffer.data(), sizeof(current_state));
    const auto& lookup_table = get_global_lookup_table();

    for (int i = 0; i < NoirLedgerConstants::MEMORY_LOOKUPS; ++i) {
        uint64_t temp_addr;
        memcpy(&temp_addr, current_state, sizeof(uint64_t));
        temp_addr *= MIX_CONSTANT_64;
        temp_addr ^= (temp_addr >> 32);
        size_t lookup_index = temp_addr % (lookup_table.size() - 64);
        
        unsigned char memory_data[64];
        memcpy(memory_data, lookup_table.data() + lookup_index, sizeof(memory_data));
        
        unsigned char chacha_key[32];
        unsigned char chacha_nonce[12];
        unsigned char nonce_material[96];
        memcpy(nonce_material, current_state, 32);
        memcpy(nonce_material + 32, memory_data, 64);
        unsigned char nonce_hash[32];
        blake3_custom_hash_direct(nonce_material, sizeof(nonce_material), nonce_hash);
        memcpy(chacha_nonce, nonce_hash, 12);
        memcpy(chacha_key, memory_data, 32);

        for(size_t j=0; j < sizeof(current_state); ++j) {
            current_state[j] ^= memory_data[j + 32];
        }
        chacha20_mix(current_state, sizeof(current_state), chacha_key, chacha_nonce);
    }
    memcpy(m_working_buffer.data(), current_state, sizeof(current_state));
    m_current_data_len = sizeof(current_state);
}

void NoirLedgerHasher::run_stage6_final_hash(unsigned char* output) {
    if (NoirLedgerDebug::is_flag_enabled(NoirLedgerDebug::DebugFlags::BLAKE3_FINAL)) {
        std::cout << "[STAGE 6] Iniciando Compressão Final (Blake3)..." << std::endl;
    }
    blake3_custom_hash_direct(m_working_buffer.data(), m_current_data_len, output);
}

// --- Implementação do restante dos métodos da classe ---

void NoirLedgerHasher::hash_gpu(const unsigned char* input, size_t input_len, unsigned char* output, size_t num_hashes) {
    switch (m_active_gpu_backend) {
#ifdef NOXIUM_ENABLE_CUDA
        case GPUBackend::CUDA:
            if (m_gpu_cuda_hasher) {
                m_gpu_cuda_hasher->hash(input, input_len, output, num_hashes);
                return;
            }
            break;
#endif
#ifdef NOXIUM_ENABLE_OPENCL
        case GPUBackend::OPENCL:
            if (m_gpu_cl_hasher) {
                m_gpu_cl_hasher->hash(input, input_len, output, num_hashes);
                return;
            }
            break;
#endif
        default:
            throw std::runtime_error("Nenhum backend de GPU ativo ou inicializado para hash_gpu.");
    }
}

std::array<uint8_t, NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES> NoirLedgerHasher::operator()(const std::vector<uint8_t>& data) {
    std::array<uint8_t, NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES> hash_output;
    hash_cpu(data.data(), data.size(), hash_output.data());
    return hash_output;
}

std::array<uint8_t, NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES> NoirLedgerHasher::operator()(const std::string& data_str) {
    std::array<uint8_t, NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES> hash_output;
    hash_cpu(reinterpret_cast<const unsigned char*>(data_str.data()), data_str.length(), hash_output.data());
    return hash_output;
}

bool NoirLedgerHasher::init_gpu(int device_id, int platform_id) {
#ifdef NOXIUM_ENABLE_CUDA
    try {
        m_gpu_cuda_hasher = std::make_unique<NoirLedgerGPU_CUDA>(*this, device_id);
        if (m_gpu_cuda_hasher->is_initialized()) {
            m_active_gpu_backend = GPUBackend::CUDA;
            std::cout << "Backend de GPU ativo: CUDA" << std::endl;
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "Falha ao inicializar backend CUDA: " << e.what() << ". Tentando OpenCL..." << std::endl;
        m_gpu_cuda_hasher.reset();
    }
#endif
#ifdef NOXIUM_ENABLE_OPENCL
    try {
        m_gpu_cl_hasher = std::make_unique<NoirLedgerGPU_CL>(*this, platform_id, device_id);
        if (m_gpu_cl_hasher->is_initialized()) {
            m_active_gpu_backend = GPUBackend::OPENCL;
            std::cout << "Backend de GPU ativo: OpenCL" << std::endl;
            return true;
        }
    } catch (const std::exception& e) {
        std::cerr << "Falha ao inicializar backend OpenCL: " << e.what() << std::endl;
        m_gpu_cl_hasher.reset();
    }
#endif
    m_active_gpu_backend = GPUBackend::NONE;
    return false;
}

bool NoirLedgerHasher::is_gpu_initialized() const {
    return m_active_gpu_backend != GPUBackend::NONE;
}

std::string NoirLedgerHasher::get_gpu_backend_name() const {
    switch (m_active_gpu_backend) {
        case GPUBackend::CUDA: return "CUDA";
        case GPUBackend::OPENCL: return "OpenCL";
        default: return "Nenhum";
    }
}

std::string NoirLedgerHasher::get_gpu_device_name() const {
#ifdef NOXIUM_ENABLE_CUDA
    if (m_active_gpu_backend == GPUBackend::CUDA && m_gpu_cuda_hasher) {
        return m_gpu_cuda_hasher->get_device_name();
    }
#endif
#ifdef NOXIUM_ENABLE_OPENCL
    if (m_active_gpu_backend == GPUBackend::OPENCL && m_gpu_cl_hasher) {
        return m_gpu_cl_hasher->get_device_name();
    }
#endif
    return "N/A";
}

const unsigned char* NoirLedgerHasher::get_lookup_table() const {
    return get_global_lookup_table().data();
}

size_t NoirLedgerHasher::get_lookup_table_size() const {
    return get_global_lookup_table().size();
}

// ============================================================================
// Funções C-Style e Helpers (Wrappers para a classe NoirLedgerHasher)
// ============================================================================

void NoirLedger_hash(const unsigned char* input, size_t input_len, unsigned char* output) {
    try {
        static NoirLedgerHasher hasher; // Instância estática para reutilização
        hasher.hash_cpu(input, input_len, output);
    } catch (const std::exception& e) {
        std::cerr << "Erro fatal em NoirLedger_hash: " << e.what() << std::endl;
        memset(output, 0xFF, NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES);
    }
}

std::vector<unsigned char> NoirLedger_hash_vector(const std::vector<unsigned char>& input_data) {
    std::vector<unsigned char> hash_output(NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES);
    NoirLedger_hash(input_data.data(), input_data.size(), hash_output.data());
    return hash_output;
}

std::vector<unsigned char> NoirLedger_hash_string(const std::string& input_str) {
    std::vector<unsigned char> hash_output(NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES);
    NoirLedger_hash(reinterpret_cast<const unsigned char*>(input_str.data()), input_str.length(), hash_output.data());
    return hash_output;
}