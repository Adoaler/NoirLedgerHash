#include "blake3_internal.h"
#include "../NoirLedger_hash/debug_flags.h" // Novo sistema de flags de depuração
#include <iostream>
#include <iomanip> // Para std::setw, std::setfill, std::hex
#include <immintrin.h> // Para intrinsics AVX2
#include <cstring> // Para memcpy
#include <cassert> // Para assert


// BLAKE3 IV
static const uint32_t BLAKE3_IV[8] = {
    0x6A09E667UL, 0xBB67AE85UL, 0x3C6EF372UL, 0xA54FF53AUL,
    0x510E527FUL, 0x9B05688CUL, 0x1F83D9EBUL, 0x5BE0CD19UL
};

// BLAKE3 Message Schedule
static const uint8_t BLAKE3_MSG_SCHEDULE[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {12, 8, 13, 9, 14, 10, 15, 11, 6, 2, 7, 3, 0, 4, 1, 5},
    {11, 15, 10, 14, 3, 7, 2, 6, 1, 5, 0, 4, 13, 9, 12, 8},
    {1, 12, 7, 10, 14, 0, 5, 15, 13, 3, 9, 2, 6, 11, 4, 8},
    {9, 14, 15, 5, 2, 8, 12, 1, 7, 10, 3, 4, 11, 0, 6, 13},
    {10, 2, 8, 12, 15, 11, 14, 6, 4, 0, 13, 7, 5, 1, 9, 3},
    {13, 7, 9, 1, 11, 14, 12, 3, 5, 0, 15, 4, 8, 6, 2, 10}
};

// Helper para converter bytes para palavras (uint32_t)
static inline void words_from_bytes(const uint8_t *bytes, uint32_t *words, size_t num_words) {
    for (size_t i = 0; i < num_words; ++i) {
        words[i] = load_u32_le(bytes + i * 4);
    }
}

#if defined(__AVX2__)

// Custom _mm256_rotr_epi32 for AVX2 compatibility
static inline __m256i rotr256_epi32(__m256i a, const int imm8) {
    return _mm256_or_si256(_mm256_slli_epi32(a, 32 - imm8), _mm256_srli_epi32(a, imm8));
}

// Função de depuração segura para o estado AVX2
static void print_state_avx2(const char* prefix, int round_idx, __m256i s0, __m256i s1, __m256i s2, __m256i s3) {
    alignas(32) uint32_t debug_s0[8], debug_s1[8], debug_s2[8], debug_s3[8];
    _mm256_store_si256((__m256i*)debug_s0, s0);
    _mm256_store_si256((__m256i*)debug_s1, s1);
    _mm256_store_si256((__m256i*)debug_s2, s2);
    _mm256_store_si256((__m256i*)debug_s3, s3);
    
    std::cout << "[" << prefix << " R" << round_idx << " AVX2] StateCols:\n";
    std::cout << std::hex << std::setfill('0');
    std::cout << "  s0: " << std::setw(8) << debug_s0[0] << " " << std::setw(8) << debug_s0[1] << " " << std::setw(8) << debug_s0[2] << " " << std::setw(8) << debug_s0[3] << "\n";
    std::cout << "  s1: " << std::setw(8) << debug_s1[0] << " " << std::setw(8) << debug_s1[1] << " " << std::setw(8) << debug_s1[2] << " " << std::setw(8) << debug_s1[3] << "\n";
    std::cout << "  s2: " << std::setw(8) << debug_s2[0] << " " << std::setw(8) << debug_s2[1] << " " << std::setw(8) << debug_s2[2] << " " << std::setw(8) << debug_s2[3] << "\n";
    std::cout << "  s3: " << std::setw(8) << debug_s3[0] << " " << std::setw(8) << debug_s3[1] << " " << std::setw(8) << debug_s3[2] << " " << std::setw(8) << debug_s3[3] << "\n";
    std::cout << std::dec;
}

// Transpõe um estado de 4x4 __m128i de linhas para colunas e vice-versa.
static inline void transpose_state_4x4(__m128i* s0, __m128i* s1, __m128i* s2, __m128i* s3) {
    _MM_TRANSPOSE4_PS(*(__m128*)s0, *(__m128*)s1, *(__m128*)s2, *(__m128*)s3);
}

// Aplica a transformação G de BLAKE3 em um conjunto de colunas/diagonais.
static inline void g_transform_avx2(__m256i& a, __m256i& b, __m256i& c, __m256i& d, __m256i mx, __m256i my) {
    a = _mm256_add_epi32(a, b);
    a = _mm256_add_epi32(a, mx);
    d = _mm256_xor_si256(d, a);
    d = rotr256_epi32(d, 16);
    c = _mm256_add_epi32(c, d);
    b = _mm256_xor_si256(b, c);
    b = rotr256_epi32(b, 12);
    a = _mm256_add_epi32(a, b);
    a = _mm256_add_epi32(a, my);
    d = _mm256_xor_si256(d, a);
    d = rotr256_epi32(d, 8);
    c = _mm256_add_epi32(c, d);
    b = _mm256_xor_si256(b, c);
    b = rotr256_epi32(b, 7);
}

static void column_round_avx2(__m256i& v_col_a, __m256i& v_col_b, __m256i& v_col_c, __m256i& v_col_d, const uint32_t m_words[16], const uint8_t schedule[16]) {
    transpose_state_4x4((__m128i*)&v_col_a, (__m128i*)&v_col_b, (__m128i*)&v_col_c, (__m128i*)&v_col_d);

    __m256i mx_col = _mm256_set_epi32(0,0,0,0, m_words[schedule[6]], m_words[schedule[4]], m_words[schedule[2]], m_words[schedule[0]]);
    __m256i my_col = _mm256_set_epi32(0,0,0,0, m_words[schedule[7]], m_words[schedule[5]], m_words[schedule[3]], m_words[schedule[1]]);

    g_transform_avx2(v_col_a, v_col_b, v_col_c, v_col_d, mx_col, my_col);

    transpose_state_4x4((__m128i*)&v_col_a, (__m128i*)&v_col_b, (__m128i*)&v_col_c, (__m128i*)&v_col_d);
}

static void diagonal_round_avx2(__m256i& v_col_a, __m256i& v_col_b, __m256i& v_col_c, __m256i& v_col_d, const uint32_t m_words[16], const uint8_t schedule[16]) {
    alignas(32) uint32_t state_flat[16];
    alignas(32) uint32_t col_a[8], col_b[8], col_c[8], col_d[8];
    
    _mm256_store_si256((__m256i*)col_a, v_col_a);
    _mm256_store_si256((__m256i*)col_b, v_col_b);
    _mm256_store_si256((__m256i*)col_c, v_col_c);
    _mm256_store_si256((__m256i*)col_d, v_col_d);
    
    state_flat[0] = col_a[0];  state_flat[1] = col_b[0];  state_flat[2] = col_c[0];  state_flat[3] = col_d[0];
    state_flat[4] = col_a[1];  state_flat[5] = col_b[1];  state_flat[6] = col_c[1];  state_flat[7] = col_d[1];
    state_flat[8] = col_a[2];  state_flat[9] = col_b[2];  state_flat[10] = col_c[2]; state_flat[11] = col_d[2];
    state_flat[12] = col_a[3]; state_flat[13] = col_b[3]; state_flat[14] = col_c[3]; state_flat[15] = col_d[3];

    __m256i opA = _mm256_set_epi32(0,0,0,0, state_flat[3], state_flat[2], state_flat[1], state_flat[0]);
    __m256i opB = _mm256_set_epi32(0,0,0,0, state_flat[4], state_flat[7], state_flat[6], state_flat[5]);
    __m256i opC = _mm256_set_epi32(0,0,0,0, state_flat[9], state_flat[8], state_flat[11], state_flat[10]);
    __m256i opD = _mm256_set_epi32(0,0,0,0, state_flat[14], state_flat[13], state_flat[12], state_flat[15]);

    __m256i mx_diag = _mm256_set_epi32(0,0,0,0, m_words[schedule[14]], m_words[schedule[12]], m_words[schedule[10]], m_words[schedule[8]]);
    __m256i my_diag = _mm256_set_epi32(0,0,0,0, m_words[schedule[15]], m_words[schedule[13]], m_words[schedule[11]], m_words[schedule[9]]);

    g_transform_avx2(opA, opB, opC, opD, mx_diag, my_diag);

    alignas(32) uint32_t diag_a[8], diag_b[8], diag_c[8], diag_d[8];
    _mm256_store_si256((__m256i*)diag_a, opA);
    _mm256_store_si256((__m256i*)diag_b, opB);
    _mm256_store_si256((__m256i*)diag_c, opC);
    _mm256_store_si256((__m256i*)diag_d, opD);

    state_flat[0] = diag_a[0];   state_flat[1] = diag_a[1];   state_flat[2] = diag_a[2];   state_flat[3] = diag_a[3];
    state_flat[4] = diag_b[3];   state_flat[5] = diag_b[0];   state_flat[6] = diag_b[1];   state_flat[7] = diag_b[2];
    state_flat[8] = diag_c[2];   state_flat[9] = diag_c[3];   state_flat[10] = diag_c[0];  state_flat[11] = diag_c[1];
    state_flat[12] = diag_d[1];  state_flat[13] = diag_d[2];  state_flat[14] = diag_d[3];  state_flat[15] = diag_d[0];

    v_col_a = _mm256_set_epi32(0,0,0,0, state_flat[12], state_flat[8], state_flat[4], state_flat[0]);
    v_col_b = _mm256_set_epi32(0,0,0,0, state_flat[13], state_flat[9], state_flat[5], state_flat[1]);
    v_col_c = _mm256_set_epi32(0,0,0,0, state_flat[14], state_flat[10], state_flat[6], state_flat[2]);
    v_col_d = _mm256_set_epi32(0,0,0,0, state_flat[15], state_flat[11], state_flat[7], state_flat[3]);
}

static void round_fn_avx2(__m256i *v_col_a, __m256i *v_col_b, __m256i *v_col_c, __m256i *v_col_d, const uint32_t m_words[16], int round_idx) {
    if (NoirLedgerDebug::is_flag_enabled(NoirLedgerDebug::DebugFlags::BLAKE3_COMPRESS) && round_idx == 0) {
        print_state_avx2("DEBUG AVX2 In", round_idx, *v_col_a, *v_col_b, *v_col_c, *v_col_d);
        std::cout << "[DEBUG AVX2 R" << round_idx << " In] Msg:   ";
        for(int i=0; i<16; ++i) std::cout << std::hex << std::setw(8) << m_words[i] << " ";
        std::cout << std::dec << "\n";
    }

    const uint8_t* schedule = BLAKE3_MSG_SCHEDULE[round_idx];
    
    column_round_avx2(*v_col_a, *v_col_b, *v_col_c, *v_col_d, m_words, schedule);
    if (NoirLedgerDebug::is_flag_enabled(NoirLedgerDebug::DebugFlags::BLAKE3_COMPRESS) && round_idx == 0) {
        print_state_avx2("DEBUG AVX2 AfterCols", round_idx, *v_col_a, *v_col_b, *v_col_c, *v_col_d);
    }

    diagonal_round_avx2(*v_col_a, *v_col_b, *v_col_c, *v_col_d, m_words, schedule);
    if (NoirLedgerDebug::is_flag_enabled(NoirLedgerDebug::DebugFlags::BLAKE3_COMPRESS) && round_idx == 0) {
        print_state_avx2("DEBUG AVX2 AfterDiags", round_idx, *v_col_a, *v_col_b, *v_col_c, *v_col_d);
    }
}

#endif // __AVX2__

// Rotação à esquerda de 32 bits (versão C puro)
static inline uint32_t rotl32_c_pure(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// Função G em C puro, agora como uma função inline em vez de uma macro
static inline void g_c_pure(uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3, uint32_t m0, uint32_t m1) {
    v0 += v1 + m0;
    v3 = rotl32_c_pure(v3 ^ v0, 16);
    v2 += v3;
    v1 = rotl32_c_pure(v1 ^ v2, 12);
    v0 += v1 + m1;
    v3 = rotl32_c_pure(v3 ^ v0, 8);
    v2 += v3;
    v1 = rotl32_c_pure(v1 ^ v2, 7);
}

// A função de compressão principal do BLAKE3
void blake3_compress_in_place(uint32_t cv[8], const uint8_t block[BLAKE3_CUSTOM_BLOCK_LEN],
                              uint8_t block_len, uint64_t counter, uint8_t flags) {
#if defined(__AVX2__)
    uint32_t block_words[16];
    words_from_bytes(block, block_words, 16);

    // Initialize state correctly in column-major format
    __m256i v_col_a = _mm256_set_epi32(0,0,0,0, (uint32_t)counter, BLAKE3_IV[0], cv[4], cv[0]);
    __m256i v_col_b = _mm256_set_epi32(0,0,0,0, (uint32_t)(counter >> 32), BLAKE3_IV[1], cv[5], cv[1]);
    __m256i v_col_c = _mm256_set_epi32(0,0,0,0, (uint32_t)block_len, BLAKE3_IV[2], cv[6], cv[2]);
    __m256i v_col_d = _mm256_set_epi32(0,0,0,0, (uint32_t)flags, BLAKE3_IV[3], cv[7], cv[3]);

    for (int r = 0; r < 7; ++r) {
        round_fn_avx2(&v_col_a, &v_col_b, &v_col_c, &v_col_d, block_words, r);
    }

    // Extract final state and compute output
    alignas(32) uint32_t final_a[8], final_b[8], final_c[8], final_d[8];
    _mm256_store_si256((__m256i*)final_a, v_col_a);
    _mm256_store_si256((__m256i*)final_b, v_col_b);
    _mm256_store_si256((__m256i*)final_c, v_col_c);
    _mm256_store_si256((__m256i*)final_d, v_col_d);

    // Reconstruct final state
    uint32_t final_state[16];
    final_state[0] = final_a[0];  final_state[1] = final_b[0];  final_state[2] = final_c[0];  final_state[3] = final_d[0];
    final_state[4] = final_a[1];  final_state[5] = final_b[1];  final_state[6] = final_c[1];  final_state[7] = final_d[1];
    final_state[8] = final_a[2];  final_state[9] = final_b[2];  final_state[10] = final_c[2]; final_state[11] = final_d[2];
    final_state[12] = final_a[3]; final_state[13] = final_b[3]; final_state[14] = final_c[3]; final_state[15] = final_d[3];

    // Apply final XOR to get chaining value
    for (size_t i = 0; i < 8; ++i) {
        cv[i] = final_state[i] ^ final_state[i + 8];
    }
#else
    uint32_t state[16];
    memcpy(state, cv, sizeof(uint32_t) * 8);
    memcpy(state + 8, BLAKE3_IV, sizeof(BLAKE3_IV));

    state[12] ^= (uint32_t)counter;
    state[13] ^= (uint32_t)(counter >> 32);
    state[14] ^= (uint32_t)block_len;
    state[15] ^= (uint32_t)flags;

    uint32_t msg_words[16];
    words_from_bytes(block, msg_words, 16);

    if (NoirLedgerDebug::is_flag_enabled(NoirLedgerDebug::DebugFlags::BLAKE3_COMPRESS)) {
        std::cout << "[DEBUG C R0 In] State: ";
        for(int i=0; i<16; ++i) std::cout << std::hex << std::setw(8) << state[i] << " ";
        std::cout << "\n[DEBUG C R0 In] Msg:   ";
        for(int i=0; i<16; ++i) std::cout << std::hex << std::setw(8) << msg_words[i] << " ";
        std::cout << std::dec << "\n";
    }

    for (int r = 0; r < 7; ++r) {
        const uint8_t* schedule = BLAKE3_MSG_SCHEDULE[r];
        // Rodadas de Colunas
        g_c_pure(state[0], state[4], state[8], state[12], msg_words[schedule[0]], msg_words[schedule[1]]);
        g_c_pure(state[1], state[5], state[9], state[13], msg_words[schedule[2]], msg_words[schedule[3]]);
        g_c_pure(state[2], state[6], state[10], state[14], msg_words[schedule[4]], msg_words[schedule[5]]);
        g_c_pure(state[3], state[7], state[11], state[15], msg_words[schedule[6]], msg_words[schedule[7]]);
        // Rodadas de Diagonais
        g_c_pure(state[0], state[5], state[10], state[15], msg_words[schedule[8]], msg_words[schedule[9]]);
        g_c_pure(state[1], state[6], state[11], state[12], msg_words[schedule[10]], msg_words[schedule[11]]);
        g_c_pure(state[2], state[7], state[8], state[13], msg_words[schedule[12]], msg_words[schedule[13]]);
        g_c_pure(state[3], state[4], state[9], state[14], msg_words[schedule[14]], msg_words[schedule[15]]);
    }

    for (int i = 0; i < 8; ++i) {
        cv[i] = state[i] ^ state[i + 8];
    }
#endif
}