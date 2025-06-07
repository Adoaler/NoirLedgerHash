/*
 * NoirLedgerHash OpenCL Kernel
 *
 * Este arquivo contém a implementação completa do algoritmo NoirLedgerHash em OpenCL C.
 * Inclui implementações de software para Blake3, AES-256 e ChaCha20,
 * adaptadas para execução em GPU.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// ============================================================================
// Definições de Constantes
// ============================================================================

#define NOXIUM_HASH_OUTPUT_SIZE_BYTES 32
#define NOXIUM_AES_KERNELS 8
#define NOXIUM_AES_ROUNDS 10
#define NOXIUM_MEMORY_LOOKUPS 16
#define NOXIUM_CHACHA_ROUNDS 8
#define NOXIUM_FP_VALUES 16
#define NOXIUM_FP_ROUNDS 16
#define NOXIUM_WORKING_MEMORY_PER_HASH_BYTES (2 * 1024 * 1024)

// ============================================================================
// Funções Auxiliares de Tipo e Rotação
// ============================================================================

// Rotação de 32 bits à direita
inline uint rotate_right(uint x, int n) {
    return (x >> n) | (x << (32 - n));
}

// Conversão de bytes para palavras (little-endian)
inline uint u8_to_u32_le(const uchar p[4]) {
    return ((uint)p[0]) | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
}

inline void u32_to_u8_le(uchar p[4], uint v) {
    p[0] = (uchar)(v);
    p[1] = (uchar)(v >> 8);
    p[2] = (uchar)(v >> 16);
    p[3] = (uchar)(v >> 24);
}

// ============================================================================
// Implementação do Blake3 (Simplificada para o Kernel)
// ============================================================================

// Constantes IV do Blake3
constant uint BLAKE3_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

// Tabela de permutação de mensagens do Blake3
constant uchar BLAKE3_MSG_PERMUTATION[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};

// Função G do Blake3
#define G(state, a, b, c, d, m1, m2) \
    state[a] = state[a] + state[b] + m1; \
    state[d] = rotate_right(state[d] ^ state[a], 16); \
    state[c] = state[c] + state[d]; \
    state[b] = rotate_right(state[b] ^ state[c], 12); \
    state[a] = state[a] + state[b] + m2; \
    state[d] = rotate_right(state[d] ^ state[a], 8); \
    state[c] = state[c] + state[d]; \
    state[b] = rotate_right(state[b] ^ state[c], 7);

// Função de compressão do Blake3
void blake3_compress_in_place(uint state[16]) {
    for (int r = 0; r < 7; ++r) {
        // Permutação da mensagem para a rodada atual
        uint schedule[16];
        for(int i=0; i<16; ++i) schedule[i] = state[BLAKE3_MSG_PERMUTATION[i]];

        // Rodada
        G(state, 0, 4, 8,  12, schedule[0],  schedule[1]);
        G(state, 1, 5, 9,  13, schedule[2],  schedule[3]);
        G(state, 2, 6, 10, 14, schedule[4],  schedule[5]);
        G(state, 3, 7, 11, 15, schedule[6],  schedule[7]);
        G(state, 0, 5, 10, 15, schedule[8],  schedule[9]);
        G(state, 1, 6, 11, 12, schedule[10], schedule[11]);
        G(state, 2, 7, 8,  13, schedule[12], schedule[13]);
        G(state, 3, 4, 9,  14, schedule[14], schedule[15]);
    }
}

// Função de hash Blake3 simplificada para uso no kernel
void blake3_hash_kernel(const uchar* input, uint input_len, uchar output[32]) {
    uint state[16];
    
    // Inicialização do estado
    for(int i=0; i<8; ++i) state[i] = BLAKE3_IV[i];
    for(int i=8; i<16; ++i) state[i] = 0;

    // Processa a entrada em blocos de 64 bytes
    uint offset = 0;
    while (offset < input_len) {
        uint block[16];
        uint bytes_to_process = min((uint)64, input_len - offset);
        
        uchar temp_block[64] = {0};
        for(uint i=0; i<bytes_to_process; ++i) temp_block[i] = input[offset + i];

        for(int i=0; i<16; ++i) block[i] = u8_to_u32_le(temp_block + i*4);

        // Mistura o bloco no estado
        for(int i=0; i<16; ++i) state[i] ^= block[i];
        
        blake3_compress_in_place(state);
        offset += 64;
    }

    // Finalização e escrita da saída
    for (int i = 0; i < 8; ++i) {
        u32_to_u8_le(output + i * 4, state[i]);
    }
}


// ============================================================================
// Implementação do AES-256 (Software)
// ============================================================================

// S-Box para AES
constant uchar sbox[256] = {
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

// Multiplicação em GF(2^8)
uchar gmul(uchar a, uchar b) {
    uchar p = 0;
    for (int i = 0; i < 8; i++) {
        if ((b & 1) != 0) {
            p ^= a;
        }
        uchar hi_bit_set = (a & 0x80);
        a <<= 1;
        if (hi_bit_set != 0) {
            a ^= 0x1b; // x^8 + x^4 + x^3 + x + 1
        }
        b >>= 1;
    }
    return p;
}

void sub_bytes(uchar state[4][4]) {
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) state[i][j] = sbox[state[i][j]];
}

void shift_rows(uchar state[4][4]) {
    uchar temp;
    // Linha 1
    temp = state[1][0]; state[1][0] = state[1][1]; state[1][1] = state[1][2]; state[1][2] = state[1][3]; state[1][3] = temp;
    // Linha 2
    temp = state[2][0]; state[2][0] = state[2][2]; state[2][2] = temp;
    temp = state[2][1]; state[2][1] = state[2][3]; state[2][3] = temp;
    // Linha 3
    temp = state[3][3]; state[3][3] = state[3][2]; state[3][2] = state[3][1]; state[3][1] = state[3][0]; state[3][0] = temp;
}

void mix_columns(uchar state[4][4]) {
    uchar t[4];
    for (int i = 0; i < 4; i++) {
        t[0] = state[0][i]; t[1] = state[1][i]; t[2] = state[2][i]; t[3] = state[3][i];
        state[0][i] = gmul(t[0], 2) ^ gmul(t[1], 3) ^ gmul(t[2], 1) ^ gmul(t[3], 1);
        state[1][i] = gmul(t[0], 1) ^ gmul(t[1], 2) ^ gmul(t[2], 3) ^ gmul(t[3], 1);
        state[2][i] = gmul(t[0], 1) ^ gmul(t[1], 1) ^ gmul(t[2], 2) ^ gmul(t[3], 3);
        state[3][i] = gmul(t[0], 3) ^ gmul(t[1], 1) ^ gmul(t[2], 1) ^ gmul(t[3], 2);
    }
}

void add_round_key(uchar state[4][4], const uchar* round_key) {
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) state[j][i] ^= round_key[i * 4 + j];
}

// Função de criptografia AES (simplificada, expansão de chave não portada)
// Usa a chave diretamente como chave de rodada para simplificar.
void aes256_encrypt_kernel(uchar data[16], const uchar key[32]) {
    uchar state[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) state[j][i] = data[i * 4 + j];

    add_round_key(state, key); // Usa os primeiros 16 bytes da chave

    for (int r = 0; r < NOXIUM_AES_ROUNDS - 1; ++r) {
        sub_bytes(state);
        shift_rows(state);
        mix_columns(state);
        add_round_key(state, key + 16); // Usa os segundos 16 bytes da chave
    }

    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, key);

    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) data[i * 4 + j] = state[j][i];
}

// ============================================================================
// Implementação do ChaCha20
// ============================================================================

#define CHACHA_QR(a, b, c, d) \
    a += b; d ^= a; d = rotate_right(d, 16); \
    c += d; b ^= c; b = rotate_right(b, 12); \
    a += b; d ^= a; d = rotate_right(d,  8); \
    c += d; b ^= c; b = rotate_right(b,  7);

void chacha_block(uint state[16]) {
    uint working_state[16];
    for(int i=0; i<16; ++i) working_state[i] = state[i];

    for (int i = 0; i < NOXIUM_CHACHA_ROUNDS; ++i) {
        CHACHA_QR(working_state[0], working_state[4], working_state[ 8], working_state[12]);
        CHACHA_QR(working_state[1], working_state[5], working_state[ 9], working_state[13]);
        CHACHA_QR(working_state[2], working_state[6], working_state[10], working_state[14]);
        CHACHA_QR(working_state[3], working_state[7], working_state[11], working_state[15]);
        CHACHA_QR(working_state[0], working_state[5], working_state[10], working_state[15]);
        CHACHA_QR(working_state[1], working_state[6], working_state[11], working_state[12]);
        CHACHA_QR(working_state[2], working_state[7], working_state[ 8], working_state[13]);
        CHACHA_QR(working_state[3], working_state[4], working_state[ 9], working_state[14]);
    }

    for (int i = 0; i < 16; ++i) state[i] += working_state[i];
}

void chacha20_mix_kernel(uchar* data, uint data_len, const uchar key[32], const uchar nonce[12]) {
    uint state[16];
    // Constantes
    state[0] = 0x61707865; state[1] = 0x3320646e; state[2] = 0x79622d32; state[3] = 0x6b206574;
    // Chave
    for (int i = 0; i < 8; ++i) state[4 + i] = u8_to_u32_le(key + i * 4);
    // Contador e Nonce
    state[12] = 1; // Contador de bloco
    for (int i = 0; i < 3; ++i) state[13 + i] = u8_to_u32_le(nonce + i * 4);

    chacha_block(state);

    uchar keystream[64];
    for(int i=0; i<16; ++i) u32_to_u8_le(keystream + i*4, state[i]);

    for (uint i = 0; i < data_len; ++i) data[i] ^= keystream[i % 64];
}


// ============================================================================
// Kernel Principal do NoirLedgerHash
// ============================================================================

__kernel void NoirLedger_hash_main(
    __global const uchar* input,
    const uint input_len,
    __global uchar* output,
    __global const uchar* lookup_table
) {
    // --- Inicialização do Work-Item ---
    int gid = get_global_id(0);
    
    // Buffers privados para cada work-item
    uchar expanded_seed[NOXIUM_WORKING_MEMORY_PER_HASH_BYTES];
    uchar working_buffer[128]; // Buffer para resultados intermediários
    uchar current_state[32];

    // --- Estágio 1: Expansão de Semente (Blake3) ---
    // TODO: Implementar XOF (Extendable Output Function) para Blake3
    // Por enquanto, usamos um hash simples e repetimos para preencher.
    uchar temp_hash[32];
    blake3_hash_kernel(input, input_len, temp_hash);
    for(int i=0; i < NOXIUM_WORKING_MEMORY_PER_HASH_BYTES / 32; ++i) {
        for(int j=0; j<32; ++j) expanded_seed[i*32 + j] = temp_hash[j];
    }

    // --- Estágio 2: Computação Paralela (AES-256) ---
    uchar aes_data_blocks[16];
    uchar aes_keys[32];
    for (int i = 0; i < NOXIUM_AES_KERNELS; ++i) {
        for(int j=0; j<16; ++j) aes_data_blocks[j] = expanded_seed[i * 16 + j];
        for(int j=0; j<32; ++j) aes_keys[j] = expanded_seed[128 + i * 32 + j];
        
        aes256_encrypt_kernel(aes_data_blocks, aes_keys);
        
        for(int j=0; j<16; ++j) working_buffer[i * 16 + j] = aes_data_blocks[j];
    }

    // --- Estágio 3: Operações de Ponto Flutuante ---
    double fp_data[NOXIUM_FP_VALUES];
    for (int i = 0; i < NOXIUM_FP_VALUES; ++i) {
        ulong u_val;
        for(int j=0; j<8; ++j) ((uchar*)&u_val)[j] = working_buffer[i*8 + j];
        fp_data[i] = (double)u_val;
    }

    const double K1 = 3.14159265358979323846;
    const double K2 = 1.61803398874989484820;
    const double K_MOD = 1000000007.0;

    for (int r = 0; r < NOXIUM_FP_ROUNDS; ++r) {
        double prev_fp_data[NOXIUM_FP_VALUES];
        for(int i=0; i<NOXIUM_FP_VALUES; ++i) prev_fp_data[i] = fp_data[i];

        for (int i = 0; i < NOXIUM_FP_VALUES; ++i) {
            int prev_idx = (i + NOXIUM_FP_VALUES - 1) % NOXIUM_FP_VALUES;
            int mix1 = (i * 5 + r) % NOXIUM_FP_VALUES;
            int mix2 = (i * 11 + r * 3) % NOXIUM_FP_VALUES;

            double term1 = prev_fp_data[i] * K1 + prev_fp_data[prev_idx];
            double term2 = sqrt(fabs(term1) + 1e-9);
            double term3 = fmod(term2 * K2 + prev_fp_data[mix1], K_MOD);
            
            double res;
            if (r % 2 == 0) res = fmod(term3 + prev_fp_data[mix2] + (double)r * K1, K_MOD);
            else res = fmod(term3 - prev_fp_data[mix2] - (double)i * K2, K_MOD);
            
            fp_data[i] = isfinite(res) ? res : fmod((double)(r*13 + i*7), K_MOD);
        }
    }
    
    uchar temp_fp_bytes[128];
    for(int i=0; i<NOXIUM_FP_VALUES; ++i) {
        for(int j=0; j<8; ++j) temp_fp_bytes[i*8+j] = ((uchar*)&fp_data[i])[j];
    }
    for (int i = 0; i < 32; ++i) {
        current_state[i] = temp_fp_bytes[i] ^ temp_fp_bytes[i + 32] ^ temp_fp_bytes[i + 64] ^ temp_fp_bytes[i + 96];
    }

    // --- Estágio 4: Lógica Mista ---
    uint logic_state[8];
    for(int i=0; i<8; ++i) {
        logic_state[i] = u8_to_u32_le(current_state + i*4);
    }

    for (int i = 0; i < 64; ++i) {
        uint idx1 = logic_state[0] % 8;
        uint idx2 = logic_state[1] % 8;
        uint r_val = logic_state[2];

        uint temp = logic_state[idx1];
        logic_state[idx1] = (logic_state[idx2] * 0x9E3779B9) + r_val;
        if (temp > logic_state[idx1]) {
            logic_state[idx2] = rotate_right(temp, (r_val % 32));
        } else {
            logic_state[idx2] = temp ^ r_val;
        }
        logic_state[2] = logic_state[idx1] + logic_state[idx2];
    }

    for(int i=0; i<8; ++i) {
        u32_to_u8_le(current_state + i*4, logic_state[i]);
    }

    // --- Estágio 5: Operações de Memória ---
    for (int i = 0; i < NOXIUM_MEMORY_LOOKUPS; ++i) {
        ulong temp_addr;
        for(int j=0; j<8; ++j) ((uchar*)&temp_addr)[j] = current_state[j];
        
        temp_addr *= 0x9E3779B97F4A7C15UL;
        temp_addr ^= (temp_addr >> 32);
        size_t lookup_index = temp_addr % (NOXIUM_LOOKUP_TABLE_SIZE_MB * 1024 * 1024 - 64);

        uchar memory_data[64];
        for(int j=0; j<64; ++j) memory_data[j] = lookup_table[lookup_index + j];

        uchar chacha_key[32];
        uchar chacha_nonce[12];
        
        // Derivação do Nonce: Usa um hash do estado atual e dos dados de memória
        // para garantir um nonce único e imprevisível para cada lookup.
        uchar nonce_material[96]; // 32 bytes do estado + 64 bytes dos dados de memória
        for(int j=0; j<32; ++j) nonce_material[j] = current_state[j];
        for(int j=0; j<64; ++j) nonce_material[32+j] = memory_data[j];
        
        uchar nonce_hash[32];
        blake3_hash_kernel(nonce_material, 96, nonce_hash);
        for(int j=0; j<12; ++j) chacha_nonce[j] = nonce_hash[j]; // Usa os primeiros 12 bytes do hash como nonce

        // A chave ainda é derivada dos primeiros 32 bytes dos dados de memória
        for(int j=0; j<32; ++j) chacha_key[j] = memory_data[j];

        // Mistura o estado com os dados de memória antes de aplicar o ChaCha20
        for(int j=0; j < 32; ++j) current_state[j] ^= memory_data[j + 32];
        
        chacha20_mix_kernel(current_state, 32, chacha_key, chacha_nonce);
    }

    // --- Estágio 6: Compressão Final ---
    uchar final_hash[32];
    blake3_hash_kernel(current_state, 32, final_hash);

    // --- Escrita da Saída ---
    for (int i = 0; i < NOXIUM_HASH_OUTPUT_SIZE_BYTES; ++i) {
        output[gid * NOXIUM_HASH_OUTPUT_SIZE_BYTES + i] = final_hash[i];
    }
}