#ifndef NOXIUM_DEBUG_FLAGS_H
#define NOXIUM_DEBUG_FLAGS_H

#include <cstdint> // Para uint32_t

namespace NoirLedgerDebug {

// Enum para as flags de depuração, usando bitmask
enum DebugFlags : uint32_t {
    NONE             = 0,
    SEED_EXPANSION   = 1 << 0, // Estágio 1
    AES_STAGE        = 1 << 1, // Estágio 2
    FP_STAGE         = 1 << 2, // Estágio 3
    LOGIC_STAGE      = 1 << 3, // Estágio 4 (Novo)
    MEMORY_STAGE     = 1 << 4, // Estágio 5
    BLAKE3_FINAL     = 1 << 5, // Estágio 6
    BLAKE3_COMPRESS  = 1 << 6, // Blake3 compress (interno)
    ALL              = 0xFFFFFFFF // Todas as flags
};

// Funções para definir e verificar as flags de depuração.
// A máscara de bits real é uma variável estática interna ao .cpp
void set_flag(DebugFlags flag, bool enable);
bool is_flag_enabled(DebugFlags flag);

} // namespace NoirLedgerDebug

#endif // NOXIUM_DEBUG_FLAGS_H