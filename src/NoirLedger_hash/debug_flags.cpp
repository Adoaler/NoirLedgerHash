#include "debug_flags.h"

namespace NoirLedgerDebug {

// Bitmask que armazena os flags de depuração ativos.
// O prefixo 's_' indica que a variável tem ligação estática (interna ao arquivo).
static uint32_t s_active_flags_mask = DebugFlags::NONE;

void set_flag(DebugFlags flag, bool enable) {
    if (enable) {
        s_active_flags_mask |= static_cast<uint32_t>(flag);
    } else {
        s_active_flags_mask &= ~static_cast<uint32_t>(flag);
    }
}

bool is_flag_enabled(DebugFlags flag) {
    return (s_active_flags_mask & static_cast<uint32_t>(flag)) != 0;
}

} // namespace NoirLedgerDebug