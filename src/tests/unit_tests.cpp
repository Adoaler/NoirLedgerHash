#include "../NoirLedger_hash/NoirLedger_hash.h"
#include "../NoirLedger_hash/debug_flags.h"
#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <iomanip> // Para std::hex, std::setw, std::setfill
#include <sstream> // Para std::stringstream

// Função auxiliar para converter um array de bytes para string hexadecimal
std::string bytes_to_hex(const std::array<uint8_t, NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES>& arr) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES; ++i) {
        ss << std::setw(2) << static_cast<int>(arr[i]);
    }
    return ss.str();
}

// Teste unitário básico para o NoirLedgerHash
void test_basic_hash_consistency() {
    std::cout << "Executando test_basic_hash_consistency..." << std::endl;

    // Desativa todos os prints de depuração para este teste para garantir saída limpa
    NoirLedgerDebug::set_flag(NoirLedgerDebug::DebugFlags::ALL, false);

    // Instância do hasher
    NoirLedgerHasher hasher;

    // Mensagem de teste
    std::string test_message = "abc";
    std::vector<uint8_t> input_data(test_message.begin(), test_message.end());

    // Hash esperado para "abc", gerado pela versão atual do algoritmo.
    std::string expected_hash_hex = "131e2072f0166165e944cd50a4c00d444fdb10e36fe63ec4d7b5e04b9f5603cb";

    // Calcula o hash da mensagem de teste.
    std::array<uint8_t, NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES> actual_hash = hasher(input_data);
    std::string actual_hash_hex = bytes_to_hex(actual_hash);

    // Compara o resultado com o valor esperado.
    std::cout << "  Mensagem: \"" << test_message << "\"" << std::endl;
    std::cout << "  Hash Esperado:  " << expected_hash_hex << std::endl;
    std::cout << "  Hash Calculado: " << actual_hash_hex << std::endl;

    if (actual_hash_hex == expected_hash_hex) {
        std::cout << "  RESULTADO: SUCESSO!" << std::endl;
    } else {
        std::cerr << "  RESULTADO: FALHA! Hash não corresponde ao esperado." << std::endl;
        throw std::runtime_error("Hash não corresponde ao esperado"); // Lança uma exceção em vez de usar exit()
    }
}

int main() {
    try {
        test_basic_hash_consistency();
    } catch (const std::exception& e) {
        std::cerr << "Teste falhou: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}