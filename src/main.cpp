#include "NoirLedger_hash/NoirLedger_hash.h"
#include "blake3_custom/blake3_custom.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <thread>
#include <atomic>
#include <regex>
#include <functional>
#include <map>

#include "NoirLedger_hash/debug_flags.h"

#ifdef NOXIUM_ENABLE_OPENCL
#include <CL/cl.h>
#endif

// Estrutura para armazenar os argumentos da linha de comando
struct AppConfig {
    enum class BenchmarkMode { CPU, GPU, DIAGNOSE_CL };
    BenchmarkMode mode = BenchmarkMode::CPU;
    int num_threads = std::thread::hardware_concurrency();
    int total_iterations = 100000;
    int gpu_platform_id = 0;
    int gpu_device_id = 0;
};

// Função para imprimir o hash em hexadecimal
void print_hash(const unsigned char* hash_val, size_t len) {
    std::cout << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        std::cout << std::setw(2) << static_cast<int>(hash_val[i]);
    }
    std::cout << std::dec << std::endl;
}

// Variável atômica para contar o total de hashes calculados
std::atomic<long long> total_hashes_calculated(0);

// Função worker para cada thread
void hash_worker(NoirLedgerHasher& hasher, const std::vector<unsigned char>& input_data, int iterations_per_thread) {
    std::array<uint8_t, NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES> hash_output_array;
    for (int i = 0; i < iterations_per_thread; ++i) {
        hasher.hash_cpu(input_data.data(), input_data.size(), hash_output_array.data());
        total_hashes_calculated.fetch_add(1, std::memory_order_relaxed);
    }
}

void diagnose_opencl() {
    std::cout << "--- Diagnóstico OpenCL ---" << std::endl;
#ifdef NOXIUM_ENABLE_OPENCL
    // ... (código de diagnóstico OpenCL como estava antes)
#else
    std::cout << "O suporte a OpenCL não foi habilitado durante a compilação." << std::endl;
#endif
}

void print_help(const char* app_name) {
    std::string sanitized_app_name = std::regex_replace(std::string(app_name), std::regex("[^a-zA-Z0-9_.-]"), "");
    std::cout << "Uso: " << sanitized_app_name << " [opções]" << std::endl;
    std::cout << "Opções:" << std::endl;
    std::cout << "  --mode <cpu|gpu>      Modo de benchmark (padrão: cpu)." << std::endl;
    std::cout << "  -t, --threads <n>     Número de threads para o modo CPU." << std::endl;
    std::cout << "  -i, --iterations <n>  Total de hashes a serem calculados." << std::endl;
    std::cout << "  --gpu-platform <id>   ID da plataforma OpenCL." << std::endl;
    std::cout << "  --gpu-device <id>     ID do dispositivo GPU." << std::endl;
    std::cout << "  --diagnose-cl         Exibe informações sobre OpenCL." << std::endl;
    std::cout << "  --debug <flag>        Ativa um flag de depuração (ex: --debug AES_STAGE)." << std::endl;
    std::cout << "  -h, --help            Exibe esta mensagem de ajuda." << std::endl;
}

void run_cpu_benchmark(NoirLedgerHasher& hasher, const AppConfig& config, const std::vector<unsigned char>& input_data) {
    int iterations_per_thread = config.total_iterations / config.num_threads;
    if (iterations_per_thread == 0) iterations_per_thread = 1;

    std::cout << "--- Iniciando Benchmark de CPU ---" << std::endl;
    std::cout << "Usando " << config.num_threads << " threads, " << config.total_iterations << " hashes no total." << std::endl;

    total_hashes_calculated = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> workers;
    try {
        for (int i = 0; i < config.num_threads; ++i) {
            workers.emplace_back(hash_worker, std::ref(hasher), std::cref(input_data), iterations_per_thread);
        }
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
    } catch (const std::exception& e) {
        std::cerr << "Erro durante a execução do benchmark de CPU: " << e.what() << std::endl;
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
        return;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << "--- Resultados do Benchmark de CPU ---" << std::endl;
    std::cout << "Tempo total: " << std::fixed << std::setprecision(6) << diff.count() << " segundos." << std::endl;
    if (diff.count() > 0) {
        double hps = total_hashes_calculated.load() / diff.count();
        std::cout << "Hashrate: " << std::fixed << std::setprecision(2) << hps << " H/s" << std::endl;
    }
}

void run_gpu_benchmark(NoirLedgerHasher& hasher, const AppConfig& config, const std::vector<unsigned char>& input_data) {
    std::cout << "--- Iniciando Benchmark de GPU ---" << std::endl;
    if (!hasher.is_gpu_initialized()) {
        std::cerr << "Erro: GPU nao inicializada." << std::endl;
        return;
    }
    std::cout << "Backend: " << hasher.get_gpu_backend_name() << " | Dispositivo: " << hasher.get_gpu_device_name() << std::endl;
    std::cout << "Calculando " << config.total_iterations << " hashes..." << std::endl;

    std::vector<unsigned char> output_hashes(config.total_iterations * NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES);
    std::vector<unsigned char> gpu_input_batch(config.total_iterations * input_data.size());
    for(int i=0; i<config.total_iterations; ++i) {
        memcpy(gpu_input_batch.data() + i * input_data.size(), input_data.data(), input_data.size());
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    hasher.hash_gpu(gpu_input_batch.data(), input_data.size(), output_hashes.data(), config.total_iterations);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    std::cout << "--- Resultados do Benchmark de GPU (" << hasher.get_gpu_backend_name() << ") ---" << std::endl;
    std::cout << "Tempo total: " << std::fixed << std::setprecision(6) << diff.count() << " segundos." << std::endl;
    if (diff.count() > 0) {
        double hps = config.total_iterations / diff.count();
        std::cout << "Hashrate: " << std::fixed << std::setprecision(2) << hps << " H/s" << std::endl;
    }
    
    std::cout << "Hash de amostra da GPU (primeiro hash): ";
    print_hash(output_hashes.data(), NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES);
}

// Mapa para associar strings de debug com os enums
const std::map<std::string, NoirLedgerDebug::DebugFlags> debug_flag_map = {
    {"SEED_EXPANSION", NoirLedgerDebug::DebugFlags::SEED_EXPANSION},
    {"AES_STAGE",      NoirLedgerDebug::DebugFlags::AES_STAGE},
    {"FP_STAGE",       NoirLedgerDebug::DebugFlags::FP_STAGE},
    {"LOGIC_STAGE",    NoirLedgerDebug::DebugFlags::LOGIC_STAGE},
    {"MEMORY_STAGE",   NoirLedgerDebug::DebugFlags::MEMORY_STAGE},
    {"BLAKE3_FINAL",   NoirLedgerDebug::DebugFlags::BLAKE3_FINAL},
    {"BLAKE3_COMPRESS",NoirLedgerDebug::DebugFlags::BLAKE3_COMPRESS},
    {"ALL",            NoirLedgerDebug::DebugFlags::ALL}
};

bool parse_arguments(int argc, char* argv[], AppConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_arg = [&]() -> const char* {
            if (i + 1 < argc) return argv[++i];
            return nullptr;
        };

        if (arg == "-h" || arg == "--help") { print_help(argv[0]); return false; }
        if (arg == "--diagnose-cl") { config.mode = AppConfig::BenchmarkMode::DIAGNOSE_CL; continue; }

        const char* value = next_arg();
        if (!value) {
            std::cerr << "Erro: Argumento '" << arg << "' requer um valor." << std::endl;
            return false;
        }

        if (arg == "--mode") {
            std::string mode_str = value;
            if (mode_str == "gpu") config.mode = AppConfig::BenchmarkMode::GPU;
            else if (mode_str == "cpu") config.mode = AppConfig::BenchmarkMode::CPU;
            else std::cerr << "Modo inválido: " << mode_str << ". Usando 'cpu'." << std::endl;
        } else if (arg == "-t" || arg == "--threads") {
            config.num_threads = std::stoi(value);
        } else if (arg == "-i" || arg == "--iterations") {
            config.total_iterations = std::stoi(value);
        } else if (arg == "--gpu-platform") {
            config.gpu_platform_id = std::stoi(value);
        } else if (arg == "--gpu-device") {
            config.gpu_device_id = std::stoi(value);
        } else if (arg == "--debug") {
            auto it = debug_flag_map.find(value);
            if (it != debug_flag_map.end()) {
                NoirLedgerDebug::set_flag(it->second, true);
            } else {
                std::cerr << "Flag de depuração inválida: " << value << std::endl;
            }
        } else {
            std::cerr << "Argumento desconhecido: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    AppConfig config;
    if (!parse_arguments(argc, argv, config)) {
        return 1;
    }

    if (config.num_threads == 0) config.num_threads = 1;

    try {
        if (config.mode == AppConfig::BenchmarkMode::DIAGNOSE_CL) {
            diagnose_opencl();
            return 0;
        }

        NoirLedgerHasher hasher;
        std::cout << "NoirLedgerHasher inicializado com sucesso." << std::endl;

        std::string input_str = "NoirLedger: Secure and Optimized Proof-of-Work by Kilo Code!";
        std::vector<unsigned char> input_data(input_str.begin(), input_str.end());

        if (config.mode == AppConfig::BenchmarkMode::GPU) {
            if (hasher.init_gpu(config.gpu_device_id, config.gpu_platform_id)) {
                run_gpu_benchmark(hasher, config, input_data);
            } else {
                std::cerr << "Falha ao inicializar a GPU. Verifique os drivers e a configuração." << std::endl;
                std::cerr << "Execute com '--diagnose-cl' para obter informações sobre OpenCL." << std::endl;
                return 1;
            }
        } else { // Modo CPU
            run_cpu_benchmark(hasher, config, input_data);
        }

        std::cout << "\nCalculando um hash de amostra da CPU para verificação..." << std::endl;
        auto final_hash_array = hasher(input_data);
        std::cout << "Hash de amostra da CPU: ";
        print_hash(final_hash_array.data(), NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES);

    } catch (const std::exception& e) {
        std::cerr << "Ocorreu um erro fatal: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}