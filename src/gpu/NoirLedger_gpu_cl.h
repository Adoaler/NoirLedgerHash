#ifndef NOXIUM_GPU_CL_H
#define NOXIUM_GPU_CL_H

#ifdef NOXIUM_ENABLE_OPENCL

#include <vector>
#include <string>
#include <CL/cl.h>

// Forward declaration para evitar incluir NoirLedger_hash.h aqui
class NoirLedgerHasher;

class NoirLedgerGPU_CL {
public:
    NoirLedgerGPU_CL(NoirLedgerHasher& hasher, int platform_id = 0, int device_id = 0);
    ~NoirLedgerGPU_CL();

    bool is_initialized() const { return initialized; }
    std::string get_device_name() const { return device_name; }

    void hash(
        const unsigned char* input,
        size_t input_len,
        unsigned char* output,
        size_t num_hashes
    );

private:
    bool initialized = false;
    NoirLedgerHasher& parent_hasher;

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_mem input_buffer = nullptr;
    cl_mem output_buffer = nullptr;
    cl_mem lookup_table_buffer = nullptr;
    size_t input_buffer_size = 0;
    size_t output_buffer_size = 0;

    std::string device_name;

    // Funções de inicialização (agora retornam void e lançam exceções)
    void select_platform(int platform_id);
    void select_device(int device_id);
    void create_context_and_queue();
    void build_program();
    void create_lookup_buffer();
    void ensure_buffers_are_allocated(size_t num_hashes, size_t input_len);
    void release_resources();
};

#endif // NOXIUM_ENABLE_OPENCL
#endif // NOXIUM_GPU_CL_H