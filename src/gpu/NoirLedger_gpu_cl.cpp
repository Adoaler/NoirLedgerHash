#ifdef NOXIUM_ENABLE_OPENCL

#include "gpu/NoirLedger_gpu_cl.h"
#include "NoirLedger_hash/NoirLedger_hash.h" // Para acessar a tabela de consulta
#include "NoirLedger_hash_cl.h" // Cabeçalho gerado pelo CMake
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Exceção customizada para erros OpenCL
class ClException : public std::runtime_error {
public:
    ClException(const char* file, int line, cl_int error)
        : std::runtime_error(std::string("OpenCL error code ") + std::to_string(error) +
                             " at " + file + ":" + std::to_string(line)) {}
};

// Macro para verificação de erros OpenCL
#define CL_CHECK(err) { \
    cl_int error = err; \
    if (error != CL_SUCCESS) { \
        throw ClException(__FILE__, __LINE__, error); \
    } \
}

NoirLedgerGPU_CL::NoirLedgerGPU_CL(NoirLedgerHasher& hasher, int platform_id, int device_id)
    : parent_hasher(hasher), initialized(false), platform(nullptr), device(nullptr),
      context(nullptr), queue(nullptr), program(nullptr), kernel(nullptr),
      input_buffer(nullptr), output_buffer(nullptr), lookup_table_buffer(nullptr),
      input_buffer_size(0), output_buffer_size(0) {
    
    try {
        select_platform(platform_id);
        select_device(device_id);
        create_context_and_queue();
        build_program();
        create_lookup_buffer();

        char name_buffer[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name_buffer), name_buffer, nullptr);
        device_name = name_buffer;

        initialized = true;
        std::cout << "NoirLedgerGPU_CL inicializado com sucesso no dispositivo: " << device_name << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Falha na inicialização do NoirLedgerGPU_CL: " << e.what() << std::endl;
        release_resources(); // Garante a limpeza em caso de falha na construção
        throw; // Relança a exceção para o chamador
    }
}

NoirLedgerGPU_CL::~NoirLedgerGPU_CL() {
    release_resources();
}

void NoirLedgerGPU_CL::select_platform(int platform_id) {
    cl_uint num_platforms;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    if (num_platforms == 0) {
        throw std::runtime_error("Nenhuma plataforma OpenCL encontrada.");
    }
    if (platform_id >= num_platforms) {
        throw std::runtime_error("ID da plataforma OpenCL inválido.");
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));
    platform = platforms[platform_id];
}

void NoirLedgerGPU_CL::select_device(int device_id) {
    cl_uint num_devices;
    cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err == CL_DEVICE_NOT_FOUND) {
         throw std::runtime_error("Nenhum dispositivo de GPU OpenCL encontrado na plataforma selecionada.");
    }
    CL_CHECK(err);
    if (device_id >= num_devices) {
        throw std::runtime_error("ID do dispositivo de GPU OpenCL inválido.");
    }

    std::vector<cl_device_id> devices(num_devices);
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr));
    device = devices[device_id];
}

void NoirLedgerGPU_CL::create_context_and_queue() {
    cl_int err;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);

    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    CL_CHECK(err);
}

void NoirLedgerGPU_CL::build_program() {
    cl_int err;
    const char* source = NoirLedgerGPU::ocl_source;
    program = clCreateProgramWithSource(context, 1, &source, nullptr, &err);
    CL_CHECK(err);

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Log de compilação OpenCL:\n" << log.data() << std::endl;
        CL_CHECK(err); // Lança a exceção após imprimir o log
    }

    kernel = clCreateKernel(program, "NoirLedger_hash_main", &err);
    CL_CHECK(err);
}

void NoirLedgerGPU_CL::create_lookup_buffer() {
    const unsigned char* table_ptr = parent_hasher.get_lookup_table();
    size_t table_size = parent_hasher.get_lookup_table_size();
    
    std::cout << "Copiando " << table_size / (1024.0 * 1024.0) << " MB da tabela de consulta para a VRAM da GPU..." << std::endl;
    cl_int err;
    lookup_table_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, table_size, (void*)table_ptr, &err);
    CL_CHECK(err);
    std::cout << "Cópia da tabela de consulta para a VRAM concluída." << std::endl;
}

void NoirLedgerGPU_CL::ensure_buffers_are_allocated(size_t num_hashes, size_t input_len) {
    size_t required_input_size = num_hashes * input_len;
    size_t required_output_size = num_hashes * NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES;

    if (input_buffer == nullptr || input_buffer_size < required_input_size) {
        if (input_buffer) clReleaseMemObject(input_buffer);
        cl_int err;
        input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, required_input_size, nullptr, &err);
        CL_CHECK(err);
        input_buffer_size = required_input_size;
    }

    if (output_buffer == nullptr || output_buffer_size < required_output_size) {
        if (output_buffer) clReleaseMemObject(output_buffer);
        cl_int err;
        output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, required_output_size, nullptr, &err);
        CL_CHECK(err);
        output_buffer_size = required_output_size;
    }
}

void NoirLedgerGPU_CL::hash(const unsigned char* input, size_t input_len, unsigned char* output, size_t num_hashes) {
    if (!initialized) {
        throw std::runtime_error("A classe NoirLedgerGPU_CL não foi inicializada corretamente.");
    }
    if (num_hashes == 0) return;

    // 1. Garantir que os buffers estejam alocados com tamanho suficiente
    ensure_buffers_are_allocated(num_hashes, input_len);

    // 2. Copiar dados de entrada para a GPU
    CL_CHECK(clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, num_hashes * input_len, input, 0, nullptr, nullptr));

    // 3. Definir argumentos do kernel
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_uint), &input_len));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &lookup_table_buffer));

    // 4. Enfileirar a execução do kernel
    size_t global_work_size = num_hashes;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr));

    // 5. Copiar resultados de volta da GPU
    CL_CHECK(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, num_hashes * NoirLedgerConstants::HASH_OUTPUT_SIZE_BYTES, output, 0, nullptr, nullptr));

    // Espera a conclusão de todas as operações na fila
    clFinish(queue);
}

void NoirLedgerGPU_CL::release_resources() {
    if (queue) clFinish(queue); // Garante que tudo terminou antes de liberar
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (input_buffer) clReleaseMemObject(input_buffer);
    if (output_buffer) clReleaseMemObject(output_buffer);
    if (lookup_table_buffer) clReleaseMemObject(lookup_table_buffer);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
    
    initialized = false;
    // Zera os ponteiros para evitar double-free
    kernel = nullptr;
    program = nullptr;
    input_buffer = nullptr;
    output_buffer = nullptr;
    lookup_table_buffer = nullptr;
    queue = nullptr;
    context = nullptr;
}

#endif // NOXIUM_ENABLE_OPENCL