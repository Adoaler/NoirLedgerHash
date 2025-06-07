cmake_minimum_required(VERSION 3.10)
project(NoirLedgerHasher LANGUAGES CXX CUDA)
# Define a política do CMake para arquiteturas CUDA para silenciar o aviso CMP0104
cmake_policy(SET CMP0104 NEW)
# Define as arquiteturas de computação CUDA. Ex: 70 para Volta, 75 para Turing, 86 para Ampere.
# Usar 70 como base garante compatibilidade com a maioria das GPUs modernas.
set(CMAKE_CUDA_ARCHITECTURES 70)
enable_testing() # Habilita CTest

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Adiciona um executável
add_executable(NoirLedger_profiler
    src/main.cpp
    src/NoirLedger_hash/NoirLedger_hash.cpp
    src/NoirLedger_hash/debug_flags.cpp # Novo arquivo de flags de depuração
    src/blake3_custom/blake3_core.cpp
    src/blake3_custom/blake3_compress.cpp
    src/gpu/NoirLedger_gpu_cl.cpp
    src/gpu/NoirLedger_gpu_cuda.cpp
)

# Adiciona um executável para os testes unitários
add_executable(NoirLedger_unit_tests
    src/tests/unit_tests.cpp
    src/NoirLedger_hash/NoirLedger_hash.cpp
    src/NoirLedger_hash/debug_flags.cpp
    src/blake3_custom/blake3_core.cpp
    src/blake3_custom/blake3_compress.cpp
    src/gpu/NoirLedger_gpu_cl.cpp
    src/gpu/NoirLedger_gpu_cuda.cpp
)

# Vincula as mesmas flags de compilação e diretórios de inclusão
target_compile_options(NoirLedger_unit_tests PRIVATE ${NOXIUM_ARCH_FLAGS})
target_include_directories(NoirLedger_unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Adiciona o teste ao CTest
add_test(NAME NoirLedgerHashBasicTest COMMAND NoirLedger_unit_tests)

# Flags de compilação baseadas no CMAKE_BUILD_TYPE
# Flags comuns para Release e Debug (específicas de arquitetura)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(NOXIUM_ARCH_FLAGS -march=native -maes -mavx2) # Sem aspas, para ser uma lista
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(NOXIUM_ARCH_FLAGS /arch:AVX2) # Sem aspas (MSVC trata flags de forma diferente)
endif()

# Flags de otimização e debug
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    # Adicionar flags de arquitetura separadamente para garantir que sejam tratadas como flags individuais
    target_compile_options(NoirLedger_profiler PRIVATE ${NOXIUM_ARCH_FLAGS})
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
    # Para MSVC, /arch:AVX2 é geralmente adicionado via target_compile_options ou CMAKE_CXX_FLAGS
    # Vamos adicioná-lo explicitamente se não estiver já em CMAKE_CXX_FLAGS_<CONFIG>
    # No entanto, a forma mais robusta é usar target_compile_options para flags de arquitetura
    target_compile_options(NoirLedger_profiler PRIVATE ${NOXIUM_ARCH_FLAGS})
endif()

# A linha target_compile_options(NoirLedger_profiler PRIVATE ${NOXIUM_ARCH_FLAGS}) foi movida para dentro dos condicionais
# para melhor controle e para evitar duplicação se CMAKE_CXX_FLAGS_<CONFIG> já as incluísse de alguma forma.
# A forma mais limpa é definir CMAKE_CXX_FLAGS_<CONFIG> apenas com otimização/debug
# e usar target_compile_options para flags de arquitetura e definições.

# Definição de pré-processador para modo Debug
target_compile_definitions(NoirLedger_profiler
    PRIVATE $<$<CONFIG:Debug>:NOXIUM_DEBUG_MODE>
)

# Inclui diretórios para headers
target_include_directories(NoirLedger_profiler PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

# --- Suporte a OpenCL ---
find_package(OpenCL)
if (OpenCL_FOUND)
    message(STATUS "OpenCL encontrado. Habilitando suporte a OpenCL.")

    # --- Geração do Header do Kernel OpenCL ---
    set(OPENCL_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/NoirLedger_hash/NoirLedger_hash.cl")
    set(OPENCL_KERNEL_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/NoirLedger_hash_cl.h")

    # Cria o diretório de saída para o cabeçalho gerado, se não existir
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")

    # Lê o conteúdo do kernel OpenCL durante a configuração do CMake
    file(READ ${OPENCL_KERNEL_SOURCE} KERNEL_CONTENT)
    
    # Cria o arquivo de cabeçalho durante a configuração do CMake
    file(WRITE ${OPENCL_KERNEL_HEADER} "/* Arquivo gerado automaticamente. NÃO EDITE. */\n")
    file(APPEND ${OPENCL_KERNEL_HEADER} "namespace NoirLedgerGPU {\n")
    file(APPEND ${OPENCL_KERNEL_HEADER} "const char* ocl_source = R\"RAW(\n")
    file(APPEND ${OPENCL_KERNEL_HEADER} "${KERNEL_CONTENT}")
    file(APPEND ${OPENCL_KERNEL_HEADER} "\n)RAW\";\n")
    file(APPEND ${OPENCL_KERNEL_HEADER} "}\n")

    # Aplica as configurações de OpenCL para todos os alvos relevantes
    set(OPENCL_TARGETS NoirLedger_profiler NoirLedger_unit_tests)
    foreach(TARGET ${OPENCL_TARGETS})
        target_include_directories(${TARGET} PRIVATE
            ${OpenCL_INCLUDE_DIRS}
            "${CMAKE_CURRENT_BINARY_DIR}/generated"
        )
        target_link_libraries(${TARGET} PRIVATE OpenCL::OpenCL)
        target_compile_definitions(${TARGET} PRIVATE NOXIUM_ENABLE_OPENCL)
    endforeach()

else()
    message(STATUS "OpenCL NÃO encontrado. O suporte a OpenCL será desabilitado.")
endif()

# --- Suporte a CUDA ---
# Define a política para CMAKE_CUDA_ARCHITECTURES para silenciar o aviso CMP0104
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

find_package(CUDA)
if (CUDA_FOUND)
    message(STATUS "CUDA encontrado. Habilitando suporte a CUDA.")
    # A linguagem CUDA já foi habilitada na chamada project()

    # --- Biblioteca Estática para o Kernel CUDA ---
    # Isso garante que o código C++ e CUDA sejam vinculados corretamente.
    add_library(NoirLedger_cuda_kernel STATIC
        src/NoirLedger_hash/NoirLedger_hash.cu
        src/gpu/NoirLedger_gpu_cuda_wrapper.cu
    )
    set_target_properties(NoirLedger_cuda_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    set(CUDA_TARGETS NoirLedger_profiler NoirLedger_unit_tests)
    foreach(TARGET ${CUDA_TARGETS})
        # Inclui os diretórios do CUDA para cada alvo
        target_include_directories(${TARGET} PRIVATE ${CUDA_TOOLKIT_INCLUDE})
        # Vincula a biblioteca do kernel e as bibliotecas principais do CUDA
        # Usando ${CUDA_LIBRARIES} para maior compatibilidade em vez de CUDA::cudart
        # Vincula a biblioteca do kernel e a biblioteca runtime do CUDA diretamente pelo nome ('cudart')
        # para máxima compatibilidade com diferentes ambientes de linker.
        target_link_libraries(${TARGET} PRIVATE NoirLedger_cuda_kernel cudart)
        target_compile_definitions(${TARGET} PRIVATE NOXIUM_ENABLE_CUDA)
    endforeach()
else()
    message(STATUS "CUDA NÃO encontrado. O suporte a CUDA será desabilitado.")
endif()

# Mensagem para o usuário sobre as flags
message(STATUS "Compilador ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "Flags CXX base para Release (antes de arch): ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Flags CXX base para Debug (antes de arch): ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Flags de arquitetura a serem aplicadas via target_compile_options: ${NOXIUM_ARCH_FLAGS}")

# Para garantir que __AVX2__ seja definido quando -mavx2 ou /arch:AVX2 é usado.
# Isso geralmente é automático, mas podemos forçar se necessário.
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     target_compile_definitions(NoirLedger_profiler PRIVATE __AVX2__)
# elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # Para MSVC, __AVX2__ é definido automaticamente com /arch:AVX2
# endif()