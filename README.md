# NoirLedgerHash: Algoritmo de Proof-of-Work Otimizado para CPU/GPU

## Objetivo do Projeto

O NoirLedgerHash é um algoritmo de Proof-of-Work (PoW) projetado para ser "ultra seguro" e "ultra otimizado" para mineração em CPUs e GPUs. Ele visa superar algoritmos existentes como o RandomX do Monero em termos de resistência a ASICs e eficiência, combinando múltiplas primitivas criptográficas e técnicas de uso intensivo de memória e ponto flutuante.

## Estrutura do Algoritmo (5 Estágios)

1.  **Expansão de Semente (Blake3-based):** Utiliza o Blake3 para expandir uma semente inicial, gerando dados de entrada para os estágios subsequentes.
2.  **Kernels de Criptografia Paralelos (AES-256):** Aplica múltiplos kernels de criptografia AES-256 em paralelo, aproveitando as instruções AES-NI quando disponíveis.
3.  **Operações de Ponto Flutuante:** Realiza uma série de operações complexas de ponto flutuante para introduzir dependências de dados e dificultar a otimização em hardware especializado.
4.  **Acesso Aleatório à Memória (Memory-Hard com ChaCha20):** Incorpora acessos aleatórios a uma grande tabela de consulta em memória, misturando os dados com ChaCha20 para garantir resistência a ASICs e GPUs com pouca memória.
5.  **Hash Final (Blake3):** O resultado dos estágios anteriores é processado por um hash final Blake3 para produzir a saída final do algoritmo.

## Como Instalar e Compilar

O projeto utiliza CMake para o sistema de build.

### Dependências do Compilador

*   Um compilador C++17 (GCC, Clang, MSVC).
*   Para suporte a otimizações SIMD (AES-NI, AVX2), o compilador deve suportá-las. As flags `-maes` e `-mavx2` (GCC/Clang) ou `/arch:AVX2` (MSVC) são usadas automaticamente pelo CMake se o compilador for compatível.
*   **Opcional:** Bibliotecas OpenCL e CUDA para suporte a GPU. O CMake tentará encontrá-las automaticamente.

### Passos de Compilação

1.  **Crie um diretório de build:**
    ```bash
    mkdir build
    cd build
    ```
2.  **Configure o projeto com CMake:**
    *   Para uma build de **Release** (otimizada):
        ```bash
        cmake -DCMAKE_BUILD_TYPE=Release ..
        ```
    *   Para uma build de **Debug** (com símbolos de depuração):
        ```bash
        cmake -DCMAKE_BUILD_TYPE=Debug ..
        ```
    *   O CMake detectará automaticamente o suporte a OpenCL e CUDA.

3.  **Compile o projeto:**
    ```bash
    cmake --build .
    # Ou, em sistemas baseados em Unix/Linux:
    # make
    ```

Após a compilação, o executável `NoirLedger_profiler` estará disponível no diretório `build/`.

## Como Rodar Testes de Benchmark

O executável `NoirLedger_profiler` pode ser usado para benchmarking e depuração.

```bash
./build/NoirLedger_profiler [OPÇÕES]
```

### Opções de Linha de Comando:

*   `--mode <cpu|gpu>`: Seleciona o modo de benchmark (padrão: `cpu`).
*   `-t <num_threads>`, `--threads <num_threads>`: Número de threads da CPU a serem usadas.
*   `-i <num_iterations>`, `--iterations <num_iterations>`: Total de hashes a serem calculados.
*   `--gpu-platform <id>`: ID da plataforma OpenCL a ser usada (padrão: 0).
*   `--gpu-device <id>`: ID do dispositivo OpenCL a ser usado (padrão: 0).
*   `--diagnose-gpu`: Exibe informações sobre as plataformas e dispositivos OpenCL disponíveis e sai.
*   `--debug-...`: Ativa prints de depuração para estágios específicos (`seed`, `aes`, `fp`, `mem`, `blake3`).
*   `-h`, `--help`: Exibe a mensagem de ajuda.

**Exemplos:**

*   Rodar benchmark de CPU com 8 threads:
    ```bash
    ./build/NoirLedger_profiler --mode cpu -t 8
    ```
*   Rodar benchmark de GPU com 1.000.000 de iterações:
    ```bash
    ./build/NoirLedger_profiler --mode gpu -i 1000000
    ```
*   Diagnosticar a configuração do OpenCL:
    ```bash
    ./build/NoirLedger_profiler --diagnose-gpu
    ```

## Descrição de Diretórios e Flags de Debug

*   `src/`: Contém o código-fonte principal.
    *   `src/NoirLedger_hash/`: Implementação do algoritmo NoirLedgerHash e suas primitivas.
    *   `src/blake3_custom/`: Implementação customizada do Blake3.
    *   `src/main.cpp`: Ponto de entrada para o executável `NoirLedger_profiler`, contendo a lógica de benchmark e processamento de argumentos.
    *   `src/NoirLedger_hash/debug_flags.h` / `src/NoirLedger_hash/debug_flags.cpp`: Contêm o sistema consolidado de flags de depuração.

As flags de depuração são controladas via linha de comando (veja "Como Rodar Testes de Benchmark"). Internamente, elas usam um sistema de bitmask para ativar/desativar prints específicos em diferentes estágios do algoritmo.

## Como Obter um Hash Final / Incorporar a Biblioteca

A biblioteca NoirLedgerHash pode ser incorporada em outros projetos C++.

### Exemplo de Uso Básico (C++)

```cpp
#include "NoirLedger_hash/NoirLedger_hash.h"
#include <iostream>
#include <vector>
#include <string>
#include <array> // Para std::array

int main() {
    // 1. Crie uma instância do NoirLedgerHasher
    // O construtor inicializa a tabela de consulta em memória (256MB).
    // Isso pode levar alguns segundos na primeira vez.
    NoirLedgerHasher hasher;

    // 2. Defina os dados de entrada
    std::string message = "Esta é a minha mensagem secreta para ser hashed pelo NoirLedgerHash!";
    std::vector<uint8_t> input_data(message.begin(), message.end());

    // 3. Calcule o hash
    // Use o operador de chamada da classe para calcular o hash.
    // Ele retorna um std::array<uint8_t, NOXIUM_HASH_OUTPUT_SIZE_BYTES>.
    std::array<uint8_t, NOXIUM_HASH_OUTPUT_SIZE_BYTES> hash_result = hasher(input_data);

    // Ou, para uma string diretamente:
    // std::array<uint8_t, NOXIUM_HASH_OUTPUT_SIZE_BYTES> hash_result_str = hasher("Outra mensagem!");

    // 4. Imprima o hash em hexadecimal
    std::cout << "Hash da mensagem: ";
    std::cout << std::hex << std::setfill('0');
    for (size_t i = 0; i < NOXIUM_HASH_OUTPUT_SIZE_BYTES; ++i) {
        std::cout << std::setw(2) << static_cast<int>(hash_result[i]);
    }
    std::cout << std::dec << std::endl; // Reset para decimal

    // O destrutor de 'hasher' (NoirLedgerHasher) chamará free_NoirLedger_globals() automaticamente
    // quando 'hasher' sair do escopo.

    return 0;
}
```

### Compilando um Projeto que Usa NoirLedgerHash

Para usar o `NoirLedgerHasher` em seu próprio projeto, você precisará:

1.  Incluir o diretório `src/` do NoirLedgerHash em seus `target_include_directories`.
2.  Vincular seu executável ou biblioteca com as fontes do NoirLedgerHash (ou com uma biblioteca estática/dinâmica se você a construir).

Exemplo de `CMakeLists.txt` para um projeto externo:

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyNoirLedgerApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(my_app main.cpp)

# Inclua os diretórios de cabeçalho do NoirLedgerHash
target_include_directories(my_app PRIVATE
    /caminho/para/NoirLedgerHash/src # Ajuste este caminho
)

# Adicione as fontes do NoirLedgerHash ao seu projeto
# Ou, se você construir o NoirLedgerHash como uma biblioteca, vincule-a aqui.
target_sources(my_app PRIVATE
    /caminho/para/NoirLedgerHash/src/NoirLedger_hash/NoirLedger_hash.cpp
    /caminho/para/NoirLedgerHash/src/NoirLedger_hash/debug_flags.cpp
    /caminho/para/NoirLedgerHash/src/blake3_custom/blake3_core.cpp
    /caminho/para/NoirLedgerHash/src/blake3_custom/blake3_compress.cpp
    # Adicione quaisquer outros arquivos .cpp necessários do NoirLedgerHash
)

# Vincule as bibliotecas necessárias (OpenCL, CUDA, etc., se usadas)
find_package(OpenCL QUIET)
if (OpenCL_FOUND)
    target_link_libraries(my_app PRIVATE OpenCL::OpenCL)
    target_compile_definitions(my_app PRIVATE NOXIUM_ENABLE_OPENCL)
endif()

find_package(CUDA QUIET)
if (CUDA_FOUND)
    target_link_libraries(my_app PRIVATE ${CUDA_LIBRARIES})
    target_compile_definitions(my_app PRIVATE NOXIUM_ENABLE_CUDA)
    enable_language(CUDA)
endif()

# Adicione flags de arquitetura se necessário (ex: -maes -mavx2)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(my_app PRIVATE -march=native -maes -mavx2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(my_app PRIVATE /arch:AVX2)
endif()
```

## Otimizações Adicionais e Benchmarks

Para uma análise de desempenho mais aprofundada, você pode:

*   **Medir Throughput:** Modificar `main.cpp` para medir o throughput (bytes/s) para diferentes tamanhos de entrada (1KB, 1MB, 100MB).
*   **Comparar Versões:** Comparar o desempenho das versões C-pura vs. SIMD (AES-NI, AVX2) para cada estágio.
*   **Análise de Memória:** Monitorar o uso de memória, especialmente da `g_lookup_table` de 256MB.
*   **Scripts de Benchmark:** Escrever scripts externos (Python, etc.) para automatizar a execução de benchmarks e gerar gráficos.

## Cobertura de Testes Automatizados

Atualmente, os testes são manuais via `main.cpp`. Para um projeto de produção, é altamente recomendável integrar uma suíte de testes unitários (ex: Google Test, Catch2) para:

*   Validar que `NoirLedger_hash("abc")` produza um valor de hash esperado conhecido.
*   Verificar a correção das funções de compressão BLAKE3 (comparando com implementações oficiais).
*   Garantir a consistência dos resultados entre diferentes plataformas e configurações SIMD.

## Portabilidade (Considerações)

*   **Endianness:** O código assume arquitetura little-endian (via `load_u32_le`, `u8to32_le`, `u32to8_le`). Para portabilidade completa para sistemas big-endian, seriam necessárias adaptações ou verificações em tempo de compilação.
*   **SIMD Fallbacks:** As implementações SIMD (AES-NI, AVX2) incluem fallbacks C-pura, garantindo que o código compile e execute mesmo em CPUs sem suporte a essas extensões. No entanto, o desempenho será significativamente menor.

---

## Licença

Este projeto é licenciado sob a Licença MIT. Veja o arquivo `LICENSE` na raiz do projeto para mais detalhes.

---

## Considerações de Segurança e Auditoria

**AVISO IMPORTANTE:** Este projeto está em fase experimental. O algoritmo NoirLedgerHash e suas implementações **não passaram por uma auditoria de segurança formal** e não devem ser usados em ambientes de produção.

A criação de um algoritmo de Proof-of-Work seguro é um desafio complexo que requer uma análise criptográfica rigorosa por especialistas independentes para se defender contra uma variedade de ataques teóricos, incluindo, mas não se limitando a:

*   **Ataques de Preimage (Preimage Attacks):** Encontrar uma entrada que produza um hash específico.
*   **Ataques de Colisão (Collision Attacks):** Encontrar duas entradas diferentes que produzam o mesmo hash.
*   **Ataques de Atalho (Shortcut Attacks):** Encontrar fraquezas em um dos estágios do algoritmo que permitam que um minerador resolva o quebra-cabeça com um esforço significativamente menor do que o esperado.

Convidamos a comunidade de segurança e criptografia a revisar o código-fonte e o design do NoirLedgerHash. Para relatar uma vulnerabilidade de forma responsável, consulte nossa [Política de Segurança](SECURITY.md).

---

## Solução de Problemas (Troubleshooting)

### Erro: `Nenhuma plataforma OpenCL encontrada` ou `Erro -1001`

Este é um dos erros mais comuns ao trabalhar com OpenCL e geralmente indica um problema de configuração do ambiente, não um bug no código. O erro `-1001` corresponde a `CL_PLATFORM_NOT_FOUND_KHR`.

Isso significa que o carregador OpenCL (`libOpenCL.so` no Linux) foi encontrado, mas ele não conseguiu localizar nenhum driver de plataforma (NVIDIA, AMD, Intel) instalado e registrado corretamente no sistema.

#### Passos para Diagnóstico e Correção:

1.  **Use a Ferramenta de Diagnóstico Interna:**
    Compile e execute o `NoirLedger_profiler` com a flag `--diagnose-gpu`.
    ```bash
    ./build/NoirLedger_profiler --diagnose-gpu
    ```
    *   Se ele mostrar "Nenhuma plataforma OpenCL encontrada", isso confirma o problema.
    *   Se ele listar plataformas e dispositivos, o problema pode ser outro (ex: permissões).

2.  **Verifique a Instalação dos Drivers da GPU:**
    *   **NVIDIA:** Certifique-se de que os drivers proprietários da NVIDIA estão instalados. O pacote `nvidia-driver` geralmente inclui o OpenCL.
    *   **AMD:** Instale os drivers `amdgpu-pro` ou os drivers open-source `mesa-opencl-icd`.
    *   **Intel:** Instale o pacote `intel-opencl-icd`.

3.  **Verifique o Registro do ICD (Installable Client Driver):**
    O OpenCL funciona procurando por arquivos `.icd` no diretório `/etc/OpenCL/vendors/`. Esses arquivos de texto simples apontam para a biblioteca `.so` do driver real.
    *   Verifique o conteúdo deste diretório:
        ```bash
        ls -l /etc/OpenCL/vendors/
        ```
    *   Verifique o conteúdo dos arquivos `.icd`. Por exemplo, para a NVIDIA, o arquivo `nvidia.icd` deve conter algo como:
        ```
        libnvidia-opencl.so.1
        ```
    *   Se o diretório ou os arquivos não existirem, a instalação do driver está incompleta. Reinstalar o pacote de drivers correto (ex: `sudo apt install nvidia-opencl-icd`) geralmente resolve isso.

4.  **Ambiente WSL (Subsistema Windows para Linux):**
    Usar OpenCL no WSL requer configuração adicional para permitir que o Linux acesse a GPU do host Windows.
    *   **Drivers do Windows:** Certifique-se de que você tem os drivers mais recentes da sua GPU instalados no Windows, com suporte para WSL.
    *   **Drivers do Linux no WSL:** Você pode precisar instalar drivers específicos para WSL dentro do seu ambiente Linux. Consulte a documentação da Microsoft e do fornecedor da sua GPU (NVIDIA/AMD/Intel) para obter as instruções mais recentes sobre computação em GPU no WSL.

5.  **Verifique com `clinfo`:**
    `clinfo` é uma ferramenta de linha de comando padrão para depurar o OpenCL.
    ```bash
    sudo apt install clinfo
    clinfo
    ```
    Se `clinfo` não conseguir encontrar nenhuma plataforma, o problema é definitivamente com a instalação do driver/runtime no nível do sistema operacional. Se `clinfo` funcionar mas o `NoirLedger_profiler` não, o problema pode ser de vinculação ou permissões.