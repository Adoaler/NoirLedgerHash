# Política de Segurança do NoirLedger

## Estado Atual

O projeto NoirLedger e seu algoritmo de Proof-of-Work, NoirLedgerHash, estão atualmente em fase **experimental e de desenvolvimento ativo**.

**NÃO USE ESTE SOFTWARE EM UM AMBIENTE DE PRODUÇÃO.**

As implementações das primitivas criptográficas (AES, ChaCha20, Blake3) foram desenvolvidas para a criação de um algoritmo de PoW único. Elas **não passaram por uma auditoria de segurança formal** e podem conter vulnerabilidades ou bugs sutis.

## Relatando uma Vulnerabilidade

A segurança do projeto é de extrema importância. Se você acredita ter encontrado uma vulnerabilidade de segurança no NoirLedgerHash ou em qualquer outro componente do projeto, nós o encorajamos a relatá-la de forma responsável.

**Por favor, não divulgue a vulnerabilidade publicamente** até que tenhamos tido a oportunidade de analisá-la e corrigi-la.

Para relatar uma vulnerabilidade, por favor, entre em contato diretamente com os mantenedores do projeto. (Métodos de contato a serem definidos, como um e-mail de segurança dedicado).

Agradecemos imensamente suas contribuições para ajudar a tornar o NoirLedger um projeto seguro e robusto.

## Escopo da Segurança

O escopo desta política de segurança cobre:

*   O algoritmo NoirLedgerHash (`src/NoirLedger_hash/`).
*   As implementações de primitivas criptográficas (`src/blake3_custom/`, etc.).
*   A lógica de benchmark e da API (`src/main.cpp`, `src/gpu/`).

## Processo de Auditoria

Reconhecemos que uma auditoria de segurança formal por terceiros independentes é um passo crucial antes que o NoirLedger possa ser considerado para uma rede principal (mainnet). Estamos comprometidos em buscar uma auditoria completa assim que o algoritmo e o código base atingirem um estado estável e de recurso completo.

A comunidade é incentivada a revisar o código-fonte e a lógica do algoritmo. Toda contribuição nesse sentido é bem-vinda.