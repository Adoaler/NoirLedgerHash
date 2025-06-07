# Script CMake para converter .cl em .h
file(READ ${KERNEL_FILE} KERNEL_CONTENT)
set(HEADER_CONTENT "/* Arquivo gerado automaticamente. NÃO EDITE. */\n")
string(APPEND HEADER_CONTENT "namespace NoirLedgerGPU {\n")
string(APPEND HEADER_CONTENT "const char* ocl_source = R\"RAW(\n")
string(APPEND HEADER_CONTENT "${KERNEL_CONTENT}")
string(APPEND HEADER_CONTENT "\n)RAW\";\n")
string(APPEND HEADER_CONTENT "}\n")
file(WRITE ${HEADER_FILE} "${HEADER_CONTENT}")