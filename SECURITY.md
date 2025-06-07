

# NoirLedger Security Policy

## Current Status

The NoirLedger project and its Proof-of-Work algorithm, **NoirLedgerHash**, are currently in an **experimental and actively developing phase**.

**DO NOT USE THIS SOFTWARE IN A PRODUCTION ENVIRONMENT.**

The implementations of cryptographic primitives (AES, ChaCha20, Blake3) were developed to support the creation of a unique PoW algorithm. These **have not undergone formal security audits** and may contain vulnerabilities or subtle bugs.

## Reporting a Vulnerability

Project security is of utmost importance. If you believe you have discovered a security vulnerability in NoirLedgerHash or any other component of the project, we encourage you to report it responsibly.

**Please do not disclose the vulnerability publicly** until we have had a chance to review and address it.

To report a vulnerability, please contact the project maintainers directly. (Contact methods to be defined, such as a dedicated security email.)

We deeply appreciate your contributions in helping to make NoirLedger a secure and robust project.

## Security Scope

This security policy applies to:

* The NoirLedgerHash algorithm (`src/NoirLedger_hash/`).
* Implementations of cryptographic primitives (`src/blake3_custom/`, etc.).
* Benchmark and API logic (`src/main.cpp`, `src/gpu/`).

## Audit Process

We acknowledge that a formal security audit by independent third parties is a crucial step before NoirLedger can be considered for a mainnet launch. We are committed to pursuing a full audit once the algorithm and codebase reach a stable and feature-complete state.

The community is encouraged to review the source code and algorithm logic. All contributions in this regard are welcome.
