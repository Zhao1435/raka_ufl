# RAKA-UFL: Round-Bound Session Protection for Multi-UAV Federated Learning

Reference implementation, executable baselines, and ProVerif models for the
paper *"Round-Bound Session Protection for Multi-UAV Federated Learning: A
Minimal Authenticated Protocol and Executable Evaluation"*.

The protocol authenticates one UAV and one FL server per training round (M1/M2),
derives a round-specific session key from a pre-shared master key and a
keyed two-nonce transcript, protects the model update with AES-GCM (M3), and
confirms server acceptance with a ciphertext-hash-bound receipt (M4). Devices
are identified on the wire by **round-derived masked pseudonyms**
(`masked_id = SHA256(real_id) XOR HMAC(K_i, task||r||pidmask)`), resolved by
the server through a per-round precomputed lookup table. The server accepts
only the next expected round, handles exact duplicates idempotently within a
bounded retry budget, and rejects stale rounds, mismatched model contexts,
revoked devices, oversized updates, and excessive hello rates.

## Layout

| File | Role |
|---|---|
| `raka_ufl_minimal.py` | Core M1–M4 client/server protocol state machines + masked-identity primitives |
| `raka_ufl_masked.py` | Standard wire form: masked-identity message types and `MaskedClient` |
| `raka_ufl_fl_server.py` | Many-to-one FL server: registry, masked resolution, per-identity endpoints, aggregation buffer, FedAvg |
| `raka_ufl_minimal_privacy.py` | Compatibility shim (original masked-identity reference names; one-to-one demo front-end) |
| `raka_ufl_experiment.py` | Fixed-vector multi-UAV functional harness over the masked wire form |
| `raka_ufl_baselines.py` | Baselines B0 (plain), B1 (TLS-1.3-PSK-equivalent channel), B2 (composite AKA) |
| `raka_ufl_pmap_port.py` | Faithful Python port of the official PMAP-D2Z Java implementation (B2p) |
| `raka_ufl_comparison.py` | Unified comparison harness: capability matrix, overhead, T1/T2 attack traces |
| `benchmark_timing.py` | Repeated wall-clock timing per client round for all five schemes (50 reps after warm-up) |
| `benchmark_tls_psk.py` | Real TLS 1.3 PSK handshake measurement via OpenSSL s_server/s_client behind a byte-counting TCP proxy |
| `lossy_retransmission.py` | T2 generalized to a Bernoulli lossy link: exact enumeration of loss patterns |
| `proverif/` | Symbolic models: masked-identity model (9 queries), master-key leak variant |
| `results/comparison_result.json` | Recorded comparison output (seed 20260924) |
| `results/timing_result.json`, `results/tls_psk_result.json`, `results/lossy_t2_result.json` | Recorded benchmark outputs |
| `test_raka_ufl_minimal.py`, `test_raka_ufl_experiment.py` | Unit tests (19 in this repo's scope) |
| `test_dynamic_pseudonym.py` | Masked-identity validation script (7 checks) |

## Reproduce

Requirements: Python ≥ 3.11, `cryptography` (see `requirements.txt`);
ProVerif 2.05 for the symbolic models.

```bash
# functional harness (default: 4 UAVs, 3 rounds, 16-dim vectors)
python raka_ufl_experiment.py

# demonstration run: 8 clients, 5 rounds, client 2 revoked
python raka_ufl_experiment.py --clients 8 --rounds 5 --vector-size 16 \
    --revoked-client 2 --seed 20260924

# five-scheme comparison (B0/B1/B2/B2p/B3) with T1/T2 attack traces
python raka_ufl_comparison.py

# repeated per-client-round timing (50 repetitions, 3 warm-up runs discarded)
python benchmark_timing.py

# real TLS 1.3 PSK handshake bytes/timing (requires openssl on PATH)
python benchmark_tls_psk.py

# lossy-link generalization of T2 (exact enumeration over loss patterns)
python lossy_retransmission.py

# unit tests and masked-identity validation
python -m unittest
python test_dynamic_pseudonym.py

# symbolic verification (ProVerif 2.05)
proverif proverif/raka_ufl_minimal_privacy.pv
proverif proverif/raka_ufl_masked_masterkey_leak.pv   # expected: secrecy queries fail
```

## Scope notes (as stated in the paper)

- The evaluation uses fixed-length floating-point vectors, one in-memory
  server, no real network and no embedded UAV platform; it demonstrates
  protocol data-path functionality and bounded admission behavior, not FL
  model accuracy, scalability, or production security.
- The construction does not provide forward secrecy (confirmed by the
  master-key leak variant), secure aggregation, or protection against
  malicious enrolled clients. Within one round the masked pseudonym is
  deterministic by design (one transaction per device per round).
- Baselines B1/B2 are protocol-semantics-level abstractions; B2p is a port of
  PMAP's official code with an appended AES-GCM upload segment that PMAP
  itself does not contain.
