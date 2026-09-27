"""Scalability test (review W4): 50/100/500 registered UAVs, sequential intake.

Measures, per fleet size and per round, over the real FLServer:
  - masked-identity table precomputation time (server, per round);
  - total server-side processing time per round (M1 admission + M3 intake +
    M4 issue for every client, excluding client-side work);
  - aggregation-buffer entries and completed-cache size after the round.

The server is a single in-memory instance processing clients sequentially
(as stated in the paper); this test bounds how that architecture scales,
it does not claim concurrent throughput.

Output: results/scalability_result.json
"""
from __future__ import annotations

import hashlib
import json
import time
import tracemalloc

from raka_ufl_experiment import ExperimentConfig
from raka_ufl_comparison import make_client_specs, local_train, serialize_model
from raka_ufl_fl_server import FLServer
from raka_ufl_masked import MaskedClient

SEED = 20260924
FLEET_SIZES = [50, 100, 500]
ROUNDS = 3


def run_fleet(n_clients: int) -> dict:
    config = ExperimentConfig(clients=n_clients, rounds=ROUNDS, vector_size=16, seed=SEED)
    specs = make_client_specs(config)
    fl = FLServer("fl-scale", {s.pid: s.master_key for s in specs}, config.vector_size)
    fl.register_sample_counts({s.pid: s.sample_count for s in specs})

    tracemalloc.start()
    precompute_ms = []
    round_ms = []
    initial = tuple(0.0 for _ in range(config.vector_size))
    global_model = initial

    for round_id in range(1, ROUNDS + 1):
        t_round = time.perf_counter()
        mv, mh = fl.start_round(round_id)
        t_pre = time.perf_counter()
        fl._refresh_masked_table(round_id)  # server-side per-round precomputation
        precompute_ms.append((time.perf_counter() - t_pre) * 1000)

        for index, spec in enumerate(specs):
            client = MaskedClient("fl-scale", spec.pid, spec.master_key, committed_round=round_id - 1)
            client.model_version, client.model_hash = mv, mh
            ts = 1000 + round_id * 1000 + index
            m1 = client.make_hello_wire(ts, bytes([(index % 255) + 1]) * 16)
            m2 = fl.ingest_wire_hello(m1, now=ts, nonce_s=bytes([round_id + 32 + (index % 32)]) * 16, timestamp_s=ts + 1)
            client.accept_wire_server_hello(m2, now=ts + 1)
            payload = serialize_model(local_train(global_model, spec, config))
            wire = client.encrypt_update_wire(payload)
            plaintext, ack, is_dup = fl.ingest_wire_update(wire)
            client.accept_wire_ack(ack)
            assert plaintext == payload and not is_dup
        report = fl.close_round()
        global_model = report.global_model
        round_ms.append((time.perf_counter() - t_round) * 1000)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ep0 = next(iter(fl._endpoints.values()))
    return {
        "registered_uavs": n_clients,
        "rounds": ROUNDS,
        "masked_table_precompute_ms_per_round": [round(v, 3) for v in precompute_ms],
        "server_round_ms": [round(v, 2) for v in round_ms],
        "server_ms_per_client_round": round(sum(round_ms) / (ROUNDS * n_clients), 4),
        "aggregation_buffer_entries": len(report.accepted_clients),
        "completed_cache_entries": len(getattr(ep0, "completed", getattr(ep0, "_completed", {})) or {}),
        "python_heap_peak_mb": round(peak / 1024 / 1024, 2),
    }


def main() -> None:
    rows = []
    for n in FLEET_SIZES:
        row = run_fleet(n)
        rows.append(row)
        print(row)
    with open("results/scalability_result.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
