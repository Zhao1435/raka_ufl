"""Repeated wall-clock timing for all five schemes (review W2).

Runs the comparison harness's run_normal() R times per scheme and reports
per-client-round wall-clock time (mean +/- std across repetitions), giving
a statistically stated local Python timing instead of the single-run
figure used before. Local timings are reference values only; they do not
establish embedded-UAV latency.

Output: results/timing_result.json
"""
from __future__ import annotations

import json
import statistics

from raka_ufl_comparison import run_normal
from raka_ufl_experiment import ExperimentConfig

SCHEMES = ["B0-plain", "B1-secure-channel", "B2-composite-aka", "B2p-pmap-port", "B3-raka-ufl"]
REPETITIONS = 50
WARMUP = 3


def main() -> None:
    config = ExperimentConfig(clients=4, rounds=3, vector_size=16, seed=20260924)
    rows = []
    for scheme in SCHEMES:
        for _ in range(WARMUP):  # discard warm-up runs (first-call allocation effects)
            run_normal(scheme, config)
        per_round_means = []
        for _ in range(REPETITIONS):
            out = run_normal(scheme, config)
            per_round_means.append(out["mean_client_round_ms"])
        rows.append({
            "scheme": scheme,
            "repetitions": REPETITIONS,
            "warmup_discarded": WARMUP,
            "client_rounds_per_run": config.clients * config.rounds,
            "mean_client_round_ms": statistics.fmean(per_round_means),
            "std_client_round_ms": statistics.stdev(per_round_means),
        })
        print(rows[-1])
    with open("results/timing_result.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()
