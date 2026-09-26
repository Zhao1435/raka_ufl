"""T2 generalized to a lossy link (review W2/W3): exact enumeration.

Instead of one scripted lost-receipt event, each client's acceptance
receipt is lost independently with probability p (a Bernoulli lossy
wireless link). Schemes without idempotent intake semantics re-submit
and double-count every lost-receipt update; RAKA-UFL (B3) returns the
cached receipt and counts each update exactly once.

With four clients there are only 2^4 = 16 loss patterns, so the
expectations below are EXACT (probability-weighted enumeration), not
Monte Carlo samples:

  E[duplicates per round] = 4p
  P(at least one double-counted update per round) = 1-(1-p)^4
  E[L2 distance of the FedAvg aggregate vs the exactly-once reference]

Output: results/lossy_t2_result.json
"""
from __future__ import annotations

import itertools
import json

from raka_ufl_comparison import make_client_specs, local_train, serialize_model, l2_distance
from raka_ufl_experiment import ExperimentConfig
from raka_ufl_fl_server import fedavg as _fedavg


def wire_value(model):
    import json as _json
    return tuple(_json.loads(serialize_model(model).decode("ascii")))


def main() -> None:
    config = ExperimentConfig(clients=4, rounds=3, vector_size=16, seed=20260924)
    specs = make_client_specs(config)
    initial = tuple(0.0 for _ in range(config.vector_size))
    trained = [(s.sample_count, wire_value(local_train(initial, s, config))) for s in specs]
    reference = _fedavg(trained)

    out = {}
    for p in (0.05, 0.10):
        exp_l2 = 0.0
        exp_dups = 0.0
        p_any_dup = 0.0
        for pattern in itertools.product([False, True], repeat=len(specs)):
            prob = 1.0
            for lost in pattern:
                prob *= p if lost else (1 - p)
            dups = sum(pattern)
            updates = []
            for (n, model), lost in zip(trained, pattern):
                updates.append((n, model))
                if lost:
                    updates.append((n, model))  # receipt lost -> retransmission counted twice
            contaminated = _fedavg(updates)
            exp_l2 += prob * l2_distance(contaminated, reference)
            exp_dups += prob * dups
            if dups:
                p_any_dup += prob
        out[f"p={p}"] = {
            "receipt_loss_probability": p,
            "clients": len(specs),
            "expected_double_counted_updates_per_round": exp_dups,
            "prob_at_least_one_double_count": p_any_dup,
            "baseline_expected_l2_vs_reference": exp_l2,
            "raka_ufl_expected_l2_vs_reference": 0.0,
        }
        print(out[f"p={p}"])
    with open("results/lossy_t2_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
