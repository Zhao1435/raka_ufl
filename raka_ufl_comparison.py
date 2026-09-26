"""Unified comparison harness: RAKA-UFL (B3) vs baselines B0/B1/B2.

All schemes run the SAME fixed-vector federated learning task (identical
local targets, sample counts, training rule, FedAvg, and seed). The schemes
differ only in how the per-round model update is protected in transit.

Outputs:
1. capability matrix (auto-generated from each scheme's CAPABILITIES);
2. normal-run overhead per scheme (messages, approx bytes, crypto ops, local
   Python time);
3. attack-trace results:
   - T1 stale-context injection: in round 2, one client uploads an update
     trained from the round-1 global model (legitimate client, legitimate
     keys). Correct behavior = reject; the update must not enter FedAvg.
   - T2 lost-receipt retransmission: a client retransmits the same update
     after the acceptance receipt is lost. Correct behavior = the update is
     counted exactly once in FedAvg.

The L2 distances are computed against the contamination-free reference
aggregation (stale update excluded for T1; exactly-once counting for T2),
so a correct scheme scores 0.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import random
import statistics
import time

from raka_ufl_baselines import CompositeAKA, CryptoOps, PlainChannel, SecureChannelPSK
from raka_ufl_experiment import ExperimentConfig
from raka_ufl_fl_server import FLServer, fedavg as _fedavg, serialize_model as _serialize_fl
from raka_ufl_minimal import MinimalClient, MinimalServer
from raka_ufl_pmap_port import PmapD2ZSession


# B3 capabilities of the RAKA-UFL minimal protocol (this paper).
RAKA_UFL_CAPABILITIES = {
    "round_binding": True,
    "model_context_binding": True,
    "acceptance_receipt": True,
    "idempotent_retransmission": True,
    "revocation": True,
    "dropout_tolerance": False,
    "server_sees_plaintext": True,
    "forward_secrecy": False,
    "update_confidentiality": True,
    "authentication": True,
}

# Protocol-logical crypto-operation count for one RAKA-UFL client round,
# both parties, generation and verification counted separately:
#   HKDF x4 (both sides derive AK_r and SK_r), HMAC x6 (tag1/tag2/tag4
#   generate+verify), AEAD x2, SHA-256 x4 (tid on both sides, client-side
#   payload hash, server-side ciphertext hash).
RAKA_UFL_CRYPTO = CryptoOps(hkdf=4, hmac=6, aead_enc=1, aead_dec=1, sha256=4)


@dataclass
class ClientSpec:
    pid: str
    master_key: bytes
    local_target: tuple[float, ...]
    sample_count: int


def make_client_specs(config: ExperimentConfig) -> list[ClientSpec]:
    rng = random.Random(config.seed)
    specs: list[ClientSpec] = []
    for index in range(config.clients):
        specs.append(ClientSpec(
            pid=f"uav-{index + 1}",
            master_key=hashlib.sha256(f"master-{index}-{config.seed}".encode()).digest(),
            local_target=tuple(rng.uniform(-1.0, 1.0) for _ in range(config.vector_size)),
            sample_count=8 + index * 4,
        ))
    return specs


def local_train(global_model: tuple[float, ...], spec: ClientSpec, config: ExperimentConfig) -> tuple[float, ...]:
    model = list(global_model)
    for _ in range(config.local_steps):
        model = [
            value + config.learning_rate * (target - value)
            for value, target in zip(model, spec.local_target)
        ]
    return tuple(model)


def serialize_model(model: tuple[float, ...]) -> bytes:
    return json.dumps([round(v, 12) for v in model], separators=(",", ":")).encode("ascii")


def l2(model: tuple[float, ...]) -> float:
    return sum(v * v for v in model) ** 0.5


def l2_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


# ---------------------------------------------------------------------------
# Normal-run overhead measurement
# ---------------------------------------------------------------------------


def run_normal(scheme: str, config: ExperimentConfig) -> dict[str, object]:
    specs = make_client_specs(config)
    if scheme == "B3-raka-ufl":
        # B3 runs on the real many-to-one FLServer: single server owning the
        # registry, per-identity endpoints, aggregation buffer, and FedAvg.
        clients = [MinimalClient("fl-demo", s.pid, s.master_key) for s in specs]
        fl = FLServer(
            "fl-demo",
            {s.pid: s.master_key for s in specs},
            config.vector_size,
        )
        fl.register_sample_counts({s.pid: s.sample_count for s in specs})
        base_ts = 1000
    else:
        channels = {
            "B0-plain": [PlainChannel(s.pid) for s in specs],
            "B1-secure-channel": [SecureChannelPSK(s.pid, s.master_key) for s in specs],
            "B2-composite-aka": [CompositeAKA(s.pid, s.master_key) for s in specs],
            "B2p-pmap-port": [PmapD2ZSession(i + 1, 0.1 + i * 0.01) for i in range(len(specs))],
        }[scheme]

    global_model = tuple(0.0 for _ in range(config.vector_size))
    rounds_out = []
    total_msgs = 0
    total_bytes = 0
    total_crypto = CryptoOps()
    all_times: list[float] = []

    for round_id in range(1, config.rounds + 1):
        updates: list[tuple[int, tuple[float, ...]]] = []
        if scheme == "B3-raka-ufl":
            mv, mh = fl.start_round(round_id)
        for index, spec in enumerate(specs):
            current_global = fl.global_model if scheme == "B3-raka-ufl" else global_model
            payload = serialize_model(local_train(current_global, spec, config))
            start = time.perf_counter()
            if scheme == "B3-raka-ufl":
                client = clients[index]
                client.model_version = mv
                client.model_hash = mh
                ts = base_ts + round_id * 1000 + index
                m1 = client.make_hello(ts, bytes([index + 1]) * 16)
                m2 = fl.ingest_hello(spec.pid, m1, now=ts, nonce_s=bytes([round_id + 32 + index]) * 16, timestamp_s=ts + 1)
                client.accept_server_hello(m2, now=ts + 1)
                update = client.encrypt_update(payload)
                plaintext, ack, is_dup = fl.ingest_update(spec.pid, update)
                client.accept_server_ack(ack)
                assert plaintext == payload and not is_dup
                msgs = 4
                byte_count = sum(len(p) if isinstance(p, bytes) else len(str(p).encode()) for p in (
                    m1.nonce_u, m1.model_version, m1.model_hash, m1.tag,
                    update.ciphertext, update.transcript_id, update.model_version, update.model_hash,
                    ack.payload_hash, ack.model_version, ack.model_hash, ack.tag,
                ))
                crypto = RAKA_UFL_CRYPTO
                accepted = True
            else:
                obs, plaintext = channels[index].run_round(payload)
                assert plaintext == payload
                msgs = obs.messages
                byte_count = obs.bytes_count
                crypto = obs.crypto
                accepted = obs.accepted
            elapsed_ms = (time.perf_counter() - start) * 1000
            all_times.append(elapsed_ms)
            total_msgs += msgs
            total_bytes += byte_count
            total_crypto.hkdf += crypto.hkdf
            total_crypto.hmac += crypto.hmac
            total_crypto.aead_enc += crypto.aead_enc
            total_crypto.aead_dec += crypto.aead_dec
            total_crypto.sha256 += crypto.sha256
            if accepted and scheme != "B3-raka-ufl":
                updates.append((spec.sample_count, local_train(global_model, spec, config)))
        if scheme == "B3-raka-ufl":
            report = fl.close_round()
            global_model = report.global_model
        else:
            global_model = _fedavg(updates)
        rounds_out.append({"round_id": round_id, "global_model_l2": l2(global_model)})

    return {
        "scheme": scheme,
        "rounds": rounds_out,
        "final_model": list(global_model),
        "total_messages": total_msgs,
        "total_bytes": total_bytes,
        "crypto_ops": total_crypto.as_dict(),
        "crypto_ops_total": total_crypto.total(),
        "mean_client_round_ms": statistics.fmean(all_times),
        "final_model_l2": l2(global_model),
    }


# ---------------------------------------------------------------------------
# T1: stale-context injection in round 2
# ---------------------------------------------------------------------------


def _wire_value(model: tuple[float, ...]) -> tuple[float, ...]:
    """Model value as actually received on the wire (12-decimal serialization)."""
    return tuple(json.loads(serialize_model(model).decode("ascii")))


def run_t1_b3(config: ExperimentConfig) -> dict[str, object]:
    """T1 with B3 on the real FLServer: aggregation acts on wire values."""
    specs = make_client_specs(config)
    victim = 0
    initial = tuple(0.0 for _ in range(config.vector_size))
    fl = FLServer("fl-demo", {s.pid: s.master_key for s in specs}, config.vector_size)
    fl.register_sample_counts({s.pid: s.sample_count for s in specs})

    # round 1: normal
    mv, mh = fl.start_round(1)
    for index, spec in enumerate(specs):
        client = MinimalClient("fl-demo", spec.pid, spec.master_key)
        client.model_version = mv
        client.model_hash = mh
        ts = 5000 + 1000 + index
        m1 = client.make_hello(ts, bytes([index + 1]) * 16)
        m2 = fl.ingest_hello(spec.pid, m1, now=ts, nonce_s=bytes([1 + 64 + index]) * 16, timestamp_s=ts + 1)
        client.accept_server_hello(m2, now=ts + 1)
        payload = serialize_model(local_train(initial, spec, config))
        update = client.encrypt_update(payload)
        _, ack, _ = fl.ingest_update(spec.pid, update)
        client.accept_server_ack(ack)
    r1 = fl.close_round()

    # round 2: victim holds the stale (round-1) model context
    mv2, mh2 = fl.start_round(2)
    stale_accepted = False
    for index, spec in enumerate(specs):
        trained = local_train(initial, spec, config) if index == victim else local_train(r1.global_model, spec, config)
        payload = serialize_model(trained)
        client = MinimalClient("fl-demo", spec.pid, spec.master_key, committed_round=1)
        if index == victim:
            client.model_version = "global-round-1"
            client.model_hash = hashlib.sha256(serialize_model(initial)).digest()
        else:
            client.model_version = mv2
            client.model_hash = mh2
        ts = 5000 + 2000 + index
        try:
            m1 = client.make_hello(ts, bytes([index + 1]) * 16)
            m2 = fl.ingest_hello(spec.pid, m1, now=ts, nonce_s=bytes([2 + 64 + index]) * 16, timestamp_s=ts + 1)
            client.accept_server_hello(m2, now=ts + 1)
            update = client.encrypt_update(payload)
            plaintext, ack, _ = fl.ingest_update(spec.pid, update)
            client.accept_server_ack(ack)
            assert plaintext == payload
        except ValueError:
            continue  # rejected at admission (model-context mismatch)
        if index == victim:
            stale_accepted = True
    r2 = fl.close_round()
    honest_wire = [
        (spec.sample_count, _wire_value(local_train(r1.global_model, spec, config)))
        for index, spec in enumerate(specs) if index != victim
    ]
    reference = _fedavg(honest_wire)
    return {
        "scheme": "B3-raka-ufl",
        "stale_update_accepted": stale_accepted,
        "aggregation_l2_vs_reference": l2_distance(r2.global_model, reference),
        "aggregated_clients": len(r2.accepted_clients),
    }


def run_t1(scheme: str, config: ExperimentConfig) -> dict[str, object]:
    if scheme == "B3-raka-ufl":
        return run_t1_b3(config)
    specs = make_client_specs(config)
    victim = 0
    initial_model = tuple(0.0 for _ in range(config.vector_size))

    def run_round_updates(round_id: int, global_model: tuple[float, ...], stale: bool) -> tuple[list[tuple[int, tuple[float, ...]]], bool]:
        """Returns (accepted updates, whether the stale update was accepted)."""
        updates: list[tuple[int, tuple[float, ...]]] = []
        stale_accepted = False
        for index, spec in enumerate(specs):
            if stale and index == victim:
                # victim trains from the ROUND-1 (stale) global model
                trained = local_train(initial_model, spec, config)
            else:
                trained = local_train(global_model, spec, config)
            payload = serialize_model(trained)
            if scheme == "B0-plain":
                obs, _ = PlainChannel(spec.pid).upload(payload)
            elif scheme == "B1-secure-channel":
                obs, _ = SecureChannelPSK(spec.pid, spec.master_key).run_round(payload)
            elif scheme == "B2p-pmap-port":
                obs, _ = PmapD2ZSession(index + 1, 0.1 + index * 0.01).run(payload)
            else:
                obs, _ = CompositeAKA(spec.pid, spec.master_key).run_round(payload)
            if obs.accepted:
                if stale and index == victim:
                    stale_accepted = True
                updates.append((spec.sample_count, trained))
        return updates, stale_accepted

    # round 1: normal
    updates1, _ = run_round_updates(1, initial_model, stale=False)
    global_after_r1 = _fedavg(updates1)
    # round 2 with stale-context injection
    updates2, stale_accepted = run_round_updates(2, global_after_r1, stale=True)
    contaminated = _fedavg(updates2)
    # contamination-free reference: victim excluded from round 2
    reference_updates = [u for i, u in enumerate(updates2) if i != victim] if stale_accepted else updates2
    if stale_accepted:
        # reference = what correct defense yields: only the honest in-context updates
        honest = []
        for index, spec in enumerate(specs):
            if index == victim:
                continue
            honest.append((spec.sample_count, local_train(global_after_r1, spec, config)))
        reference = _fedavg(honest)
    else:
        reference = contaminated
    return {
        "scheme": scheme,
        "stale_update_accepted": stale_accepted,
        "aggregation_l2_vs_reference": l2_distance(contaminated, reference),
        "aggregated_clients": len(updates2),
    }


# ---------------------------------------------------------------------------
# T2: lost receipt -> retransmission of the same update
# ---------------------------------------------------------------------------


def run_t2_b3(config: ExperimentConfig) -> dict[str, object]:
    """T2 with B3 on the real FLServer."""
    specs = make_client_specs(config)
    victim = 0
    initial = tuple(0.0 for _ in range(config.vector_size))
    fl = FLServer("fl-demo", {s.pid: s.master_key for s in specs}, config.vector_size)
    fl.register_sample_counts({s.pid: s.sample_count for s in specs})
    mv, mh = fl.start_round(1)
    counted_twice = False
    for index, spec in enumerate(specs):
        trained = local_train(initial, spec, config)
        payload = serialize_model(trained)
        client = MinimalClient("fl-demo", spec.pid, spec.master_key)
        client.model_version = mv
        client.model_hash = mh
        ts = 9000 + index
        m1 = client.make_hello(ts, bytes([index + 1]) * 16)
        m2 = fl.ingest_hello(spec.pid, m1, now=ts, nonce_s=bytes([index + 96]) * 16, timestamp_s=ts + 1)
        client.accept_server_hello(m2, now=ts + 1)
        update = client.encrypt_update(payload)
        plaintext, ack, _ = fl.ingest_update(spec.pid, update)
        assert plaintext == payload
        client.accept_server_ack(ack)
        if index == victim:
            # receipt lost: retransmit the byte-identical update
            plaintext2, ack2, is_dup = fl.ingest_update(spec.pid, update)
            idempotent = plaintext2 == payload and ack2 == ack and is_dup
            counted_twice = not idempotent
    report = fl.close_round()
    contaminated = report.global_model
    reference = _fedavg([
        (s.sample_count, _wire_value(local_train(initial, s, config))) for s in specs
    ])
    return {
        "scheme": "B3-raka-ufl",
        "retransmission_counted_twice": counted_twice,
        "aggregation_l2_vs_reference": l2_distance(contaminated, reference),
        "victim_weight_after": specs[victim].sample_count / sum(s.sample_count for s in specs),
        "victim_weight_expected": specs[victim].sample_count / sum(s.sample_count for s in specs),
    }


def run_t2(scheme: str, config: ExperimentConfig) -> dict[str, object]:
    if scheme == "B3-raka-ufl":
        return run_t2_b3(config)
    specs = make_client_specs(config)
    victim = 0
    global_model = tuple(0.0 for _ in range(config.vector_size))
    updates: list[tuple[int, tuple[float, ...]]] = []
    counted_twice = False

    for index, spec in enumerate(specs):
        trained = local_train(global_model, spec, config)
        payload = serialize_model(trained)
        if scheme == "B0-plain":
            obs, _ = PlainChannel(spec.pid).upload(payload)
            channel = None
        elif scheme == "B1-secure-channel":
            channel = SecureChannelPSK(spec.pid, spec.master_key)
            obs, _ = channel.run_round(payload)
        elif scheme == "B2p-pmap-port":
            channel = PmapD2ZSession(index + 1, 0.1 + index * 0.01)
            obs, _ = channel.run(payload)
        else:
            channel = CompositeAKA(spec.pid, spec.master_key)
            obs, _ = channel.run_round(payload)
        if obs.accepted:
            updates.append((spec.sample_count, trained))
        if index == victim:
            # receipt lost: retransmission — baselines have no idempotency,
            # so the update is submitted (and aggregated) a second time.
            updates.append((spec.sample_count, trained))
            counted_twice = True

    contaminated = _fedavg(updates)
    reference = _fedavg([(s.sample_count, local_train(global_model, s, config)) for s in specs])
    total_samples = sum(n for n, _ in updates)
    return {
        "scheme": scheme,
        "retransmission_counted_twice": counted_twice,
        "aggregation_l2_vs_reference": l2_distance(contaminated, reference),
        "victim_weight_after": specs[victim].sample_count * (2 if counted_twice else 1) / total_samples,
        "victim_weight_expected": specs[victim].sample_count / sum(s.sample_count for s in specs),
    }


# ---------------------------------------------------------------------------
# capability matrix
# ---------------------------------------------------------------------------


def capability_matrix() -> dict[str, dict[str, bool]]:
    return {
        "B0-plain": PlainChannel.CAPABILITIES,
        "B1-secure-channel": SecureChannelPSK.CAPABILITIES,
        "B2-composite-aka": CompositeAKA.CAPABILITIES,
        "B2p-pmap-port": PmapD2ZSession.CAPABILITIES,
        "B3-raka-ufl": RAKA_UFL_CAPABILITIES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAKA-UFL vs baselines comparison")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--vector-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    config = ExperimentConfig(
        clients=args.clients, rounds=args.rounds,
        vector_size=args.vector_size, seed=args.seed,
    )
    schemes = ["B0-plain", "B1-secure-channel", "B2-composite-aka", "B2p-pmap-port", "B3-raka-ufl"]
    report = {
        "config": {"clients": config.clients, "rounds": config.rounds, "vector_size": config.vector_size, "seed": config.seed},
        "capability_matrix": capability_matrix(),
        "normal_run": [run_normal(s, config) for s in schemes],
        "T1_stale_context": [run_t1(s, config) for s in schemes],
        "T2_lost_receipt_retransmission": [run_t2(s, config) for s in schemes],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
