"""Small end-to-end experiment for the recommended RAKA-UFL scope.

The experiment intentionally uses a deterministic vector model instead of a
deep-learning framework. It measures protocol overhead while exercising
multiple UAV identities, local updates, FedAvg, idempotent retransmission,
and server-side revocation.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import random
import secrets
import statistics
import time
from typing import Iterable

from raka_ufl_fl_server import FLServer, fedavg, serialize_model
from raka_ufl_minimal import ModelUpdate
from raka_ufl_masked import MaskedClient, WireUpdate, WireAck, WireHello

# Backward-compatible alias (tests import _fedavg from this module).
_fedavg = fedavg


@dataclass(frozen=True)
class ExperimentConfig:
    clients: int = 4
    rounds: int = 3
    vector_size: int = 16
    local_steps: int = 2
    learning_rate: float = 0.05
    seed: int = 20260924
    revoked_client: int | None = None


@dataclass(frozen=True)
class RoundMetrics:
    round_id: int
    attempted_clients: int
    accepted_clients: int
    revoked_clients: int
    duplicate_uploads: int
    total_messages: int
    total_bytes: int
    mean_handshake_ms: float
    mean_upload_ms: float
    mean_total_ms: float
    global_model_l2: float


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    rounds: tuple[RoundMetrics, ...]
    final_model: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "config": {
                "clients": self.config.clients,
                "rounds": self.config.rounds,
                "vector_size": self.config.vector_size,
                "local_steps": self.config.local_steps,
                "learning_rate": self.config.learning_rate,
                "seed": self.config.seed,
                "revoked_client": self.config.revoked_client,
            },
            "rounds": [
                {
                    "round_id": item.round_id,
                    "attempted_clients": item.attempted_clients,
                    "accepted_clients": item.accepted_clients,
                    "revoked_clients": item.revoked_clients,
                    "duplicate_uploads": item.duplicate_uploads,
                    "total_messages": item.total_messages,
                    "total_bytes": item.total_bytes,
                    "mean_handshake_ms": item.mean_handshake_ms,
                    "mean_upload_ms": item.mean_upload_ms,
                    "mean_total_ms": item.mean_total_ms,
                    "global_model_l2": item.global_model_l2,
                }
                for item in self.rounds
            ],
            "final_model": list(self.final_model),
        }


@dataclass
class _ClientRecord:
    client_id: str
    client: MaskedClient
    local_target: tuple[float, ...]
    sample_count: int
    revoked: bool = False


class MultiUAVExperiment:
    def __init__(self, config: ExperimentConfig):
        if config.clients <= 0 or config.rounds <= 0 or config.vector_size <= 0:
            raise ValueError("clients、rounds 和 vector_size 必须为正数")
        if config.local_steps <= 0 or config.learning_rate <= 0:
            raise ValueError("local_steps 和 learning_rate 必须为正数")
        if config.revoked_client is not None and not 0 <= config.revoked_client < config.clients:
            raise ValueError("revoked_client 超出客户端范围")
        self.config = config
        self.rng = random.Random(config.seed)
        self.clients = self._make_clients()
        # The real many-to-one FL server: one instance holding the registry,
        # per-identity protocol endpoints, the aggregation buffer, and the
        # global model lifecycle.
        registry = {record.client_id: record.client.master_key for record in self.clients}
        self.fl_server = FLServer(
            "fl-demo",
            registry,
            config.vector_size,
            revoked={record.client_id for record in self.clients if record.revoked},
        )
        self.fl_server.register_sample_counts(
            {record.client_id: record.sample_count for record in self.clients}
        )

    def _make_clients(self) -> list[_ClientRecord]:
        records: list[_ClientRecord] = []
        for index in range(self.config.clients):
            client_id = f"uav-{index + 1}"
            master_key = secrets.token_bytes(32)
            client = MaskedClient("fl-demo", client_id, master_key)
            target = tuple(
                self.rng.uniform(-1.0, 1.0)
                for _ in range(self.config.vector_size)
            )
            records.append(_ClientRecord(
                client_id,
                client,
                target,
                8 + index * 4,
                self.config.revoked_client == index,
            ))
        return records

    def _local_train(self, record: _ClientRecord) -> tuple[float, ...]:
        model = list(self.fl_server.global_model)
        for _ in range(self.config.local_steps):
            model = [
                value + self.config.learning_rate * (target - value)
                for value, target in zip(model, record.local_target)
            ]
        return tuple(model)

    @staticmethod
    def _serialize_model(model: Iterable[float]) -> bytes:
        return serialize_model(model)

    @staticmethod
    def _model_l2(model: Iterable[float]) -> float:
        return sum(value * value for value in model) ** 0.5

    @staticmethod
    def _message_bytes(hello: WireHello, update: WireUpdate, ack: WireAck) -> int:
        return sum(len(part) if isinstance(part, bytes) else len(str(part).encode("utf-8")) for part in (
            hello.masked_id,
            hello.nonce_u,
            hello.model_version,
            hello.model_hash,
            hello.tag,
            update.masked_id,
            update.ciphertext,
            update.transcript_id,
            update.model_version,
            update.model_hash,
            ack.masked_id,
            ack.payload_hash,
            ack.model_version,
            ack.model_hash,
            ack.tag,
        ))

    def run(self) -> ExperimentResult:
        round_metrics: list[RoundMetrics] = []
        for round_id in range(1, self.config.rounds + 1):
            attempted = 0
            accepted = 0
            revoked = 0
            duplicate_uploads = 0
            message_count = 0
            byte_count = 0
            handshake_times: list[float] = []
            upload_times: list[float] = []
            total_times: list[float] = []

            mv, mh = self.fl_server.start_round(round_id)

            for index, record in enumerate(self.clients):
                attempted += 1
                if record.revoked:
                    revoked += 1
                    m1 = record.client.make_hello_wire(round_id * 1000 + index, bytes([index + 1]) * 16)
                    try:
                        self.fl_server.ingest_wire_hello(
                            m1,
                            now=round_id * 1000 + index,
                            timestamp_s=round_id * 1000 + index + 1,
                        )
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("服务器未拒绝已吊销 UAV")
                    continue
                record.client.model_version = mv
                record.client.model_hash = mh
                start = time.perf_counter()
                timestamp_u = round_id * 1000 + index
                nonce_u = bytes([index + 1]) * 16
                m1 = record.client.make_hello_wire(timestamp_u, nonce_u)
                m2 = self.fl_server.ingest_wire_hello(
                    m1,
                    now=timestamp_u,
                    nonce_s=bytes([round_id + 32 + index]) * 16,
                    timestamp_s=timestamp_u + 1,
                )
                record.client.accept_wire_server_hello(m2, now=timestamp_u + 1)
                handshake_ms = (time.perf_counter() - start) * 1000

                local_model = self._local_train(record)
                payload = self._serialize_model(local_model)
                upload_start = time.perf_counter()
                update = record.client.encrypt_update_wire(payload)
                plaintext, ack, is_dup = self.fl_server.ingest_wire_update(update)
                if is_dup:
                    raise AssertionError("首次上传被误判为重复")
                if plaintext != payload:
                    raise AssertionError("服务器解密后的模型更新不一致")
                upload_ms = (time.perf_counter() - upload_start) * 1000
                record.client.accept_wire_ack(ack)
                if record.client.committed_round != round_id:
                    raise AssertionError("客户端轮次没有正确提交")
                accepted += 1
                message_count += 4
                byte_count += self._message_bytes(m1, update, ack)
                handshake_times.append(handshake_ms)
                upload_times.append(upload_ms)
                total_times.append((time.perf_counter() - start) * 1000)

                duplicate_result, duplicate_ack, is_dup2 = self.fl_server.ingest_wire_update(update)
                if not is_dup2 or duplicate_result != payload or duplicate_ack != ack:
                    raise AssertionError("重复上传没有返回一致的幂等结果")
                duplicate_uploads += 1
                message_count += 1
                byte_count += (
                    len(update.masked_id)
                    + len(update.ciphertext)
                    + len(update.transcript_id)
                    + len(update.model_version.encode("utf-8"))
                    + len(update.model_hash)
                )

            report = self.fl_server.close_round()
            if len(report.accepted_clients) != accepted:
                raise AssertionError("FL 服务器聚合参与数与接受数不一致")
            round_metrics.append(RoundMetrics(
                round_id=round_id,
                attempted_clients=attempted,
                accepted_clients=accepted,
                revoked_clients=revoked,
                duplicate_uploads=duplicate_uploads,
                total_messages=message_count,
                total_bytes=byte_count,
                mean_handshake_ms=statistics.fmean(handshake_times),
                mean_upload_ms=statistics.fmean(upload_times),
                mean_total_ms=statistics.fmean(total_times),
                global_model_l2=self._model_l2(report.global_model),
            ))
        return ExperimentResult(self.config, tuple(round_metrics), self.fl_server.global_model)


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentResult:
    return MultiUAVExperiment(config or ExperimentConfig()).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal RAKA-UFL multi-UAV experiment")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--vector-size", type=int, default=16)
    parser.add_argument("--local-steps", type=int, default=2)
    parser.add_argument("--revoked-client", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    result = run_experiment(ExperimentConfig(
        clients=args.clients,
        rounds=args.rounds,
        vector_size=args.vector_size,
        local_steps=args.local_steps,
        revoked_client=args.revoked_client,
        seed=args.seed,
    ))
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
