"""FLServer: the aggregation-side component of the RAKA-UFL harness.

Before this module, the experiment driver (harness loop) performed FedAvg over
per-UAV MinimalServer instances — one protocol endpoint per UAV with no shared
server. FLServer turns that into the real many-to-one architecture: ONE server
holding the global model, the per-identity protocol endpoints, the per-round
aggregation buffer, the participation set, and revocation.

Protocol message formats and the MinimalServer state machine are unchanged;
FLServer indexes endpoint state by pid and owns the FL-level lifecycle:
  start_round(round_id)        -> configure model context (v, h) on endpoints
  ingest_hello(pid, m1)        -> revocation gate, then delegate M1 to endpoint
  ingest_update(pid, update)   -> delegate M3; fresh updates enter the buffer,
                                  byte-identical retransmissions are idempotent
  close_round()                -> sample-weighted FedAvg, advance global model
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Optional

from raka_ufl_minimal import (
    ClientHello,
    MinimalServer,
    ModelUpdate,
    ServerAck,
    ServerHello,
    _identity_digest,
    _round_pad,
    _xor32,
    DIGEST_SIZE,
)
from raka_ufl_masked import WireHello, WireServerHello, WireUpdate, WireAck


def fedavg(updates: list[tuple[int, tuple[float, ...]]]) -> tuple[float, ...]:
    total_samples = sum(sample_count for sample_count, _ in updates)
    if total_samples <= 0:
        raise ValueError("FedAvg 样本数必须为正数")
    width = len(updates[0][1])
    return tuple(
        sum(sample_count * model[index] for sample_count, model in updates) / total_samples
        for index in range(width)
    )


def serialize_model(model: Iterable[float]) -> bytes:
    return json.dumps(
        [round(value, 12) for value in model],
        separators=(",", ":"),
    ).encode("ascii")


@dataclass(frozen=True)
class RoundReport:
    round_id: int
    accepted_clients: tuple[str, ...]
    duplicate_tids: int
    global_model: tuple[float, ...]
    model_version: str
    model_hash_hex: str


class FLServer:
    """One server, many UAVs: registry + protocol endpoints + aggregation.

    registry maps pid -> master_key (one MinimalServer endpoint per identity,
    preserving the protocol's per-identity state machine; FLServer adds the
    shared FL lifecycle on top).
    """

    def __init__(
        self,
        task_id: str,
        registry: dict[str, bytes],
        vector_size: int,
        revoked: Optional[set[str]] = None,
    ):
        if not registry:
            raise ValueError("注册表不能为空")
        if vector_size <= 0:
            raise ValueError("vector_size 必须为正数")
        self.task_id = task_id
        self.registry = dict(registry)
        self._revoked = set(revoked or set())
        unknown = self._revoked - set(self.registry)
        if unknown:
            raise ValueError(f"吊销集合含未注册 UAV: {unknown}")
        self.global_model = tuple(0.0 for _ in range(vector_size))
        self.committed_round = 0
        # masked-identity layer: registry key is real_id; endpoints are keyed by
        # the identity digest and use keyed transcript ids (HMAC tid).
        self._digests = {real_id: _identity_digest(real_id) for real_id in self.registry}
        self._digest_to_real = {d.hex(): real_id for real_id, d in self._digests.items()}
        self._endpoints = {
            d.hex(): MinimalServer(task_id, d.hex(), key, transcript_key=key)
            for real_id, (d, key) in ((r, (self._digests[r], k)) for r, k in self.registry.items())
        }
        self._masked_table: dict[bytes, str] = {}
        self._masked_table_round: Optional[int] = None
        self._round_id = 0
        self._round_open = False
        self._updates: list[tuple[str, int, tuple[float, ...]]] = []
        self._accepted_tids: set[bytes] = set()
        self._duplicate_tids = 0
        self._sample_counts: dict[str, int] = {}

    # ---- FL-level context ----

    @property
    def model_version(self) -> str:
        return f"global-round-{self._round_id}"

    @property
    def model_hash(self) -> bytes:
        return hashlib.sha256(serialize_model(self.global_model)).digest()

    def is_revoked(self, pid: str) -> bool:
        return pid in self._revoked

    def revoke(self, pid: str) -> None:
        if pid not in self.registry:
            raise ValueError(f"未注册的 UAV: {pid}")
        self._revoked.add(pid)

    # ---- round lifecycle ----

    def start_round(self, round_id: int) -> tuple[str, bytes]:
        if round_id != self.committed_round + 1:
            raise ValueError("FL 服务器只推进到下一轮")
        if self._round_open:
            raise RuntimeError("上一轮尚未关闭")
        self._round_id = round_id
        self._round_open = True
        self._updates = []
        self._accepted_tids = set()
        self._duplicate_tids = 0
        mv, mh = self.model_version, self.model_hash
        for ep in self._endpoints.values():
            ep.model_version = mv
            ep.model_hash = mh
        return mv, mh

    def _refresh_masked_table(self, round_id: int) -> None:
        """Per-round precomputation: masked_id -> real_id for non-revoked UAVs."""
        if self._masked_table_round == round_id:
            return
        self._masked_table = {}
        for real_id, key in self.registry.items():
            if real_id in self._revoked:
                continue  # revoked UAVs are unresolvable from this round on
            pad = _round_pad(key, self.task_id, round_id)
            self._masked_table[_xor32(self._digests[real_id], pad)] = real_id
        self._masked_table_round = round_id

    def resolve_masked_id(self, masked_id: bytes, round_id: int) -> str:
        if len(masked_id) != DIGEST_SIZE:
            raise ValueError("动态假名格式无效")
        self._refresh_masked_table(round_id)
        real_id = self._masked_table.get(masked_id)
        if real_id is None:
            raise ValueError("动态假名不在本轮注册表中（伪造、跨轮或已吊销）")
        return real_id

    def ingest_wire_hello(
        self,
        wire: WireHello,
        now: Optional[int] = None,
        nonce_s: Optional[bytes] = None,
        timestamp_s: Optional[int] = None,
    ) -> WireServerHello:
        real_id = self.resolve_masked_id(wire.masked_id, wire.round_id)
        canonical = ClientHello(
            task_id=wire.task_id,
            pid=self._digests[real_id].hex(),
            round_id=wire.round_id,
            timestamp_u=wire.timestamp_u,
            nonce_u=wire.nonce_u,
            model_version=wire.model_version,
            model_hash=wire.model_hash,
            tag=wire.tag,
        )
        m2 = self.ingest_hello(real_id, canonical, now=now, nonce_s=nonce_s, timestamp_s=timestamp_s)
        return WireServerHello(
            m2.task_id, wire.masked_id, m2.round_id, m2.timestamp_u,
            m2.timestamp_s, m2.nonce_u, m2.nonce_s, m2.transcript_id,
            m2.model_version, m2.model_hash, m2.tag,
        )

    def ingest_wire_update(self, wire: WireUpdate) -> tuple[bytes, WireAck, bool]:
        real_id = self.resolve_masked_id(wire.masked_id, wire.round_id)
        canonical = ModelUpdate(
            wire.task_id, self._digests[real_id].hex(), wire.round_id,
            wire.transcript_id, wire.model_version, wire.model_hash, wire.ciphertext,
        )
        plaintext, ack, is_dup = self.ingest_update(real_id, canonical)
        return plaintext, WireAck(
            ack.task_id, wire.masked_id, ack.round_id, ack.transcript_id,
            ack.payload_hash, ack.model_version, ack.model_hash, ack.tag,
        ), is_dup

    def ingest_hello(
        self,
        pid: str,
        hello: ClientHello,
        now: Optional[int] = None,
        nonce_s: Optional[bytes] = None,
        timestamp_s: Optional[int] = None,
    ) -> ServerHello:
        if pid in self._revoked:
            raise ValueError("UAV 已被服务器吊销")
        digest_hex = self._digests[pid].hex() if pid in self._digests else pid
        endpoint = self._endpoints.get(digest_hex)
        if endpoint is None:
            raise ValueError("未注册的 UAV")
        return endpoint.accept_client_hello(
            hello, now=now, nonce_s=nonce_s, timestamp_s=timestamp_s
        )

    def ingest_update(self, pid: str, update: ModelUpdate) -> tuple[bytes, ServerAck, bool]:
        """Returns (plaintext, ack, is_duplicate). Only fresh transcripts enter
        the aggregation buffer; exact retransmissions return the cached receipt
        without double-counting."""
        if pid in self._revoked:
            raise ValueError("UAV 已被服务器吊销")
        digest_hex = self._digests[pid].hex() if pid in self._digests else pid
        endpoint = self._endpoints.get(digest_hex)
        if endpoint is None:
            raise ValueError("未注册的 UAV")
        plaintext, ack = endpoint.receive_update(update)
        if update.transcript_id in self._accepted_tids:
            self._duplicate_tids += 1
            return plaintext, ack, True
        self._accepted_tids.add(update.transcript_id)
        parsed = tuple(json.loads(plaintext.decode("ascii")))
        sample_count = self._sample_counts.get(pid, 1)
        self._updates.append((pid, sample_count, parsed))
        return plaintext, ack, False

    def register_sample_counts(self, sample_counts: dict[str, int]) -> None:
        self._sample_counts = dict(sample_counts)

    def close_round(self) -> RoundReport:
        if not self._round_open:
            raise RuntimeError("没有进行中的轮次")
        if not self._updates:
            raise RuntimeError(f"第 {self._round_id} 轮没有可用更新")
        self.global_model = fedavg([
            (sample_count, model) for _, sample_count, model in self._updates
        ])
        self.committed_round = self._round_id
        self._round_open = False
        mv, mh = self.model_version, self.model_hash
        return RoundReport(
            round_id=self._round_id,
            accepted_clients=tuple(pid for pid, _, _ in self._updates),
            duplicate_tids=self._duplicate_tids,
            global_model=self.global_model,
            model_version=mv,
            model_hash_hex=mh.hex(),
        )
