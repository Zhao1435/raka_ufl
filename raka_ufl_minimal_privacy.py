"""Dynamic-pseudonym privacy extension for the RAKA-UFL minimal protocol.

Optional module: replaces the fixed pseudonym on the wire with a per-round
round-derived masked identity, while leaving the core M1-M4 state machine
untouched.

Construction (no random nonce on the wire for the pseudonym):

    pad_i     = HMAC(K_i, task || r || "pid-mask")     (keyed, round-derived)
    digest    = SHA256(real_id)                        (32-byte stable identity)
    masked_id = digest XOR pad_i

Wire-level M1 carries masked_id (32 bytes) instead of the fixed pid.

Server-side resolution is O(1) by precomputation: at the start of each round
the server computes masked_j = digest_j XOR pad_j for every registered UAV
and builds a reverse lookup table. A forged or cross-round masked_id simply
misses the table (rejected before any tag verification), so the pseudonym
layer itself becomes an extra round-binding check. Revocation works by
removing the entry from the registry — next round the UAV's pseudonym is no
longer in the table.

Properties:
- Unlinkable across rounds: pad_i changes with r; without K_i an observer
  cannot compute pad_i, so successive masked_id values are indistinguishable
  from random.
- Deterministic within one round: retransmissions of M1 in the same round
  carry the same masked_id (recognizable as same-device retry). The protocol
  already admits at most one transaction per round per device, so the
  observable leakage is "this device participated once this round".
- No desynchronization: nothing about the pseudonym depends on delivery of
  previous messages (unlike rotating-pseudonym schemes such as PMAP).

The canonical ClientHello keeps pid = digest.hex (stable identity), so the
authentication transcript binds the stable identity, not the mask; existing
MinimalServer verification logic is reused unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from typing import Optional

from raka_ufl_minimal import (
    ClientHello,
    MinimalClient,
    MinimalServer,
    ModelUpdate,
    ServerAck,
    ServerHello,
)


DIGEST_SIZE = 32


def _identity_digest(real_id: str) -> bytes:
    return hashlib.sha256(real_id.encode("utf-8")).digest()


def _round_pad(master_key: bytes, task_id: str, round_id: int) -> bytes:
    return hmac.new(
        master_key,
        b"pid-mask|" + task_id.encode("utf-8") + b"|" + round_id.to_bytes(8, "big"),
        hashlib.sha256,
    ).digest()


def _xor32(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


@dataclass(frozen=True)
class WireHelloP:
    """Wire-format M1 with the round-derived dynamic pseudonym."""

    task_id: str
    masked_id: bytes
    round_id: int
    timestamp_u: int
    nonce_u: bytes
    model_version: str
    model_hash: bytes
    tag: bytes


@dataclass(frozen=True)
class WireServerHelloP:
    """Wire-format M2: the pid field carries masked_id, never the digest."""

    task_id: str
    masked_id: bytes
    round_id: int
    timestamp_u: int
    timestamp_s: int
    nonce_u: bytes
    nonce_s: bytes
    transcript_id: bytes
    model_version: str
    model_hash: bytes
    tag: bytes


@dataclass(frozen=True)
class WireUpdateP:
    """Wire-format M3: pid field carries masked_id."""

    task_id: str
    masked_id: bytes
    round_id: int
    transcript_id: bytes
    model_version: str
    model_hash: bytes
    ciphertext: bytes


@dataclass(frozen=True)
class WireAckP:
    """Wire-format M4: pid field carries masked_id."""

    task_id: str
    masked_id: bytes
    round_id: int
    transcript_id: bytes
    payload_hash: bytes
    model_version: str
    model_hash: bytes
    tag: bytes


class PrivacyClient(MinimalClient):
    """MinimalClient variant: sends the round-derived masked identity on the
    wire. The canonical ClientHello keeps pid = SHA256(real_id).hex, so the
    authentication transcript binds the stable identity digest."""

    def __init__(self, task_id: str, real_id: str, master_key: bytes, **kwargs):
        digest_hex = _identity_digest(real_id).hex()
        # keyed transcript id (HMAC) is REQUIRED in this extension: a public
        # hash-only tid would be an offline dictionary verifier for the digest
        # (all other tid fields are visible on the wire in M1/M2).
        super().__init__(task_id, digest_hex, master_key, transcript_key=master_key, **kwargs)
        self.real_id = real_id
        self._last_masked_id: Optional[bytes] = None

    def make_hello_masked(
        self, timestamp_u: int, nonce_u: Optional[bytes] = None
    ) -> WireHelloP:
        hello = self.make_hello(timestamp_u, nonce_u)
        pad = _round_pad(self.master_key, self.task_id, hello.round_id)
        masked_id = _xor32(_identity_digest(self.real_id), pad)
        self._last_masked_id = masked_id
        return WireHelloP(
            task_id=hello.task_id,
            masked_id=masked_id,
            round_id=hello.round_id,
            timestamp_u=hello.timestamp_u,
            nonce_u=hello.nonce_u,
            model_version=hello.model_version,
            model_hash=hello.model_hash,
            tag=hello.tag,
        )

    def accept_wire_server_hello(self, wire: WireServerHelloP, now: Optional[int] = None):
        if wire.masked_id != self._last_masked_id:
            raise ValueError("M2 假名与当前事务不匹配")
        canonical = ServerHello(
            wire.task_id, self.pid, wire.round_id, wire.timestamp_u,
            wire.timestamp_s, wire.nonce_u, wire.nonce_s, wire.transcript_id,
            wire.model_version, wire.model_hash, wire.tag,
        )
        return self.accept_server_hello(canonical, now=now)

    def encrypt_update_wire(self, model_update: bytes) -> WireUpdateP:
        update = self.encrypt_update(model_update)
        return WireUpdateP(
            update.task_id, self._last_masked_id, update.round_id,
            update.transcript_id, update.model_version, update.model_hash,
            update.ciphertext,
        )

    def accept_wire_ack(self, wire: WireAckP) -> None:
        if wire.masked_id != self._last_masked_id:
            raise ValueError("M4 假名与当前事务不匹配")
        canonical = ServerAck(
            wire.task_id, self.pid, wire.round_id, wire.transcript_id,
            wire.payload_hash, wire.model_version, wire.model_hash, wire.tag,
        )
        self.accept_server_ack(canonical)


class PrivacyServer:
    """Server front-end with per-round precomputed pseudonym lookup table.

    registry: real_id -> master_key. One MinimalServer instance per identity
    (mirrors the reference harness, one client-server pair per UAV)."""

    def __init__(self, task_id: str, registry: dict[str, bytes], **server_kwargs):
        if not registry:
            raise ValueError("注册表不能为空")
        self.task_id = task_id
        self.registry = dict(registry)
        self._digests = {real_id: _identity_digest(real_id) for real_id in registry}
        self._servers = {
            self._digests[real_id]: MinimalServer(
                task_id, self._digests[real_id].hex(), key,
                transcript_key=key, **server_kwargs
            )
            for real_id, key in self.registry.items()
        }
        self._table_round: Optional[int] = None
        self._table: dict[bytes, bytes] = {}

    def _refresh_table(self, round_id: int) -> None:
        """Per-round precomputation: masked -> digest for every registered UAV."""
        if self._table_round == round_id:
            return
        self._table = {}
        for real_id, key in self.registry.items():
            pad = _round_pad(key, self.task_id, round_id)
            self._table[_xor32(self._digests[real_id], pad)] = self._digests[real_id]
        self._table_round = round_id

    def resolve(self, masked_id: bytes, round_id: int) -> bytes:
        if len(masked_id) != DIGEST_SIZE:
            raise ValueError("动态假名格式无效")
        self._refresh_table(round_id)
        digest = self._table.get(masked_id)
        if digest is None:
            raise ValueError("动态假名不在本轮注册表中（伪造、跨轮或已吊销）")
        return digest

    def accept_wire_hello(
        self,
        wire: WireHelloP,
        now: Optional[int] = None,
        nonce_s: Optional[bytes] = None,
        timestamp_s: Optional[int] = None,
    ) -> WireServerHelloP:
        digest = self.resolve(wire.masked_id, wire.round_id)
        hello = ClientHello(
            task_id=wire.task_id,
            pid=digest.hex(),
            round_id=wire.round_id,
            timestamp_u=wire.timestamp_u,
            nonce_u=wire.nonce_u,
            model_version=wire.model_version,
            model_hash=wire.model_hash,
            tag=wire.tag,
        )
        m2 = self._servers[digest].accept_client_hello(
            hello, now=now, nonce_s=nonce_s, timestamp_s=timestamp_s
        )
        return WireServerHelloP(
            m2.task_id, wire.masked_id, m2.round_id, m2.timestamp_u,
            m2.timestamp_s, m2.nonce_u, m2.nonce_s, m2.transcript_id,
            m2.model_version, m2.model_hash, m2.tag,
        )

    def receive_wire_update(self, wire: WireUpdateP) -> tuple[bytes, WireAckP]:
        digest = self.resolve(wire.masked_id, wire.round_id)
        canonical = ModelUpdate(
            wire.task_id, digest.hex(), wire.round_id, wire.transcript_id,
            wire.model_version, wire.model_hash, wire.ciphertext,
        )
        plaintext, ack = self._servers[digest].receive_update(canonical)
        return plaintext, WireAckP(
            ack.task_id, wire.masked_id, ack.round_id, ack.transcript_id,
            ack.payload_hash, ack.model_version, ack.model_hash, ack.tag,
        )

    def server_for(self, real_id: str) -> MinimalServer:
        return self._servers[self._digests[real_id]]
