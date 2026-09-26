"""Round-derived masked identities for the RAKA-UFL protocol.

Unified form: the wire identity in M1-M4 is ALWAYS the per-round masked
identity (this is the protocol's standard wire form, not an option):

    pad_i     = HMAC(K_i, task || r || "pid-mask")     (keyed, round-derived)
    digest    = SHA256(real_id)                        (32-byte stable identity)
    masked_id = digest XOR pad_i

Server-side resolution is O(1) by per-round precomputation. A forged or
cross-round masked_id misses the table and is rejected before any tag
verification. The transcript id is keyed (HMAC under the master key): a public
hash-only tid would be an offline dictionary verifier for the digest.

Properties:
- Unlinkable across rounds: pad_i changes with r; without K_i successive
  masked_id values are indistinguishable from random.
- Deterministic within one round: same-round M1 retransmissions carry the same
  masked_id (one transaction per device per round).
- No desynchronization: nothing depends on delivery of previous messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from raka_ufl_minimal import (
    ClientHello,
    MinimalClient,
    MinimalServer,
    ModelUpdate,
    ServerAck,
    ServerHello,
    _identity_digest,
    _round_pad,
    _xor32,
    DIGEST_SIZE,
)


@dataclass(frozen=True)
class WireHello:
    """Wire-format M1: the identity field carries the round-derived masked_id."""

    task_id: str
    masked_id: bytes
    round_id: int
    timestamp_u: int
    nonce_u: bytes
    model_version: str
    model_hash: bytes
    tag: bytes


@dataclass(frozen=True)
class WireServerHello:
    """Wire-format M2: the identity field carries masked_id, never the digest."""

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
class WireUpdate:
    """Wire-format M3: the identity field carries masked_id."""

    task_id: str
    masked_id: bytes
    round_id: int
    transcript_id: bytes
    model_version: str
    model_hash: bytes
    ciphertext: bytes


@dataclass(frozen=True)
class WireAck:
    """Wire-format M4: the identity field carries masked_id."""

    task_id: str
    masked_id: bytes
    round_id: int
    transcript_id: bytes
    payload_hash: bytes
    model_version: str
    model_hash: bytes
    tag: bytes


class MaskedClient(MinimalClient):
    """Standard RAKA-UFL client: puts the round-derived masked identity on the
    wire. The canonical ClientHello keeps pid = SHA256(real_id).hex, so the
    authentication transcript binds the stable identity digest, not the mask."""

    def __init__(self, task_id: str, real_id: str, master_key: bytes, **kwargs):
        digest_hex = _identity_digest(real_id).hex()
        super().__init__(task_id, digest_hex, master_key, transcript_key=master_key, **kwargs)
        self.real_id = real_id
        self._last_masked_id: Optional[bytes] = None

    def make_hello_wire(self, timestamp_u: int, nonce_u: Optional[bytes] = None) -> WireHello:
        hello = self.make_hello(timestamp_u, nonce_u)
        pad = _round_pad(self.master_key, self.task_id, hello.round_id)
        masked_id = _xor32(_identity_digest(self.real_id), pad)
        self._last_masked_id = masked_id
        return WireHello(
            task_id=hello.task_id,
            masked_id=masked_id,
            round_id=hello.round_id,
            timestamp_u=hello.timestamp_u,
            nonce_u=hello.nonce_u,
            model_version=hello.model_version,
            model_hash=hello.model_hash,
            tag=hello.tag,
        )

    def accept_wire_server_hello(self, wire: WireServerHello, now: Optional[int] = None):
        if wire.masked_id != self._last_masked_id:
            raise ValueError("M2 假名与当前事务不匹配")
        canonical = ServerHello(
            wire.task_id, self.pid, wire.round_id, wire.timestamp_u,
            wire.timestamp_s, wire.nonce_u, wire.nonce_s, wire.transcript_id,
            wire.model_version, wire.model_hash, wire.tag,
        )
        return self.accept_server_hello(canonical, now=now)

    def encrypt_update_wire(self, model_update: bytes) -> WireUpdate:
        update = self.encrypt_update(model_update)
        return WireUpdate(
            update.task_id, self._last_masked_id, update.round_id,
            update.transcript_id, update.model_version, update.model_hash,
            update.ciphertext,
        )

    def accept_wire_ack(self, wire: WireAck) -> None:
        if wire.masked_id != self._last_masked_id:
            raise ValueError("M4 假名与当前事务不匹配")
        canonical = ServerAck(
            wire.task_id, self.pid, wire.round_id, wire.transcript_id,
            wire.payload_hash, wire.model_version, wire.model_hash, wire.tag,
        )
        self.accept_server_ack(canonical)


def resolve_masked(
    registry: dict[str, bytes],
    digests: dict[str, bytes],
    task_id: str,
    masked_id: bytes,
    round_id: int,
) -> bytes:
    """Resolve one masked_id against the per-round table (built by caller)."""
    if len(masked_id) != DIGEST_SIZE:
        raise ValueError("动态假名格式无效")
    for real_id, key in registry.items():
        pad = _round_pad(key, task_id, round_id)
        if _xor32(digests[real_id], pad) == masked_id:
            return digests[real_id]
    raise ValueError("动态假名不在本轮注册表中（伪造、跨轮或已吊销）")
