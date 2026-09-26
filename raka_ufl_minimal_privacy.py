"""Compatibility shim for the round-derived masked-identity mechanism.

The unified implementation lives in raka_ufl_masked.py (wire message types +
MaskedClient) and raka_ufl_fl_server.py (masked resolution inside FLServer).
This module keeps the original names used by test_dynamic_pseudonym.py:
PrivacyClient (= MaskedClient), the WireHelloP family (= WireHello family),
and a minimal one-to-one demonstration front-end PrivacyServer.
"""

from __future__ import annotations

from typing import Optional

from raka_ufl_minimal import (
    ClientHello,
    MinimalServer,
    ModelUpdate,
    _identity_digest,
    _round_pad,
    _xor32,
    DIGEST_SIZE,
)
from raka_ufl_masked import (
    MaskedClient,
    WireHello,
    WireServerHello,
    WireUpdate,
    WireAck,
)

# Original names kept for the validation script and external references.
PrivacyClient = MaskedClient
WireHelloP = WireHello
WireServerHelloP = WireServerHello
WireUpdateP = WireUpdate
WireAckP = WireAck


def _make_hello_masked(self, timestamp_u: int, nonce_u: Optional[bytes] = None) -> WireHello:
    return self.make_hello_wire(timestamp_u, nonce_u)


MaskedClient.make_hello_masked = _make_hello_masked  # backward-compatible alias


class PrivacyServer:
    """One-to-one demonstration front-end with per-round pseudonym lookup.

    registry: real_id -> master_key. One MinimalServer endpoint per identity.
    (The many-to-one aggregation architecture lives in FLServer; this class
    remains as the minimal masked-resolution reference used by the tests.)"""

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
        wire: WireHello,
        now: Optional[int] = None,
        nonce_s: Optional[bytes] = None,
        timestamp_s: Optional[int] = None,
    ) -> WireServerHello:
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
        return WireServerHello(
            m2.task_id, wire.masked_id, m2.round_id, m2.timestamp_u,
            m2.timestamp_s, m2.nonce_u, m2.nonce_s, m2.transcript_id,
            m2.model_version, m2.model_hash, m2.tag,
        )

    def receive_wire_update(self, wire: WireUpdate) -> tuple[bytes, WireAck]:
        digest = self.resolve(wire.masked_id, wire.round_id)
        canonical = ModelUpdate(
            wire.task_id, digest.hex(), wire.round_id, wire.transcript_id,
            wire.model_version, wire.model_hash, wire.ciphertext,
        )
        plaintext, ack = self._servers[digest].receive_update(canonical)
        return plaintext, WireAck(
            ack.task_id, wire.masked_id, ack.round_id, ack.transcript_id,
            ack.payload_hash, ack.model_version, ack.model_hash, ack.tag,
        )

    def server_for(self, real_id: str) -> MinimalServer:
        return self._servers[self._digests[real_id]]
