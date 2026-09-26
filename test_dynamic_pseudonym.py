"""Validation for the dynamic-pseudonym extension (round-derived mask)."""

from __future__ import annotations

import hashlib
import secrets
import time

from raka_ufl_minimal import MinimalClient, MinimalServer
from raka_ufl_minimal_privacy import PrivacyClient, PrivacyServer, _identity_digest


def check(name: str, cond: bool) -> None:
    print(f"{'PASS' if cond else 'FAIL'}: {name}")
    if not cond:
        raise AssertionError(name)


def main() -> None:
    task = "fl-demo"
    registry = {f"uav-real-{i}": secrets.token_bytes(32) for i in range(4)}
    server = PrivacyServer(task, registry)

    # --- 1. full M1-M4 flow with dynamic pseudonym ---
    real_id = "uav-real-0"
    key = registry[real_id]
    client = PrivacyClient(task, real_id, key)
    inner_server = server.server_for(real_id)
    mv = "global-round-1"
    mh = hashlib.sha256(b"model-0").digest()
    client.model_version = mv
    client.model_hash = mh
    inner_server.model_version = mv
    inner_server.model_hash = mh
    ts = 1000
    w1 = client.make_hello_masked(ts, b"\x01" * 16)
    m2w = server.accept_wire_hello(w1, now=ts, nonce_s=b"\x02" * 16, timestamp_s=ts + 1)
    client.accept_wire_server_hello(m2w, now=ts + 1)
    update_w = client.encrypt_update_wire(b'[0.1, 0.2]')
    plaintext, ack_w = server.receive_wire_update(update_w)
    client.accept_wire_ack(ack_w)
    check("完整 M1-M4 流程（动态假名，全线遮蔽）", plaintext == b'[0.1, 0.2]' and client.committed_round == 1)

    # --- 2. unlinkability across rounds + digest never on the wire ---
    client2 = PrivacyClient(task, real_id, key, committed_round=1)
    client2.model_version = mv
    client2.model_hash = mh
    w2 = client2.make_hello_masked(2000, b"\x03" * 16)
    check("跨轮假名不同", w1.masked_id != w2.masked_id)
    check("跨轮恢复同一身份", server.resolve(w2.masked_id, 2) == _identity_digest(real_id))
    digest_hex = _identity_digest(real_id).hex().encode()
    wire_fields = bytes(w1.masked_id) + w1.nonce_u + m2w.masked_id + m2w.transcript_id + update_w.masked_id + update_w.ciphertext + ack_w.masked_id + ack_w.payload_hash
    check("身份摘要明文不出现在线上字段", _identity_digest(real_id) not in wire_fields and digest_hex not in wire_fields)

    # --- 3. forged masked id rejected before tag verification ---
    forged = type(w1)(task, secrets.token_bytes(32), 1, ts, b"\x01" * 16, mv, mh, w1.tag)
    try:
        server.accept_wire_hello(forged, now=ts, nonce_s=b"\x04" * 16, timestamp_s=ts + 1)
        check("伪造假名拒绝", False)
    except ValueError:
        check("伪造假名拒绝", True)

    # --- 4. cross-round replay of masked id rejected by table miss ---
    try:
        server.resolve(w1.masked_id, 2)  # round-1 pseudonym in round 2
        check("跨轮重放假名拒绝", False)
    except ValueError:
        check("跨轮重放假名拒绝", True)

    # --- 5. revoked (removed from registry) UAV rejected at resolution ---
    registry2 = {k: v for k, v in registry.items() if k != real_id}
    server2 = PrivacyServer(task, registry2)
    try:
        server2.resolve(w2.masked_id, 2)
        check("吊销后假名拒绝", False)
    except ValueError:
        check("吊销后假名拒绝", True)

    # --- 6. overhead comparison: fixed pid vs masked id on M1 ---
    fixed_client = MinimalClient(task, "uav-1", key)
    fixed_hello = fixed_client.make_hello(ts, b"\x01" * 16)
    fixed_m1_bytes = len(fixed_hello.pid.encode()) + len(fixed_hello.nonce_u) + len(fixed_hello.model_hash) + len(fixed_hello.tag) + len(fixed_hello.model_version.encode())
    dyn_m1_bytes = len(w1.masked_id) + len(w1.nonce_u) + len(w1.model_hash) + len(w1.tag) + len(w1.model_version.encode())
    print(f"M1 字段字节: 固定假名≈{fixed_m1_bytes}B  动态假名≈{dyn_m1_bytes}B  增量={dyn_m1_bytes - fixed_m1_bytes}B")

    # --- 7. server resolution cost: precompute vs per-message lookup ---
    n_fleet = 50
    big_registry = {f"uav-real-{i}": secrets.token_bytes(32) for i in range(n_fleet)}
    big_server = PrivacyServer(task, big_registry)
    start = time.perf_counter()
    big_server._refresh_table(2)  # per-round precomputation
    precompute_ms = (time.perf_counter() - start) * 1000
    probe_masked = next(iter(big_server._table.keys()))
    start = time.perf_counter()
    for _ in range(1000):
        big_server.resolve(probe_masked, 2)
    lookup_us = (time.perf_counter() - start) * 1000
    print(f"服务器定位（{n_fleet} 架机群）: 每轮预计算 {precompute_ms:.3f}ms（一次性），之后查表 {lookup_us:.3f}ms/1000 次")

    print("\nALL DYNAMIC-PSEUDONYM CHECKS PASSED")


if __name__ == "__main__":
    main()
