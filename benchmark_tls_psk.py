"""Real TLS 1.3 PSK baseline measurement via the OpenSSL CLI stack.

Motivation (review W3/W4): the paper's B1 was an abstract secure channel.
This script measures a REAL TLS 1.3 PSK handshake (openssl s_server /
s_client, TLS_AES_128_GCM_SHA256, session tickets disabled) over loopback,
with a Python TCP proxy logging every TCP chunk (direction, size, time).

Separation rule: chunks transferred before the payload send time belong to
the handshake phase; chunks at/after it are application-data records.
Handshake time = first client-byte -> last handshake-phase client byte
(client Finished), which excludes process-spawn overhead.

Note: openssl s_client (3.5.6) exits right after a PSK-only handshake on
Windows (session-print quirk), so application-data bytes are NOT counted
from the wire here; per-record AEAD framing overhead is a protocol
constant (22 bytes per RFC 8446 Section 5.2) and is stated as such.

Outputs results/tls_psk_result.json.
"""
from __future__ import annotations

import json
import secrets
import socket
import statistics
import subprocess
import threading
import time

OPENSSL = "openssl"
CIPHERSUITE = "TLS_AES_128_GCM_SHA256"
SERVER_PORT = 44331
PROXY_PORT = 44332
PAYLOAD = b"x" * 350  # approx. serialized 16-dim update size used in the harness
PAYLOAD_DELAY_S = 0.02
N_CONN = 200


def run_connection(psk_hex: str) -> dict | None:
    server = subprocess.Popen(
        [OPENSSL, "s_server", "-accept", str(SERVER_PORT), "-nocert",
         "-psk", psk_hex, "-ciphersuites", CIPHERSUITE, "-tls1_3",
         "-num_tickets", "0", "-naccept", "1", "-rev", "-quiet"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    chunks: list[tuple[float, str, int]] = []

    def pump(src: socket.socket, dst: socket.socket, key: str, done: threading.Event):
        try:
            while not done.is_set():
                data = src.recv(65536)
                if not data:
                    break
                chunks.append((time.perf_counter(), key, len(data)))
                dst.sendall(data)
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    proxy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    proxy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    proxy.bind(("127.0.0.1", PROXY_PORT))
    proxy.listen(1)
    proxy.settimeout(10)

    client = subprocess.Popen(
        [OPENSSL, "s_client", "-connect", f"127.0.0.1:{PROXY_PORT}",
         "-psk", psk_hex, "-ciphersuites", CIPHERSUITE, "-tls1_3", "-quiet"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    conn, _ = proxy.accept()
    upstream = None
    for _ in range(500):
        try:
            upstream = socket.create_connection(("127.0.0.1", SERVER_PORT), timeout=0.2)
            break
        except OSError:
            time.sleep(0.01)
    if upstream is None:
        server.kill()
        client.kill()
        return None
    done = threading.Event()
    threading.Thread(target=pump, args=(conn, upstream, "c2s", done), daemon=True).start()
    threading.Thread(target=pump, args=(upstream, conn, "s2c", done), daemon=True).start()

    time.sleep(PAYLOAD_DELAY_S)
    t_send = time.perf_counter()
    try:
        client.stdin.write(PAYLOAD + b"\n")
        client.stdin.flush()
    except OSError:
        done.set()
        client.kill()
        server.kill()
        conn.close()
        upstream.close()
        proxy.close()
        print("connection dropped (client exited before payload)")
        return None
    time.sleep(0.05)
    done.set()
    try:
        client.stdin.close()
    except OSError:
        pass
    try:
        client.wait(timeout=10)
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        client.kill()
        server.kill()
    conn.close()
    upstream.close()
    proxy.close()

    hs = [c for c in chunks if c[0] < t_send]
    app = [c for c in chunks if c[0] >= t_send]
    hs_c2s = [c for c in hs if c[1] == "c2s"]
    if not hs or not hs_c2s:
        return None
    import os
    if os.environ.get("TLS_DBG"):
        t0 = chunks[0][0]
        for t, k, n in chunks:
            print(f"  {(t - t0) * 1000:8.2f}ms {k} {n}B{' (app)' if t >= t_send else ''}")
    return {
        "hs_c2s_bytes": sum(c[2] for c in hs_c2s),
        "hs_s2c_bytes": sum(c[2] for c in hs if c[1] == "s2c"),
        "app_c2s_bytes": sum(c[2] for c in app if c[1] == "c2s"),
        "app_s2c_bytes": sum(c[2] for c in app if c[1] == "s2c"),
        "handshake_ms": (hs_c2s[-1][0] - hs_c2s[0][0]) * 1000,
    }


def main() -> None:
    psk_hex = secrets.token_hex(32)
    rows = []
    for i in range(N_CONN):
        r = run_connection(psk_hex)
        if r:
            rows.append(r)
        if i % 50 == 0:
            print(f"connection {i}: {r}")
    result = {
        "stack": subprocess.run([OPENSSL, "version"], capture_output=True, text=True).stdout.strip(),
        "ciphersuite": CIPHERSUITE,
        "session_tickets": "disabled (-num_tickets 0)",
        "payload_bytes": len(PAYLOAD),
        "connections_ok": len(rows),
        "connections_total": N_CONN,
        "handshake_c2s_bytes_mean": statistics.fmean(r["hs_c2s_bytes"] for r in rows),
        "handshake_s2c_bytes_mean": statistics.fmean(r["hs_s2c_bytes"] for r in rows),
        "handshake_total_bytes_mean": statistics.fmean(r["hs_c2s_bytes"] + r["hs_s2c_bytes"] for r in rows),
        "app_record_c2s_bytes_mean": statistics.fmean(r["app_c2s_bytes"] for r in rows),
        "handshake_ms_mean": statistics.fmean(r["handshake_ms"] for r in rows),
        "handshake_ms_std": statistics.stdev(r["handshake_ms"] for r in rows),
    }
    with open("results/tls_psk_result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
