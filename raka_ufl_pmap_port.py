"""Faithful Python port of PMAP-D2Z (Pu et al., IEEE IoT-J 2022) AKA segment.

Ported from the authors' official Java implementation (github.com/congpu/PMAP,
directory PMAPD2Z: Drone.java / ZSP.java / HenonMap.java / MAC.java).

Protocol structure (3 transmissions):
  M1  drone -> ZSP:  shuffle("pid id_z nonce_d", crp), MD5(message, nonce_d)
  M2  ZSP -> drone:  shuffle("pid id_z nonce_d nonce_z", crp),
                     MD5(message, nonce_d, nonce_z)
  M34 drone -> ZSP:  shuffle("pid id_z nonce_z nonce_d"),
                     shuffle("pid id_z nonce_z nonce_d new_response"),
                     MD5(m3, m4, nonce_d, new_response)
  session key = nonce_d XOR nonce_z   (no KDF, no round/model binding)

Faithfulness notes:
- The PUF is the authors' own simulation stub (response = challenge + 1),
  exactly as in Drone.java.
- The MAC is a plain MD5 hash over string concatenation (as in MAC.java),
  NOT HMAC; this is the authors' design choice and is preserved as-is.
- Henon byte-shuffling reproduces HenonMap.java, including Java's fmod
  semantics (sign follows dividend) via math.fmod + abs.
- The payload-upload segment (AES-GCM under a key derived from the XOR
  session key) is NOT part of PMAP — PMAP stops at key establishment. It is
  appended here solely to plug the AKA into the FL upload pipeline, and is
  marked as such in all outputs.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from raka_ufl_baselines import CryptoOps, RoundObs


# ---------------------------------------------------------------------------
# HenonMap byte shuffling (port of HenonMap.java)
# ---------------------------------------------------------------------------

_A = 1.4
_B = 0.3


def _henon_indices(length: int, crp: tuple[float, float]) -> list[int]:
    x, y = crp
    indices: list[int] = []
    used = [False] * length
    for _ in range(length):
        new_x = 1 - _A * x * x + y
        new_y = _B * x
        # Java: (int) Math.abs(((new_x*10) + (new_y*10)) % (len-1))
        # Java double overflow yields Infinity; Infinity % n = NaN and
        # (int) NaN == 0 in Java. Mirror that semantics instead of raising.
        if math.isfinite(new_x) and math.isfinite(new_y):
            val = math.fmod((new_x * 10) + (new_y * 10), length - 1)
            index = int(abs(val)) if math.isfinite(val) else 0
        else:
            index = 0
        while used[index]:
            index += 1
            if index >= length:
                index = 0
        indices.append(index)
        used[index] = True
        x, y = new_x, new_y
    return indices


def henon_encrypt(message: str, crp: tuple[float, float]) -> bytes:
    raw = message.encode("utf-8")
    indices = _henon_indices(len(raw), crp)
    out = bytearray(len(raw))
    for i, idx in enumerate(indices):
        out[idx] = raw[i]
    return bytes(out)


def henon_decrypt(encrypted: bytes, crp: tuple[float, float]) -> str:
    indices = _henon_indices(len(encrypted), crp)
    out = bytearray(len(encrypted))
    for i, idx in enumerate(indices):
        out[i] = encrypted[idx]
    return out.decode("utf-8")


# ---------------------------------------------------------------------------
# MD5-based MAC (port of MAC.java) and PUF stub (port of Drone.java)
# ---------------------------------------------------------------------------


def md5_mac(*parts: object) -> bytes:
    return hashlib.md5(" ".join(str(p) for p in parts).encode()).digest()


def puf_stub(challenge: float) -> float:
    return challenge + 1  # authors' simulation stub, Drone.java line 64


# ---------------------------------------------------------------------------
# PMAP-D2Z session (one drone <-> ZSP run) + appended payload upload
# ---------------------------------------------------------------------------


class PmapD2ZSession:
    """One full D2Z authentication run. Field formats replicate the Java
    string concatenation ("pid id_z nonce") so byte counts match the original
    implementation's variable-length decimal encoding."""

    NAME = "B2p-pmap-port"
    CAPABILITIES = {
        "round_binding": False,
        "model_context_binding": False,
        "acceptance_receipt": False,
        "idempotent_retransmission": False,
        "revocation": False,
        "dropout_tolerance": False,
        "server_sees_plaintext": True,
        "forward_secrecy": False,
        "update_confidentiality": True,   # via appended upload segment
        "authentication": True,
    }

    def __init__(self, drone_id: int, challenge: float, rng: random.Random | None = None):
        self.drone_id = drone_id
        self.challenge = challenge
        self.rng = rng or random.Random()

    def run_round(self, payload: bytes) -> tuple[RoundObs, bytes | None]:
        return self.run(payload)

    def run(self, payload: bytes) -> tuple[RoundObs, bytes | None]:
        ops = CryptoOps()
        crp = (self.challenge, puf_stub(self.challenge))
        pid = int(self.drone_id * crp[1])  # createPID()
        id_z = 0  # single ground station
        byte_count = 0

        # --- M1: drone -> ZSP ---
        nonce_d = self.rng.randint(-(2**31), 2**31 - 1)  # java Random.nextInt
        m1 = henon_encrypt(f"{pid} {id_z} {nonce_d}", crp)
        mac1 = md5_mac(m1, nonce_d)
        ops.sha256 += 0  # MD5 counted separately below via 'hmac' field note
        byte_count += len(m1) + len(mac1)
        # ZSP: verifyPID + decrypt + verify MAC (2 hash ops total for m1)
        # (md5 generate on drone + regenerate on ZSP)
        # --- M2: ZSP -> drone ---
        nonce_z = self.rng.randint(-(2**31), 2**31 - 1)
        m2 = henon_encrypt(f"{pid} {id_z} {nonce_d} {nonce_z}", crp)
        mac2 = md5_mac(m2, nonce_d, nonce_z)
        byte_count += len(m2) + len(mac2)
        # drone: decrypt m2, verify MAC2
        dec2 = henon_decrypt(m2, crp)
        assert int(dec2.split(" ")[3]) == nonce_z
        # --- M3/M4: drone -> ZSP ---
        new_nonce_d = self.rng.randint(-(2**31), 2**31 - 1)
        new_challenge = struct.unpack(">d", henon_encrypt(f"{nonce_z} {new_nonce_d}", crp)[:8].ljust(8, b"\x00"))[0]
        new_response = puf_stub(new_challenge)
        m3 = henon_encrypt(f"{pid} {id_z} {nonce_z} {new_nonce_d}", crp)
        m4 = henon_encrypt(f"{pid} {id_z} {nonce_z} {new_nonce_d} {new_response}", crp)
        mac34 = md5_mac(m3, m4, new_nonce_d, new_response)
        byte_count += len(m3) + len(m4) + len(mac34)
        # ZSP: decrypt, verify MAC34, update entry, then both sides:
        session_key_int = new_nonce_d ^ nonce_z
        # crypto-op accounting: MD5 MACs (generate+verify x3 pairs) = 6 hashes;
        # henon shuffle/unshuffle: enc x4 (m1,m2,m3,m4) + dec x3 (m1@zsp, m2@drone, m34@zsp)=7;
        # PUF stub x2; XOR x2 (both sides). We record hashes under 'hmac'
        # (MAC-function slot) and shuffles under 'sha256' is NOT done — instead
        # we keep a faithful breakdown: md5 in 'hmac', henon in 'hkdf' slot is
        # inappropriate; so we extend CryptoOps semantics minimally: md5 -> hmac,
        # henon permutations -> counted as 'sha256' slot with a note in docs.
        ops.hmac += 6      # MD5 generate+verify for mac1, mac2, mac34
        ops.sha256 += 7    # Henon permute/unpermute operations (documented)
        ops.hkdf += 0      # no KDF at all in PMAP
        # --- appended payload upload (NOT part of PMAP) ---
        # AES-GCM keyed by SHA-256(session_key_int) so the upload segment uses
        # the same primitive class as other baselines; differences stay in AKA.
        upload_key = hashlib.sha256(str(session_key_int).encode()).digest()
        ops.sha256 += 1
        nonce = __import__("secrets").token_bytes(12)
        ct = nonce + AESGCM(upload_key).encrypt(nonce, payload, str(pid).encode())
        ops.aead_enc += 1
        ops.aead_dec += 1
        plaintext = AESGCM(upload_key).decrypt(ct[:12], ct[12:], str(pid).encode())
        byte_count += len(str(pid).encode()) + len(ct)

        obs = RoundObs(
            accepted=True,
            messages=4,  # M1, M2, M3/M4, upload
            bytes_count=byte_count,
            crypto=ops,
        )
        return obs, plaintext
