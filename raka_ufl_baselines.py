"""Comparison baselines for the RAKA-UFL protocol evaluation.

Implements three transport-protection baselines that run the SAME federated
learning task as the RAKA-UFL M1-M4 protocol, differing only in how the model
update is protected in transit:

- B0 PlainChannel: plaintext upload, no protection at all.
- B1 SecureChannelPSK: abstract secure channel with TLS 1.3 PSK-equivalent
  semantics (mutual authentication via PSK-tagged nonces, HKDF session key,
  AES-GCM with per-connection sequence nonces). NOT a real TLS stack; the
  channel has no FL-round or model-context semantics.
- B2 CompositeAKA: composite baseline abstracting the shared skeleton of
  published UAV/IoD AKA protocols (liteA4/PMAP family): a two-message
  PSK+nonce mutual authentication deriving a session key (bound to nonces
  only, not to rounds or model context), then AES-GCM upload with the
  pseudonym as associated data. No acceptance receipt, no idempotent cache.

Measurement conventions:
- bytes: approximate protocol-field + payload byte count (no transport
  headers), matching the accounting style of raka_ufl_experiment.py.
- crypto_ops: total logical cryptographic operations per client per round,
  counting BOTH parties (generation and verification counted separately).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def _hkdf(master_key: bytes, info: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"baseline",
        info=info,
    ).derive(master_key)


def _mac(key: bytes, body: bytes) -> bytes:
    return hmac.new(key, body, hashlib.sha256).digest()


@dataclass
class CryptoOps:
    hkdf: int = 0
    hmac: int = 0
    aead_enc: int = 0
    aead_dec: int = 0
    sha256: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hkdf": self.hkdf,
            "hmac": self.hmac,
            "aead_enc": self.aead_enc,
            "aead_dec": self.aead_dec,
            "sha256": self.sha256,
        }

    def total(self) -> int:
        return sum(self.as_dict().values())


@dataclass
class RoundObs:
    """Per-client per-round observation for one scheme."""

    accepted: bool
    messages: int
    bytes_count: int
    crypto: CryptoOps = field(default_factory=CryptoOps)


# ---------------------------------------------------------------------------
# B0: plaintext channel
# ---------------------------------------------------------------------------


class PlainChannel:
    """B0: model updates are uploaded as cleartext; the server aggregates
    whatever it receives. One message per client per round."""

    NAME = "B0-plain"
    CAPABILITIES = {
        "round_binding": False,
        "model_context_binding": False,
        "acceptance_receipt": False,
        "idempotent_retransmission": False,
        "revocation": False,
        "dropout_tolerance": False,
        "server_sees_plaintext": True,
        "forward_secrecy": False,
        "update_confidentiality": False,
        "authentication": False,
    }

    def __init__(self, pid: str):
        self.pid = pid

    def upload(self, payload: bytes) -> tuple[RoundObs, bytes]:
        wire = self.pid.encode("utf-8") + b"|" + payload
        obs = RoundObs(accepted=True, messages=1, bytes_count=len(wire))
        return obs, payload  # server gets plaintext directly

    def run_round(self, payload: bytes) -> tuple[RoundObs, bytes | None]:
        return self.upload(payload)


# ---------------------------------------------------------------------------
# B1: abstract secure channel (TLS 1.3 PSK-equivalent semantics)
# ---------------------------------------------------------------------------


class SecureChannelPSK:
    """B1: generic secure channel. Both sides authenticate via PSK-tagged
    nonces and derive a session key that binds ONLY the nonces (no FL round,
    no model context). Updates are AES-GCM protected with per-connection
    sequence nonces (TLS record semantics). The server accepts any
    well-formed ciphertext from an authenticated peer; there is no acceptance
    receipt and no idempotent cache."""

    NAME = "B1-secure-channel"
    CAPABILITIES = {
        "round_binding": False,
        "model_context_binding": False,
        "acceptance_receipt": False,
        "idempotent_retransmission": False,
        "revocation": False,
        "dropout_tolerance": False,
        "server_sees_plaintext": True,
        "forward_secrecy": False,  # pure-PSK mode, no (EC)DHE
        "update_confidentiality": True,
        "authentication": True,
    }

    def __init__(self, pid: str, master_key: bytes):
        self.pid = pid
        self.master_key = master_key

    def run_round(self, payload: bytes) -> tuple[RoundObs, bytes | None]:
        ops = CryptoOps()
        # --- handshake (2 messages) ---
        nonce_c = secrets.token_bytes(16)
        nonce_s = secrets.token_bytes(16)
        auth_key = _hkdf(self.master_key, b"b1|auth")
        ops.hkdf += 2  # both parties derive the auth key
        tag_c = _mac(auth_key, b"b1|hello|" + self.pid.encode() + nonce_c)
        ops.hmac += 2  # client generates, server verifies
        tag_s = _mac(auth_key, b"b1|resp|" + self.pid.encode() + nonce_c + nonce_s)
        ops.hmac += 2
        session_key = _hkdf(self.master_key, b"b1|session|" + nonce_c + nonce_s)
        ops.hkdf += 2  # both parties derive the session key
        handshake_bytes = (
            len(self.pid.encode()) + len(nonce_c) + len(tag_c)
            + len(nonce_s) + len(tag_s)
        )
        # --- upload (1 message) ---
        # Each run_round models a FRESH connection (new handshake, new key),
        # matching per-round TLS usage. TLS record sequence numbers protect
        # ordering/replay only WITHIN one connection; an application-level
        # resend after a lost ACK is a NEW connection carrying the same
        # update, which the server has no idempotency key to recognize.
        nonce = secrets.token_bytes(12)
        ct = nonce + AESGCM(session_key).encrypt(nonce, payload, self.pid.encode())
        ops.aead_enc += 1
        wire = self.pid.encode() + ct
        ops.aead_dec += 1
        plaintext = AESGCM(session_key).decrypt(ct[:12], ct[12:], self.pid.encode())
        obs = RoundObs(
            accepted=True,
            messages=3,
            bytes_count=handshake_bytes + len(wire),
            crypto=ops,
        )
        return obs, plaintext


# ---------------------------------------------------------------------------
# B2: composite AKA baseline (liteA4/PMAP-family skeleton, PSK variant)
# ---------------------------------------------------------------------------


class CompositeAKA:
    """B2: two-message mutual AKA deriving a session key bound to nonces
    only, then one AES-GCM upload with the pseudonym as associated data.
    The server aggregates upon successful decryption: no round check, no
    model-context check, no acceptance receipt, no duplicate cache."""

    NAME = "B2-composite-aka"
    CAPABILITIES = {
        "round_binding": False,
        "model_context_binding": False,
        "acceptance_receipt": False,
        "idempotent_retransmission": False,
        "revocation": False,
        "dropout_tolerance": False,
        "server_sees_plaintext": True,
        "forward_secrecy": False,
        "update_confidentiality": True,
        "authentication": True,
    }

    def __init__(self, pid: str, master_key: bytes):
        self.pid = pid
        self.master_key = master_key

    def run_round(self, payload: bytes) -> tuple[RoundObs, bytes | None]:
        ops = CryptoOps()
        auth_key = _hkdf(self.master_key, b"b2|auth|" + self.pid.encode())
        ops.hkdf += 2  # both parties derive the per-identity auth key
        # --- AKE message 1: client -> server ---
        nonce_c = secrets.token_bytes(16)
        tag1 = _mac(auth_key, b"b2|m1|" + self.pid.encode() + nonce_c)
        ops.hmac += 2  # generate + verify
        # --- AKE message 2: server -> client ---
        nonce_s = secrets.token_bytes(16)
        tag2 = _mac(auth_key, b"b2|m2|" + self.pid.encode() + nonce_c + nonce_s)
        ops.hmac += 2
        session_key = _hkdf(self.master_key, b"b2|session|" + nonce_c + nonce_s)
        ops.hkdf += 2
        ake_bytes = (
            len(self.pid.encode()) + len(nonce_c) + len(tag1)
            + len(nonce_s) + len(tag2)
        )
        # --- upload: AES-GCM, AAD binds pid only ---
        nonce_aead = secrets.token_bytes(12)
        ct = nonce_aead + AESGCM(session_key).encrypt(nonce_aead, payload, self.pid.encode())
        ops.aead_enc += 1
        ops.aead_dec += 1
        wire = self.pid.encode() + ct
        plaintext = AESGCM(session_key).decrypt(
            ct[:12], ct[12:], self.pid.encode()
        )
        obs = RoundObs(
            accepted=True,
            messages=3,
            bytes_count=ake_bytes + len(wire),
            crypto=ops,
        )
        return obs, plaintext
