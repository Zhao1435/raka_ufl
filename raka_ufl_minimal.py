"""Minimal UAV-FL key agreement reference implementation.

The protocol uses one pre-shared master key for UAV/server authentication,
derives one session key per task round, and protects one model update with
AES-GCM. The server accepts a round only once and rejects stale rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict, deque
import hashlib
import hmac
import json
import secrets
import time
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


DOMAIN = b"RAKA-UFL/minimal-v1"
NONCE_U_SIZE = 16
NONCE_S_SIZE = 16
AEAD_NONCE_SIZE = 12
TAG_SIZE = 32
MODEL_HASH_SIZE = 32
DEFAULT_MODEL_VERSION = "model-v1"
DEFAULT_MODEL_HASH = hashlib.sha256(b"RAKA-UFL/default-model-v1").digest()
MAX_MODEL_UPDATE_SIZE = 1_048_576
MAX_HELLO_REQUESTS = 10
HELLO_WINDOW_SECONDS = 60.0
MAX_COMPLETED_RETRIES = 3


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hkdf(master_key: bytes, label: bytes, context: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=DOMAIN,
        info=DOMAIN + b"|" + label + b"|" + context,
    ).derive(master_key)


def _mac(key: bytes, body: bytes) -> bytes:
    return hmac.new(key, body, hashlib.sha256).digest()


def _context(task_id: str, pid: str, round_id: int) -> bytes:
    return _canonical([task_id, pid, round_id])


def _model_context(model_version: str, model_hash: bytes) -> bytes:
    return _canonical([model_version, model_hash.hex()])


def _auth_key(master_key: bytes, task_id: str, pid: str, round_id: int) -> bytes:
    return _hkdf(master_key, b"auth", _context(task_id, pid, round_id))


def _transcript_id(
    task_id: str,
    pid: str,
    round_id: int,
    timestamp_u: int,
    timestamp_s: int,
    nonce_u: bytes,
    nonce_s: bytes,
    model_version: str,
    model_hash: bytes,
    key: bytes | None = None,
) -> bytes:
    body = _canonical([
        DOMAIN.hex(),
        "transcript",
        task_id,
        pid,
        round_id,
        timestamp_u,
        timestamp_s,
        nonce_u.hex(),
        nonce_s.hex(),
        model_version,
        model_hash.hex(),
    ])
    # Optional keyed variant (HMAC): used by the dynamic-pseudonym extension,
    # where pid carries a secret identity digest and a public hash-only tid
    # would otherwise allow offline dictionary verification of the digest.
    if key is not None:
        return _mac(key, body)
    return hashlib.sha256(body).digest()


def _session_key(master_key: bytes, transcript_id: bytes) -> bytes:
    return _hkdf(master_key, b"session", transcript_id)


def _m1_body(message: "ClientHello") -> bytes:
    return _canonical([
        DOMAIN.hex(),
        "M1",
        "U2S",
        message.task_id,
        message.pid,
        message.round_id,
        message.timestamp_u,
        message.nonce_u.hex(),
        message.model_version,
        message.model_hash.hex(),
    ])


def _m2_body(message: "ServerHello") -> bytes:
    return _canonical([
        DOMAIN.hex(),
        "M2",
        "S2U",
        message.task_id,
        message.pid,
        message.round_id,
        message.timestamp_u,
        message.timestamp_s,
        message.nonce_u.hex(),
        message.nonce_s.hex(),
        message.transcript_id.hex(),
        message.model_version,
        message.model_hash.hex(),
    ])


def _upload_aad(
    task_id: str,
    pid: str,
    round_id: int,
    transcript_id: bytes,
    model_version: str,
    model_hash: bytes,
) -> bytes:
    return _canonical([
        DOMAIN.hex(),
        "M3",
        task_id,
        pid,
        round_id,
        transcript_id.hex(),
        model_version,
        model_hash.hex(),
    ])


def _ack_body(ack: "ServerAck") -> bytes:
    return _canonical([
        DOMAIN.hex(),
        "M4",
        "S2U",
        ack.task_id,
        ack.pid,
        ack.round_id,
        ack.transcript_id.hex(),
        ack.payload_hash.hex(),
        ack.model_version,
        ack.model_hash.hex(),
    ])


@dataclass(frozen=True)
class ClientHello:
    task_id: str
    pid: str
    round_id: int
    timestamp_u: int
    nonce_u: bytes
    model_version: str
    model_hash: bytes
    tag: bytes


@dataclass(frozen=True)
class ServerHello:
    task_id: str
    pid: str
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
class ModelUpdate:
    task_id: str
    pid: str
    round_id: int
    transcript_id: bytes
    model_version: str
    model_hash: bytes
    ciphertext: bytes


@dataclass(frozen=True)
class ServerAck:
    task_id: str
    pid: str
    round_id: int
    transcript_id: bytes
    payload_hash: bytes
    model_version: str
    model_hash: bytes
    tag: bytes


@dataclass(frozen=True)
class ClientSession:
    round_id: int
    transcript_id: bytes
    session_key: bytes
    model_version: str
    model_hash: bytes


@dataclass(frozen=True)
class ServerSession:
    round_id: int
    transcript_id: bytes
    session_key: bytes
    model_version: str
    model_hash: bytes


class MinimalClient:
    def __init__(
        self,
        task_id: str,
        pid: str,
        master_key: bytes,
        committed_round: int = 0,
        max_clock_skew: int = 30,
        model_version: str = DEFAULT_MODEL_VERSION,
        model_hash: bytes = DEFAULT_MODEL_HASH,
        max_model_update_size: int = MAX_MODEL_UPDATE_SIZE,
        transcript_key: bytes | None = None,
    ):
        _validate_master_key(master_key)
        if committed_round < 0 or max_clock_skew < 0 or max_model_update_size <= 0:
            raise ValueError("committed_round 和 max_clock_skew 无效")
        _validate_model_context(model_version, model_hash)
        self.task_id = task_id
        self.pid = pid
        self.master_key = master_key
        self.committed_round = committed_round
        self.max_clock_skew = max_clock_skew
        self.model_version = model_version
        self.model_hash = model_hash
        self.max_model_update_size = max_model_update_size
        self._transcript_key = transcript_key
        self._pending_hello: Optional[ClientHello] = None
        self._pending_session: Optional[ClientSession] = None
        self._pending_payload_hash: Optional[bytes] = None
        self._pending_update: Optional[ModelUpdate] = None

    def make_hello(self, timestamp_u: int, nonce_u: Optional[bytes] = None) -> ClientHello:
        if self._pending_hello is not None:
            return self._pending_hello
        nonce_u = nonce_u or secrets.token_bytes(NONCE_U_SIZE)
        if len(nonce_u) != NONCE_U_SIZE:
            raise ValueError("nonce_u 必须为 16 字节")
        message = ClientHello(
            self.task_id,
            self.pid,
            self.committed_round + 1,
            timestamp_u,
            nonce_u,
            self.model_version,
            self.model_hash,
            b"",
        )
        tag = _mac(_auth_key(self.master_key, self.task_id, self.pid, message.round_id), _m1_body(message))
        self._pending_hello = ClientHello(
            message.task_id,
            message.pid,
            message.round_id,
            message.timestamp_u,
            message.nonce_u,
            message.model_version,
            message.model_hash,
            tag,
        )
        return self._pending_hello

    def accept_server_hello(
        self,
        message: ServerHello,
        now: Optional[int] = None,
    ) -> ClientSession:
        hello = self._pending_hello
        if hello is None or message.task_id != self.task_id or message.pid != self.pid:
            raise ValueError("M2 与当前客户端事务不匹配")
        if message.round_id != hello.round_id or message.timestamp_u != hello.timestamp_u:
            raise ValueError("M2 轮次或时间戳不匹配")
        if message.nonce_u != hello.nonce_u:
            raise ValueError("M2 nonce_u 不匹配")
        if message.model_version != self.model_version or not hmac.compare_digest(message.model_hash, self.model_hash):
            raise ValueError("M2 模型上下文不匹配")
        _validate_nonce(message.nonce_s, "nonce_s")
        if len(message.transcript_id) != 32 or len(message.tag) != TAG_SIZE:
            raise ValueError("M2 完整性字段长度无效")
        if now is not None and abs(now - message.timestamp_s) > self.max_clock_skew:
            raise ValueError("M2 时间戳超出允许时钟偏差")
        expected_tid = _transcript_id(
            self.task_id,
            self.pid,
            message.round_id,
            message.timestamp_u,
            message.timestamp_s,
            message.nonce_u,
            message.nonce_s,
            message.model_version,
            message.model_hash,
            key=self._transcript_key,
        )
        if not hmac.compare_digest(expected_tid, message.transcript_id):
            raise ValueError("M2 transcript_id 校验失败")
        auth_key = _auth_key(self.master_key, self.task_id, self.pid, message.round_id)
        if not hmac.compare_digest(_mac(auth_key, _m2_body(message)), message.tag):
            raise ValueError("M2 服务端认证失败")
        session = ClientSession(
            message.round_id,
            message.transcript_id,
            _session_key(self.master_key, message.transcript_id),
            message.model_version,
            message.model_hash,
        )
        self._pending_session = session
        return session

    def encrypt_update(self, model_update: bytes) -> ModelUpdate:
        if self._pending_session is None:
            raise RuntimeError("尚未完成 M2 验证")
        if self._pending_update is not None:
            return self._pending_update
        if not isinstance(model_update, bytes) or len(model_update) > self.max_model_update_size:
            raise ValueError("模型更新超过最大允许尺寸")
        nonce = secrets.token_bytes(AEAD_NONCE_SIZE)
        ciphertext = nonce + AESGCM(self._pending_session.session_key).encrypt(
            nonce,
            model_update,
            _upload_aad(
                self.task_id,
                self.pid,
                self._pending_session.round_id,
                self._pending_session.transcript_id,
                self._pending_session.model_version,
                self._pending_session.model_hash,
            ),
        )
        self._pending_payload_hash = hashlib.sha256(ciphertext).digest()
        self._pending_update = ModelUpdate(
            self.task_id,
            self.pid,
            self._pending_session.round_id,
            self._pending_session.transcript_id,
            self._pending_session.model_version,
            self._pending_session.model_hash,
            ciphertext,
        )
        return self._pending_update

    def accept_server_ack(self, ack: ServerAck) -> None:
        if self._pending_session is None:
            raise RuntimeError("没有等待确认的会话")
        if (
            ack.task_id != self.task_id
            or ack.pid != self.pid
            or ack.round_id != self._pending_session.round_id
            or ack.transcript_id != self._pending_session.transcript_id
            or ack.model_version != self._pending_session.model_version
            or ack.model_hash != self._pending_session.model_hash
        ):
            raise ValueError("M4 与当前会话不匹配")
        if len(ack.payload_hash) != 32 or len(ack.tag) != TAG_SIZE:
            raise ValueError("M4 完整性字段长度无效")
        if self._pending_payload_hash is None:
            raise RuntimeError("没有待确认的模型更新")
        if not hmac.compare_digest(ack.payload_hash, self._pending_payload_hash):
            raise ValueError("M4 未绑定当前模型更新")
        session_key = self._pending_session.session_key
        if not hmac.compare_digest(_mac(session_key, _ack_body(ack)), ack.tag):
            raise ValueError("M4 服务端确认失败")
        self.committed_round = ack.round_id
        self._pending_hello = None
        self._pending_session = None
        self._pending_payload_hash = None
        self._pending_update = None


class MinimalServer:
    def __init__(
        self,
        task_id: str,
        pid: str,
        master_key: bytes,
        committed_round: int = 0,
        max_clock_skew: int = 30,
        completed_cache_ttl: float = 600.0,
        completed_cache_limit: int = 1024,
        model_version: str = DEFAULT_MODEL_VERSION,
        model_hash: bytes = DEFAULT_MODEL_HASH,
        max_model_update_size: int = MAX_MODEL_UPDATE_SIZE,
        max_hello_requests: int = MAX_HELLO_REQUESTS,
        hello_window_seconds: float = HELLO_WINDOW_SECONDS,
        max_completed_retries: int = MAX_COMPLETED_RETRIES,
        revoked: bool = False,
        transcript_key: bytes | None = None,
    ):
        _validate_master_key(master_key)
        if (
            committed_round < 0
            or max_clock_skew < 0
            or completed_cache_ttl <= 0
            or completed_cache_limit <= 0
            or max_model_update_size <= 0
            or max_hello_requests <= 0
            or hello_window_seconds <= 0
            or max_completed_retries < 0
        ):
            raise ValueError("服务器状态或完成缓存参数无效")
        _validate_model_context(model_version, model_hash)
        self.task_id = task_id
        self.pid = pid
        self.master_key = master_key
        self.committed_round = committed_round
        self.max_clock_skew = max_clock_skew
        self.completed_cache_ttl = completed_cache_ttl
        self.completed_cache_limit = completed_cache_limit
        self.model_version = model_version
        self.model_hash = model_hash
        self.max_model_update_size = max_model_update_size
        self.max_hello_requests = max_hello_requests
        self.hello_window_seconds = hello_window_seconds
        self.max_completed_retries = max_completed_retries
        self.revoked = revoked
        self._transcript_key = transcript_key
        self._hello_requests: deque[float] = deque()
        self._completed_retries: dict[bytes, int] = {}
        self._pending_hello: Optional[ClientHello] = None
        self._pending_server_hello: Optional[ServerHello] = None
        self._pending_session: Optional[ServerSession] = None
        self._completed: OrderedDict[bytes, tuple[int, bytes, bytes, ServerAck, float]] = OrderedDict()

    def _prune_completed(self, now: float) -> None:
        expired = [
            transcript_id
            for transcript_id, (_, _, _, _, completed_at) in self._completed.items()
            if now - completed_at >= self.completed_cache_ttl
        ]
        for transcript_id in expired:
            self._completed.pop(transcript_id, None)
            self._completed_retries.pop(transcript_id, None)
        while len(self._completed) > self.completed_cache_limit:
            evicted_id, _ = self._completed.popitem(last=False)
            self._completed_retries.pop(evicted_id, None)

    def accept_client_hello(
        self,
        message: ClientHello,
        now: Optional[int] = None,
        nonce_s: Optional[bytes] = None,
        timestamp_s: Optional[int] = None,
    ) -> ServerHello:
        if self.revoked:
            raise ValueError("UAV 已被服务器吊销")
        if message.task_id != self.task_id or message.pid != self.pid:
            raise ValueError("M1 身份或任务不匹配")
        if message.model_version != self.model_version or not hmac.compare_digest(message.model_hash, self.model_hash):
            raise ValueError("M1 模型上下文不匹配")
        if message.round_id != self.committed_round + 1:
            raise ValueError("M1 轮次已过期或跳跃")
        _validate_nonce(message.nonce_u, "nonce_u")
        if len(message.tag) != TAG_SIZE:
            raise ValueError("M1 认证字段长度无效")
        if now is not None and abs(now - message.timestamp_u) > self.max_clock_skew:
            raise ValueError("M1 时间戳超出允许时钟偏差")
        auth_key = _auth_key(self.master_key, self.task_id, self.pid, message.round_id)
        if not hmac.compare_digest(_mac(auth_key, _m1_body(message)), message.tag):
            raise ValueError("M1 无人机认证失败")
        monotonic_now = time.monotonic()
        while self._hello_requests and monotonic_now - self._hello_requests[0] >= self.hello_window_seconds:
            self._hello_requests.popleft()
        if len(self._hello_requests) >= self.max_hello_requests and self._pending_hello != message:
            raise ValueError("M1 请求频率超过限制")
        if self._pending_hello != message:
            self._hello_requests.append(monotonic_now)
        if self._pending_hello == message and self._pending_server_hello is not None:
            return self._pending_server_hello
        if self._pending_hello is not None:
            raise ValueError("已有未完成的当前轮次事务")
        nonce_s = nonce_s or secrets.token_bytes(NONCE_S_SIZE)
        _validate_nonce(nonce_s, "nonce_s")
        if timestamp_s is None:
            raise ValueError("timestamp_s 必须显式提供")
        transcript_id = _transcript_id(
            self.task_id,
            self.pid,
            message.round_id,
            message.timestamp_u,
            timestamp_s,
            message.nonce_u,
            nonce_s,
            message.model_version,
            message.model_hash,
            key=self._transcript_key,
        )
        unsigned = ServerHello(
            self.task_id,
            self.pid,
            message.round_id,
            message.timestamp_u,
            timestamp_s,
            message.nonce_u,
            nonce_s,
            transcript_id,
            message.model_version,
            message.model_hash,
            b"",
        )
        server_hello = ServerHello(
            unsigned.task_id,
            unsigned.pid,
            unsigned.round_id,
            unsigned.timestamp_u,
            unsigned.timestamp_s,
            unsigned.nonce_u,
            unsigned.nonce_s,
            unsigned.transcript_id,
            unsigned.model_version,
            unsigned.model_hash,
            _mac(auth_key, _m2_body(unsigned)),
        )
        self._pending_hello = message
        self._pending_server_hello = server_hello
        self._pending_session = ServerSession(
            message.round_id,
            transcript_id,
            _session_key(self.master_key, transcript_id),
            message.model_version,
            message.model_hash,
        )
        return server_hello

    def receive_update(self, update: ModelUpdate) -> tuple[bytes, ServerAck]:
        current = time.monotonic()
        self._prune_completed(current)
        if update.task_id != self.task_id or update.pid != self.pid:
            raise ValueError("模型更新身份或任务不匹配")
        ciphertext_hash = hashlib.sha256(update.ciphertext).digest()
        completed = self._completed.get(update.transcript_id)
        if completed is not None:
            self._completed.move_to_end(update.transcript_id)
            if (
                completed[0] != update.round_id
                or not hmac.compare_digest(completed[1], ciphertext_hash)
            ):
                raise ValueError("重复模型更新与已完成事务不匹配")
            retries = self._completed_retries.get(update.transcript_id, 0)
            if retries >= self.max_completed_retries:
                raise ValueError("完成事务重传次数超过限制")
            self._completed_retries[update.transcript_id] = retries + 1
            return completed[2], completed[3]
        session = self._pending_session
        server_hello = self._pending_server_hello
        if session is None or server_hello is None:
            raise ValueError("没有等待中的会话")
        if (
            update.round_id != session.round_id
            or update.transcript_id != session.transcript_id
            or update.model_version != session.model_version
            or update.model_hash != session.model_hash
        ):
            raise ValueError("模型更新与当前会话不匹配")
        if len(update.ciphertext) <= AEAD_NONCE_SIZE or len(update.ciphertext) > self.max_model_update_size + AEAD_NONCE_SIZE + 16:
            raise ValueError("模型更新密文长度无效")
        try:
            plaintext = AESGCM(session.session_key).decrypt(
                update.ciphertext[:AEAD_NONCE_SIZE],
                update.ciphertext[AEAD_NONCE_SIZE:],
                _upload_aad(
                    server_hello.task_id,
                    server_hello.pid,
                    server_hello.round_id,
                    server_hello.transcript_id,
                    server_hello.model_version,
                    server_hello.model_hash,
                ),
            )
        except (InvalidTag, ValueError) as exc:
            raise ValueError("模型更新认证失败") from exc
        payload_hash = ciphertext_hash
        unsigned_ack = ServerAck(
            self.task_id,
            self.pid,
            update.round_id,
            update.transcript_id,
            payload_hash,
            update.model_version,
            update.model_hash,
            b"",
        )
        ack = ServerAck(
            unsigned_ack.task_id,
            unsigned_ack.pid,
            unsigned_ack.round_id,
            unsigned_ack.transcript_id,
            unsigned_ack.payload_hash,
            unsigned_ack.model_version,
            unsigned_ack.model_hash,
            _mac(session.session_key, _ack_body(unsigned_ack)),
        )
        self.committed_round = update.round_id
        self._completed[update.transcript_id] = (
            update.round_id,
            payload_hash,
            plaintext,
            ack,
            current,
        )
        self._completed_retries[update.transcript_id] = 0
        self._prune_completed(current)
        self._pending_hello = None
        self._pending_server_hello = None
        self._pending_session = None
        return plaintext, ack


def _validate_master_key(master_key: bytes) -> None:
    if not isinstance(master_key, bytes) or len(master_key) < 16:
        raise ValueError("master_key 至少需要 16 字节")


def _validate_nonce(nonce: bytes, name: str) -> None:
    if not isinstance(nonce, bytes) or len(nonce) != NONCE_U_SIZE:
        raise ValueError(f"{name} 必须为 16 字节")


def _validate_model_context(model_version: str, model_hash: bytes) -> None:
    if not isinstance(model_version, str) or not model_version or len(model_version.encode("utf-8")) > 128:
        raise ValueError("model_version 必须为 1 至 128 字节")
    if not isinstance(model_hash, bytes) or len(model_hash) != MODEL_HASH_SIZE:
        raise ValueError("model_hash 必须为 32 字节 SHA-256")


# ---------------------------------------------------------------------------
# Round-derived masked identity primitives (standard wire identity form)
# ---------------------------------------------------------------------------

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
