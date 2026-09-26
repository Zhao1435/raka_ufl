import unittest
from dataclasses import replace

from raka_ufl_minimal import MinimalClient, MinimalServer


class MinimalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.master_key = b"master-key-for-uav-fl" * 2
        self.client = MinimalClient("task-1", "uav-1", self.master_key)
        self.server = MinimalServer("task-1", "uav-1", self.master_key)

    def establish(self):
        m1 = self.client.make_hello(100, b"u" * 16)
        m2 = self.server.accept_client_hello(
            m1,
            now=100,
            nonce_s=b"s" * 16,
            timestamp_s=101,
        )
        client_session = self.client.accept_server_hello(m2, now=101)
        return m1, m2, client_session

    def test_both_parties_derive_same_round_key(self):
        _, m2, client_session = self.establish()
        self.assertIsNotNone(self.server._pending_session)
        assert self.server._pending_session is not None
        self.assertEqual(client_session.session_key, self.server._pending_session.session_key)
        self.assertEqual(client_session.transcript_id, m2.transcript_id)

    def test_model_update_is_authenticated_and_server_returns_ack(self):
        _, _, _ = self.establish()
        update = self.client.encrypt_update(b"model-update")
        plaintext, ack = self.server.receive_update(update)
        self.assertEqual(plaintext, b"model-update")
        self.client.accept_server_ack(ack)
        self.assertEqual(self.client.committed_round, 1)
        self.assertEqual(self.server.committed_round, 1)

    def test_tampered_server_hello_is_rejected(self):
        _, m2, _ = self.establish()
        with self.assertRaises(ValueError):
            self.client.accept_server_hello(replace(m2, tag=b"x" * 32))
        self.assertEqual(self.client.committed_round, 0)

    def test_tampered_model_update_is_rejected(self):
        self.establish()
        update = self.client.encrypt_update(b"model-update")
        tampered = replace(
            update,
            ciphertext=update.ciphertext[:-1] + bytes([update.ciphertext[-1] ^ 1]),
        )
        with self.assertRaises(ValueError):
            self.server.receive_update(tampered)
        self.assertEqual(self.server.committed_round, 0)

    def test_model_update_transcript_is_bound_to_aad(self):
        self.establish()
        update = self.client.encrypt_update(b"model-update")
        changed = replace(update, transcript_id=b"x" * 32)
        with self.assertRaises(ValueError):
            self.server.receive_update(changed)
        self.assertEqual(self.server.committed_round, 0)

    def test_ack_must_bind_the_sent_ciphertext(self):
        self.establish()
        update = self.client.encrypt_update(b"model-update")
        _, ack = self.server.receive_update(update)
        with self.assertRaises(ValueError):
            self.client.accept_server_ack(replace(ack, payload_hash=b"x" * 32))
        self.assertEqual(self.client.committed_round, 0)

    def test_replayed_round_is_rejected_after_commit(self):
        m1, _, _ = self.establish()
        update = self.client.encrypt_update(b"model-update")
        _, ack = self.server.receive_update(update)
        self.client.accept_server_ack(ack)
        with self.assertRaises(ValueError):
            self.server.accept_client_hello(
                m1,
                now=102,
                nonce_s=b"t" * 16,
                timestamp_s=103,
            )

    def test_cross_task_message_is_rejected(self):
        m1 = self.client.make_hello(100, b"u" * 16)
        other_server = MinimalServer("other-task", "uav-1", self.master_key)
        with self.assertRaises(ValueError):
            other_server.accept_client_hello(m1, now=100, nonce_s=b"s" * 16, timestamp_s=101)

    def test_m1_replay_is_idempotent_before_update(self):
        m1 = self.client.make_hello(100, b"u" * 16)
        first = self.server.accept_client_hello(
            m1,
            now=100,
            nonce_s=b"s" * 16,
            timestamp_s=101,
        )
        second = self.server.accept_client_hello(
            m1,
            now=100,
            nonce_s=b"x" * 16,
            timestamp_s=102,
        )
        self.assertEqual(first, second)

    def test_model_hash_is_bound_to_handshake_and_upload(self):
        model_hash = b"h" * 32
        client = MinimalClient("task-1", "uav-1", self.master_key, model_version="v7", model_hash=model_hash)
        server = MinimalServer("task-1", "uav-1", self.master_key, model_version="v7", model_hash=model_hash)
        hello = client.make_hello(100, b"u" * 16)
        response = server.accept_client_hello(hello, now=100, nonce_s=b"s" * 16, timestamp_s=101)
        client.accept_server_hello(response, now=101)
        update = client.encrypt_update(b"weights")
        self.assertEqual((update.model_version, update.model_hash), ("v7", model_hash))
        self.assertEqual(server.receive_update(update)[0], b"weights")

    def test_server_rejects_wrong_model_hash_before_session(self):
        hello = self.client.make_hello(100, b"u" * 16)
        with self.assertRaisesRegex(ValueError, "模型上下文"):
            self.server.accept_client_hello(
                replace(hello, model_hash=b"x" * 32),
                now=100,
                nonce_s=b"s" * 16,
                timestamp_s=101,
            )
        self.assertIsNone(self.server._pending_session)

    def test_server_rejects_revoked_device_at_admission(self):
        revoked_server = MinimalServer("task-1", "uav-1", self.master_key, revoked=True)
        hello = self.client.make_hello(100, b"u" * 16)
        with self.assertRaisesRegex(ValueError, "吊销"):
            revoked_server.accept_client_hello(
                hello,
                now=100,
                nonce_s=b"s" * 16,
                timestamp_s=101,
            )

    def test_server_limits_model_update_size(self):
        client = MinimalClient("task-1", "uav-1", self.master_key, max_model_update_size=4)
        server = MinimalServer("task-1", "uav-1", self.master_key, max_model_update_size=4)
        hello = client.make_hello(100, b"u" * 16)
        response = server.accept_client_hello(hello, now=100, nonce_s=b"s" * 16, timestamp_s=101)
        client.accept_server_hello(response, now=101)
        with self.assertRaisesRegex(ValueError, "尺寸"):
            client.encrypt_update(b"12345")

    def test_server_limits_hello_rate_and_completed_retries(self):
        limited = MinimalServer(
            "task-1", "uav-1", self.master_key,
            max_hello_requests=1, max_completed_retries=1,
        )
        first = self.client.make_hello(100, b"u" * 16)
        other_client = MinimalClient("task-1", "uav-1", self.master_key)
        second = other_client.make_hello(101, b"v" * 16)
        response = limited.accept_client_hello(first, now=100, nonce_s=b"s" * 16, timestamp_s=101)
        with self.assertRaisesRegex(ValueError, "请求频率"):
            limited.accept_client_hello(second, now=101, nonce_s=b"t" * 16, timestamp_s=102)
        self.client.accept_server_hello(response, now=101)
        update = self.client.encrypt_update(b"weights")
        _, ack = limited.receive_update(update)
        self.assertEqual(limited.receive_update(update)[1], ack)
        with self.assertRaisesRegex(ValueError, "重传次数"):
            limited.receive_update(update)


if __name__ == "__main__":
    unittest.main()
