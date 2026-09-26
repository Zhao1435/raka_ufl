import unittest

from raka_ufl_experiment import ExperimentConfig, MultiUAVExperiment, _fedavg, run_experiment


class ExperimentTests(unittest.TestCase):
    def test_fedavg_uses_sample_count_weights(self):
        result = _fedavg([(1, (0.0, 2.0)), (3, (4.0, 6.0))])
        self.assertEqual(result, (3.0, 5.0))

    def test_multi_uav_rounds_isolate_device_state(self):
        result = run_experiment(ExperimentConfig(clients=3, rounds=2, vector_size=4, seed=7))
        self.assertEqual(len(result.rounds), 2)
        for metrics in result.rounds:
            self.assertEqual(metrics.attempted_clients, 3)
            self.assertEqual(metrics.accepted_clients, 3)
            self.assertEqual(metrics.revoked_clients, 0)
            self.assertEqual(metrics.duplicate_uploads, 3)
            self.assertEqual(metrics.total_messages, 15)
            self.assertGreater(metrics.total_bytes, 0)
        self.assertTrue(any(value != 0.0 for value in result.final_model))

    def test_revoked_uav_is_rejected_every_round(self):
        result = run_experiment(ExperimentConfig(
            clients=4,
            rounds=2,
            vector_size=4,
            revoked_client=1,
            seed=11,
        ))
        for metrics in result.rounds:
            self.assertEqual(metrics.attempted_clients, 4)
            self.assertEqual(metrics.accepted_clients, 3)
            self.assertEqual(metrics.revoked_clients, 1)
            self.assertEqual(metrics.duplicate_uploads, 3)

    def test_repeated_seed_reproduces_model_and_metrics(self):
        config = ExperimentConfig(clients=3, rounds=2, vector_size=5, seed=19)
        first = run_experiment(config)
        second = run_experiment(config)
        self.assertEqual(first.final_model, second.final_model)
        self.assertEqual(
            [(item.accepted_clients, item.total_messages, item.total_bytes) for item in first.rounds],
            [(item.accepted_clients, item.total_messages, item.total_bytes) for item in second.rounds],
        )

    def test_invalid_revocation_index_is_rejected(self):
        with self.assertRaises(ValueError):
            MultiUAVExperiment(ExperimentConfig(clients=2, revoked_client=2))


if __name__ == "__main__":
    unittest.main()
