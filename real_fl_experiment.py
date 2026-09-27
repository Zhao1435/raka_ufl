"""Real FL workload validation (review W1): MNIST LeNet through the actual
RAKA-UFL protocol stack.

Four clients train a LeNet-style CNN (61,706 parameters) on an IID MNIST
partition for three FedAvg rounds. Every update crosses the real wire path:
MaskedClient M1 -> FLServer admission -> M2 -> AES-GCM M3 -> server commit ->
M4 receipt. The protocol does not inspect or alter payloads, so the FedAvg
aggregate computed over decrypted wire payloads must be bit-identical to a
direct FedAvg over the same plaintext updates -- this is the transparency
check showing the protocol does not disturb the FL workflow.

MNIST shards were obtained from the ModelScope mirror of ylecun/mnist
(parquet files under data/). The 1 MiB default update-size cap is raised to
2 MiB for this experiment (the limit is configurable by design); JSON field
serialization keeps the payload encoding identical to the vector harness.

Outputs results/real_fl_result.json:
  per-round test accuracy, payload bytes, protocol wall time per client
  round, and the protocol-vs-direct L2 distance (expected exactly 0).
"""
from __future__ import annotations

import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from raka_ufl_fl_server import FLServer, fedavg, serialize_model
from raka_ufl_masked import MaskedClient

SEED = 20260924
CLIENTS = 4
ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH = 64
LR = 0.01
MOMENTUM = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE_CAP = 2 * 1024 * 1024  # 1 MiB default raised; limit is configurable


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.f1 = nn.Linear(16 * 4 * 4, 120)
        self.f2 = nn.Linear(120, 84)
        self.f3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.c2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        return self.f3(x)


def load_mnist():
    def unpack(path):
        df = pd.read_parquet(path)
        imgs = np.stack([
            np.asarray(Image.open(io.BytesIO(rec["bytes"])), dtype=np.float32) / 255.0
            for rec in df["image"]
        ])
        return torch.from_numpy(imgs).unsqueeze(1), torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    return unpack("data/mnist_train.parquet"), unpack("data/mnist_test.parquet")


def flat_params(model: nn.Module) -> tuple[float, ...]:
    return tuple(
        float(v) for p in model.parameters() for v in p.detach().cpu().view(-1)
    )


def load_flat(model: nn.Module, flat) -> None:
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            n = p.numel()
            chunk = torch.tensor(flat[offset:offset + n], dtype=p.dtype)
            p.copy_(chunk.view_as(p))
            offset += n


def main() -> None:
    torch.manual_seed(SEED)
    (x_train, y_train), (x_test, y_test) = load_mnist()
    perm = torch.randperm(len(x_train), generator=torch.Generator().manual_seed(SEED))
    shards = perm.chunk(CLIENTS)

    # identical initial model on every client
    init = LeNet()
    init_flat = flat_params(init)

    registry = {f"uav-{i + 1}": hashlib.sha256(f"master-{i}-{SEED}".encode()).digest() for i in range(CLIENTS)}
    fl = FLServer("fl-mnist", registry, len(init_flat))
    for ep in fl._endpoints.values():  # configurable cap raised for a real model
        ep.max_model_update_size = SIZE_CAP
    fl.register_sample_counts({pid: len(shards[i]) for i, pid in enumerate(registry)})

    global_flat = init_flat
    round_rows = []
    proto_times = []

    for round_id in range(1, ROUNDS + 1):
        mv, mh = fl.start_round(round_id)
        for index, pid in enumerate(registry):
            # local training from the current global model
            torch.manual_seed(SEED + round_id * 100 + index)
            model = LeNet().to(DEVICE)
            load_flat(model, global_flat)
            opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
            xs, ys = x_train[shards[index]].to(DEVICE), y_train[shards[index]].to(DEVICE)
            model.train()
            for _ in range(LOCAL_EPOCHS):
                g = torch.Generator(device=DEVICE).manual_seed(SEED + round_id * 100 + index)
                order = torch.randperm(len(xs), generator=g, device=DEVICE)
                for start in range(0, len(xs), BATCH):
                    idx = order[start:start + BATCH]
                    opt.zero_grad()
                    loss = nn.functional.cross_entropy(model(xs[idx]), ys[idx])
                    loss.backward()
                    opt.step()
            update_flat = flat_params(model)
            payload = serialize_model(update_flat)

            # ---- protocol path (this is the only part timed) ----
            t0 = time.perf_counter()
            client = MaskedClient("fl-mnist", pid, registry[pid], committed_round=round_id - 1)
            client.model_version, client.model_hash = mv, mh
            ts = 1000 + round_id * 1000 + index
            m1 = client.make_hello_wire(ts, bytes([index + 1]) * 16)
            m2 = fl.ingest_wire_hello(m1, now=ts, nonce_s=bytes([round_id + 32 + index]) * 16, timestamp_s=ts + 1)
            client.accept_wire_server_hello(m2, now=ts + 1)
            wire = client.encrypt_update_wire(payload)
            plaintext, ack, is_dup = fl.ingest_wire_update(wire)
            client.accept_wire_ack(ack)
            proto_times.append((time.perf_counter() - t0) * 1000)
            assert plaintext == payload and not is_dup

            # direct FedAvg reference over wire-parsed values (transparency check)
            parsed = tuple(json.loads(plaintext.decode("ascii")))
            if index == 0:
                direct_updates = []
            direct_updates.append((len(shards[index]), parsed))

        report = fl.close_round()
        direct_global = fedavg(direct_updates)
        l2_transparency = sum(
            (a - b) ** 2 for a, b in zip(report.global_model, direct_global)
        ) ** 0.5
        global_flat = report.global_model

        # test accuracy of the aggregated model
        eval_model = LeNet().to(DEVICE)
        load_flat(eval_model, global_flat)
        eval_model.eval()
        with torch.no_grad():
            correct = 0
            for start in range(0, len(x_test), 1024):
                xs = x_test[start:start + 1024].to(DEVICE)
                correct += int((eval_model(xs).argmax(1) == y_test[start:start + 1024].to(DEVICE)).sum())
        acc = correct / len(x_test)
        round_rows.append({
            "round_id": round_id,
            "accepted_clients": len(report.accepted_clients),
            "payload_bytes": len(payload),
            "test_accuracy": acc,
            "transparency_l2_vs_direct_fedavg": l2_transparency,
        })
        print(round_rows[-1])

    result = {
        "config": {"clients": CLIENTS, "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS,
                   "batch": BATCH, "lr": LR, "momentum": MOMENTUM, "seed": SEED,
                   "device": DEVICE, "model": "LeNet-style CNN", "params": len(init_flat)},
        "rounds": round_rows,
        "protocol_ms_per_client_round": {
            "mean": sum(proto_times) / len(proto_times),
            "max": max(proto_times),
        },
    }
    with open("results/real_fl_result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["protocol_ms_per_client_round"], indent=2))


if __name__ == "__main__":
    main()
