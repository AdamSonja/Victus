import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import socketio


class LogisticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def flatten_params(model):
    flat = []
    shapes = []
    for p in model.parameters():
        shapes.append(tuple(p.shape))
        flat.extend(p.data.numpy().ravel().tolist())
    return flat, shapes


def set_params(model, flat, shapes):
    offset = 0
    for p, shape in zip(model.parameters(), shapes):
        size = math.prod(shape)
        vals = np.array(flat[offset:offset+size], dtype=np.float32).reshape(shape)
        p.data.copy_(torch.from_numpy(vals))
        offset += size


def flatten_grads(model):
    flat = []
    for p in model.parameters():
        flat.extend(p.grad.numpy().ravel().tolist())
    return flat


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    print(f"Distributed Accuracy: {100 * correct / total:.2f}%")
    model.train()


def main(url):
    sio = socketio.Client()
    model = LogisticNet()
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

    shard_start = 0
    shard_end = len(train_dataset)
    current_step = 0
    weight_shapes = None
    training_started = False

    @sio.event
    def connect():
        print("Connected to coordinator")
        flat, shapes = flatten_params(model)
        nonlocal weight_shapes
        weight_shapes = shapes
        sio.emit("init", {"weights": flat, "shapes": shapes})

    @sio.on("shard_assignment")
    def on_shard(data):
        nonlocal shard_start, shard_end
        shard_start = data["start"]
        shard_end = data["end"]
        print(f"Assigned shard {shard_start} - {shard_end}")

    @sio.on("training_start")
    def on_start():
        nonlocal training_started
        training_started = True
        print("Training started")
        sio.emit("request_batch")

    @sio.on("weights")
    def on_weights(data):
        nonlocal current_step
        current_step = data["step_id"]
        set_params(model, data["updated_weights"], weight_shapes)

        if current_step % 50 == 0:
            evaluate(model, test_loader)

        if training_started:
            sio.emit("request_batch")

    @sio.on("batch_assigned")
    def on_batch(data):
        bidx = data["batch_index"]
        bs = 64

        local_size = shard_end - shard_start
        start = shard_start + ((bidx * bs) % local_size)

        images = []
        labels = []

        for i in range(bs):
            idx = start + i
            if idx >= shard_end:
                idx = shard_start + (idx - shard_end)
            img, lab = train_dataset[idx]
            images.append(img)
            labels.append(lab)

        images = torch.stack(images)
        labels = torch.tensor(labels)

        model.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()

        sio.emit("gradients", {
            "step_id": current_step,
            "gradients": flatten_grads(model),
            "batch_size": bs
        })

    @sio.on("training_complete")
    def done(data):
        print("Training finished at step", data["step_id"])
        sio.disconnect()

    sio.connect(url)
    sio.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:3000")
    main(parser.parse_args().url)