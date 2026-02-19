# Distributed ML Phase 1

This project implements PHASE 1 of a distributed AI training system (pure distributed ML, no blockchain/zk/tokens).

Structure

project/
├── coordinator/
│   ├── server.js        # Node.js Express + Socket.io coordinator
│   ├── package.json
│
├── worker/
│   ├── worker.py        # Python Socket.io worker
│
├── single_node_test/
│   ├── train.py         # Single-node logistic regression on MNIST (CPU)
│
├── requirements.txt

Notes

- Single-node training (fast): run the script in `single_node_test/train.py`. By default it uses a subset of 10000 training samples to keep runtime short. You may increase with `--subset` but runtime may exceed 30s on slower CPUs.
- Coordinator: `cd coordinator && npm install && npm start`
- Worker: `pip install -r requirements.txt` then `python worker/worker.py --url http://<coordinator-host>:3000`

Coordinator responsibilities
- Accept worker connections
- Maintain active worker list
- Assign batch indices
- Receive gradients, average per-step across active workers (or timeout), apply SGD update, broadcast updated weights

Worker responsibilities
- Connect, receive weights, request batch, compute gradients for assigned batch, send gradients back

Restrictions
- CPU-only PyTorch
- No blockchain, no zk, no token logic, no merkle trees
