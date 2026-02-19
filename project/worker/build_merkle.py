import argparse
import time
import json
import pickle
import random
import os
from typing import List

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from merkle import hash_data, build_merkle_tree, compute_root, get_proof, verify_proof


def hash_dataset(dataset, shard_start: int, shard_end: int, batch_size: int = 1024) -> List[str]:
    """Hash samples in dataset[shard_start:shard_end] and return list of hex hashes."""
    total = shard_end - shard_start
    loader = DataLoader(Subset(dataset, list(range(shard_start, shard_end))), batch_size=batch_size, shuffle=False, num_workers=0)
    leaves = []
    for batch in loader:
        images, labels = batch
        # images: [B,1,28,28], labels: [B]
        arr = images.numpy()
        labs = labels.numpy()
        for i in range(arr.shape[0]):
            raw = arr[i].tobytes() + bytes([int(labs[i])])
            leaves.append(hash_data(raw))
    return leaves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_start', type=int, default=0)
    parser.add_argument('--shard_end', type=int, default=60000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--save_tree', action='store_true')
    args = parser.parse_args()

    shard_start = args.shard_start
    shard_end = args.shard_end

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    if shard_start < 0 or shard_end > len(dataset) or shard_start >= shard_end:
        raise ValueError('Invalid shard range')

    print('Hashing dataset...')
    t0 = time.time()
    leaves = hash_dataset(dataset, shard_start, shard_end, batch_size=args.batch_size)
    print(f'Hashed {len(leaves)} samples in {time.time()-t0:.2f}s')

    print('Building Merkle tree...')
    t1 = time.time()
    tree = build_merkle_tree(leaves)
    root = tree[-1][0] if tree and tree[-1] else ''
    print(f'Built tree in {time.time()-t1:.2f}s')

    print(f'Merkle root: {root}')

    # save root metadata
    out = {
        'root': root,
        'shard_start': shard_start,
        'shard_end': shard_end,
    }
    with open('dataset_root.json', 'w') as f:
        json.dump(out, f)

    if args.save_tree:
        with open('dataset_tree.pkl', 'wb') as f:
            pickle.dump(tree, f)

    # pick random index in shard
    idx = random.randint(0, len(leaves)-1)
    print(f'Testing proof for index {idx}...')
    proof = get_proof(tree, idx)
    leaf = leaves[idx]
    valid = verify_proof(leaf, proof, root, idx)
    print(f'Proof valid: {valid}')
    if not valid:
        raise SystemExit('Proof verification failed')


if __name__ == '__main__':
    main()
