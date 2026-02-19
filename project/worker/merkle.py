import hashlib
from typing import List


def hash_data(data_bytes: bytes) -> str:
    """Return SHA256 hex digest of data_bytes."""
    return hashlib.sha256(data_bytes).hexdigest()


def hash_pair(left_hex: str, right_hex: str) -> str:
    """Hash two hex digests (left || right) and return hex digest."""
    left = bytes.fromhex(left_hex)
    right = bytes.fromhex(right_hex)
    return hashlib.sha256(left + right).hexdigest()


def build_merkle_tree(leaves: List[str]) -> List[List[str]]:
    """Build full Merkle tree as list of levels.

    levels[0] = leaves (hex strings)
    levels[-1][0] = root
    If odd, duplicate last node.
    """
    if not leaves:
        return [[]]

    levels = [list(leaves)]
    current = levels[0]
    while len(current) > 1:
        if len(current) % 2 == 1:
            current = current + [current[-1]]
        parents = []
        for i in range(0, len(current), 2):
            parents.append(hash_pair(current[i], current[i+1]))
        levels.append(parents)
        current = parents
    return levels


def compute_root(leaves: List[str]) -> str:
    """Compute root only (optimized)."""
    if not leaves:
        return ''
    current = list(leaves)
    while len(current) > 1:
        if len(current) % 2 == 1:
            current = current + [current[-1]]
        parents = []
        for i in range(0, len(current), 2):
            parents.append(hash_pair(current[i], current[i+1]))
        current = parents
    return current[0]


def get_proof(tree: List[List[str]], index: int) -> List[str]:
    """Return proof (list of sibling hex hashes) for leaf at index.

    Proof order: from leaf level up to but not including root.
    """
    proof = []
    idx = index
    for level in tree[:-1]:
        length = len(level)
        # if odd, implicit duplication handled by logic when selecting sibling
        if idx % 2 == 0:
            sib_idx = idx + 1
        else:
            sib_idx = idx - 1

        if sib_idx >= length:
            # sibling is the duplicated last
            sib_idx = length - 1

        proof.append(level[sib_idx])
        idx = idx // 2
    return proof


def verify_proof(leaf_hex: str, proof: List[str], root_hex: str, index: int) -> bool:
    """Verify proof for given leaf hex, proof (sibling hex list), root and original index."""
    cur = leaf_hex
    idx = index
    for sib in proof:
        if idx % 2 == 0:
            # cur is left
            cur = hash_pair(cur, sib)
        else:
            # cur is right
            cur = hash_pair(sib, cur)
        idx = idx // 2
    return cur == root_hex


__all__ = [
    'hash_data',
    'hash_pair',
    'build_merkle_tree',
    'compute_root',
    'get_proof',
    'verify_proof',
]
