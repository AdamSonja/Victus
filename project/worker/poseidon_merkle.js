const { buildPoseidon } = require('circomlibjs');

async function buildPoseidonHasher() {
  const poseidon = await buildPoseidon();
  return poseidon;
}

function poseidonHash(poseidon, left, right) {
  // left/right are BigInt
  const res = poseidon([left, right]);
  // convert to BigInt decimal via F.toString
  const asDec = poseidon.F.toString(res);
  return BigInt(asDec);
}

function buildMerkleTree(poseidon, leaves, depth) {
  const size = 1 << depth;
  if (leaves.length > size) throw new Error('Number of leaves exceeds 2^depth');

  // pad leaves to size with 0n
  const L = new Array(size).fill(0n);
  for (let i = 0; i < leaves.length; i++) L[i] = BigInt(leaves[i]);

  const levels = [];
  levels.push(L);

  let current = L;
  for (let d = 0; d < depth; d++) {
    const parents = new Array(current.length / 2).fill(0n);
    for (let i = 0; i < current.length; i += 2) {
      const left = BigInt(current[i]);
      const right = BigInt(current[i+1]);
      parents[i/2] = poseidonHash(poseidon, left, right);
    }
    levels.push(parents);
    current = parents;
  }

  return levels;
}

function computeRoot(tree) {
  if (!tree || tree.length === 0) return 0n;
  const top = tree[tree.length - 1];
  return BigInt(top[0]);
}

function getProof(tree, index, depth) {
  const pathElements = [];
  const pathIndices = [];
  let idx = index;
  for (let d = 0; d < depth; d++) {
    const level = tree[d];
    const siblingIndex = (idx % 2 === 0) ? idx + 1 : idx - 1;
    // if siblingIndex out of bounds (shouldn't happen because padded), pick last
    const sib = level[siblingIndex] !== undefined ? BigInt(level[siblingIndex]) : BigInt(level[level.length - 1]);
    pathElements.push(sib);
    pathIndices.push(idx % 2);
    idx = Math.floor(idx / 2);
  }
  return { pathElements, pathIndices };
}

function verifyProof(poseidon, leaf, proof, root, depth) {
  let cur = BigInt(leaf);
  for (let i = 0; i < depth; i++) {
    const sib = BigInt(proof.pathElements[i]);
    const idx = proof.pathIndices[i];
    if (idx === 0) {
      cur = poseidonHash(poseidon, cur, sib);
    } else {
      cur = poseidonHash(poseidon, sib, cur);
    }
  }
  return BigInt(cur) === BigInt(root);
}

module.exports = {
  buildPoseidonHasher,
  poseidonHash,
  buildMerkleTree,
  computeRoot,
  getProof,
  verifyProof,
};
