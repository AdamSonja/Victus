pragma circom 2.0.0;

include "circomlib/circuits/poseidon.circom";

template MerkleInclusion(depth) {

    // Private inputs
    signal input leaf;
    signal input pathElements[depth];
    signal input pathIndices[depth];

    // Public input
    signal input root;

    signal current[depth + 1];
    signal delta[depth];

    component hashes[depth];

    // Start from leaf
    current[0] <== leaf;

    for (var i = 0; i < depth; i++) {

        // Enforce pathIndices is binary (0 or 1)
        pathIndices[i] * (1 - pathIndices[i]) === 0;

        hashes[i] = Poseidon(2);

        // delta = sibling - current
        delta[i] <== pathElements[i] - current[i];

        // If index == 0 → current left, sibling right
        // If index == 1 → sibling left, current right
        hashes[i].inputs[0] <== current[i] + pathIndices[i] * delta[i];
        hashes[i].inputs[1] <== pathElements[i] - pathIndices[i] * delta[i];

        current[i + 1] <== hashes[i].out;
    }

    // Enforce final computed root equals provided root
    root === current[depth];
}

// Make root PUBLIC for Solidity verifier
component main { public [root] } = MerkleInclusion(16);