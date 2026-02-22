// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IVerifier {
    function verifyProof(
        uint[2] memory a,
        uint[2][2] memory b,
        uint[2] memory c,
        uint[1] memory input
    ) external view returns (bool);
}

contract TrainingReward {

    address[] public workers;
    mapping(address => bool) public hasClaimed;

    bytes32 public merkleRoot;
    uint256 public rewardPerWorker;

    IVerifier public verifier;

    constructor(
        address[] memory _workers,
        bytes32 _merkleRoot,
        address _verifier
    ) payable {
        require(_workers.length == 3, "Require exactly 3 workers");
        require(msg.value > 0, "Reward must be funded");

        workers = _workers;
        merkleRoot = _merkleRoot;
        verifier = IVerifier(_verifier);

        rewardPerWorker = msg.value / _workers.length;
    }

    function isWorker(address user) public view returns (bool) {
        for (uint i = 0; i < workers.length; i++) {
            if (workers[i] == user) return true;
        }
        return false;
    }

    function claimReward(
        uint[2] memory a,
        uint[2][2] memory b,
        uint[2] memory c,
        uint[1] memory input
    ) external {

        require(isWorker(msg.sender), "Not authorized worker");
        require(!hasClaimed[msg.sender], "Already claimed");

        // Ensure proof is for stored Merkle root
        require(bytes32(input[0]) == merkleRoot, "Root mismatch");

        bool valid = verifier.verifyProof(a, b, c, input);
        require(valid, "Invalid proof");

        hasClaimed[msg.sender] = true;
        payable(msg.sender).transfer(rewardPerWorker);
    }
}