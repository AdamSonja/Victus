const { ethers } = require("hardhat");

async function main() {
  const [deployer, worker1, worker2, worker3] = await ethers.getSigners();

  console.log("Deploying with:", deployer.address);

  // Deploy Verifier
  const Verifier = await ethers.getContractFactory("Groth16Verifier");
  const verifier = await Verifier.deploy();
  await verifier.waitForDeployment();
  const verifierAddress = await verifier.getAddress();

  console.log("Verifier deployed to:", verifierAddress);

  // Worker addresses
  const workers = [
    worker1.address,
    worker2.address,
    worker3.address
  ];

  console.log("Workers:");
  console.log("Worker1:", worker1.address);
  console.log("Worker2:", worker2.address);
  console.log("Worker3:", worker3.address);

  // Replace with your real Merkle root
  const merkleRoot = "0x02e69196f244e7988614db267166567fa61c01d075a7bc512a2f53192ef3efb2";

  // Deploy TrainingReward
  const TrainingReward = await ethers.getContractFactory("TrainingReward");

  const contract = await TrainingReward.deploy(
    workers,
    merkleRoot,
    verifierAddress,
    {
      value: ethers.parseEther("1.0"),
    }
  );

  await contract.waitForDeployment();
  const contractAddress = await contract.getAddress();

  console.log("TrainingReward deployed to:", contractAddress);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});