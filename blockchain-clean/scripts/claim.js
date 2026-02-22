const { ethers } = require("hardhat");
const fs = require("fs");

async function main() {

  const workerIndex = process.env.WORKER;

  if (!workerIndex || !["1", "2", "3"].includes(workerIndex)) {
    console.log("Windows:");
    console.log("$env:WORKER=1; npx hardhat run scripts/claim.js --network localhost");
    process.exit(1);
  }

  const [deployer, worker1, worker2, worker3] = await ethers.getSigners();

  const workers = {
    "1": worker1,
    "2": worker2,
    "3": worker3
  };

  const selectedWorker = workers[workerIndex];

  console.log(`Worker ${workerIndex} claiming reward...`);
  console.log("Address:", selectedWorker.address);

  const contractAddress = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512";

  const TrainingReward = await ethers.getContractFactory("TrainingReward");
  const contract = TrainingReward.attach(contractAddress);
  const contractAsWorker = contract.connect(selectedWorker);

  // 🔥 BALANCE BEFORE
  const balanceBefore = await ethers.provider.getBalance(selectedWorker.address);
  console.log("Balance Before:", ethers.formatEther(balanceBefore), "ETH");

  const contractBalanceBefore = await ethers.provider.getBalance(contractAddress);
  console.log("Contract Balance Before:", ethers.formatEther(contractBalanceBefore), "ETH");

  const proof = JSON.parse(fs.readFileSync("../proof.json"));
  const publicSignals = JSON.parse(fs.readFileSync("../public.json"));

  const a = proof.pi_a.slice(0, 2);

  const b = [
    [proof.pi_b[0][1], proof.pi_b[0][0]],
    [proof.pi_b[1][1], proof.pi_b[1][0]],
  ];

  const c = proof.pi_c.slice(0, 2);

  const tx = await contractAsWorker.claimReward(a, b, c, publicSignals);
  await tx.wait();

  console.log("Reward claimed successfully!");

  // 🔥 BALANCE AFTER
  const balanceAfter = await ethers.provider.getBalance(selectedWorker.address);
  console.log("Balance After:", ethers.formatEther(balanceAfter), "ETH");

  const contractBalanceAfter = await ethers.provider.getBalance(contractAddress);
  console.log("Contract Balance After:", ethers.formatEther(contractBalanceAfter), "ETH");

  const rewardReceived = balanceAfter - balanceBefore;
  console.log("Reward Received:", ethers.formatEther(rewardReceived), "ETH");
}

main().catch((error) => {
  console.error("ERROR:", error);
  process.exitCode = 1;
});