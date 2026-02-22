const fs = require("fs");
const path = require("path");
const minimist = require("minimist");
const {
  buildPoseidonHasher,
  buildMerkleTree,
  computeRoot,
  getProof,
  verifyProof,
} = require("./poseidon_merkle");

const depth = 16;
const capacity = 1 << depth;

function readIDXImages(imagesPath) {
  const buf = fs.readFileSync(imagesPath);
  const num = buf.readUInt32BE(4);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  const imgs = [];
  let offset = 16;
  const imageSize = rows * cols;

  for (let i = 0; i < num; i++) {
    imgs.push(buf.slice(offset, offset + imageSize));
    offset += imageSize;
  }
  return imgs;
}

function readIDXLabels(labelsPath) {
  const buf = fs.readFileSync(labelsPath);
  const num = buf.readUInt32BE(4);
  const labels = [];
  for (let i = 0; i < num; i++) {
    labels.push(buf.readUInt8(8 + i));
  }
  return labels;
}

async function main() {
  const args = minimist(process.argv.slice(2));
  const shard_start = args.shard_start ? Number(args.shard_start) : 0;
  const shard_end = args.shard_end ? Number(args.shard_end) : 20000;

  if (shard_end <= shard_start) {
    console.error("Invalid shard range");
    process.exit(1);
  }

  console.log("Building Poseidon Merkle tree...");

  // 🔥 Correct dataset path (two levels up)
  const rawDir = path.join(__dirname, "..", "..", "data", "MNIST", "raw");
  const imagesPath = path.join(rawDir, "train-images-idx3-ubyte");
  const labelsPath = path.join(rawDir, "train-labels-idx1-ubyte");

  if (!fs.existsSync(imagesPath) || !fs.existsSync(labelsPath)) {
    console.error("MNIST raw files not found at", rawDir);
    process.exit(1);
  }

  const images = readIDXImages(imagesPath);
  const labels = readIDXLabels(labelsPath);

  const shardSize = shard_end - shard_start;
  if (shardSize > capacity) {
    console.error(`Shard size ${shardSize} exceeds capacity ${capacity}`);
    process.exit(1);
  }

  const poseidon = await buildPoseidonHasher();
  const Fp = BigInt(poseidon.F.p.toString());

  // Initialize all leaves with 0n (padding)
  const leaves = new Array(capacity).fill(0n);

  for (let i = shard_start; i < shard_end; i++) {
    const imgBuf = images[i];
    const lab = labels[i];

    const dataBuf = Buffer.concat([imgBuf, Buffer.from([lab & 0xff])]);
    const hex = dataBuf.toString("hex");

    let big = BigInt("0x" + hex) % Fp;

    const h = poseidon([big, 0n]);
    const hBig = BigInt(poseidon.F.toString(h));

    leaves[i - shard_start] = hBig;
  }

  const tree = buildMerkleTree(poseidon, leaves, depth);
  const root = computeRoot(tree);

  console.log("Root:", root.toString());

  const rootObj = {
    root: root.toString(),
    depth: depth,
    shard_start: shard_start,
    shard_end: shard_end,
  };

  fs.writeFileSync(
    path.join(__dirname, "dataset_root.json"),
    JSON.stringify(rootObj, null, 2)
  );

  // Generate random proof
  const randIndex = Math.floor(Math.random() * shardSize);
  console.log("Testing proof for index", randIndex);

  const proof = getProof(tree, randIndex, depth);
  const leaf = leaves[randIndex];

  const ok = verifyProof(poseidon, leaf, proof, root, depth);
  console.log("Proof valid:", ok);

  if (!ok) {
    throw new Error("Proof verification failed");
  }

  const zkInput = {
    leaf: leaf.toString(),
    pathElements: proof.pathElements.map((x) => x.toString()),
    pathIndices: proof.pathIndices,
    root: root.toString(),
  };

  fs.writeFileSync(
    path.join(__dirname, "zk_input.json"),
    JSON.stringify(zkInput, null, 2)
  );

  console.log("zk_input.json created.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});