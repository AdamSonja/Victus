const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

const PORT = 3000;
const TOTAL_SAMPLES = 60000;
const EXPECTED_WORKERS = 3;   // 👈 SET THIS//IMPORTANT
const MAX_STEPS = 500;
const LR = 0.01;

let workers = {};
let batchIndex = 0;
let globalWeights = null;
let weightShapes = null;
let stepId = 0;
let trainingStarted = false;

const aggregators = {};

function now() { return Date.now(); }

io.on("connection", (socket) => {
  console.log("Worker connected:", socket.id);
  workers[socket.id] = { lastSeen: now() };

  // Barrier check
  if (Object.keys(workers).length === EXPECTED_WORKERS) {
    assignShards();
    trainingStarted = true;
    io.emit("training_start");
    console.log("All workers ready. Training started.");
  }

  socket.on("init", (data) => {
    if (!globalWeights) {
      globalWeights = Float64Array.from(data.weights);
      weightShapes = data.shapes;
      console.log("Global weights initialized");
    }
  });

  socket.on("request_batch", () => {
    if (!trainingStarted) return;
    socket.emit("batch_assigned", { batch_index: batchIndex++ });
  });

  socket.on("gradients", (data) => {
    workers[socket.id].lastSeen = now();
    const sid = data.step_id;

    if (!aggregators[sid]) {
      aggregators[sid] = {
        accum: new Float64Array(data.gradients.length),
        count: 0,
        target: EXPECTED_WORKERS,
        timeout: setTimeout(() => finalizeStep(sid), 5000)
      };
    }

    const agg = aggregators[sid];

    for (let i = 0; i < data.gradients.length; i++) {
      agg.accum[i] += data.gradients[i];
    }

    agg.count++;

    if (agg.count >= agg.target) {
      clearTimeout(agg.timeout);
      finalizeStep(sid);
    }
  });

  socket.on("disconnect", () => {
    console.log("Worker disconnected:", socket.id);
    delete workers[socket.id];
  });
});

function assignShards() {
  const ids = Object.keys(workers);
  const shardSize = Math.floor(TOTAL_SAMPLES / ids.length);

  ids.forEach((id, index) => {
    const start = index * shardSize;
    const end = (index === ids.length - 1)
      ? TOTAL_SAMPLES
      : start + shardSize;

    workers[id].shardStart = start;
    workers[id].shardEnd = end;

    io.to(id).emit("shard_assignment", { start, end });

    console.log(`Shard assigned to ${id}: ${start} - ${end}`);
  });
}

function finalizeStep(sid) {
  const agg = aggregators[sid];
  if (!agg || !globalWeights) return;

  for (let i = 0; i < globalWeights.length; i++) {
    const avg = agg.accum[i] / agg.count;
    globalWeights[i] -= LR * avg;
  }

  stepId++;
  console.log(`Step ${stepId} updated`);

  if (stepId >= MAX_STEPS) {
    console.log("Training complete");
    io.emit("training_complete", { step_id: stepId });
    return;
  }

  io.emit("weights", {
    updated_weights: Array.from(globalWeights),
    step_id: stepId
  });

  delete aggregators[sid];
}

server.listen(PORT, "0.0.0.0", () => {
  console.log("Coordinator running on port", PORT);
});