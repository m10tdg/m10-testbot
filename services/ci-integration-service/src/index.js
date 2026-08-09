import express from "express";
import "dotenv/config";
import { triggerRun, getRunStatus } from "./routes/triggerRun.js";
import { connectProducer } from "./kafka.js";

const app = express();
app.use(express.json());

app.get("/health", (req, res) => res.json({ status: "ok", service: "ci-integration-service" }));

// This is the single REST endpoint any CI system (or you, with curl) calls to
// start an AI test run - matches the "Trigger M10" step in your architecture doc.
app.post("/runs", triggerRun);
app.get("/runs/:runId", getRunStatus);

const PORT = process.env.PORT || 4003;

async function start() {
  await connectProducer();
  app.listen(PORT, () => {
    console.log(`[ci-integration-service] listening on http://localhost:${PORT}`);
  });
}

start().catch((err) => {
  console.error("[ci-integration-service] failed to start:", err);
  process.exit(1);
});
