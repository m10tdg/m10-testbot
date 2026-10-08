import express from "express";
import multer from "multer";
import "dotenv/config";
import { triggerRun, getRunStatus } from "./routes/triggerRun.js";
import { connectProducer } from "./kafka.js";

const app = express();
app.use(express.json());
const baselineUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024, files: 1 },
  fileFilter: (req, file, callback) => {
    if (file.fieldname !== "baseline") {
      return callback(Object.assign(new Error("Upload the image in the 'baseline' field"), { statusCode: 400 }));
    }
    if (!["image/png", "image/jpeg"].includes(file.mimetype)) {
      return callback(Object.assign(new Error("baseline must be a PNG or JPEG image"), { statusCode: 400 }));
    }
    callback(null, true);
  },
});

app.get("/health", (req, res) => res.json({ status: "ok", service: "ci-integration-service" }));

// This is the single REST endpoint any CI system (or you, with curl) calls to
// start an AI test run - matches the "Trigger M10" step in your architecture doc.
app.post("/runs", baselineUpload.single("baseline"), triggerRun);
app.get("/runs/:runId", getRunStatus);

app.use((err, req, res, next) => {
  if (res.headersSent) return next(err);
  if (err instanceof multer.MulterError) {
    const status = err.code === "LIMIT_FILE_SIZE" ? 413 : 400;
    return res.status(status).json({ error: err.message });
  }
  if (err.statusCode) {
    return res.status(err.statusCode).json({ error: err.message });
  }
  console.error("[ci-integration-service] request failed:", err);
  res.status(500).json({ error: "Request failed" });
});

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
