import express from "express";
import multer from "multer";
import "dotenv/config";
import { uploadDocument } from "./routes/upload.js";
import { connectProducer } from "./kafka.js";

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.get("/health", (req, res) => res.json({ status: "ok", service: "document-service" }));

// multer reads the uploaded file into memory as req.file, and puts the other
// form fields (tenantId, projectId, docType) onto req.body.
app.post("/documents/upload", upload.single("file"), uploadDocument);

const PORT = process.env.PORT || 4002;

async function start() {
  await connectProducer(); // connect to Kafka BEFORE accepting requests
  app.listen(PORT, () => {
    console.log(`[document-service] listening on http://localhost:${PORT}`);
  });
}

start().catch((err) => {
  console.error("[document-service] failed to start:", err);
  process.exit(1);
});
