import { v4 as uuid } from "uuid";
import { PutObjectCommand } from "@aws-sdk/client-s3";
import { pool } from "../db.js";
import { producer } from "../kafka.js";
import { s3 } from "../s3.js";

// This is the endpoint a developer, or a GitHub Actions / GitLab CI / Jenkins pipeline,
// calls to kick off an AI test run. Same placeholder note as the other services:
// tenantId/projectId come from the request body until Cognito auth exists.
export async function triggerRun(req, res) {
  try {
    const { tenantId, projectId, url, prompt, runSource } = req.body;
    if (!tenantId || !projectId || !url || !prompt) {
      return res.status(400).json({ error: "tenantId, projectId, url and prompt are required" });
    }

    const runId = uuid();
    const baseline = req.file;
    let baselineS3Path = null;

    if (baseline) {
      const isPng = baseline.buffer.subarray(0, 8).equals(
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
      );
      const isJpeg = baseline.buffer[0] === 0xff
        && baseline.buffer[1] === 0xd8
        && baseline.buffer[2] === 0xff;

      if (!isPng && !isJpeg) {
        return res.status(400).json({ error: "baseline must be a valid PNG or JPEG image" });
      }

      const extension = isPng ? "png" : "jpg";
      baselineS3Path = `${tenantId}/${projectId}/${runId}/baseline.${extension}`;
      await s3.send(new PutObjectCommand({
        Bucket: process.env.S3_ARTIFACTS_BUCKET,
        Key: baselineS3Path,
        Body: baseline.buffer,
        ContentType: isPng ? "image/png" : "image/jpeg",
      }));
    }

    await pool.query(
      `INSERT INTO test_runs (id, tenant_id, project_id, run_source, url, prompt, status, baseline_s3_path)
       VALUES ($1,$2,$3,$4,$5,$6,'queued',$7)`,
      [runId, tenantId, projectId, runSource || "ui", url, prompt, baselineS3Path]
    );

    await producer.send({
      topic: "test.requested",
      messages: [{
        value: JSON.stringify({
          eventType: "test.requested",
          tenantId, projectId, runId, url, prompt,
          runSource: runSource || "ui",
          baselineS3Path,
          correlationId: uuid(),
          timestamp: new Date().toISOString(),
        }),
      }],
    });

    // Returns immediately - the orchestrator processes this asynchronously.
    res.status(202).json({ runId, status: "queued", baselineUploaded: Boolean(baselineS3Path) });
  } catch (err) {
    console.error("[ci-integration-service] triggerRun failed:", err);
    res.status(500).json({ error: err.message });
  }
}

export async function getRunStatus(req, res) {
  const { runId } = req.params;
  const runResult = await pool.query(`SELECT * FROM test_runs WHERE id = $1`, [runId]);
  if (runResult.rows.length === 0) {
    return res.status(404).json({ error: "run not found" });
  }
  const reportResult = await pool.query(
    `SELECT s3_path, critical_count, warning_count FROM reports WHERE run_id = $1`,
    [runId]
  );
  res.json({ ...runResult.rows[0], report: reportResult.rows[0] || null });
}
