import { v4 as uuid } from "uuid";
import { pool } from "../db.js";
import { producer } from "../kafka.js";

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

    await pool.query(
      `INSERT INTO test_runs (id, tenant_id, project_id, run_source, url, prompt, status)
       VALUES ($1,$2,$3,$4,$5,$6,'queued')`,
      [runId, tenantId, projectId, runSource || "ui", url, prompt]
    );

    await producer.send({
      topic: "test.requested",
      messages: [{
        value: JSON.stringify({
          eventType: "test.requested",
          tenantId, projectId, runId, url, prompt,
          runSource: runSource || "ui",
          correlationId: uuid(),
          timestamp: new Date().toISOString(),
        }),
      }],
    });

    // Returns immediately - the orchestrator processes this asynchronously.
    res.status(202).json({ runId, status: "queued" });
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
