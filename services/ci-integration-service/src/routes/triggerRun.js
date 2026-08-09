// services/ci-integration-service/src/routes/triggerRun.js
export async function triggerRun(req, res) {
  const { tenantId, projectId } = req.tenantContext;
  const { url, prompt, environment, runSource } = req.body; // runSource: 'github-actions' etc.
  const runId = uuid();

  await db.query(
    `INSERT INTO test_runs (id, tenant_id, project_id, run_source, url, prompt, status)
     VALUES ($1,$2,$3,$4,$5,$6,'queued')`,
    [runId, tenantId, projectId, runSource, url, prompt]
  );

  await producer.send({
    topic: "test.requested",
    messages: [{ value: JSON.stringify({
      eventType: "test.requested", tenantId, projectId, runId, url, prompt, runSource,
      correlationId: uuid(), timestamp: new Date().toISOString(),
    })}],
  });

  // Async per your doc's "Asynchronous Execution" section — return runId immediately, don't block
  res.status(202).json({ runId, status: "queued", pollUrl: `/runs/${runId}` });
}

export async function getRunStatus(req, res) {
  const { runId } = req.params;
  const result = await db.query(`SELECT status, report_url FROM test_runs r
    LEFT JOIN reports rep ON rep.run_id = r.id WHERE r.id = $1`, [runId]);
  res.json(result.rows[0]);
}