import { GetObjectCommand, ListObjectsV2Command } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { pool } from "../db.js";
import { publicS3, s3 } from "../s3.js";

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const SIGNED_URL_TTL_SECONDS = 900;

function badRequest(message) {
  const error = new Error(message);
  error.statusCode = 400;
  throw error;
}

function notFound(message) {
  const error = new Error(message);
  error.statusCode = 404;
  throw error;
}

function requireUuid(value, name) {
  if (typeof value !== "string" || !UUID_PATTERN.test(value)) {
    badRequest(`${name} must be a valid UUID`);
  }
  return value;
}

function requireTenant(req) {
  return requireUuid(req.query.tenantId, "tenantId");
}

function pagination(req) {
  const limit = Number(req.query.limit ?? 25);
  const offset = Number(req.query.offset ?? 0);
  if (!Number.isInteger(limit) || limit < 1 || limit > 100) {
    badRequest("limit must be an integer from 1 to 100");
  }
  if (!Number.isInteger(offset) || offset < 0) {
    badRequest("offset must be a non-negative integer");
  }
  return { limit, offset };
}

async function signedDownloadUrl(bucket, key) {
  return getSignedUrl(
    publicS3,
    new GetObjectCommand({ Bucket: bucket, Key: key }),
    { expiresIn: SIGNED_URL_TTL_SECONDS }
  );
}

async function signedReportRows(rows) {
  const bucket = process.env.S3_ARTIFACTS_BUCKET;
  return Promise.all(rows.map(async ({ s3_path, ...report }) => ({
    ...report,
    downloadUrl: await signedDownloadUrl(bucket, s3_path),
  })));
}

async function requireProject(projectId, tenantId) {
  const result = await pool.query(
    `SELECT id, tenant_id, name, slug, qdrant_collection, created_at
     FROM projects WHERE id = $1 AND tenant_id = $2`,
    [projectId, tenantId]
  );
  if (!result.rows[0]) notFound("project not found");
  return result.rows[0];
}

async function requireRun(runId, tenantId) {
  const result = await pool.query(
    `SELECT id, tenant_id, project_id, run_source, url, prompt, baseline_s3_path,
            status, created_at, completed_at
     FROM test_runs WHERE id = $1 AND tenant_id = $2`,
    [runId, tenantId]
  );
  if (!result.rows[0]) notFound("run not found");
  return result.rows[0];
}

export async function getDashboard(req, res) {
  const tenantId = requireTenant(req);
  const [counts, projects] = await Promise.all([
    pool.query(
      `SELECT
         (SELECT COUNT(*)::int FROM projects WHERE tenant_id = $1) AS project_count,
         (SELECT COUNT(*)::int FROM documents WHERE tenant_id = $1) AS document_count,
         (SELECT COUNT(*)::int FROM test_runs WHERE tenant_id = $1) AS run_count,
         (SELECT COUNT(*)::int FROM reports WHERE tenant_id = $1) AS report_count,
         COALESCE((
           SELECT jsonb_object_agg(status, count)
           FROM (
             SELECT status, COUNT(*)::int AS count
             FROM test_runs WHERE tenant_id = $1 GROUP BY status
           ) run_statuses
         ), '{}'::jsonb) AS run_status_counts`,
      [tenantId]
    ),
    pool.query(
      `SELECT p.id, p.name, p.slug, p.created_at,
         (SELECT COUNT(*)::int FROM documents d
          WHERE d.project_id = p.id AND d.tenant_id = p.tenant_id) AS document_count,
         (SELECT COUNT(*)::int FROM test_runs tr
          WHERE tr.project_id = p.id AND tr.tenant_id = p.tenant_id) AS run_count,
         (SELECT jsonb_build_object(
             'id', tr.id, 'status', tr.status, 'url', tr.url,
             'createdAt', tr.created_at, 'completedAt', tr.completed_at
           )
          FROM test_runs tr
          WHERE tr.project_id = p.id AND tr.tenant_id = p.tenant_id
          ORDER BY tr.created_at DESC LIMIT 1) AS latest_run
       FROM projects p
       WHERE p.tenant_id = $1
       ORDER BY p.created_at DESC`,
      [tenantId]
    ),
  ]);

  res.json({ summary: counts.rows[0], projects: projects.rows });
}

export async function getProject(req, res) {
  const tenantId = requireTenant(req);
  const projectId = requireUuid(req.params.projectId, "projectId");
  const project = await requireProject(projectId, tenantId);
  const result = await pool.query(
    `SELECT
       (SELECT COUNT(*)::int FROM documents WHERE project_id = $1 AND tenant_id = $2) AS document_count,
       (SELECT COUNT(*)::int FROM test_runs WHERE project_id = $1 AND tenant_id = $2) AS run_count,
       (SELECT COUNT(*)::int FROM reports WHERE project_id = $1 AND tenant_id = $2) AS report_count`,
    [projectId, tenantId]
  );
  res.json({ ...project, ...result.rows[0] });
}

export async function listProjectDocuments(req, res) {
  const tenantId = requireTenant(req);
  const projectId = requireUuid(req.params.projectId, "projectId");
  const { limit, offset } = pagination(req);
  await requireProject(projectId, tenantId);
  const result = await pool.query(
    `SELECT id, project_id, name, doc_type, s3_path, status, created_at
     FROM documents
     WHERE project_id = $1 AND tenant_id = $2
     ORDER BY created_at DESC
     LIMIT $3 OFFSET $4`,
    [projectId, tenantId, limit, offset]
  );
  const bucket = process.env.S3_DOCUMENTS_BUCKET;
  const documents = await Promise.all(result.rows.map(async (document) => ({
    ...document,
    downloadUrl: await signedDownloadUrl(bucket, document.s3_path),
  })));
  res.json({ items: documents, limit, offset });
}

export async function listProjectRuns(req, res) {
  const tenantId = requireTenant(req);
  const projectId = requireUuid(req.params.projectId, "projectId");
  const { limit, offset } = pagination(req);
  await requireProject(projectId, tenantId);

  const values = [projectId, tenantId];
  let statusFilter = "";
  if (req.query.status !== undefined) {
    if (typeof req.query.status !== "string" || !req.query.status.trim()) {
      badRequest("status must be a non-empty string");
    }
    values.push(req.query.status.trim());
    statusFilter = `AND tr.status = $${values.length}`;
  }
  values.push(limit, offset);

  const result = await pool.query(
    `SELECT tr.id, tr.project_id, tr.run_source, tr.url, tr.prompt,
            tr.status, tr.created_at, tr.completed_at,
            r.id AS report_id, r.status AS report_status, r.severity,
            r.critical_count, r.warning_count, r.format, r.s3_path AS report_s3_path
     FROM test_runs tr
     LEFT JOIN LATERAL (
       SELECT id, status, severity, critical_count, warning_count, format, s3_path
       FROM reports WHERE run_id = tr.id AND tenant_id = tr.tenant_id
       ORDER BY created_at DESC LIMIT 1
     ) r ON TRUE
     WHERE tr.project_id = $1 AND tr.tenant_id = $2 ${statusFilter}
     ORDER BY tr.created_at DESC
     LIMIT $${values.length - 1} OFFSET $${values.length}`,
    values
  );
  const items = await Promise.all(result.rows.map(async (run) => {
    const { report_s3_path: reportPath, ...publicRun } = run;
    if (reportPath) {
      publicRun.report = {
        id: publicRun.report_id,
        status: publicRun.report_status,
        severity: publicRun.severity,
        critical_count: publicRun.critical_count,
        warning_count: publicRun.warning_count,
        format: publicRun.format,
        downloadUrl: await signedDownloadUrl(process.env.S3_ARTIFACTS_BUCKET, reportPath),
      };
    } else {
      publicRun.report = null;
    }
    delete publicRun.report_id;
    delete publicRun.report_status;
    delete publicRun.severity;
    delete publicRun.critical_count;
    delete publicRun.warning_count;
    delete publicRun.format;
    return publicRun;
  }));
  res.json({ items, limit, offset });
}

export async function getRun(req, res) {
  const tenantId = requireTenant(req);
  const runId = requireUuid(req.params.runId, "runId");
  const run = await requireRun(runId, tenantId);
  const result = await pool.query(
    `SELECT id, status, severity, s3_path, format,
            critical_count, warning_count, created_at
     FROM reports WHERE run_id = $1 AND tenant_id = $2
     ORDER BY created_at DESC`,
    [runId, tenantId]
  );
  res.json({ ...run, reports: await signedReportRows(result.rows) });
}

export async function listRunArtifacts(req, res) {
  const tenantId = requireTenant(req);
  const runId = requireUuid(req.params.runId, "runId");
  const run = await requireRun(runId, tenantId);
  const bucket = process.env.S3_ARTIFACTS_BUCKET;
  const prefix = `${tenantId}/${run.project_id}/${runId}/`;
  const artifacts = [];
  let continuationToken;

  do {
    const result = await s3.send(new ListObjectsV2Command({
      Bucket: bucket,
      Prefix: prefix,
      ContinuationToken: continuationToken,
    }));
    for (const object of result.Contents || []) {
      if (!object.Key || object.Key.endsWith("/")) continue;
      artifacts.push({
        name: object.Key.slice(prefix.length),
        key: object.Key,
        size: object.Size,
        lastModified: object.LastModified,
        downloadUrl: await signedDownloadUrl(bucket, object.Key),
      });
    }
    continuationToken = result.IsTruncated ? result.NextContinuationToken : undefined;
  } while (continuationToken);

  res.json({ runId, items: artifacts });
}

export async function listReports(req, res) {
  const tenantId = requireTenant(req);
  const { limit, offset } = pagination(req);
  const values = [tenantId];
  let projectFilter = "";
  if (req.query.projectId !== undefined) {
    values.push(requireUuid(req.query.projectId, "projectId"));
    projectFilter = `AND r.project_id = $${values.length}`;
  }
  values.push(limit, offset);
  const result = await pool.query(
    `SELECT r.id, r.project_id, r.run_id, r.status, r.severity, r.format,
            r.critical_count, r.warning_count, r.s3_path, r.created_at,
            p.name AS project_name, tr.url AS run_url, tr.run_source
     FROM reports r
     JOIN projects p ON p.id = r.project_id AND p.tenant_id = r.tenant_id
     JOIN test_runs tr ON tr.id = r.run_id AND tr.tenant_id = r.tenant_id
     WHERE r.tenant_id = $1 ${projectFilter}
     ORDER BY r.created_at DESC
     LIMIT $${values.length - 1} OFFSET $${values.length}`,
    values
  );
  res.json({ items: await signedReportRows(result.rows), limit, offset });
}

export async function getReport(req, res) {
  const tenantId = requireTenant(req);
  const reportId = requireUuid(req.params.reportId, "reportId");
  const result = await pool.query(
    `SELECT r.id, r.project_id, r.run_id, r.status, r.severity, r.format,
            r.critical_count, r.warning_count, r.s3_path, r.created_at,
            p.name AS project_name, tr.url AS run_url, tr.run_source
     FROM reports r
     JOIN projects p ON p.id = r.project_id AND p.tenant_id = r.tenant_id
     JOIN test_runs tr ON tr.id = r.run_id AND tr.tenant_id = r.tenant_id
     WHERE r.id = $1 AND r.tenant_id = $2`,
    [reportId, tenantId]
  );
  if (!result.rows[0]) notFound("report not found");
  res.json((await signedReportRows(result.rows))[0]);
}
