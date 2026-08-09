import { v4 as uuid } from "uuid";
import { QdrantClient } from "@qdrant/js-client-rest";
import { pool } from "../db.js";

const qdrant = new QdrantClient({ url: process.env.QDRANT_URL,checkCompatibility: false });

// NOTE: in the real system, `tenantId` comes from the logged-in user's JWT via
// the Tenant Resolver middleware. We haven't built auth yet, so for now the
// caller just puts tenantId directly in the request body. This is clearly
// unsafe for production and is a placeholder we will remove once Cognito is wired in.
export async function createProject(req, res) {
  try {
    const { tenantId, name, slug } = req.body;
    if (!tenantId || !name || !slug) {
      return res.status(400).json({ error: "tenantId, name and slug are required" });
    }

    const projectId = uuid();
    const collectionName = `${tenantId}_${projectId}`;

    await qdrant.createCollection(collectionName, {
      vectors: { size: 1536, distance: "Cosine" },
    });

    await pool.query(
      `INSERT INTO projects (id, tenant_id, name, slug, qdrant_collection) VALUES ($1,$2,$3,$4,$5)`,
      [projectId, tenantId, name, slug, collectionName]
    );

    res.status(201).json({ projectId, qdrantCollection: collectionName });
  } catch (err) {
    console.error("[project-service] createProject failed:", err);
    res.status(500).json({ error: err.message });
  }
}

export async function listProjects(req, res) {
  const { tenantId } = req.query;
  const result = await pool.query(
    `SELECT id, name, slug, qdrant_collection, created_at FROM projects WHERE tenant_id = $1`,
    [tenantId]
  );
  res.json(result.rows);
}
