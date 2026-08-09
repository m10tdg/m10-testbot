// services/project-service/src/routes/createProject.js
import { QdrantClient } from "@qdrant/js-client-rest";
const qdrant = new QdrantClient({ url: process.env.QDRANT_URL });

export async function createProject(req, res) {
  const { tenantId } = req.tenantContext;
  const { name, slug } = req.body;
  const projectId = uuid();
  const collectionName = `${tenantId}_${projectId}`;

  await qdrant.createCollection(collectionName, {
    vectors: { size: 1536, distance: "Cosine" },
  });

  await db.query(
    `INSERT INTO projects (id, tenant_id, name, slug, qdrant_collection) VALUES ($1,$2,$3,$4,$5)`,
    [projectId, tenantId, name, slug, collectionName]
  );

  res.status(201).json({ projectId, qdrantCollection: collectionName });
}