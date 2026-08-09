import { PutObjectCommand } from "@aws-sdk/client-s3";
import { v4 as uuid } from "uuid";
import { s3 } from "../s3.js";
import { pool } from "../db.js";
import { producer } from "../kafka.js";

// NOTE: same placeholder as project-service - tenantId/projectId come from the
// request body directly for now, until Cognito + Tenant Resolver exist.
export async function uploadDocument(req, res) {
  try {
    const { tenantId, projectId, docType } = req.body;
    const file = req.file; // populated by multer, see index.js

    if (!tenantId || !projectId || !file) {
      return res.status(400).json({ error: "tenantId, projectId and a file are required" });
    }

    const docId = uuid();
    const key = `${tenantId}/${projectId}/documents/${file.originalname}`;

    await s3.send(new PutObjectCommand({
      Bucket: process.env.S3_DOCUMENTS_BUCKET,
      Key: key,
      Body: file.buffer,
      ContentType: file.mimetype,
    }));

    await pool.query(
      `INSERT INTO documents (id, tenant_id, project_id, name, doc_type, s3_path, status)
       VALUES ($1, $2, $3, $4, $5, $6, 'uploaded')`,
      [docId, tenantId, projectId, file.originalname, docType || "requirement", key]
    );

    await producer.send({
      topic: "document.uploaded",
      messages: [{
        value: JSON.stringify({
          eventType: "document.uploaded",
          tenantId,
          projectId,
          documentId: docId,
          s3Path: key,
          correlationId: uuid(),
          timestamp: new Date().toISOString(),
        }),
      }],
    });

    res.status(202).json({ documentId: docId, s3Path: key, status: "uploaded" });
  } catch (err) {
    console.error("[document-service] upload failed:", err);
    res.status(500).json({ error: err.message });
  }
}
