// services/document-service/src/routes/upload.js
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { Kafka } from "kafkajs";
import { v4 as uuid } from "uuid";
import db from "../db.js"; // pg pool, sets `SET app.current_tenant` per request — see 5.4

const s3 = new S3Client({ endpoint: process.env.S3_ENDPOINT }); // localstack locally
const kafka = new Kafka({ brokers: [process.env.KAFKA_BROKER] });
const producer = kafka.producer();

export async function uploadDocument(req, res) {
  const { tenantId, projectId } = req.tenantContext; // set by tenant-resolver middleware
  const file = req.file; // multer
  const docId = uuid();
  const key = `${tenantId}/${projectId}/documents/${file.originalname}`;

  await s3.send(new PutObjectCommand({
    Bucket: "m10-documents",
    Key: key,
    Body: file.buffer,
    ContentType: file.mimetype,
  }));

  await db.query(
    `INSERT INTO documents (id, tenant_id, project_id, name, doc_type, s3_path, status)
     VALUES ($1, $2, $3, $4, $5, $6, 'uploaded')`,
    [docId, tenantId, projectId, file.originalname, req.body.docType, key]
  );

  await producer.send({
    topic: "document.uploaded",
    messages: [{
      value: JSON.stringify({
        eventType: "document.uploaded",
        tenantId, projectId,
        documentId: docId,
        s3Path: key,
        correlationId: uuid(),
        timestamp: new Date().toISOString(),
      }),
    }],
  });

  res.status(202).json({ documentId: docId, status: "uploaded" });
}