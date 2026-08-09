"""
embedding-service
-----------------
Listens for `document.uploaded` events on Kafka. For each one:
  1. downloads the document from S3 (Localstack locally)
  2. chunks the text
  3. embeds each chunk with OpenAI
  4. writes the vectors into the project's Qdrant collection
  5. publishes `knowledge.indexed`

Run it with:  python main.py
(see README.md for full step-by-step instructions)
"""

import json
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

import boto3
import openai
from kafka import KafkaConsumer, KafkaProducer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

openai.api_key = os.environ["OPENAI_API_KEY"]

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["AWS_REGION"],
)

qdrant = QdrantClient(url=os.environ["QDRANT_URL"])

producer = KafkaProducer(
    bootstrap_servers=os.environ["KAFKA_BROKER"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def extract_text(raw_bytes: bytes, filename: str) -> str:
    """v1: plain text files only. We'll add PDF/DOCX extraction in a later step -
    for now, upload a .txt file to test the pipeline end to end."""
    if filename.lower().endswith(".txt"):
        return raw_bytes.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type for '{filename}'. Use a .txt file for now.")


def semantic_chunk(text: str, max_words: int = 200) -> list[str]:
    """Chunk on blank-line paragraph boundaries, merging short paragraphs together
    up to max_words. Simple and predictable - good enough to prove the pipeline works."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len((current + " " + p).split()) > max_words:
            if current:
                chunks.append(current)
            current = p
        else:
            current = f"{current} {p}".strip()
    if current:
        chunks.append(current)
    return chunks or [text.strip()]


def embed(text: str) -> list[float]:
    resp = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding


def handle_document_uploaded(event: dict):
    tenant_id = event["tenantId"]
    project_id = event["projectId"]
    doc_id = event["documentId"]
    s3_path = event["s3Path"]
    collection = f"{tenant_id}_{project_id}"

    print(f"[embedding-service] processing document {doc_id} -> collection {collection}")

    obj = s3.get_object(Bucket=os.environ["S3_DOCUMENTS_BUCKET"], Key=s3_path)
    raw_bytes = obj["Body"].read()
    filename = s3_path.split("/")[-1]
    text = extract_text(raw_bytes, filename)

    chunks = semantic_chunk(text)
    print(f"[embedding-service] split into {len(chunks)} chunk(s)")

    points = []
    for i, chunk in enumerate(chunks):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embed(chunk),
            payload={
                "tenant_id": tenant_id,
                "project_id": project_id,
                "document_id": doc_id,
                "chunk_index": i,
                "type": "requirement",
                "text": chunk,
            },
        ))

    qdrant.upsert(collection_name=collection, points=points)
    print(f"[embedding-service] wrote {len(points)} vector(s) into '{collection}'")

    producer.send("knowledge.indexed", {
        "eventType": "knowledge.indexed",
        "tenantId": tenant_id,
        "projectId": project_id,
        "documentId": doc_id,
        "chunkCount": len(chunks),
        "correlationId": event.get("correlationId"),
    })
    producer.flush()


def main():
    consumer = KafkaConsumer(
        "document.uploaded",
        bootstrap_servers=os.environ["KAFKA_BROKER"],
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="embedding-service",
        auto_offset_reset="earliest",
    )
    print("[embedding-service] waiting for document.uploaded events...")
    for msg in consumer:
        try:
            handle_document_uploaded(msg.value)
        except Exception as e:
            # Never let one bad document crash the whole consumer.
            print(f"[embedding-service] FAILED on {msg.value.get('documentId')}: {e}")


if __name__ == "__main__":
    main()
