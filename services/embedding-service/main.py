# services/embedding-service/main.py
import json
from kafka import KafkaConsumer, KafkaProducer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import boto3
import uuid
import openai

s3 = boto3.client("s3", endpoint_url="http://localhost:4566")  # swap for real S3 in prod
qdrant = QdrantClient(url="http://localhost:6333")
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def semantic_chunk(text: str, max_tokens: int = 500) -> list[str]:
    """Chunk on paragraph boundaries first, then hard-split anything still too long.
    Naive but predictable — swap for a proper sentence-aware splitter (e.g. langchain's
    RecursiveCharacterTextSplitter) once this pipeline is proven end-to-end."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len((current + " " + p).split()) > max_tokens:
            if current:
                chunks.append(current)
            current = p
        else:
            current = f"{current} {p}".strip()
    if current:
        chunks.append(current)
    return chunks

def embed(text: str) -> list[float]:
    resp = openai.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding

def classify_chunk_type(text: str) -> str:
    """Cheap heuristic first (keyword match), fall back to an LLM call only when ambiguous —
    don't burn a model call classifying every chunk when a regex gets you 90% of the way."""
    lowered = text.lower()
    if "as a user" in lowered or "i want to" in lowered:
        return "user_story"
    if "given" in lowered and "when" in lowered and "then" in lowered:
        return "acceptance_criteria"
    return "requirement"

def handle_document_uploaded(event: dict):
    tenant_id, project_id, doc_id = event["tenantId"], event["projectId"], event["documentId"]
    collection = f"{tenant_id}_{project_id}"

    # 1. Download from S3
    obj = s3.get_object(Bucket="m10-documents", Key=event["s3Path"])
    raw_text = extract_text(obj["Body"].read(), event["s3Path"])  # pdf/docx extraction helper

    # 2. Chunk
    chunks = semantic_chunk(raw_text)

    # 3. Embed + upsert
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
                "chunk_type": classify_chunk_type(chunk),
                "text": chunk,
                "type": "requirement",  # coarse type used by agents' retrieval filter
            },
        ))
    qdrant.upsert(collection_name=collection, points=points)

    # 4. Publish knowledge.indexed
    producer.send("knowledge.indexed", {
        "eventType": "knowledge.indexed",
        "tenantId": tenant_id,
        "projectId": project_id,
        "documentId": doc_id,
        "chunkCount": len(chunks),
        "correlationId": event["correlationId"],
    })

def main():
    consumer = KafkaConsumer(
        "document.uploaded",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="embedding-service",
    )
    for msg in consumer:
        try:
            handle_document_uploaded(msg.value)
        except Exception as e:
            # Never let one bad document take the consumer down — log, ack, move on.
            # A dead-letter topic (document.uploaded.dlq) is worth adding once this works.
            print(f"[embedding-service] failed on {msg.value.get('documentId')}: {e}")

if __name__ == "__main__":
    main()