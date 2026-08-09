# agents/orchestrator/main.py
import json, uuid
from kafka import KafkaConsumer, KafkaProducer
from qdrant_client import QdrantClient
from graph import build_graph

qdrant = QdrantClient(url="http://localhost:6333")
graph = build_graph(qdrant)
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def handle_test_requested(event: dict):
    initial_state = {
        "run_id": event["runId"],
        "tenant_id": event["tenantId"],
        "project_id": event["projectId"],
        "url": event["url"],
        "prompt": event["prompt"],
        "correlation_id": event["correlationId"],
        "status": "queued",
    }

    final_state = graph.invoke(initial_state)

    producer.send("report.ready", {
        "eventType": "report.ready",
        "tenantId": final_state["tenant_id"],
        "projectId": final_state["project_id"],
        "runId": final_state["run_id"],
        "reportUrl": final_state.get("report_url"),
        "criticalCount": 1 if final_state.get("severity") == "critical" else 0,
        "warningCount": 1 if final_state.get("severity") == "warning" else 0,
        "correlationId": final_state["correlation_id"],
    })

def main():
    consumer = KafkaConsumer(
        "test.requested",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="orchestrator",
    )
    for msg in consumer:
        handle_test_requested(msg.value)

if __name__ == "__main__":
    main()