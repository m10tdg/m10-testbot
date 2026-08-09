"""
orchestrator - THE entry point for the whole AI agent pipeline.

Run it with:  python main.py
(from inside the agents/orchestrator/ folder - see README.md)

It listens for `test.requested` events on Kafka, runs the full LangGraph
pipeline (scenario -> execution -> visual analysis -> analysis -> reporting),
and publishes `report.ready` when done.
"""

import json
import os

from dotenv import load_dotenv
load_dotenv()

from kafka import KafkaConsumer, KafkaProducer
from graph import build_graph

graph = build_graph()

producer = KafkaProducer(
    bootstrap_servers=os.environ["KAFKA_BROKER"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def handle_test_requested(event: dict):
    print(f"\n=== starting run {event['runId']} ===")

    initial_state = {
        "run_id": event["runId"],
        "tenant_id": event["tenantId"],
        "project_id": event["projectId"],
        "url": event["url"],
        "prompt": event["prompt"],
        "correlation_id": event.get("correlationId"),
        "status": "queued",
    }

    final_state = graph.invoke(initial_state)

    producer.send("report.ready", {
        "eventType": "report.ready",
        "tenantId": final_state["tenant_id"],
        "projectId": final_state["project_id"],
        "runId": final_state["run_id"],
        "reportUrl": final_state.get("report_url"),
        "severity": final_state.get("severity"),
        "correlationId": final_state.get("correlation_id"),
    })
    producer.flush()

    print(f"=== run {event['runId']} finished: {final_state['status']} ===\n")


def main():
    consumer = KafkaConsumer(
        "test.requested",
        bootstrap_servers=os.environ["KAFKA_BROKER"],
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="orchestrator",
        auto_offset_reset="earliest",
    )
    print("[orchestrator] waiting for test.requested events...")
    for msg in consumer:
        try:
            handle_test_requested(msg.value)
        except Exception as e:
            print(f"[orchestrator] run {msg.value.get('runId')} crashed: {e}")


if __name__ == "__main__":
    main()
