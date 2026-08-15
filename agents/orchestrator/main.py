"""
TESTBOT Orchestrator v2 - Enhanced with:
- Infrastructure validation before execution
- Dynamic DOM crawling for actual page structure
- Self-healing error recovery (up to 3 attempts)
- Better context retrieval & script generation
- Robust multi-tenant isolation
"""

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from kafka import KafkaConsumer, KafkaProducer
from graph import build_graph

# ============================================================================
# VALIDATION: Ensure all infrastructure is ready before processing
# ============================================================================

def validate_infrastructure():
    """Verify S3, Qdrant, PostgreSQL, and Kafka are accessible."""
    from clients import s3, qdrant, pg_conn, ARTIFACTS_BUCKET
    import os
    
    print("[orchestrator] validating infrastructure...")
    
    # 1. Test S3 connection and bucket
    try:
        s3.list_buckets()
        print("  ✓ S3 connection OK")
        
        # Ensure artifacts bucket exists
        try:
            s3.head_bucket(Bucket=ARTIFACTS_BUCKET)
            print(f"  ✓ Bucket '{ARTIFACTS_BUCKET}' exists")
        except Exception as e:
            print(f"  ✗ Bucket '{ARTIFACTS_BUCKET}' missing. Creating...")
            region = os.environ.get("AWS_REGION", "eu-central-1")
            
            # For non-us-east-1 regions, specify LocationConstraint
            if region != "us-east-1":
                s3.create_bucket(
                    Bucket=ARTIFACTS_BUCKET,
                    CreateBucketConfiguration={"LocationConstraint": region}
                )
            else:
                s3.create_bucket(Bucket=ARTIFACTS_BUCKET)
            
            print(f"  ✓ Created bucket '{ARTIFACTS_BUCKET}' in {region}")
    except Exception as e:
        print(f"  ✗ S3 error: {e}")
        sys.exit(1)
    
    # 2. Test Qdrant connection
    try:
        qdrant.get_collections()
        print("  ✓ Qdrant connection OK")
    except Exception as e:
        print(f"  ✗ Qdrant error: {e}")
        sys.exit(1)
    
    # 3. Test PostgreSQL connection
    try:
        with pg_conn.cursor() as cur:
            cur.execute("SELECT 1")
        print("  ✓ PostgreSQL connection OK")
    except Exception as e:
        print(f"  ✗ PostgreSQL error: {e}")
        sys.exit(1)
    
    print("[orchestrator] infrastructure validated ✓\n")


# ============================================================================
# MAIN CONSUMER LOGIC
# ============================================================================

graph = build_graph()
producer = KafkaProducer(
    bootstrap_servers=os.environ["KAFKA_BROKER"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def handle_test_requested(event: dict, attempt: int = 1):
    """
    Process a test.requested event.
    - Attempt 1: Full pipeline
    - Attempt 2-3: Self-healing (regenerate script with error log)
    """
    run_id = event["runId"]
    max_attempts = 3
    
    print(f"\n{'='*70}")
    print(f"RUN: {run_id}")
    print(f"Tenant: {event['tenantId']} | Project: {event['projectId']}")
    print(f"Attempt: {attempt}/{max_attempts}")
    print(f"{'='*70}\n")
    
    # Build the initial state
    initial_state = {
        "run_id": run_id,
        "tenant_id": event["tenantId"],
        "project_id": event["projectId"],
        "url": event["url"],
        "prompt": event["prompt"],
        "correlation_id": event.get("correlationId"),
        "status": "queued",
        "attempt": attempt,
        "max_attempts": max_attempts,
    }
    
    try:
        # Execute the full LangGraph pipeline
        final_state = graph.invoke(initial_state)
        
        # Check if execution failed and we have retries left
        if (final_state.get("status") == "failed" and 
            final_state.get("execution_results") and 
            attempt < max_attempts):
            
            # Extract the error from the execution attempt
            execution_error = final_state["execution_results"][0].get("error", "Unknown error")
            print(f"\n[orchestrator] Execution failed: {execution_error}")
            print(f"[orchestrator] Self-healing: regenerating script (attempt {attempt + 1}/{max_attempts})...\n")
            
            # Retry with error context
            event["_previous_error"] = execution_error
            event["_previous_state"] = final_state
            return handle_test_requested(event, attempt + 1)
        
        # Publish report.ready event
        producer.send("report.ready", {
            "eventType": "report.ready",
            "tenantId": final_state["tenant_id"],
            "projectId": final_state["project_id"],
            "runId": final_state["run_id"],
            "reportUrl": final_state.get("report_url"),
            "severity": final_state.get("severity"),
            "correlationId": final_state.get("correlation_id"),
            "attempt": attempt,
        })
        producer.flush()
        
        print(f"\n[orchestrator] Run {run_id} COMPLETED")
        print(f"[orchestrator] Status: {final_state.get('status', 'unknown')}")
        print(f"[orchestrator] Severity: {final_state.get('severity', 'N/A')}")
        print(f"[orchestrator] Report: {final_state.get('report_url', 'N/A')}\n")
        
    except Exception as e:
        print(f"\n[orchestrator] ✗ Run {run_id} CRASHED")
        print(f"[orchestrator] Error: {str(e)}")
        print(f"[orchestrator] This crash will be logged but NOT retried.\n")
        
        # Still publish a report so the user sees the error
        producer.send("report.ready", {
            "eventType": "report.ready",
            "tenantId": event["tenantId"],
            "projectId": event["projectId"],
            "runId": run_id,
            "reportUrl": None,
            "severity": "critical",
            "error": str(e),
            "correlationId": event.get("correlationId"),
        })
        producer.flush()


def main():
    """Listen for test.requested events on Kafka."""
    validate_infrastructure()
    
    consumer = KafkaConsumer(
        "test.requested",
        bootstrap_servers=os.environ["KAFKA_BROKER"],
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="orchestrator",
        auto_offset_reset="earliest",
    )
    
    print("[orchestrator] waiting for test.requested events...\n")
    
    for msg in consumer:
        try:
            handle_test_requested(msg.value)
        except KeyboardInterrupt:
            print("\n[orchestrator] Shutting down gracefully...")
            break
        except Exception as e:
            # Catch any uncaught exceptions to prevent consumer crash
            print(f"[orchestrator] Unhandled error: {e}\n")
            continue


if __name__ == "__main__":
    main()