"""
TESTBOT Orchestrator v3 - Fixed Self-Healing

Improvements from v2:
1. Proper retry detection (check execution_results, not just status)
2. Full context passing on retry (error + screenshot + DOM)
3. Crawler re-runs on retry to get fresh page state
4. Better logging of retry flow
5. Retry context properly merged into event for next attempt
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


def should_retry(state: dict, attempt: int, max_attempts: int) -> bool:
    """
    Determine if we should retry based on execution results.
    
    Retry if:
    - Execution failed AND we have retries left
    - Status is "failed" OR execution_results show failures
    """
    if attempt >= max_attempts:
        print(f"[orchestrator] max retries reached ({max_attempts})")
        return False
    
    # Check execution results for failures
    execution_results = state.get("execution_results", [])
    if execution_results:
        has_failures = any(not r.get("passed", True) for r in execution_results)
        if has_failures:
            print(f"[orchestrator] execution had failures, eligible for retry")
            return True
    
    # Also check status flag
    if state.get("status") == "failed":
        print(f"[orchestrator] status=failed, eligible for retry")
        return True
    
    return False


def build_retry_context(event: dict, state: dict) -> dict:
    """
    Build enhanced context for retry attempt.
    
    Includes:
    - Previous error message
    - Previous state (for analysis context)
    - Screenshot path (for visual debugging)
    - DOM snapshot (for selector matching)
    - Attempt count
    """
    execution_results = state.get("execution_results", [])
    
    retry_context = {
        "attempt_number": state.get("attempt", 1),
        "previous_error": None,
        "previous_screenshot": None,
        "previous_dom": None,
        "execution_log": [],
    }
    
    # Extract error details from execution results
    if execution_results:
        result = execution_results[0]
        retry_context["previous_error"] = result.get("error")
        retry_context["previous_error_context"] = result.get("error_context")
        retry_context["previous_screenshot"] = result.get("screenshot_s3_path")
        retry_context["previous_dom"] = result.get("dom_s3_path")
        retry_context["console_errors"] = result.get("console_errors", [])
    
    return retry_context


def handle_test_requested(event: dict, attempt: int = 1):
    """
    Process a test.requested event with self-healing retry logic.
    
    - Attempt 1: Full pipeline with fresh crawl
    - Attempt 2-3: Self-healing (regenerate script with error log, re-crawl)
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
        
        # Retry context (if this is a retry)
        "is_retry": attempt > 1,
        "retry_context": event.get("_retry_context"),
    }
    
    try:
        # Execute the full LangGraph pipeline
        print(f"[orchestrator] invoking graph (attempt {attempt}/{max_attempts})...")
        final_state = graph.invoke(initial_state)
        
        # Check if we should retry
        if should_retry(final_state, attempt, max_attempts):
            retry_context = build_retry_context(event, final_state)
            
            print(f"\n[orchestrator] SELF-HEALING TRIGGERED")
            print(f"[orchestrator] Error: {retry_context['previous_error']}")
            print(f"[orchestrator] Regenerating script with error context...")
            print(f"[orchestrator] Re-crawling page to get fresh DOM...\n")
            
            # Merge retry context back into event for next attempt
            event["_retry_context"] = retry_context
            
            # Recursive call for retry
            return handle_test_requested(event, attempt + 1)
        
        # Execution succeeded or max retries reached - publish report
        print(f"\n[orchestrator] PIPELINE COMPLETE (attempt {attempt})")
        print(f"[orchestrator] Status: {final_state.get('status', 'unknown')}")
        print(f"[orchestrator] Severity: {final_state.get('severity', 'N/A')}")
        
        producer.send("report.ready", {
            "eventType": "report.ready",
            "tenantId": final_state["tenant_id"],
            "projectId": final_state["project_id"],
            "runId": final_state["run_id"],
            "reportUrl": final_state.get("report_url"),
            "severity": final_state.get("severity"),
            "correlationId": final_state.get("correlation_id"),
            "attempt": attempt,
            "attempts_total": max_attempts,
        })
        producer.flush()
        
        print(f"[orchestrator] Report: {final_state.get('report_url', 'N/A')}\n")
        
    except Exception as e:
        print(f"\n[orchestrator] ✗ Run {run_id} CRASHED")
        print(f"[orchestrator] Error: {str(e)}")
        print(f"[orchestrator] Attempt: {attempt}/{max_attempts}")
        print(f"[orchestrator] This crash will be logged but NOT auto-retried.\n")
        
        # Still publish a report so the user sees the error
        producer.send("report.ready", {
            "eventType": "report.ready",
            "tenantId": event["tenantId"],
            "projectId": event["projectId"],
            "runId": run_id,
            "reportUrl": None,
            "severity": "critical",
            "error": str(e),
            "error_type": type(e).__name__,
            "correlationId": event.get("correlationId"),
            "attempt": attempt,
        })
        producer.flush()


def main():
    """Listen for test.requested events on Kafka with reconnection."""
    validate_infrastructure()
    
    print("[orchestrator] waiting for test.requested events...\n")
    
    while True:
        consumer = None
        try:
            consumer = KafkaConsumer(
                "test.requested",
                bootstrap_servers=os.environ["KAFKA_BROKER"],
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                group_id="orchestrator",
                auto_offset_reset="earliest",
                session_timeout_ms=10000,
                request_timeout_ms=40000,
                connections_max_idle_ms=540000,
            )
            
            for msg in consumer:
                try:
                    handle_test_requested(msg.value)
                except KeyboardInterrupt:
                    print("\n[orchestrator] Shutting down gracefully...")
                    return
                except Exception as e:
                    print(f"[orchestrator] Error processing message: {e}\n")
                    continue
        
        except ValueError as e:
            if "Invalid file descriptor" in str(e):
                print(f"[orchestrator] Kafka selector error (Windows issue), reconnecting in 5s...")
                import time
                time.sleep(5)
                continue
            else:
                raise
        
        except Exception as e:
            print(f"[orchestrator] Kafka consumer error: {e}")
            print("[orchestrator] Reconnecting in 5s...")
            import time
            time.sleep(5)
            continue
        
        finally:
            if consumer:
                try:
                    consumer.close()
                except:
                    pass


if __name__ == "__main__":
    main()