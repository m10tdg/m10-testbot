import uuid
from playwright.sync_api import sync_playwright
from clients import s3, ARTIFACTS_BUCKET

# SECURITY NOTE: the script we're exec()-ing here was written by an LLM based on a
# user's prompt - it is effectively untrusted code. Running it directly in this
# process is fine for local development. In production (Phase 7 of the build guide)
# this whole node becomes "launch a throwaway Kubernetes Job, run it there, collect
# the results, destroy the pod" specifically so untrusted code never touches a
# process that has real credentials or access to other tenants' data.


def execution_agent_node(state: dict) -> dict:
    print(f"[execution-agent] running generated script for run {state['run_id']}")

    console_errors: list[str] = []
    passed = True
    error_message = None

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        page.goto(state["url"], wait_until="load")

        try:
            # The generated script body only uses `page` and `console_errors` -
            # this is the contract defined in the Scenario Agent's system prompt.
            exec(state["playwright_script"], {"page": page, "console_errors": console_errors})
        except Exception as e:
            passed = False
            error_message = str(e)
            print(f"[execution-agent] generated script raised an error: {e}")

        screenshot_bytes = page.screenshot(full_page=True)
        dom_html = page.content()
        browser.close()

    run_id, tenant_id, project_id = state["run_id"], state["tenant_id"], state["project_id"]
    screenshot_key = f"{tenant_id}/{project_id}/{run_id}/screenshot.png"
    dom_key = f"{tenant_id}/{project_id}/{run_id}/dom.html"

    s3.put_object(Bucket=ARTIFACTS_BUCKET, Key=screenshot_key, Body=screenshot_bytes, ContentType="image/png")
    s3.put_object(Bucket=ARTIFACTS_BUCKET, Key=dom_key, Body=dom_html.encode("utf-8"), ContentType="text/html")

    execution_results = [{
        "pageUrl": state["url"],
        "passed": passed,
        "error": error_message,
        "consoleErrors": len(console_errors),
        "screenshot_s3_path": screenshot_key,
        "dom_s3_path": dom_key,
    }]
    
    print(f"[execution-agent] execution results: {execution_results}")
    print(f"[execution-agent] done. passed={passed}, consoleErrors={len(console_errors)}")

    return {
        **state,
        "execution_results": execution_results,
        "status": "executing" if passed else "failed",
    }
