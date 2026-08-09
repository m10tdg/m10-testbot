# agents/execution_agent/node.py
import subprocess, json, tempfile, os
import boto3

s3 = boto3.client("s3")

def execution_agent_node(state: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        script_path = os.path.join(tmp, "generated.spec.ts")
        with open(script_path, "w") as f:
            f.write(state["playwright_script"])

        # In production this subprocess call becomes "create an ephemeral EKS pod, run this,
        # collect artifacts, destroy pod" per your doc's Browser Infrastructure section.
        # Locally, run it directly for fast iteration.
        result = subprocess.run(
            ["npx", "playwright", "test", script_path,
             "--reporter=json",
             f"--output={tmp}/artifacts"],
            capture_output=True, text=True, cwd=tmp,
        )

        report = json.loads(result.stdout) if result.stdout else {}
        execution_results = []

        for spec in report.get("suites", []):
            for test in spec.get("specs", []):
                page_result = {
                    "pageUrl": state["url"],
                    "loadTimeMs": test.get("duration", 0),
                    "consoleErrors": 0,   # populate from page.on('console') hook in the script itself
                    "networkFailures": 0,
                    "passed": test.get("ok", False),
                }
                # upload artifacts (screenshot, dom, trace) captured by Playwright to S3
                for artifact_name in ["screenshot.png", "dom.html", "trace.zip"]:
                    local_path = os.path.join(tmp, "artifacts", artifact_name)
                    if os.path.exists(local_path):
                        key = f"{state['tenant_id']}/{state['project_id']}/{state['run_id']}/{artifact_name}"
                        s3.upload_file(local_path, "m10-artifacts", key)
                        if artifact_name == "screenshot.png":
                            page_result["screenshot_s3_path"] = key
                        if artifact_name == "dom.html":
                            page_result["dom_s3_path"] = key

                execution_results.append(page_result)

        return {
            **state,
            "execution_results": execution_results,
            "status": "executing" if result.returncode == 0 else "failed",
            "error": None if result.returncode == 0 else result.stderr,
        }