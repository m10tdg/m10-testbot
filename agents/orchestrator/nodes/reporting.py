from jinja2 import Template
from clients import s3, ARTIFACTS_BUCKET, pg_conn

REPORT_TEMPLATE = Template("""
<!doctype html>
<html>
<head><meta charset="utf-8"><title>TESTBOT Report</title></head>
<body style="font-family: sans-serif; max-width: 700px; margin: 40px auto;">
  <h1>TESTBOT Report</h1>
  <p><strong>Run:</strong> {{ run_id }}</p>
  <p><strong>URL tested:</strong> {{ url }}</p>
  <p><strong>Severity:</strong> {{ severity }}</p>
  <h2>Recommendation</h2>
  <p>{{ recommendation }}</p>
  <h2>Visual Diffs</h2>
  <ul>
    {% for d in visual_diffs %}
    <li>{{ d.page }} — {{ d.difference_percent }}% difference ({{ d.verdict }})</li>
    {% endfor %}
  </ul>
</body>
</html>
""")


def reporting_agent_node(state: dict) -> dict:
    print(f"[reporting-agent] building report for run {state['run_id']}")

    html = REPORT_TEMPLATE.render(
        run_id=state["run_id"],
        url=state["url"],
        severity=state.get("severity", "info"),
        recommendation=state.get("recommendation", "No issues detected."),
        visual_diffs=state.get("visual_diffs") or [],
    )

    key = f"{state['tenant_id']}/{state['project_id']}/{state['run_id']}/report.html"
    s3.put_object(Bucket=ARTIFACTS_BUCKET, Key=key, Body=html.encode("utf-8"), ContentType="text/html")

    report_url = f"{s3.meta.endpoint_url}/{ARTIFACTS_BUCKET}/{key}"
    print(f"[reporting-agent] report written to {report_url}")

    critical_count = 1 if state.get("severity") == "critical" else 0
    warning_count = 1 if state.get("severity") == "warning" else 0

    with pg_conn.cursor() as cur:
        cur.execute(
            """INSERT INTO reports (tenant_id, run_id, s3_path, format, critical_count, warning_count)
               VALUES (%s, %s, %s, 'html', %s, %s)""",
            (state["tenant_id"], state["run_id"], key, critical_count, warning_count),
        )
        cur.execute(
            "UPDATE test_runs SET status = 'completed', completed_at = now() WHERE id = %s",
            (state["run_id"],),
        )

    return {**state, "report_url": report_url, "status": "completed"}
