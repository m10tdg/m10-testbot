# agents/reporting_agent/node.py
import boto3, json, uuid
from jinja2 import Template

s3 = boto3.client("s3")

REPORT_TEMPLATE = Template("""
<h1>TESTBOT Report — Run {{ run_id }}</h1>
<h2>Summary</h2>
<p>Severity: {{ severity }}</p>
<p>{{ recommendation }}</p>
<h2>Visual Diffs</h2>
<ul>{% for d in visual_diffs %}<li>{{ d.page }}: {{ d.difference_percent }}%</li>{% endfor %}</ul>
""")

def reporting_agent_node(state: dict) -> dict:
    html = REPORT_TEMPLATE.render(**state)
    key = f"{state['tenant_id']}/{state['project_id']}/{state['run_id']}/report.html"
    s3.put_object(Bucket="m10-artifacts", Key=key, Body=html.encode("utf-8"))

    junit_xml = build_junit_xml(state)  # implement per your CI's expected schema
    junit_key = key.replace("report.html", "report.junit.xml")
    s3.put_object(Bucket="m10-artifacts", Key=junit_key, Body=junit_xml.encode("utf-8"))

    return {
        **state,
        "report_url": f"https://m10-artifacts.s3.amazonaws.com/{key}",
        "status": "completed",
    }