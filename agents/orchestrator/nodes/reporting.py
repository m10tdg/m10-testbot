"""
Enhanced Reporting Agent

Generates professional HTML reports with:
- Execution summary
- Visual diffs with analysis
- Recommendations
- Error context for debugging
- Links to artifacts (screenshots, DOM)
"""

from datetime import datetime
from jinja2 import Template
from clients import s3, ARTIFACTS_BUCKET, pg_conn


REPORT_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>TESTBOT Report - {{ run_id }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container { max-width: 900px; margin: 0 auto; padding: 20px; }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .header h1 { font-size: 28px; margin-bottom: 10px; }
        .meta {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
            font-size: 14px;
        }
        .meta-item { background: rgba(255,255,255,0.1); padding: 10px; border-radius: 4px; }
        .meta-label { font-weight: 600; }
        
        .section {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .section h2 {
            font-size: 18px;
            margin-bottom: 15px;
            color: #667eea;
        }
        
        .status-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 13px;
            margin-right: 10px;
        }
        .status-passed { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }
        .status-warning { background: #fff3cd; color: #856404; }
        .status-info { background: #d1ecf1; color: #0c5460; }
        
        .result-item {
            background: #f9f9f9;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 6px;
            border-left: 3px solid #ddd;
        }
        .result-item.error { border-left-color: #dc3545; }
        .result-item.warning { border-left-color: #ffc107; }
        .result-item.success { border-left-color: #28a745; }
        
        .result-item strong { color: #333; }
        .result-item .details {
            margin-top: 8px;
            font-size: 13px;
            color: #666;
            font-family: "Monaco", "Menlo", "Ubuntu Mono", monospace;
            background: white;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        
        .artifacts {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        .artifact-link {
            display: inline-block;
            padding: 6px 12px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 12px;
            transition: background 0.2s;
        }
        .artifact-link:hover { background: #764ba2; }
        
        .visual-diff {
            background: #f9f9f9;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 6px;
            border-left: 3px solid #17a2b8;
        }
        .visual-diff strong { color: #333; }
        .visual-diff .verdict {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: 600;
            margin-top: 8px;
        }
        .verdict.no-change { background: #d4edda; color: #155724; }
        .verdict.minor { background: #fff3cd; color: #856404; }
        .verdict.significant { background: #f8d7da; color: #721c24; }
        
        .recommendation {
            background: #e8f4f8;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin-top: 15px;
            border-radius: 4px;
        }
        .recommendation strong { color: #17a2b8; }
        
        .footer {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f5f5f5;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>TESTBOT Test Report</h1>
            <div class="meta">
                <div class="meta-item"><span class="meta-label">Run ID:</span> {{ run_id }}</div>
                <div class="meta-item"><span class="meta-label">Status:</span> <span class="status-badge status-{{ status }}">{{ status | upper }}</span></div>
                <div class="meta-item"><span class="meta-label">URL:</span> {{ url }}</div>
                <div class="meta-item"><span class="meta-label">Executed:</span> {{ timestamp }}</div>
            </div>
        </div>
        
        {% if execution_results %}
        <div class="section">
            <h2>Execution Results</h2>
            {% for result in execution_results %}
            <div class="result-item {% if result.passed %}success{% else %}error{% endif %}">
                <strong>{{ result.pageUrl }}</strong>
                <span class="status-badge status-{% if result.passed %}passed{% else %}failed{% endif %}">
                    {{ "✓ PASSED" if result.passed else "✗ FAILED" }}
                </span>
                
                {% if result.error %}
                <div class="details">
                    <strong>Error:</strong> {{ result.error }}
                    {% if result.error_context %}
                    <br><strong>Context:</strong> {{ result.error_context }}
                    {% endif %}
                </div>
                {% endif %}
                
                {% if result.screenshot_s3_path or result.dom_s3_path %}
                <div class="artifacts">
                    {% if result.screenshot_s3_path %}
                    <a href="{{ s3_endpoint }}/{{ s3_bucket }}/{{ result.screenshot_s3_path }}" class="artifact-link" target="_blank">📸 Screenshot</a>
                    {% endif %}
                    {% if result.dom_s3_path %}
                    <a href="{{ s3_endpoint }}/{{ s3_bucket }}/{{ result.dom_s3_path }}" class="artifact-link" target="_blank">🔍 DOM</a>
                    {% endif %}
                </div>
                {% endif %}
                
                {% if result.console_errors %}
                <div class="details" style="margin-top: 10px;">
                    <strong>Console Errors ({{ result.console_errors | length }}):</strong>
                    {% for err in result.console_errors[:3] %}
                    <br>• {{ err }}
                    {% endfor %}
                    {% if result.console_errors | length > 3 %}
                    <br>... and {{ result.console_errors | length - 3 }} more
                    {% endif %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if visual_diffs %}
        <div class="section">
            <h2>Visual Analysis</h2>
            {% for diff in visual_diffs %}
            <div class="visual-diff">
                <strong>{{ diff.page }}</strong>
                <div style="margin-top: 8px; font-size: 13px;">
                    <div>Pixel Difference: <strong>{{ diff.difference_percent }}%</strong></div>
                    <div class="verdict verdict-{{ diff.verdict }}">{{ diff.verdict | replace('_', ' ') | upper }}</div>
                </div>
                
                {% if diff.ai_analysis %}
                <div class="details" style="margin-top: 10px;">
                    <strong>AI Analysis:</strong><br>
                    {{ diff.ai_analysis }}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if recommendation %}
        <div class="recommendation">
            <strong>Recommendation:</strong>
            <p style="margin-top: 8px;">{{ recommendation }}</p>
        </div>
        {% endif %}
        
        <div class="section">
            <h2>Summary</h2>
            <table>
                <tr>
                    <td>Severity Level</td>
                    <td><span class="status-badge status-{{ severity }}">{{ severity | upper }}</span></td>
                </tr>
                <tr>
                    <td>Tests Run</td>
                    <td>{{ execution_results | length }}</td>
                </tr>
                <tr>
                    <td>Passed</td>
                    <td>{{ execution_results | selectattr('passed') | list | length }}</td>
                </tr>
                <tr>
                    <td>Failed</td>
                    <td>{{ execution_results | rejectattr('passed') | list | length }}</td>
                </tr>
                <tr>
                    <td>Visual Diffs</td>
                    <td>{{ visual_diffs | selectattr('needs_review') | list | length }}</td>
                </tr>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by MyTester</p>
            <p>{{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
""")


def reporting_agent_node(state: dict) -> dict:
    """
    Generate a comprehensive HTML report with all findings.
    """
    print(f"\n[reporting-agent] building report for run {state['run_id']}")
    
    # Prepare data for template
    template_data = {
        "run_id": state["run_id"],
        "url": state.get("url", "Unknown"),
        "status": state.get("status", "unknown"),
        "severity": state.get("severity", "info"),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "execution_results": state.get("execution_results", []),
        "visual_diffs": state.get("visual_diffs", []),
        "recommendation": state.get("recommendation", "All checks passed successfully."),
        "s3_endpoint": os.environ.get("S3_ENDPOINT", "http://localhost:4566"),
        "s3_bucket": ARTIFACTS_BUCKET,
    }
    
    # Render HTML
    html = REPORT_TEMPLATE.render(**template_data)
    
    # Save to S3
    tenant_id = state["tenant_id"]
    project_id = state["project_id"]
    run_id = state["run_id"]
    key = f"{tenant_id}/{project_id}/{run_id}/report.html"
    
    try:
        s3.put_object(
            Bucket=ARTIFACTS_BUCKET,
            Key=key,
            Body=html.encode("utf-8"),
            ContentType="text/html"
        )
        report_url = f"{os.environ.get('S3_ENDPOINT', 'http://localhost:4566')}/{ARTIFACTS_BUCKET}/{key}"
        print(f"[reporting-agent] report written to {report_url}")
    except Exception as e:
        print(f"[reporting-agent] error saving report: {e}")
        report_url = None
    
    # Save metadata to PostgreSQL
    try:
        with pg_conn.cursor() as cur:
            # Insert report metadata
            critical_count = 1 if state.get("severity") == "critical" else 0
            warning_count = 1 if state.get("severity") == "warning" else 0
            
            cur.execute("""
                INSERT INTO reports (tenant_id, project_id, run_id, s3_path, format, critical_count, warning_count, created_at)
                VALUES (%s, %s, %s, %s, 'html', %s, %s, NOW())
            """, (tenant_id, project_id, run_id, key, critical_count, warning_count))
            
            # Update run status
            cur.execute("""
                UPDATE test_runs 
                SET status = %s, completed_at = NOW()
                WHERE id = %s
            """, (state.get("status", "completed"), run_id))
            
            pg_conn.commit()
    except Exception as e:
        print(f"[reporting-agent] warning: could not update database: {e}")
    
    return {
        **state,
        "report_url": report_url,
        "status": "completed",
    }


# Need to import os for environment variables
import os