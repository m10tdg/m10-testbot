# agents/visual_analysis_agent/node.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import boto3

s3 = boto3.client("s3")

def compute_pixel_diff(current_path: str, baseline_path: str) -> dict:
    current = cv2.imread(current_path)
    baseline = cv2.imread(baseline_path)
    if current.shape != baseline.shape:
        current = cv2.resize(current, (baseline.shape[1], baseline.shape[0]))

    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    gray_baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(gray_current, gray_baseline, full=True)
    difference_percent = round((1 - score) * 100, 2)
    return {"difference_percent": difference_percent}

def summarize_screenshot_semantically(screenshot_s3_path: str, dom_summary: dict) -> str:
    """This is the piece that distinguishes TESTBOT from pixel-diff tools per the doc:
    describe the page in plain language using DOM structure, not just report a diff %."""
    from openai import OpenAI
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": f"Summarize this page's key UI components in one sentence, "
                        f"given this DOM structure: {dom_summary}"
        }],
    )
    return completion.choices[0].message.content

def visual_analysis_agent_node(state: dict) -> dict:
    visual_diffs = []
    for page_result in state["execution_results"]:
        current_local = f"/tmp/{state['run_id']}_current.png"
        baseline_key = f"{state['tenant_id']}/{state['project_id']}/baselines/{page_result['pageUrl']}.png"

        s3.download_file("m10-artifacts", page_result["screenshot_s3_path"], current_local)
        try:
            baseline_local = f"/tmp/{state['run_id']}_baseline.png"
            s3.download_file("m10-artifacts", baseline_key, baseline_local)
        except s3.exceptions.ClientError:
            # No baseline yet — this run's screenshot BECOMES the baseline. Not a diff.
            s3.upload_file(current_local, "m10-artifacts", baseline_key)
            continue

        diff = compute_pixel_diff(current_local, baseline_local)
        diff["page"] = page_result["pageUrl"]

        # Only escalate to LLM-based semantic judgement if pixel diff crosses a noise threshold —
        # don't burn a model call on a 0.1% anti-aliasing difference.
        if diff["difference_percent"] > 1.0:
            diff["semantic_summary"] = summarize_screenshot_semantically(
                page_result["screenshot_s3_path"], dom_summary={}
            )
            diff["verdict"] = "needs_review"
        else:
            diff["verdict"] = "expected_change"

        visual_diffs.append(diff)

    return {**state, "visual_diffs": visual_diffs}